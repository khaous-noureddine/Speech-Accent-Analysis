"""Jointly train LibriSpeech CTC and prompt- or word-level SupCon."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, Wav2Vec2CTCTokenizer

from accented_asr.adaptation.data import (
    L2ArcticAdaptationDataset,
    PromptBatchSampler,
    WordBatchSampler,
    WordContrastiveDataset,
    collate_auxiliary,
)
from accented_asr.adaptation.model import SupConLoss
from accented_asr.adaptation.train import load_manifest
from accented_asr.asr.data import CTCCollator, LibriSpeechDataset
from accented_asr.asr.model import build_asr_model
from accented_asr.asr.train import evaluate, tri_stage_lr_factor
from accented_asr.joint.model import (
    AccentClassifier,
    ProjectionHead,
    accent_classification_loss,
    augmented_view_supcon_loss,
    balanced_shuffled_labels,
    configure_backbone_trainability,
    contrastive_loss,
    multidomain_ctc_supcon_losses,
)


@dataclass(frozen=True)
class JointConfig:
    name: str
    fold: str
    heldout_accent: str
    seeds: tuple[int, ...]
    backbone_name: str
    tokenizer_path: str
    librispeech_train_parquet: str
    librispeech_dev_parquet: str
    contrastive_unit: str
    contrastive_parquet: str
    sample_rate: int
    max_librispeech_duration_s: float
    max_contrastive_duration_s: float
    validate_audio: bool
    num_workers: int
    librispeech_batch_size: int
    prompts_per_batch: int
    speakers_per_prompt: int
    contrastive_batches_per_cycle: int
    contrastive_dev_batches: int
    projection_hidden_size: int
    projection_size: int
    temperature: float
    supcon_weight: float
    auxiliary_objective: str
    auxiliary_weight: float
    classifier_hidden_size: int
    grl_scale: float
    augmentation_noise_std: float
    augmentation_time_mask_ratio: float
    frozen_transformer_layers: int
    freeze_feature_encoder: bool
    head_warmup_epochs: int | None
    head_warmup_steps: int | None
    max_steps: int
    backbone_lr: float
    head_lr: float
    projection_lr: float
    weight_decay: float
    scheduler_warmup_ratio: float
    scheduler_hold_ratio: float
    final_lr_scale: float
    gradient_clip: float
    mixed_precision: bool
    gradient_checkpointing: bool
    output_dir: str
    device: str
    log_every_steps: int
    eval_every_steps: int
    tensorboard: bool
    tensorboard_subdir: str

    @property
    def l2_parquet(self) -> str:
        """Backward-compatible alias used by existing experiment tests."""
        return self.contrastive_parquet

    @property
    def max_l2_duration_s(self) -> float:
        return self.max_contrastive_duration_s

    @property
    def l2_batches_per_cycle(self) -> int:
        return self.contrastive_batches_per_cycle

    @property
    def l2_dev_batches(self) -> int:
        return self.contrastive_dev_batches


def load_config(path: Path) -> JointConfig:
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if set(document) != {"experiment", "joint_training"}:
        raise ValueError("Joint config requires experiment and joint_training.")
    experiment = document["experiment"]
    joint = document["joint_training"]
    data, sampler, model, training, evaluation = (
        joint["data"], joint["sampler"], joint["model"],
        joint["training"], joint["evaluation"],
    )
    config = JointConfig(
        name=experiment["name"], fold=experiment["fold"],
        heldout_accent=experiment["heldout_accent"],
        seeds=tuple(experiment["seeds"]), backbone_name=model["model_name"],
        tokenizer_path=model["tokenizer_path"],
        librispeech_train_parquet=data["librispeech_train_parquet"],
        librispeech_dev_parquet=data["librispeech_dev_parquet"],
        contrastive_unit=data.get("contrastive_unit", "prompt"),
        contrastive_parquet=data.get("contrastive_parquet", data.get("l2_parquet")),
        sample_rate=data["sample_rate"],
        max_librispeech_duration_s=data["max_librispeech_duration_s"],
        max_contrastive_duration_s=data.get(
            "max_contrastive_duration_s", data.get("max_l2_duration_s")
        ),
        validate_audio=data["validate_audio"], num_workers=data["num_workers"],
        librispeech_batch_size=sampler["librispeech_batch_size"],
        prompts_per_batch=sampler["prompts_per_batch"],
        speakers_per_prompt=sampler["speakers_per_prompt"],
        contrastive_batches_per_cycle=sampler.get(
            "contrastive_batches_per_cycle", sampler.get("l2_batches_per_cycle")
        ),
        contrastive_dev_batches=sampler.get(
            "contrastive_dev_batches", sampler.get("l2_dev_batches")
        ),
        projection_hidden_size=model["projection_hidden_size"],
        projection_size=model["projection_size"], temperature=model["temperature"],
        supcon_weight=training["supcon_weight"],
        auxiliary_objective=training.get("auxiliary_objective", "parallel_supcon"),
        auxiliary_weight=training.get("auxiliary_weight", 0.0),
        classifier_hidden_size=model.get("classifier_hidden_size", 512),
        grl_scale=training.get("grl_scale", 1.0),
        augmentation_noise_std=training.get("augmentation_noise_std", 0.005),
        augmentation_time_mask_ratio=training.get("augmentation_time_mask_ratio", 0.05),
        frozen_transformer_layers=model["frozen_transformer_layers"],
        freeze_feature_encoder=model["freeze_feature_encoder"],
        gradient_checkpointing=model["gradient_checkpointing"],
        head_warmup_epochs=training.get("head_warmup_epochs"),
        head_warmup_steps=training.get("head_warmup_steps"),
        max_steps=training["max_steps"], backbone_lr=training["backbone_lr"],
        head_lr=training["head_lr"], projection_lr=training["projection_lr"],
        weight_decay=training["weight_decay"],
        scheduler_warmup_ratio=training["scheduler_warmup_ratio"],
        scheduler_hold_ratio=training["scheduler_hold_ratio"],
        final_lr_scale=training["final_lr_scale"],
        gradient_clip=training["gradient_clip"],
        mixed_precision=training["mixed_precision"],
        output_dir=training["output_dir"], device=training["device"],
        log_every_steps=training["log_every_steps"],
        eval_every_steps=evaluation["eval_every_steps"],
        tensorboard=training["tensorboard"],
        tensorboard_subdir=training["tensorboard_subdir"],
    )
    if experiment["task"] not in {"joint_ctc_supcon", "joint_ctc_auxiliary"}:
        raise ValueError("Joint runner requires a supported joint CTC task.")
    supported = {
        "parallel_supcon", "augmented_view_supcon", "accent_mtl",
        "accent_dat", "multidomain_ctc", "multidomain_ctc_supcon",
        "shuffled_parallel_supcon",
    }
    if config.auxiliary_objective not in supported:
        raise ValueError(f"Unsupported auxiliary objective: {config.auxiliary_objective}")
    if config.contrastive_unit not in {"prompt", "word"}:
        raise ValueError("contrastive_unit must be prompt or word.")
    if config.contrastive_unit == "prompt" and config.fold != config.heldout_accent:
        raise ValueError("fold and heldout_accent must match.")
    if not config.contrastive_parquet or config.max_contrastive_duration_s is None:
        raise ValueError("Joint config requires a contrastive parquet and duration limit.")
    if evaluation["selection_metric"] != "librispeech_dev_wer":
        raise ValueError("Joint checkpoints must be selected by LibriSpeech dev WER.")
    if (
        config.supcon_weight < 0
        or config.auxiliary_weight < 0
        or config.temperature <= 0
    ):
        raise ValueError("Auxiliary weights must be non-negative and temperature positive.")
    if (config.head_warmup_epochs is None) == (config.head_warmup_steps is None):
        raise ValueError(
            "Configure exactly one of head_warmup_epochs or head_warmup_steps."
        )
    if config.head_warmup_epochs is not None and config.head_warmup_epochs <= 0:
        raise ValueError("head_warmup_epochs must be positive.")
    if config.head_warmup_steps is not None and config.head_warmup_steps <= 0:
        raise ValueError("head_warmup_steps must be positive.")
    if config.max_steps <= 0 or config.eval_every_steps <= 0:
        raise ValueError("Training and evaluation step counts must be positive.")
    if config.scheduler_warmup_ratio + config.scheduler_hold_ratio >= 1:
        raise ValueError("Scheduler warmup + hold ratios must be below one.")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    path, model, projection, classifier, optimizer, scheduler, metadata
) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "projection_state_dict": projection.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metadata": metadata,
    }, path)


def evaluate_supcon(model, projection, criterion, loader, device, mixed_precision):
    model.eval()
    projection.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            with torch.amp.autocast(
                "cuda", enabled=mixed_precision and device.type == "cuda"
            ):
                loss = contrastive_loss(model, projection, criterion, batch)
            total += float(loss)
    return total / len(loader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed not in config.seeds:
        raise ValueError(f"Seed {args.seed} is not declared in {config.seeds}.")
    seed_everything(args.seed)
    root = args.repository_root.resolve()
    run_dir = root / config.output_dir / f"seed={args.seed}"
    if args.smoke:
        run_dir /= "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    libri_batch_size = min(config.librispeech_batch_size, 2) if args.smoke else config.librispeech_batch_size
    prompts_per_batch = min(config.prompts_per_batch, 2) if args.smoke else config.prompts_per_batch
    speakers_per_prompt = min(config.speakers_per_prompt, 2) if args.smoke else config.speakers_per_prompt

    contrastive_dir = (root / config.contrastive_parquet).parent
    if config.contrastive_unit == "prompt":
        split_metadata = load_manifest(contrastive_dir)
        resolved_heldout_accent = str(split_metadata["heldout_l1"]).casefold()
        if resolved_heldout_accent != config.heldout_accent.casefold():
            raise ValueError("L2 manifest does not match the held-out accent.")
    else:
        report_path = contrastive_dir / "validation_report.json"
        split_metadata = json.loads(report_path.read_text(encoding="utf-8"))
        if split_metadata.get("status") != "passed":
            raise ValueError(f"Word split validation did not pass: {report_path}")
        resolved_heldout_accent = str(split_metadata["heldout_accent"]).casefold()
        if config.heldout_accent not in {"from_report", resolved_heldout_accent}:
            raise ValueError("Word manifest does not match the configured held-out accent.")

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(root / config.tokenizer_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.backbone_name)
    libri_kwargs = dict(
        repository_root=root, sample_rate=config.sample_rate,
        max_duration_s=config.max_librispeech_duration_s,
        validate_audio=config.validate_audio,
    )
    libri_train = LibriSpeechDataset(
        root / config.librispeech_train_parquet, **libri_kwargs
    )
    libri_dev = LibriSpeechDataset(
        root / config.librispeech_dev_parquet, **libri_kwargs
    )
    libri_collator = CTCCollator(feature_extractor, tokenizer, config.sample_rate)
    generator = torch.Generator().manual_seed(args.seed)
    libri_loader = DataLoader(
        libri_train, batch_size=libri_batch_size, shuffle=True,
        generator=generator, num_workers=0 if args.smoke else config.num_workers,
        collate_fn=libri_collator, pin_memory=True, drop_last=True,
    )
    libri_dev_loader = DataLoader(
        libri_dev, batch_size=libri_batch_size, shuffle=False,
        num_workers=0 if args.smoke else config.num_workers,
        collate_fn=libri_collator, pin_memory=True,
    )

    contrastive_kwargs = dict(
        parquet_path=root / config.contrastive_parquet, repository_root=root,
        sample_rate=config.sample_rate,
        max_audio_len_s=config.max_contrastive_duration_s,
        validate_audio=config.validate_audio,
    )
    if config.contrastive_unit == "prompt":
        prompt_kwargs = {
            **contrastive_kwargs, "manifest_sha256": split_metadata["sha256"]
        }
        contrastive_train = L2ArcticAdaptationDataset(split="train", **prompt_kwargs)
        contrastive_dev = L2ArcticAdaptationDataset(split="dev", **prompt_kwargs)
        contrastive_train_sampler = PromptBatchSampler(
            contrastive_train, prompts_per_batch=prompts_per_batch,
            speakers_per_prompt=speakers_per_prompt,
            batches_per_epoch=config.contrastive_batches_per_cycle, seed=args.seed,
        )
        contrastive_dev_sampler = PromptBatchSampler(
            contrastive_dev, prompts_per_batch=prompts_per_batch,
            speakers_per_prompt=speakers_per_prompt,
            batches_per_epoch=1 if args.smoke else config.contrastive_dev_batches,
            seed=args.seed + 10_000,
        )
    else:
        contrastive_train = WordContrastiveDataset(split="train", **contrastive_kwargs)
        contrastive_dev = WordContrastiveDataset(split="dev", **contrastive_kwargs)
        contrastive_train_sampler = WordBatchSampler(
            contrastive_train, words_per_batch=prompts_per_batch,
            accents_per_word=speakers_per_prompt,
            batches_per_epoch=config.contrastive_batches_per_cycle, seed=args.seed,
        )
        contrastive_dev_sampler = WordBatchSampler(
            contrastive_dev, words_per_batch=prompts_per_batch,
            accents_per_word=speakers_per_prompt,
            batches_per_epoch=1 if args.smoke else config.contrastive_dev_batches,
            seed=args.seed + 10_000,
        )
    accent_names = sorted(
        set(
            contrastive_train.frame[
                "native_language" if config.contrastive_unit == "prompt" else "accent"
            ].astype(str)
        )
    )
    accent_to_id = {name: index for index, name in enumerate(accent_names)}

    contrastive_collator = partial(
        collate_auxiliary,
        tokenizer=(
            tokenizer
            if config.auxiliary_objective in {
                "multidomain_ctc", "multidomain_ctc_supcon"
            }
            else None
        ),
        accent_to_id=accent_to_id,
    )
    contrastive_loader = DataLoader(
        contrastive_train, batch_sampler=contrastive_train_sampler,
        num_workers=0 if args.smoke else config.num_workers,
        collate_fn=contrastive_collator, pin_memory=True,
    )
    contrastive_dev_loader = DataLoader(
        contrastive_dev, batch_sampler=contrastive_dev_sampler,
        num_workers=0 if args.smoke else config.num_workers,
        collate_fn=contrastive_collator, pin_memory=True,
    )

    model, transfer = build_asr_model(
        backbone_name=config.backbone_name, vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id, stage2_checkpoint=None,
        gradient_checkpointing=config.gradient_checkpointing,
        mask_time_prob=0.05, mask_time_length=10,
        mask_feature_prob=0.008, mask_feature_length=64,
        layerdrop=0.1, activation_dropout=0.1,
    )
    projection = ProjectionHead(
        model.config.hidden_size, config.projection_hidden_size,
        config.projection_size,
    )
    classifier = AccentClassifier(
        model.config.hidden_size, config.classifier_hidden_size, len(accent_names)
    )
    criterion = SupConLoss(config.temperature)
    requested_device = config.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Joint training requests CUDA but it is unavailable.")
    device = torch.device(requested_device)
    model.to(device)
    projection.to(device)
    classifier.to(device)
    configure_backbone_trainability(
        model, head_only=True,
        frozen_transformer_layers=config.frozen_transformer_layers,
        freeze_feature_encoder=config.freeze_feature_encoder,
    )
    optimizer = torch.optim.AdamW([
        {"params": model.wav2vec2.parameters(), "lr": config.backbone_lr},
        {"params": model.lm_head.parameters(), "lr": config.head_lr},
        {"params": projection.parameters(), "lr": config.projection_lr},
        {"params": classifier.parameters(), "lr": config.projection_lr},
    ], weight_decay=config.weight_decay, betas=(0.9, 0.98))
    target_steps = min(config.max_steps, 2) if args.smoke else config.max_steps
    if args.smoke:
        head_warmup_steps = 1
    elif config.head_warmup_steps is not None:
        head_warmup_steps = config.head_warmup_steps
    else:
        head_warmup_steps = config.head_warmup_epochs * len(libri_loader)
    if head_warmup_steps >= target_steps:
        raise ValueError("The head-only warm-up must be shorter than the training run.")
    scheduler_warmup_steps = int(config.scheduler_warmup_ratio * target_steps)
    scheduler_hold_steps = int(config.scheduler_hold_ratio * target_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: tri_stage_lr_factor(
            step, warmup_steps=scheduler_warmup_steps,
            hold_steps=scheduler_hold_steps, total_steps=target_steps,
            final_lr_scale=config.final_lr_scale,
        ),
    )
    resolved = {
        **asdict(config), "seed": args.seed, "smoke": args.smoke,
        "effective_librispeech_batch_size": libri_batch_size,
        "effective_prompts_per_batch": prompts_per_batch,
        "effective_speakers_per_prompt": speakers_per_prompt,
        "head_warmup_steps": head_warmup_steps,
        "resolved_heldout_accent": resolved_heldout_accent,
        "split_manifest_sha256": split_metadata.get("sha256"),
        "initialization": transfer,
        "accent_labels": accent_to_id,
    }
    (run_dir / "config.resolved.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    writer = None
    if config.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(run_dir / config.tensorboard_subdir)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.mixed_precision and device.type == "cuda"
    )
    global_step = 0
    best_wer = float("inf")
    contrastive_cycle = 0
    contrastive_iterator = iter(contrastive_loader)
    latest_metadata = resolved
    libri_epoch = 0
    while global_step < target_steps:
        libri_epoch += 1
        for libri_batch in libri_loader:
            if global_step >= target_steps:
                break
            model.train()
            projection.train()
            classifier.train()
            libri_batch = {key: value.to(device) for key, value in libri_batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=config.mixed_precision and device.type == "cuda"
            ):
                ctc_loss = model(**libri_batch).loss
            if not torch.isfinite(ctc_loss):
                raise RuntimeError(f"Non-finite CTC loss at step {global_step}.")
            scaler.scale(ctc_loss).backward()
            auxiliary_value = 0.0
            weighted_auxiliary_value = 0.0
            supcon_weight = (
                config.supcon_weight
                if config.auxiliary_objective in {
                    "parallel_supcon", "multidomain_ctc_supcon",
                    "shuffled_parallel_supcon",
                }
                else 0.0
            )
            other_auxiliary_weight = (
                0.0
                if config.auxiliary_objective == "parallel_supcon"
                else config.auxiliary_weight
            )
            joint_active = (
                global_step >= head_warmup_steps
                and (supcon_weight > 0 or other_auxiliary_weight > 0)
            )
            supcon_value = 0.0
            accented_ctc_value = 0.0
            if joint_active:
                try:
                    contrastive_batch = next(contrastive_iterator)
                except StopIteration:
                    contrastive_cycle += 1
                    contrastive_train_sampler.set_epoch(contrastive_cycle)
                    contrastive_iterator = iter(contrastive_loader)
                    contrastive_batch = next(contrastive_iterator)
                contrastive_batch = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in contrastive_batch.items()
                }
                with torch.amp.autocast(
                    "cuda", enabled=config.mixed_precision and device.type == "cuda"
                ):
                    if config.auxiliary_objective in {
                        "parallel_supcon", "shuffled_parallel_supcon"
                    }:
                        if config.auxiliary_objective == "shuffled_parallel_supcon":
                            contrastive_batch = {
                                **contrastive_batch,
                                "labels": balanced_shuffled_labels(
                                    contrastive_batch["labels"],
                                    seed=args.seed * 1_000_003 + global_step,
                                ),
                            }
                        auxiliary_loss = contrastive_loss(
                            model, projection, criterion, contrastive_batch
                        )
                        weighted_auxiliary = supcon_weight * auxiliary_loss
                        supcon_value = float(auxiliary_loss.detach())
                    elif config.auxiliary_objective == "multidomain_ctc_supcon":
                        accented_ctc_loss, auxiliary_loss = (
                            multidomain_ctc_supcon_losses(
                                model, projection, criterion, contrastive_batch
                            )
                        )
                        weighted_auxiliary = (
                            other_auxiliary_weight * accented_ctc_loss
                            + supcon_weight * auxiliary_loss
                        )
                        accented_ctc_value = float(accented_ctc_loss.detach())
                        supcon_value = float(auxiliary_loss.detach())
                    elif config.auxiliary_objective == "augmented_view_supcon":
                        auxiliary_loss = augmented_view_supcon_loss(
                            model, projection, criterion, contrastive_batch,
                            noise_std=config.augmentation_noise_std,
                            time_mask_ratio=config.augmentation_time_mask_ratio,
                        )
                        weighted_auxiliary = other_auxiliary_weight * auxiliary_loss
                    elif config.auxiliary_objective in {"accent_mtl", "accent_dat"}:
                        auxiliary_loss = accent_classification_loss(
                            model, classifier, contrastive_batch,
                            adversarial_scale=(
                                config.grl_scale
                                if config.auxiliary_objective == "accent_dat"
                                else None
                            ),
                        )
                        weighted_auxiliary = other_auxiliary_weight * auxiliary_loss
                    else:
                        auxiliary_loss = model(
                            input_values=contrastive_batch["audio"],
                            attention_mask=contrastive_batch["attention_mask"],
                            labels=contrastive_batch["ctc_labels"],
                        ).loss
                        weighted_auxiliary = other_auxiliary_weight * auxiliary_loss
                if not torch.isfinite(auxiliary_loss):
                    raise RuntimeError(f"Non-finite auxiliary loss at step {global_step}.")
                if (
                    config.auxiliary_objective == "multidomain_ctc_supcon"
                    and not torch.isfinite(accented_ctc_loss)
                ):
                    raise RuntimeError(
                        f"Non-finite accented CTC loss at step {global_step}."
                    )
                scaler.scale(weighted_auxiliary).backward()
                auxiliary_value = float(auxiliary_loss.detach())
                weighted_auxiliary_value = float(weighted_auxiliary.detach())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters())
                + list(projection.parameters())
                + list(classifier.parameters()),
                config.gradient_clip,
            )
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                print(json.dumps({
                    "event": "optimizer_step_skipped", "global_step": global_step,
                    "reason": "mixed_precision_overflow",
                }), flush=True)
                continue
            scheduler.step()
            global_step += 1
            if global_step == head_warmup_steps:
                configure_backbone_trainability(
                    model, head_only=False,
                    frozen_transformer_layers=config.frozen_transformer_layers,
                    freeze_feature_encoder=config.freeze_feature_encoder,
                )
                print(json.dumps({
                    "event": "joint_training_enabled", "global_step": global_step,
                    "frozen_transformer_layers": config.frozen_transformer_layers,
                    "freeze_feature_encoder": config.freeze_feature_encoder,
                }, sort_keys=True), flush=True)
            total_value = float(ctc_loss.detach()) + weighted_auxiliary_value
            if writer:
                writer.add_scalar("train/ctc_loss", float(ctc_loss.detach()), global_step)
                writer.add_scalar("train/auxiliary_loss", auxiliary_value, global_step)
                if config.auxiliary_objective == "multidomain_ctc_supcon":
                    writer.add_scalar(
                        "train/accented_ctc_loss", accented_ctc_value, global_step
                    )
                    writer.add_scalar("train/supcon_loss", supcon_value, global_step)
                writer.add_scalar("train/total_loss", total_value, global_step)
            if global_step % config.log_every_steps == 0 or global_step == target_steps:
                print(json.dumps({
                    "event": "train_progress", "global_step": global_step,
                    "libri_epoch": libri_epoch, "joint_active": joint_active,
                    "train": {"ctc_loss": float(ctc_loss.detach()),
                              "auxiliary_loss": auxiliary_value,
                              "accented_ctc_loss": accented_ctc_value,
                              "supcon_loss": supcon_value,
                              "loss": total_value},
                }, sort_keys=True), flush=True)
            if global_step % config.eval_every_steps == 0 or global_step == target_steps:
                dev = evaluate(
                    model, libri_dev_loader, tokenizer, device,
                    config.mixed_precision, max_batches=1 if args.smoke else None,
                )
                contrastive_dev_loss = None
                contrastive_metric = f"{config.auxiliary_objective}_loss"
                if joint_active and config.auxiliary_objective in {
                    "parallel_supcon", "multidomain_ctc_supcon",
                    "shuffled_parallel_supcon",
                }:
                    contrastive_dev_loss = evaluate_supcon(
                        model, projection, criterion, contrastive_dev_loader, device,
                        config.mixed_precision,
                    )
                metrics = {
                    "global_step": global_step, "libri_epoch": libri_epoch,
                    "dev": {
                        "librispeech": dev,
                        contrastive_metric: contrastive_dev_loss,
                    },
                }
                with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                print(json.dumps(metrics, sort_keys=True), flush=True)
                latest_metadata = {**resolved, **metrics}
                save_checkpoint(
                    run_dir / "checkpoint_latest.pt", model, projection, classifier,
                    optimizer, scheduler, latest_metadata,
                )
                if dev["wer"] < best_wer:
                    best_wer = dev["wer"]
                    save_checkpoint(
                        run_dir / "checkpoint_best.pt", model, projection, classifier,
                        optimizer, scheduler, latest_metadata,
                    )
                if writer:
                    writer.add_scalar("dev/librispeech_wer", dev["wer"], global_step)
                    if contrastive_dev_loss is not None:
                        writer.add_scalar(
                            f"dev/{contrastive_metric}", contrastive_dev_loss, global_step
                        )
                    writer.flush()
    save_checkpoint(
        run_dir / "checkpoint_final.pt", model, projection, classifier,
        optimizer, scheduler, latest_metadata,
    )
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
