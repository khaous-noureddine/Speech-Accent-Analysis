"""Jointly train LibriSpeech CTC and L2-ARCTIC prompt-level SupCon."""

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
    collate_adaptation,
)
from accented_asr.adaptation.model import SupConLoss
from accented_asr.adaptation.train import load_manifest
from accented_asr.asr.data import CTCCollator, LibriSpeechDataset
from accented_asr.asr.model import build_asr_model
from accented_asr.asr.train import evaluate, tri_stage_lr_factor
from accented_asr.joint.model import (
    ProjectionHead,
    configure_backbone_trainability,
    contrastive_loss,
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
    l2_parquet: str
    sample_rate: int
    max_librispeech_duration_s: float
    max_l2_duration_s: float
    validate_audio: bool
    num_workers: int
    librispeech_batch_size: int
    prompts_per_batch: int
    speakers_per_prompt: int
    l2_batches_per_cycle: int
    l2_dev_batches: int
    projection_hidden_size: int
    projection_size: int
    temperature: float
    supcon_weight: float
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
        l2_parquet=data["l2_parquet"], sample_rate=data["sample_rate"],
        max_librispeech_duration_s=data["max_librispeech_duration_s"],
        max_l2_duration_s=data["max_l2_duration_s"],
        validate_audio=data["validate_audio"], num_workers=data["num_workers"],
        librispeech_batch_size=sampler["librispeech_batch_size"],
        prompts_per_batch=sampler["prompts_per_batch"],
        speakers_per_prompt=sampler["speakers_per_prompt"],
        l2_batches_per_cycle=sampler["l2_batches_per_cycle"],
        l2_dev_batches=sampler["l2_dev_batches"],
        projection_hidden_size=model["projection_hidden_size"],
        projection_size=model["projection_size"], temperature=model["temperature"],
        supcon_weight=training["supcon_weight"],
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
    if experiment["task"] != "joint_ctc_supcon":
        raise ValueError("Joint runner requires task=joint_ctc_supcon.")
    if config.fold != config.heldout_accent:
        raise ValueError("fold and heldout_accent must match.")
    if evaluation["selection_metric"] != "librispeech_dev_wer":
        raise ValueError("Joint checkpoints must be selected by LibriSpeech dev WER.")
    if config.supcon_weight < 0 or config.temperature <= 0:
        raise ValueError("SupCon weight must be non-negative and temperature positive.")
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


def save_checkpoint(path, model, projection, optimizer, scheduler, metadata) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "projection_state_dict": projection.state_dict(),
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

    fold_dir = (root / config.l2_parquet).parent
    manifest = load_manifest(fold_dir)
    if str(manifest["heldout_l1"]).casefold() != config.heldout_accent.casefold():
        raise ValueError("L2 manifest does not match the held-out accent.")

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

    l2_kwargs = dict(
        parquet_path=root / config.l2_parquet, repository_root=root,
        sample_rate=config.sample_rate, max_audio_len_s=config.max_l2_duration_s,
        manifest_sha256=manifest["sha256"], validate_audio=config.validate_audio,
    )
    l2_train = L2ArcticAdaptationDataset(split="train", **l2_kwargs)
    l2_dev = L2ArcticAdaptationDataset(split="dev", **l2_kwargs)
    l2_train_sampler = PromptBatchSampler(
        l2_train, prompts_per_batch=prompts_per_batch,
        speakers_per_prompt=speakers_per_prompt,
        batches_per_epoch=config.l2_batches_per_cycle, seed=args.seed,
    )
    l2_dev_sampler = PromptBatchSampler(
        l2_dev, prompts_per_batch=prompts_per_batch,
        speakers_per_prompt=speakers_per_prompt,
        batches_per_epoch=1 if args.smoke else config.l2_dev_batches,
        seed=args.seed + 10_000,
    )
    l2_collator = partial(collate_adaptation, tokenizer=None)
    l2_loader = DataLoader(
        l2_train, batch_sampler=l2_train_sampler,
        num_workers=0 if args.smoke else config.num_workers,
        collate_fn=l2_collator, pin_memory=True,
    )
    l2_dev_loader = DataLoader(
        l2_dev, batch_sampler=l2_dev_sampler,
        num_workers=0 if args.smoke else config.num_workers,
        collate_fn=l2_collator, pin_memory=True,
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
    criterion = SupConLoss(config.temperature)
    requested_device = config.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Joint training requests CUDA but it is unavailable.")
    device = torch.device(requested_device)
    model.to(device)
    projection.to(device)
    configure_backbone_trainability(
        model, head_only=True,
        frozen_transformer_layers=config.frozen_transformer_layers,
        freeze_feature_encoder=config.freeze_feature_encoder,
    )
    optimizer = torch.optim.AdamW([
        {"params": model.wav2vec2.parameters(), "lr": config.backbone_lr},
        {"params": model.lm_head.parameters(), "lr": config.head_lr},
        {"params": projection.parameters(), "lr": config.projection_lr},
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
        "split_manifest_sha256": manifest["sha256"], "initialization": transfer,
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
    l2_cycle = 0
    l2_iterator = iter(l2_loader)
    latest_metadata = resolved
    libri_epoch = 0
    while global_step < target_steps:
        libri_epoch += 1
        for libri_batch in libri_loader:
            if global_step >= target_steps:
                break
            model.train()
            projection.train()
            libri_batch = {key: value.to(device) for key, value in libri_batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=config.mixed_precision and device.type == "cuda"
            ):
                ctc_loss = model(**libri_batch).loss
            if not torch.isfinite(ctc_loss):
                raise RuntimeError(f"Non-finite CTC loss at step {global_step}.")
            scaler.scale(ctc_loss).backward()
            supcon_value = 0.0
            joint_active = (
                global_step >= head_warmup_steps and config.supcon_weight > 0
            )
            if joint_active:
                try:
                    l2_batch = next(l2_iterator)
                except StopIteration:
                    l2_cycle += 1
                    l2_train_sampler.set_epoch(l2_cycle)
                    l2_iterator = iter(l2_loader)
                    l2_batch = next(l2_iterator)
                l2_batch = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in l2_batch.items()
                }
                with torch.amp.autocast(
                    "cuda", enabled=config.mixed_precision and device.type == "cuda"
                ):
                    supcon = contrastive_loss(
                        model, projection, criterion, l2_batch
                    )
                    weighted_supcon = config.supcon_weight * supcon
                if not torch.isfinite(supcon):
                    raise RuntimeError(f"Non-finite SupCon loss at step {global_step}.")
                scaler.scale(weighted_supcon).backward()
                supcon_value = float(supcon.detach())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(projection.parameters()),
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
            total_value = float(ctc_loss.detach()) + config.supcon_weight * supcon_value
            if writer:
                writer.add_scalar("train/ctc_loss", float(ctc_loss.detach()), global_step)
                writer.add_scalar("train/supcon_loss", supcon_value, global_step)
                writer.add_scalar("train/total_loss", total_value, global_step)
            if global_step % config.log_every_steps == 0 or global_step == target_steps:
                print(json.dumps({
                    "event": "train_progress", "global_step": global_step,
                    "libri_epoch": libri_epoch, "joint_active": joint_active,
                    "train": {"ctc_loss": float(ctc_loss.detach()),
                              "supcon_loss": supcon_value,
                              "loss": total_value},
                }, sort_keys=True), flush=True)
            if global_step % config.eval_every_steps == 0 or global_step == target_steps:
                dev = evaluate(
                    model, libri_dev_loader, tokenizer, device,
                    config.mixed_precision, max_batches=1 if args.smoke else None,
                )
                l2_dev_loss = evaluate_supcon(
                    model, projection, criterion, l2_dev_loader, device,
                    config.mixed_precision,
                ) if config.supcon_weight > 0 and global_step >= head_warmup_steps else None
                metrics = {
                    "global_step": global_step, "libri_epoch": libri_epoch,
                    "dev": {"librispeech": dev, "l2_supcon_loss": l2_dev_loss},
                }
                with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                print(json.dumps(metrics, sort_keys=True), flush=True)
                latest_metadata = {**resolved, **metrics}
                save_checkpoint(
                    run_dir / "checkpoint_latest.pt", model, projection,
                    optimizer, scheduler, latest_metadata,
                )
                if dev["wer"] < best_wer:
                    best_wer = dev["wer"]
                    save_checkpoint(
                        run_dir / "checkpoint_best.pt", model, projection,
                        optimizer, scheduler, latest_metadata,
                    )
                if writer:
                    writer.add_scalar("dev/librispeech_wer", dev["wer"], global_step)
                    if l2_dev_loss is not None:
                        writer.add_scalar("dev/l2_supcon_loss", l2_dev_loss, global_step)
                    writer.flush()
    save_checkpoint(
        run_dir / "checkpoint_final.pt", model, projection,
        optimizer, scheduler, latest_metadata,
    )
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
