"""Adapt a LibriSpeech ASR checkpoint to leakage-safe L2-ARCTIC folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import Wav2Vec2CTCTokenizer

from accented_asr.adaptation.data import (
    L2ArcticAdaptationDataset,
    PromptBatchSampler,
    WordBatchSampler,
    WordContrastiveDataset,
    collate_auxiliary,
)
from accented_asr.adaptation.model import SupConLoss
from accented_asr.adaptation.train import load_manifest
from accented_asr.asr.model import build_asr_model
from accented_asr.asr.train import evaluate
from accented_asr.joint.model import (
    ProjectionHead,
    accented_ctc_loss,
    configure_backbone_trainability,
    contrastive_loss,
    multidomain_ctc_supcon_losses,
)


OBJECTIVES = {"ctc", "ctc_supcon", "ctc_word_supcon"}


@dataclass(frozen=True)
class SequentialL2Config:
    name: str
    fold: str
    heldout_accent: str
    seeds: tuple[int, ...]
    backbone_name: str
    tokenizer_path: str
    source_checkpoint: str
    parquet_path: str
    contrastive_unit: str
    word_parquet_path: str | None
    sample_rate: int
    max_audio_len_s: float
    max_word_len_s: float | None
    validate_audio: bool
    num_workers: int
    prompts_per_batch: int
    speakers_per_prompt: int
    words_per_batch: int | None
    accents_per_word: int | None
    batches_per_epoch: int
    dev_batch_size: int
    projection_hidden_size: int
    projection_size: int
    temperature: float
    freeze_feature_encoder: bool
    gradient_checkpointing: bool
    warmup_epochs: int
    adaptation_epochs: int
    early_stopping_patience: int
    supcon_weight: float
    supcon_ramp_ratio: float
    backbone_lr: float
    head_lr: float
    projection_lr: float
    weight_decay: float
    lr_warmup_ratio: float
    gradient_clip: float
    mixed_precision: bool
    output_dir: str
    device: str
    log_every_steps: int
    selection_metric: str
    tensorboard: bool
    tensorboard_subdir: str


def load_config(path: Path) -> SequentialL2Config:
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if set(document) != {"experiment", "sequential_l2_adaptation"}:
        raise ValueError(
            "Sequential L2 config requires experiment and sequential_l2_adaptation."
        )
    experiment = document["experiment"]
    section = document["sequential_l2_adaptation"]
    data = section["data"]
    sampler = section["sampler"]
    model = section["model"]
    training = section["training"]
    development = section["development"]
    config = SequentialL2Config(
        name=experiment["name"],
        fold=experiment["fold"],
        heldout_accent=experiment["heldout_accent"],
        seeds=tuple(experiment["seeds"]),
        backbone_name=model["model_name"],
        tokenizer_path=model["tokenizer_path"],
        source_checkpoint=model["source_checkpoint"],
        parquet_path=data["parquet_path"],
        contrastive_unit=data.get("contrastive_unit", "prompt"),
        word_parquet_path=data.get("word_parquet_path"),
        sample_rate=data["sample_rate"],
        max_audio_len_s=data["max_audio_len_s"],
        max_word_len_s=data.get("max_word_len_s"),
        validate_audio=data["validate_audio"],
        num_workers=data["num_workers"],
        prompts_per_batch=sampler["prompts_per_batch"],
        speakers_per_prompt=sampler["speakers_per_prompt"],
        words_per_batch=sampler.get("words_per_batch"),
        accents_per_word=sampler.get("accents_per_word"),
        batches_per_epoch=sampler["batches_per_epoch"],
        dev_batch_size=development["batch_size"],
        projection_hidden_size=model["projection_hidden_size"],
        projection_size=model["projection_size"],
        temperature=model["temperature"],
        freeze_feature_encoder=model["freeze_feature_encoder"],
        gradient_checkpointing=model["gradient_checkpointing"],
        warmup_epochs=training["warmup_epochs"],
        adaptation_epochs=training["adaptation_epochs"],
        early_stopping_patience=training["early_stopping_patience"],
        supcon_weight=training["supcon_weight"],
        supcon_ramp_ratio=training["supcon_ramp_ratio"],
        backbone_lr=training["backbone_lr"],
        head_lr=training["head_lr"],
        projection_lr=training["projection_lr"],
        weight_decay=training["weight_decay"],
        lr_warmup_ratio=training["lr_warmup_ratio"],
        gradient_clip=training["gradient_clip"],
        mixed_precision=training["mixed_precision"],
        output_dir=training["output_dir"],
        device=training["device"],
        log_every_steps=training["log_every_steps"],
        selection_metric=development["selection_metric"],
        tensorboard=training["tensorboard"],
        tensorboard_subdir=training["tensorboard_subdir"],
    )
    if config.fold != config.heldout_accent:
        raise ValueError("fold and heldout_accent must match.")
    if config.contrastive_unit not in {"prompt", "word"}:
        raise ValueError("contrastive_unit must be prompt or word.")
    if config.contrastive_unit == "word" and (
        not config.word_parquet_path
        or config.max_word_len_s is None
        or config.words_per_batch is None
        or config.accents_per_word is None
    ):
        raise ValueError(
            "Word-level sequential adaptation requires word_parquet_path, "
            "max_word_len_s, words_per_batch, and accents_per_word."
        )
    if config.selection_metric != "l2_dev_wer":
        raise ValueError("Sequential adaptation must select on l2_dev_wer.")
    if not config.seeds:
        raise ValueError("At least one seed is required.")
    if min(config.warmup_epochs, config.adaptation_epochs) <= 0:
        raise ValueError("Warm-up and adaptation epochs must be positive.")
    if config.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive.")
    if not 0 < config.supcon_ramp_ratio <= 1:
        raise ValueError("supcon_ramp_ratio must be in (0, 1].")
    if config.supcon_weight < 0 or config.temperature <= 0:
        raise ValueError("SupCon weight must be non-negative and temperature positive.")
    return config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_with_warmup(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps and step <= warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def supcon_weight_at_step(
    step: int, *, maximum: float, ramp_steps: int
) -> float:
    return maximum * min(1.0, step / max(1, ramp_steps))


def save_checkpoint(path: Path, *, model, projection, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "projection_state_dict": projection.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_model_checkpoint(model, path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has no model_state_dict: {path}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint.get("metadata", {})


def build_loaders(config: SequentialL2Config, root: Path, seed: int, smoke: bool):
    parquet = root / config.parquet_path
    manifest = load_manifest(parquet.parent)
    if str(manifest["heldout_l1"]).casefold() != config.fold.casefold():
        raise ValueError("The L2 manifest does not match the held-out fold.")
    kwargs = dict(
        parquet_path=parquet,
        repository_root=root,
        sample_rate=config.sample_rate,
        max_audio_len_s=config.max_audio_len_s,
        manifest_sha256=manifest["sha256"],
        validate_audio=config.validate_audio,
    )
    dev = L2ArcticAdaptationDataset(split="dev", **kwargs)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(root / config.tokenizer_path)
    collator = partial(collate_auxiliary, tokenizer=tokenizer)
    sampler = None
    train_loader = None
    if config.contrastive_unit == "prompt":
        train = L2ArcticAdaptationDataset(split="train", **kwargs)
        sampler = PromptBatchSampler(
            train,
            prompts_per_batch=4 if smoke else config.prompts_per_batch,
            speakers_per_prompt=(
                min(2, config.speakers_per_prompt)
                if smoke
                else config.speakers_per_prompt
            ),
            batches_per_epoch=2 if smoke else config.batches_per_epoch,
            seed=seed,
        )
        train_loader = DataLoader(
            train,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=0 if smoke else config.num_workers,
            pin_memory=True,
        )
    dev_loader = DataLoader(
        dev,
        batch_size=config.dev_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0 if smoke else config.num_workers,
        pin_memory=True,
    )
    word_sampler = None
    word_train_loader = None
    if config.contrastive_unit == "word":
        word_parquet = root / str(config.word_parquet_path)
        report_path = word_parquet.parent / "validation_report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"Missing word-data validation report: {report_path}")
        report = json.loads(report_path.read_text())
        if report.get("status") != "passed":
            raise ValueError(f"Word-data validation did not pass: {report_path}")
        if str(report.get("heldout_accent", "")).casefold() != config.fold.casefold():
            raise ValueError(
                "Word-data held-out accent does not match the sequential fold: "
                f"{report.get('heldout_accent')!r} != {config.fold!r}"
            )
        word_train = WordContrastiveDataset(
            word_parquet,
            split="train",
            repository_root=root,
            sample_rate=config.sample_rate,
            max_audio_len_s=float(config.max_word_len_s),
            validate_audio=config.validate_audio,
        )
        word_sampler = WordBatchSampler(
            word_train,
            words_per_batch=min(2, int(config.words_per_batch)) if smoke else int(config.words_per_batch),
            accents_per_word=min(2, int(config.accents_per_word)) if smoke else int(config.accents_per_word),
            batches_per_epoch=2 if smoke else config.batches_per_epoch,
            seed=seed,
        )
        word_train_loader = DataLoader(
            word_train,
            batch_sampler=word_sampler,
            collate_fn=collator,
            num_workers=0 if smoke else config.num_workers,
            pin_memory=True,
        )
    return (
        manifest,
        sampler,
        train_loader,
        dev_loader,
        tokenizer,
        word_sampler,
        word_train_loader,
    )


def evaluate_l2(model, loader, tokenizer, device, mixed_precision, smoke):
    def translated():
        for index, batch in enumerate(loader):
            if smoke and index >= 1:
                break
            yield {
                "input_values": batch["audio"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["ctc_labels"],
            }

    return evaluate(model, translated(), tokenizer, device, mixed_precision)


def train_warmup(
    config, root, seed, smoke, model, projection, sampler, train_loader,
    dev_loader, tokenizer, device, run_dir, base_metadata,
    word_sampler=None, word_train_loader=None,
):
    configure_backbone_trainability(
        model, head_only=True, frozen_transformer_layers=0,
        freeze_feature_encoder=config.freeze_feature_encoder,
    )
    optimizer = torch.optim.AdamW(
        model.lm_head.parameters(), lr=config.head_lr, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")
    epochs = 1 if smoke else config.warmup_epochs
    global_step = 0
    active_sampler = word_sampler if config.contrastive_unit == "word" else sampler
    active_loader = word_train_loader if config.contrastive_unit == "word" else train_loader
    if active_sampler is None or active_loader is None:
        raise ValueError("Warm-up requires a training sampler and loader.")
    for epoch in range(1, epochs + 1):
        active_sampler.set_epoch(epoch)
        model.train()
        for batch in active_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=config.mixed_precision and device.type == "cuda"):
                ctc = accented_ctc_loss(model, batch)
            scaler.scale(ctc).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.lm_head.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if global_step % config.log_every_steps == 0:
                print(
                    json.dumps(
                        {
                            "event": "warmup_progress",
                            "epoch": epoch,
                            "global_step": global_step,
                            "ctc_loss": float(ctc.detach()),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    dev = evaluate_l2(model, dev_loader, tokenizer, device, config.mixed_precision, smoke)
    metadata = {
        **base_metadata,
        "stage": "warmup",
        "global_step": global_step,
        "dev": dev,
    }
    destination = run_dir / "warmup" / "checkpoint.pt"
    save_checkpoint(destination, model=model, projection=projection, metadata=metadata)
    print(json.dumps({"event": "warmup_complete", "checkpoint": str(destination), "dev": dev}, sort_keys=True), flush=True)


def train_adaptation(config, root, seed, smoke, objective, model, projection, sampler,
                     train_loader, dev_loader, tokenizer, device, run_dir, base_metadata,
                     word_sampler=None, word_train_loader=None):
    warmup = run_dir / "warmup" / "checkpoint.pt"
    if not warmup.is_file():
        raise FileNotFoundError(f"Run the shared warm-up first: {warmup}")
    load_model_checkpoint(model, warmup)
    configure_backbone_trainability(
        model, head_only=False, frozen_transformer_layers=0,
        freeze_feature_encoder=config.freeze_feature_encoder,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.wav2vec2.parameters(), "lr": config.backbone_lr},
            {"params": model.lm_head.parameters(), "lr": config.head_lr},
            {"params": projection.parameters(), "lr": config.projection_lr},
        ],
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98),
    )
    epochs = 1 if smoke else config.adaptation_epochs
    criterion = SupConLoss(config.temperature)
    if objective == "ctc_word_supcon" and config.contrastive_unit != "word":
        raise ValueError("ctc_word_supcon requires contrastive_unit=word.")
    if objective == "ctc_supcon" and config.contrastive_unit != "prompt":
        raise ValueError("ctc_supcon requires contrastive_unit=prompt.")
    if objective == "ctc_word_supcon" and word_train_loader is None:
        raise ValueError("ctc_word_supcon requires a word-level contrastive loader.")
    active_loader = word_train_loader if objective == "ctc_word_supcon" else train_loader
    active_sampler = word_sampler if objective == "ctc_word_supcon" else sampler
    if active_loader is None or active_sampler is None:
        raise ValueError("Adaptation requires a training sampler and loader.")
    total_steps = epochs * len(active_loader)
    lr_warmup_steps = int(config.lr_warmup_ratio * total_steps)
    ramp_steps = int(config.supcon_ramp_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_with_warmup(
            step, warmup_steps=lr_warmup_steps, total_steps=total_steps
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")
    objective_dir = run_dir / objective
    objective_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if config.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(objective_dir / config.tensorboard_subdir)
    best_wer = float("inf")
    stale_epochs = 0
    global_step = 0
    latest_metadata = base_metadata
    for epoch in range(1, epochs + 1):
        active_sampler.set_epoch(epoch)
        model.train()
        projection.train()
        for batch in active_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=config.mixed_precision and device.type == "cuda"):
                if objective == "ctc_word_supcon":
                    ctc = accented_ctc_loss(model, batch)
                    supcon = contrastive_loss(model, projection, criterion, batch)
                else:
                    ctc, supcon = multidomain_ctc_supcon_losses(
                        model, projection, criterion, batch
                    )
                weight = (
                    supcon_weight_at_step(global_step, maximum=config.supcon_weight, ramp_steps=ramp_steps)
                    if objective in {"ctc_supcon", "ctc_word_supcon"} else 0.0
                )
                loss = ctc + weight * supcon
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite sequential adaptation loss at step {global_step}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(projection.parameters()),
                config.gradient_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            if writer:
                writer.add_scalar("train/ctc_loss", float(ctc.detach()), global_step)
                writer.add_scalar("train/supcon_loss", float(supcon.detach()), global_step)
                writer.add_scalar("train/supcon_weight", weight, global_step)
                writer.add_scalar("train/total_loss", float(loss.detach()), global_step)
            if global_step % config.log_every_steps == 0:
                print(json.dumps({"event": "train_progress", "objective": objective, "epoch": epoch, "global_step": global_step, "ctc_loss": float(ctc.detach()), "supcon_loss": float(supcon.detach()), "supcon_weight": weight, "loss": float(loss.detach())}, sort_keys=True), flush=True)
        dev = evaluate_l2(model, dev_loader, tokenizer, device, config.mixed_precision, smoke)
        latest_metadata = {
            **base_metadata,
            "stage": "adaptation",
            "objective": objective,
            "contrastive_unit": config.contrastive_unit,
            "epoch": epoch,
            "global_step": global_step,
            "warmup_checkpoint": str(warmup.relative_to(root)),
            "warmup_checkpoint_sha256": file_sha256(warmup),
            "dev": dev,
        }
        with (objective_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(latest_metadata, sort_keys=True) + "\n")
        save_checkpoint(objective_dir / "checkpoint_latest.pt", model=model, projection=projection, metadata=latest_metadata)
        if dev["wer"] < best_wer:
            best_wer = dev["wer"]
            stale_epochs = 0
            save_checkpoint(objective_dir / "checkpoint_best.pt", model=model, projection=projection, metadata=latest_metadata)
        else:
            stale_epochs += 1
        print(json.dumps({"event": "development", "objective": objective, "epoch": epoch, "global_step": global_step, "dev": dev, "best_wer": best_wer}, sort_keys=True), flush=True)
        if stale_epochs >= config.early_stopping_patience:
            print(json.dumps({"event": "early_stopping", "objective": objective, "epoch": epoch, "patience": config.early_stopping_patience}, sort_keys=True), flush=True)
            break
    save_checkpoint(objective_dir / "checkpoint_final.pt", model=model, projection=projection, metadata=latest_metadata)
    (objective_dir / "_SUCCESS").touch()
    if writer:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--stage", required=True, choices=("warmup", "adapt"))
    parser.add_argument("--objective", choices=sorted(OBJECTIVES))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.stage == "adapt" and args.objective is None:
        parser.error("--objective is required with --stage adapt")
    if args.stage == "warmup" and args.objective is not None:
        parser.error("--objective is only valid with --stage adapt")
    config = load_config(args.config)
    if args.seed not in config.seeds:
        raise ValueError(f"Seed {args.seed} is not declared in {config.seeds}.")
    seed_everything(args.seed)
    root = args.repository_root.resolve()
    source = root / config.source_checkpoint
    if not source.is_file():
        raise FileNotFoundError(source)
    (
        manifest,
        sampler,
        train_loader,
        dev_loader,
        tokenizer,
        word_sampler,
        word_train_loader,
    ) = build_loaders(
        config, root, args.seed, args.smoke
    )
    model, _ = build_asr_model(
        backbone_name=config.backbone_name,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        stage2_checkpoint=None,
        gradient_checkpointing=config.gradient_checkpointing,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_feature_prob=0.008,
        mask_feature_length=64,
        layerdrop=0.1,
        activation_dropout=0.1,
    )
    source_metadata = load_model_checkpoint(model, source)
    projection = ProjectionHead(
        model.config.hidden_size,
        config.projection_hidden_size,
        config.projection_size,
    )
    requested_device = config.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Sequential L2 adaptation requests CUDA but none is available.")
    device = torch.device(requested_device)
    model.to(device)
    projection.to(device)
    run_dir = root / config.output_dir / f"seed={args.seed}"
    if args.smoke:
        run_dir = run_dir / "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    base_metadata = {
        **asdict(config),
        "seed": args.seed,
        "smoke": args.smoke,
        "source_checkpoint_sha256": file_sha256(source),
        "source_checkpoint_metadata": source_metadata,
        "split_manifest_sha256": manifest["sha256"],
    }
    (run_dir / "config.resolved.json").write_text(
        json.dumps(base_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.stage == "warmup":
        train_warmup(
            config, root, args.seed, args.smoke, model, projection, sampler,
            train_loader, dev_loader, tokenizer, device, run_dir, base_metadata,
            word_sampler, word_train_loader,
        )
    else:
        train_adaptation(
            config, root, args.seed, args.smoke, args.objective, model,
            projection, sampler, train_loader, dev_loader, tokenizer, device,
            run_dir, base_metadata, word_sampler, word_train_loader,
        )


if __name__ == "__main__":
    main()
