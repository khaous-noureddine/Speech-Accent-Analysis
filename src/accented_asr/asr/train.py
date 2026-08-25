"""Fine-tune one Stage 2 encoder for CTC ASR on LibriSpeech train-clean-100."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2CTCTokenizer,
    get_linear_schedule_with_warmup,
)

from accented_asr.asr.data import CTCCollator, LibriSpeechDataset
from accented_asr.asr.model import build_asr_model


@dataclass
class TrainConfig:
    experiment_name: str
    objective: str
    fold: str
    seeds: tuple[int, ...]
    train_parquet: str
    dev_parquet: str
    stage2_output_dir: str
    output_dir: str
    backbone_name: str
    tokenizer_path: str
    sample_rate: int
    max_duration_s: float
    validate_audio: bool
    num_workers: int
    batch_size: int
    epochs: int
    max_steps: int | None
    backbone_lr: float
    head_lr: float
    weight_decay: float
    warmup_ratio: float
    gradient_clip: float
    mixed_precision: bool
    gradient_checkpointing: bool
    tensorboard: bool
    tensorboard_subdir: str
    device: str
    eval_every_steps: int


def load_config(path: Path) -> TrainConfig:
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if set(document) != {"experiment", "stage3_finetuning"}:
        raise ValueError("Stage 3 config requires experiment and stage3_finetuning.")
    experiment = document["experiment"]
    stage3 = document["stage3_finetuning"]
    data, model, training, evaluation = (
        stage3["data"], stage3["model"], stage3["training"], stage3["evaluation"]
    )
    config = TrainConfig(
        experiment_name=experiment["name"], objective=experiment["objective"],
        fold=experiment["fold"], seeds=tuple(experiment["seeds"]),
        train_parquet=data["train_parquet"], dev_parquet=data["dev_parquet"],
        sample_rate=data["sample_rate"], max_duration_s=data["max_duration_s"],
        validate_audio=data["validate_audio"], num_workers=data["num_workers"],
        stage2_output_dir=model["stage2_output_dir"],
        backbone_name=model["model_name"], tokenizer_path=model["tokenizer_path"],
        gradient_checkpointing=model["gradient_checkpointing"],
        output_dir=training["output_dir"], batch_size=training["batch_size"],
        epochs=training["epochs"], max_steps=training["max_steps"],
        backbone_lr=training["backbone_lr"], head_lr=training["head_lr"],
        weight_decay=training["weight_decay"], warmup_ratio=training["warmup_ratio"],
        gradient_clip=training["gradient_clip"],
        mixed_precision=training["mixed_precision"], tensorboard=training["tensorboard"],
        tensorboard_subdir=training["tensorboard_subdir"], device=training["device"],
        eval_every_steps=evaluation["eval_every_steps"],
    )
    if experiment["stage"] != 3:
        raise ValueError("This runner only supports stage=3.")
    if config.objective not in {"supcon-only", "supcon-ctc", "ctc-only"}:
        raise ValueError(f"Unknown Stage 2 objective: {config.objective}")
    if evaluation["selection_metric"] != "dev_wer":
        raise ValueError("Stage 3 checkpoints must be selected by dev_wer.")
    if config.eval_every_steps <= 0:
        raise ValueError("eval_every_steps must be positive.")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def word_error_counts(prediction: str, reference: str) -> tuple[int, int]:
    predicted, expected = prediction.lower().split(), reference.lower().split()
    previous = list(range(len(predicted) + 1))
    for row, ref_word in enumerate(expected, start=1):
        current = [row]
        for column, pred_word in enumerate(predicted, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (ref_word != pred_word),
            ))
        previous = current
    return previous[-1], len(expected)


def evaluate(model, loader, tokenizer, device, mixed_precision, max_batches=None):
    model.eval()
    total_loss, errors, words, batches = 0.0, 0, 0, 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=mixed_precision and device.type == "cuda"):
                outputs = model(**batch)
            total_loss += float(outputs.loss)
            predictions = tokenizer.batch_decode(outputs.logits.argmax(dim=-1))
            label_ids = batch["labels"].clone()
            label_ids[label_ids == -100] = tokenizer.pad_token_id
            references = tokenizer.batch_decode(label_ids, group_tokens=False)
            for prediction, reference in zip(predictions, references):
                edit_errors, reference_words = word_error_counts(prediction, reference)
                errors += edit_errors
                words += reference_words
            batches += 1
    return {"loss": total_loss / batches, "wer": errors / words}


def save_checkpoint(path, model, optimizer, scheduler, metadata):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metadata": metadata,
    }, path)


def should_evaluate_step(
    global_step: int, *, interval: int, target_steps: int, smoke: bool
) -> bool:
    if global_step >= target_steps:
        return True
    return not smoke and global_step % interval == 0


def evaluate_and_save(
    *, model, dev_loader, tokenizer, device, mixed_precision, run_dir,
    optimizer, scheduler, resolved, writer, epoch, global_step,
    train_loss, best_wer, smoke,
):
    dev = evaluate(
        model, dev_loader, tokenizer, device, mixed_precision,
        max_batches=1 if smoke else None,
    )
    metrics = {
        "epoch": epoch,
        "global_step": global_step,
        "train": {"loss": train_loss},
        "dev": dev,
    }
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True))
    if writer:
        writer.add_scalar("dev/loss", dev["loss"], global_step)
        writer.add_scalar("dev/wer", dev["wer"], global_step)
        writer.flush()
    metadata = {**resolved, **metrics}
    save_checkpoint(
        run_dir / "checkpoint_latest.pt", model, optimizer, scheduler, metadata
    )
    if dev["wer"] < best_wer:
        best_wer = dev["wer"]
        save_checkpoint(
            run_dir / "checkpoint_best.pt", model, optimizer, scheduler, metadata
        )
    model.train()
    return best_wer, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed not in config.seeds:
        raise ValueError(f"Seed {args.seed} is not declared in {config.seeds}.")
    seed_everything(args.seed)
    root = args.repository_root.resolve()
    run_dir = root / config.output_dir / f"seed={args.seed}"
    stage2_checkpoint = root / config.stage2_output_dir / f"seed={args.seed}/checkpoint_best.pt"
    if args.smoke:
        run_dir = run_dir / "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(root / config.tokenizer_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.backbone_name)
    collator = CTCCollator(feature_extractor, tokenizer, config.sample_rate)
    dataset_kwargs = dict(
        repository_root=root, sample_rate=config.sample_rate,
        max_duration_s=config.max_duration_s, validate_audio=config.validate_audio,
    )
    train_dataset = LibriSpeechDataset(root / config.train_parquet, **dataset_kwargs)
    dev_dataset = LibriSpeechDataset(root / config.dev_parquet, **dataset_kwargs)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator,
        num_workers=0 if args.smoke else config.num_workers, collate_fn=collator,
        pin_memory=True, drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0 if args.smoke else config.num_workers, collate_fn=collator,
        pin_memory=True,
    )
    requested_device = config.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Stage 3 requests CUDA but it is unavailable.")
    device = torch.device(requested_device)
    model, transfer = build_asr_model(
        backbone_name=config.backbone_name, vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id, stage2_checkpoint=stage2_checkpoint,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    expected_transfer = {
        "stage2_seed": args.seed,
        "stage2_fold": config.fold,
        "stage2_backbone_name": config.backbone_name,
    }
    mismatches = {
        key: (transfer.get(key), expected)
        for key, expected in expected_transfer.items()
        if transfer.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Stage 2 checkpoint metadata mismatch: {mismatches}")
    model.to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.wav2vec2.parameters(), "lr": config.backbone_lr},
        {"params": model.lm_head.parameters(), "lr": config.head_lr},
    ], weight_decay=config.weight_decay, betas=(0.9, 0.98))
    configured_steps = config.max_steps or config.epochs * len(train_loader)
    target_steps = min(configured_steps, 2) if args.smoke else configured_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(config.warmup_ratio * target_steps), target_steps
    )
    resolved = {
        **asdict(config), "seed": args.seed, "smoke": args.smoke,
        "stage2_checkpoint": str(stage2_checkpoint.relative_to(root)),
        "stage2_transfer": transfer,
    }
    (run_dir / "config.resolved.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    writer = None
    if config.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(run_dir / config.tensorboard_subdir)
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")
    global_step, best_wer = 0, float("inf")
    loss_since_eval, batches_since_eval = 0.0, 0
    epochs = 1 if args.smoke else config.epochs
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=config.mixed_precision and device.type == "cuda"):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Stage 3 CTC loss at step {global_step}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_was_run = scaler.get_scale() >= scale_before_step
            if not optimizer_was_run:
                print(
                    json.dumps({
                        "event": "optimizer_step_skipped",
                        "reason": "mixed_precision_overflow",
                        "global_step": global_step,
                    }, sort_keys=True)
                )
                continue
            scheduler.step()
            global_step += 1
            loss_since_eval += float(loss.detach())
            batches_since_eval += 1
            if writer:
                writer.add_scalar("train/loss", float(loss.detach()), global_step)
            if should_evaluate_step(
                global_step,
                interval=config.eval_every_steps,
                target_steps=target_steps,
                smoke=args.smoke,
            ):
                best_wer, metadata = evaluate_and_save(
                    model=model, dev_loader=dev_loader, tokenizer=tokenizer,
                    device=device, mixed_precision=config.mixed_precision,
                    run_dir=run_dir, optimizer=optimizer, scheduler=scheduler,
                    resolved=resolved, writer=writer, epoch=epoch,
                    global_step=global_step,
                    train_loss=loss_since_eval / batches_since_eval,
                    best_wer=best_wer, smoke=args.smoke,
                )
                loss_since_eval, batches_since_eval = 0.0, 0
            if global_step >= target_steps:
                break
        if global_step >= target_steps:
            break
    save_checkpoint(run_dir / "checkpoint_final.pt", model, optimizer, scheduler, metadata)
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
