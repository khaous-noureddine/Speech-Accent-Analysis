"""Train one manifest-bound L2-ARCTIC adaptation run."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import Wav2Vec2CTCTokenizer, get_linear_schedule_with_warmup

from accented_asr.adaptation.data import (
    L2ArcticAdaptationDataset,
    PromptBatchSampler,
    collate_adaptation,
)
from accented_asr.adaptation.model import AdaptationModel
from accented_asr.data.l2_arctic_splits import validate_manifest


CONDITION_TO_MODE = {"A": "supcon_ctc", "E": "supcon_only", "F": "ctc_only"}
FOLDS = {"arabic", "chinese", "hindi", "korean", "spanish", "vietnamese"}


@dataclass
class TrainConfig:
    backbone_name: str = "facebook/wav2vec2-base"
    data_root: str = "data/processed/l2_arctic_leave_one_accent_out"
    output_root: str = "outputs/resubmission/stage2"
    tokenizer_path: str = "configs/tokenizers/librispeech_char"
    sample_rate: int = 16_000
    max_audio_len_s: float = 10.0
    prompts_per_batch: int = 8
    speakers_per_prompt: int = 5
    batches_per_epoch: int = 100
    dev_prompts_per_batch: int = 8
    dev_speakers_per_prompt: int = 5
    dev_batches: int = 12
    epochs: int = 50
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    temperature: float = 0.1
    ctc_weight: float = 0.1
    projection_hidden_size: int = 512
    projection_size: int = 256
    frozen_transformer_layers: int = 9
    num_workers: int = 2
    mixed_precision: bool = True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: Path) -> TrainConfig:
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(values) - set(TrainConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    return TrainConfig(**values)


def load_manifest(fold_dir: Path) -> dict:
    path = fold_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    parquet_hashes = set(
        pd.read_parquet(fold_dir / "corpus.parquet", columns=["split_manifest_sha256"])[
            "split_manifest_sha256"
        ].astype(str)
    )
    if parquet_hashes != {manifest["sha256"]}:
        raise ValueError("The fold Parquet does not match its manifest SHA-256.")
    return manifest


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_epoch(
    *, model, loader, sampler, device, mode, ctc_weight, optimizer=None,
    scheduler=None, mixed_precision=False, gradient_clip=1.0, epoch=0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sampler.set_epoch(epoch)
    totals = {"loss": 0.0, "supcon_loss": 0.0, "ctc_loss": 0.0}
    scaler = torch.amp.GradScaler(
        "cuda", enabled=training and mixed_precision and device.type == "cuda"
    )
    for batch in loader:
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(
            "cuda", enabled=mixed_precision and device.type == "cuda"
        ):
            outputs = model(batch["audio"], batch["attention_mask"])
            losses = model.compute_losses(batch, outputs, mode=mode, ctc_weight=ctc_weight)
        if not torch.isfinite(losses["loss"]):
            raise RuntimeError(f"Non-finite {mode} loss: {losses}")
        if training:
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        for key in totals:
            totals[key] += float(losses[key].detach())
    return {key: value / len(loader) for key, value in totals.items()}


def save_checkpoint(path: Path, *, model, optimizer, scheduler, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", choices=sorted(CONDITION_TO_MODE), required=True)
    parser.add_argument("--fold", choices=sorted(FOLDS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.smoke:
        config.epochs = 1
        config.batches_per_epoch = 2
        config.dev_batches = 1
        config.prompts_per_batch = 4
        config.dev_prompts_per_batch = 4
        config.max_audio_len_s = min(config.max_audio_len_s, 6.0)
        config.num_workers = 0

    seed_everything(args.seed)
    repository_root = args.repository_root.resolve()
    fold_dir = repository_root / config.data_root / args.fold
    manifest = load_manifest(fold_dir)
    if manifest["heldout_l1"].lower() != args.fold:
        raise ValueError(f"Fold {args.fold} contains held-out L1 {manifest['heldout_l1']}.")
    mode = CONDITION_TO_MODE[args.condition]
    tokenizer = None
    if mode != "supcon_only":
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            repository_root / config.tokenizer_path
        )
        if len(tokenizer) != 32 or tokenizer.pad_token_id != 0:
            raise ValueError("The fixed CTC tokenizer must have 32 tokens and blank ID 0.")

    dataset_args = {
        "parquet_path": fold_dir / "corpus.parquet",
        "repository_root": repository_root,
        "sample_rate": config.sample_rate,
        "max_audio_len_s": config.max_audio_len_s,
        "manifest_sha256": manifest["sha256"],
    }
    train_dataset = L2ArcticAdaptationDataset(**dataset_args, split="train")
    dev_dataset = L2ArcticAdaptationDataset(**dataset_args, split="dev")
    train_sampler = PromptBatchSampler(
        train_dataset,
        prompts_per_batch=config.prompts_per_batch,
        speakers_per_prompt=config.speakers_per_prompt,
        batches_per_epoch=config.batches_per_epoch,
        seed=args.seed,
    )
    dev_sampler = PromptBatchSampler(
        dev_dataset,
        prompts_per_batch=config.dev_prompts_per_batch,
        speakers_per_prompt=config.dev_speakers_per_prompt,
        batches_per_epoch=config.dev_batches,
        seed=args.seed + 10_000,
    )
    collate = partial(collate_adaptation, tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler, collate_fn=collate,
        num_workers=config.num_workers, pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_sampler=dev_sampler, collate_fn=collate,
        num_workers=config.num_workers, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptationModel(
        backbone_name=config.backbone_name,
        vocab_size=32,
        projection_hidden_size=config.projection_hidden_size,
        projection_size=config.projection_size,
        temperature=config.temperature,
        frozen_transformer_layers=config.frozen_transformer_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = config.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(config.warmup_steps, total_steps),
        num_training_steps=total_steps,
    )
    run_dir = repository_root / config.output_root / args.condition / args.fold / f"seed={args.seed}"
    if args.smoke:
        run_dir = run_dir / "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        **asdict(config), "condition": args.condition, "loss_mode": mode,
        "fold": args.fold, "seed": args.seed, "heldout_l1": manifest["heldout_l1"],
        "split_manifest_sha256": manifest["sha256"], "smoke": args.smoke,
    }
    (run_dir / "config.resolved.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    best_loss = float("inf")
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model=model, loader=train_loader, sampler=train_sampler, device=device,
            mode=mode, ctc_weight=config.ctc_weight, optimizer=optimizer,
            scheduler=scheduler, mixed_precision=config.mixed_precision,
            gradient_clip=config.gradient_clip, epoch=epoch,
        )
        with torch.no_grad():
            dev_metrics = run_epoch(
                model=model, loader=dev_loader, sampler=dev_sampler, device=device,
                mode=mode, ctc_weight=config.ctc_weight,
                mixed_precision=config.mixed_precision, epoch=epoch,
            )
        metrics = {"epoch": epoch, "train": train_metrics, "dev": dev_metrics}
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(json.dumps(metrics, sort_keys=True))
        metadata = {**resolved, **metrics, "global_step": epoch * len(train_loader)}
        save_checkpoint(
            run_dir / "checkpoint_final.pt", model=model, optimizer=optimizer,
            scheduler=scheduler, metadata=metadata,
        )
        if dev_metrics["loss"] < best_loss:
            best_loss = dev_metrics["loss"]
            save_checkpoint(
                run_dir / "checkpoint_best.pt", model=model, optimizer=optimizer,
                scheduler=scheduler, metadata=metadata,
            )


if __name__ == "__main__":
    main()
