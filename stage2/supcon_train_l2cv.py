"""
stage2/supcon_train_l2cv.py

Stage 2 supervised contrastive training on L2-ARCTIC CV only.

Training:
  split=train

Monitoring:
  split=dev by default

Test split:
  never used here

Robustness:
  corrupted / unreadable audio files are filtered by SupConL2ArcticCVDataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import Wav2Vec2CTCTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent))

from stage2.supcon_data_l2cv import (
    SupConL2ArcticCVDataset,
    SupConL2CVBatchSampler,
    collate_supcon_l2cv,
    collate_supcon_l2cv_with_tokenizer,
)
from stage2.supcon_xlsr import SupConXLSR
from stage2.supcon_evaluate import run_eval


def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() == "true":
        return True
    if str(v).lower() == "false":
        return False
    raise argparse.ArgumentTypeError("Expected boolean value: true or false.")


def to_jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float):
        if np.isnan(x):
            return None
        return x
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    return x


def save_checkpoint(
    model: SupConXLSR,
    optimizer: torch.optim.Optimizer,
    scheduler,
    save_dir: Path,
    epoch: int,
    global_step: int,
    train_metrics: dict,
    eval_metrics: dict | None,
    args,
    name: str | None = None,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = save_dir / (name if name is not None else f"checkpoint_epoch{epoch:03d}.pt")

    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_metrics": to_jsonable(train_metrics),
            "eval_metrics": to_jsonable(eval_metrics),
            "args": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in vars(args).items()
            },
        },
        ckpt_path,
    )

    logger.info(f"Checkpoint saved → {ckpt_path}")
    return ckpt_path


def get_metric(eval_result, metric_name: str) -> float:
    if eval_result is None:
        return float("nan")

    if not hasattr(eval_result, metric_name):
        raise ValueError(f"EvalResult has no metric '{metric_name}'")

    return float(getattr(eval_result, metric_name))


def is_better_metric(current: float, best: float, metric_name: str) -> bool:
    if np.isnan(current):
        return False

    if np.isnan(best):
        return True

    lower_is_better = {
        "alignment_pos_proj",
        "alignment_pos_backbone",
        "alignment_ratio_proj",
        "alignment_ratio_backbone",
        "uniformity_proj",
        "uniformity_backbone",
    }

    if metric_name in lower_is_better:
        return current < best

    return current > best


def log_eval_to_tensorboard(writer: SummaryWriter, eval_result, epoch: int) -> None:
    """
    Logs all available eval curves, including:
      alignment_pos / alignment_neg / alignment_ratio / alignment_cos
      uniformity
      retrieval@1 / retrieval@5 / retrieval@10
      probe metrics if present
    """

    if eval_result is None:
        return

    eval_dict = eval_result.to_dict()

    for key, value in eval_dict.items():
        if value is None:
            continue

        try:
            value = float(value)
        except Exception:
            continue

        if np.isnan(value):
            continue

        writer.add_scalar(f"Eval/{key}", value, epoch)

    # Keep backward compatibility with existing tensorboard_scalars().
    for tag, value in eval_result.tensorboard_scalars().items():
        try:
            value = float(value)
        except Exception:
            continue

        if not np.isnan(value):
            writer.add_scalar(tag, value, epoch)


def train_one_epoch(
    model: SupConXLSR,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    use_ctc: bool,
    use_mixed_precision: bool,
    grad_clip: float,
) -> dict:
    model.train()

    use_amp = use_mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    total_loss = 0.0
    total_supcon_loss = 0.0
    total_ctc_loss = 0.0
    n_batches = 0

    for batch in loader:
        audio = batch["audio"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(audio, attention_mask=attention_mask)

            if use_ctc:
                input_lengths = model.backbone._get_feat_extract_output_lengths(
                    attention_mask.sum(dim=-1).long()
                ).long()

                losses = model.compute_loss(
                    embeddings=out["embeddings"],
                    labels=labels,
                    ctc_logits=out["ctc_logits"],
                    ctc_targets=batch["ctc_targets"].to(device),
                    ctc_input_lengths=input_lengths,
                    ctc_target_lengths=batch["ctc_target_lengths"].to(device),
                )
            else:
                losses = model.compute_loss(
                    embeddings=out["embeddings"],
                    labels=labels,
                )

            if not torch.isfinite(losses["loss"]):
                logger.error(
                    f"NaN/Inf loss detected | "
                    f"loss={losses['loss']} | "
                    f"supcon={losses['supcon_loss']} | "
                    f"ctc={losses['ctc_loss']}"
                )
                raise RuntimeError("Stopping because loss is NaN/Inf.")

        if use_amp:
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        scheduler.step()

        total_loss += float(losses["loss"].item())
        total_supcon_loss += float(losses["supcon_loss"].item())
        total_ctc_loss += float(losses["ctc_loss"].item())
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "supcon_loss": total_supcon_loss / max(n_batches, 1),
        "ctc_loss": total_ctc_loss / max(n_batches, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 SupCon training on L2-ARCTIC CV only."
    )

    # Data
    parser.add_argument("--parquet_path", type=Path, required=True)
    parser.add_argument("--sample_rate", type=int, default=16_000)
    parser.add_argument("--max_audio_len_s", type=float, default=10.0)
    parser.add_argument("--label_col", type=str, default="prompt_id")
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        choices=["train", "dev", "test"],
        help="Split used for training. Default: train.",
    )
    parser.add_argument(
        "--dev_split",
        type=str,
        default="dev",
        choices=["train", "dev", "test"],
        help="Split used for monitoring eval. Default: dev.",
    )

    parser.add_argument(
        "--validate_audio",
        type=str2bool,
        default=True,
        help="If true, drop unreadable audio files at dataset construction.",
    )

    # Sampler
    parser.add_argument("--k_utterances", type=int, default=15)
    parser.add_argument("--s_speakers", type=int, default=12)
    parser.add_argument("--n_batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    # Model
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-large-xlsr-53")
    parser.add_argument("--proj_hidden_dim", type=int, default=512)
    parser.add_argument("--proj_out_dim", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=32)
    parser.add_argument("--min_frozen_layer", type=int, default=0)
    parser.add_argument("--max_frozen_layer", type=int, default=18)
    parser.add_argument("--ctc_lambda", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_ctc", type=str2bool, default=True)
    parser.add_argument("--tokenizer", type=str, default="facebook/wav2vec2-large-960h")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--use_mixed_precision", type=str2bool, default=False)

    # Checkpointing
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--tensorboard_dir", type=Path, required=True)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)

    # Monitoring eval
    parser.add_argument("--eval_every_n_epochs", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--eval_n_neg_samples", type=int, default=1000)
    parser.add_argument("--retrieval_ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument(
        "--eval_metrics",
        type=str,
        nargs="+",
        default=["alignment", "uniformity", "retrieval_at_5"],
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        default="retrieval_at_5_backbone",
        help="Metric used to select checkpoint_best.pt.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info("=" * 80)
    logger.info("Stage 2 SupCon L2-ARCTIC CV training")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    if args.dev_split == "test":
        raise ValueError("Do not use split='test' for monitoring / checkpoint selection.")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if args.use_ctc:
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.tokenizer)
        logger.info(f"Tokenizer: {args.tokenizer} | vocab_size={len(tokenizer)}")
        logger.info(f"Tokenizer pad_token_id={tokenizer.pad_token_id}")

    train_collate = (
        partial(collate_supcon_l2cv_with_tokenizer, tokenizer=tokenizer)
        if args.use_ctc
        else collate_supcon_l2cv
    )

    eval_collate = collate_supcon_l2cv

    train_dataset = SupConL2ArcticCVDataset(
        parquet_path=args.parquet_path,
        split=args.train_split,
        sample_rate=args.sample_rate,
        max_audio_len_s=args.max_audio_len_s,
        label_col=args.label_col,
        validate_audio=args.validate_audio,
    )

    dev_dataset = SupConL2ArcticCVDataset(
        parquet_path=args.parquet_path,
        split=args.dev_split,
        sample_rate=args.sample_rate,
        max_audio_len_s=args.max_audio_len_s,
        label_col=args.label_col,
        validate_audio=args.validate_audio,
    )

    train_sampler = SupConL2CVBatchSampler(
        dataset=train_dataset,
        k_utterances=args.k_utterances,
        s_speakers=args.s_speakers,
        n_batches=args.n_batches,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=train_collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=eval_collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logger.info(
        f"Monitoring eval uses split='{args.dev_split}' from {args.parquet_path}"
    )
    logger.info(
        f"Audio drops: train={train_dataset.audio_filter_report.n_dropped}, "
        f"dev={dev_dataset.audio_filter_report.n_dropped}"
    )

    model = SupConXLSR(
        model_name=args.model_name,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_out_dim=args.proj_out_dim,
        vocab_size=args.vocab_size,
        ctc_lambda=args.ctc_lambda,
        temperature=args.temperature,
        min_frozen_layer=args.min_frozen_layer,
        max_frozen_layer=args.max_frozen_layer,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(
        f"Training schedule: {args.epochs} epochs × "
        f"{len(train_loader)} batches = {total_steps} steps"
    )

    writer = SummaryWriter(log_dir=str(args.tensorboard_dir))
    logger.info(f"TensorBoard → {args.tensorboard_dir}")

    best_score = float("nan")
    best_epoch = None
    global_step = 0

    last_train_metrics = None
    last_eval_metrics_dict = None

    writer.add_scalar("Data/train_dropped_audio", train_dataset.audio_filter_report.n_dropped, 0)
    writer.add_scalar("Data/dev_dropped_audio", dev_dataset.audio_filter_report.n_dropped, 0)
    writer.add_scalar("Data/train_rows", len(train_dataset), 0)
    writer.add_scalar("Data/dev_rows", len(dev_dataset), 0)

    for epoch in range(1, args.epochs + 1):
        train_sampler.rng.seed(args.seed + epoch)

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_ctc=args.use_ctc,
            use_mixed_precision=args.use_mixed_precision,
            grad_clip=args.grad_clip,
        )

        last_train_metrics = train_metrics
        global_step += len(train_loader)

        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={train_metrics['loss']:.4f} | "
            f"supcon={train_metrics['supcon_loss']:.4f} | "
            f"ctc={train_metrics['ctc_loss']:.4f}"
        )

        writer.add_scalar("Train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("Train/supcon_loss", train_metrics["supcon_loss"], epoch)
        writer.add_scalar("Train/ctc_loss", train_metrics["ctc_loss"], epoch)
        writer.add_scalar("Optim/lr", scheduler.get_last_lr()[0], epoch)

        eval_result = None
        eval_metrics_dict = None

        if epoch % args.eval_every_n_epochs == 0:
            eval_result = run_eval(
                model=model,
                loader=dev_loader,
                device=device,
                retrieval_ks=args.retrieval_ks,
                metrics=args.eval_metrics,
                n_neg_samples=args.eval_n_neg_samples,
            )

            eval_result.log(prefix=f"Epoch {epoch:03d}")
            eval_metrics_dict = eval_result.to_dict()
            last_eval_metrics_dict = eval_metrics_dict

            log_eval_to_tensorboard(writer, eval_result, epoch)

            current_score = get_metric(eval_result, args.best_metric)

            logger.info(
                f"Best metric check: {args.best_metric}={current_score:.6f} "
                f"| previous_best={best_score}"
            )

            if is_better_metric(current_score, best_score, args.best_metric):
                best_score = current_score
                best_epoch = epoch

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    save_dir=args.save_dir,
                    epoch=epoch,
                    global_step=global_step,
                    train_metrics=train_metrics,
                    eval_metrics=eval_metrics_dict,
                    args=args,
                    name="checkpoint_best.pt",
                )

                logger.info(
                    f"✓ New best checkpoint at epoch {epoch:03d}: "
                    f"{args.best_metric}={best_score:.6f}"
                )

        if epoch % args.save_every_n_epochs == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                save_dir=args.save_dir,
                epoch=epoch,
                global_step=global_step,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics_dict,
                args=args,
            )

    if last_train_metrics is None:
        raise RuntimeError("No training epoch completed.")

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=args.save_dir,
        epoch=args.epochs,
        global_step=global_step,
        train_metrics=last_train_metrics,
        eval_metrics=last_eval_metrics_dict,
        args=args,
        name="checkpoint_final.pt",
    )

    metadata = {
        "best_epoch": best_epoch,
        "best_metric": args.best_metric,
        "best_score": None if np.isnan(best_score) else float(best_score),
        "final_epoch": args.epochs,
        "global_step": global_step,
        "train_split": args.train_split,
        "dev_split": args.dev_split,
        "train_rows": len(train_dataset),
        "dev_rows": len(dev_dataset),
        "train_dropped_audio": train_dataset.audio_filter_report.n_dropped,
        "dev_dropped_audio": dev_dataset.audio_filter_report.n_dropped,
        "train_bad_audio_paths": train_dataset.audio_filter_report.bad_paths,
        "dev_bad_audio_paths": dev_dataset.audio_filter_report.bad_paths,
    }

    metadata_path = args.save_dir / "training_summary.json"
    metadata_path.write_text(json.dumps(to_jsonable(metadata), indent=2))
    logger.info(f"Training summary → {metadata_path}")

    writer.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()