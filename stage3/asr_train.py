"""
asr_train.py

Stage 3 — CTC fine-tuning for downstream ASR.

This script fine-tunes a Wav2Vec2ForCTC model on one of two possible
Stage 3 training datasets:

  - LibriSpeech: standard English ASR fine-tuning.
  - AESRC: accented English ASR fine-tuning.

Model initialisation
--------------------
The model is always instantiated from --model_name.

Two modes are supported:

  1. Vanilla baseline:
     If --stage2_checkpoint is not provided, the model keeps the HuggingFace
     pretrained wav2vec2 backbone.

  2. Stage 2 initialisation:
     If --stage2_checkpoint is provided, the script loads the HuggingFace model
     first, then replaces its wav2vec2 backbone weights with the Stage 2
     contrastive checkpoint.

Training
--------
The wav2vec2 backbone is fine-tuned with backbone_lr.
The CTC lm_head is fine-tuned with head_lr.
No explicit layer freezing is applied in this script.

Training can be controlled either by epochs or by optimizer updates:

  - epoch-based: use --epochs
  - step-based: use --max_steps

If --max_steps is provided, it overrides epoch-based stopping.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available — metrics only logged to console.")

try:
    import evaluate
    _wer_metric = evaluate.load("wer")
    EVALUATE_AVAILABLE = True
except Exception:
    EVALUATE_AVAILABLE = False
    logger.warning("'evaluate' not found — WER will be skipped during eval.")

sys.path.insert(0, str(Path(__file__).parent.parent))
from stage3.asr_data import build_loaders


def _load_stage2_backbone(
    model: Wav2Vec2ForCTC,
    checkpoint_path: Path,
) -> Wav2Vec2ForCTC:
    """
    Copy backbone weights from a Stage 2 checkpoint into Wav2Vec2ForCTC.

    Key mapping:
        Stage 2  "backbone.*"  →  Stage 3  "wav2vec2.*"

    Everything else is discarded.
    lm_head is NOT touched.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage 2 checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading Stage 2 backbone from: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    stage2_state = ckpt["model"]
    ctc_state = model.state_dict()

    mapped: dict[str, torch.Tensor] = {}
    skipped_absent: list[str] = []
    skipped_shape: list[str] = []

    for k, v in stage2_state.items():
        if not k.startswith("backbone."):
            continue

        new_k = "wav2vec2." + k[len("backbone."):]

        if new_k not in ctc_state:
            skipped_absent.append(new_k)
            continue

        if ctc_state[new_k].shape != v.shape:
            skipped_shape.append(
                f"{new_k}: want {ctc_state[new_k].shape}, got {v.shape}"
            )
            continue

        mapped[new_k] = v

    missing, _ = model.load_state_dict(mapped, strict=False)
    lm_missing = [k for k in missing if "lm_head" in k]

    logger.info(f"  Mapped         : {len(mapped)} tensors")
    logger.info(f"  Skipped absent : {len(skipped_absent)}")
    logger.info(f"  Skipped shape  : {len(skipped_shape)}")
    logger.info(f"  lm_head random : {len(lm_missing)} tensors (expected)")
    logger.info(f"  Stage 2 epoch  : {ckpt.get('epoch', '?')}")

    if skipped_shape:
        for s in skipped_shape:
            logger.warning(f"    {s}")

    return model


def build_model(args, processor: Wav2Vec2Processor) -> Wav2Vec2ForCTC:
    """
    Instantiate Wav2Vec2ForCTC from --model_name.
    If --stage2_checkpoint is given, overwrite backbone weights from it.
    """
    logger.info(f"Instantiating Wav2Vec2ForCTC from: {args.model_name}")

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name,
        vocab_size=len(processor.tokenizer),
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
    )

    if args.stage2_checkpoint is not None:
        model = _load_stage2_backbone(model, args.stage2_checkpoint)
    else:
        logger.info("No Stage 2 checkpoint — backbone stays HF-pretrained.")

    return model


def get_total_training_steps(args, steps_per_epoch: int) -> int:
    """
    Determine total number of optimizer updates.

    If max_steps is provided, training is step-based.
    Otherwise, training is epoch-based.
    """
    if args.max_steps is not None:
        return int(args.max_steps)

    return int(args.epochs * steps_per_epoch)


def build_optimizer_and_scheduler(
    model: Wav2Vec2ForCTC,
    args,
    steps_per_epoch: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """
    Two LR groups:
      wav2vec2 backbone  → backbone_lr
      lm_head            → head_lr
    """
    backbone_params = [p for p in model.wav2vec2.parameters() if p.requires_grad]
    head_params = list(model.lm_head.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )

    total_steps = get_total_training_steps(args, steps_per_epoch)
    warmup_steps = int(args.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(
        f"Optimizer: AdamW  backbone_lr={args.backbone_lr:.1e}  "
        f"head_lr={args.head_lr:.1e}  wd={args.weight_decay}"
    )
    logger.info(f"Scheduler: linear warmup {warmup_steps} / {total_steps} steps")

    if args.max_steps is not None:
        logger.info(f"Training mode: step-based, max_steps={args.max_steps}")
    else:
        logger.info(f"Training mode: epoch-based, epochs={args.epochs}")

    return optimizer, scheduler


def compute_wer(
    model: Wav2Vec2ForCTC,
    loader,
    processor: Wav2Vec2Processor,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    if not EVALUATE_AVAILABLE:
        return float("nan")

    model.eval()
    all_preds: list[str] = []
    all_labels: list[str] = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            logits = model(
                input_values=batch["input_values"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits

            pred_ids = torch.argmax(logits, dim=-1)
            preds = processor.batch_decode(pred_ids)

            label_ids = batch["labels"].clone()
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            refs = processor.batch_decode(label_ids, group_tokens=False)

            all_preds.extend(preds)
            all_labels.extend(refs)

    wer = _wer_metric.compute(predictions=all_preds, references=all_labels)
    model.train()

    return float(wer)


def save_training_checkpoint(
    output_dir: Path,
    epoch: int,
    global_step: int,
    model: Wav2Vec2ForCTC,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    args,
    name: str,
) -> Path:
    ckpt_path = output_dir / name

    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
        },
        ckpt_path,
    )

    logger.info(f"  Checkpoint → {ckpt_path}")
    return ckpt_path


def run_eval_and_maybe_save_best(
    model: Wav2Vec2ForCTC,
    eval_loader,
    processor: Wav2Vec2Processor,
    device: torch.device,
    args,
    output_dir: Path,
    best_wer: float,
    writer: Optional["SummaryWriter"],
    global_step: int,
    epoch: int,
) -> float:
    logger.info("  Computing WER on dev set …")

    wer = compute_wer(
        model,
        eval_loader,
        processor,
        device,
        max_batches=args.eval_max_batches,
    )

    logger.info(f"  WER = {wer:.4f}")

    if writer:
        writer.add_scalar("Eval/wer", wer, global_step)
        writer.add_scalar("Eval/wer_by_epoch", wer, epoch)

    if wer < best_wer:
        best_wer = wer
        best_path = output_dir / "best_model"
        model.save_pretrained(best_path)
        processor.save_pretrained(best_path)
        logger.info(f"  ✓ New best WER={best_wer:.4f} → {best_path}")

    return best_wer


def should_save_at_step(args, global_step: int) -> bool:
    if args.save_every_steps is None:
        return False
    return global_step > 0 and global_step % args.save_every_steps == 0


def should_eval_at_step(args, global_step: int) -> bool:
    if args.eval_every_steps is None:
        return False
    return global_step > 0 and global_step % args.eval_every_steps == 0


def train(
    model: Wav2Vec2ForCTC,
    train_loader,
    eval_loader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    processor: Wav2Vec2Processor,
    device: torch.device,
    args,
    writer: Optional["SummaryWriter"] = None,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_wer = float("inf")
    stop_training = False

    steps_per_epoch = len(train_loader)
    target_steps = args.max_steps if args.max_steps is not None else args.epochs * steps_per_epoch

    logger.info(f"Steps per epoch : {steps_per_epoch:,}")
    logger.info(f"Target steps    : {target_steps:,}")

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_loss = 0.0
        epoch_batches = 0

        for batch in train_loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                lr_bb = optimizer.param_groups[0]["lr"]
                lr_hd = optimizer.param_groups[1]["lr"]

                logger.info(
                    f"Epoch {epoch:03d} | step {global_step:06d}/{target_steps:06d} | "
                    f"loss={loss.item():.4f} | "
                    f"lr_bb={lr_bb:.2e}  lr_hd={lr_hd:.2e}"
                )

                if writer:
                    writer.add_scalar("Train/loss", loss.item(), global_step)
                    writer.add_scalar("LR/backbone", lr_bb, global_step)
                    writer.add_scalar("LR/head", lr_hd, global_step)

            # Step-based checkpointing
            if should_save_at_step(args, global_step):
                save_training_checkpoint(
                    output_dir=output_dir,
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    name=f"checkpoint_step{global_step:06d}.pt",
                )

            # Step-based eval
            if should_eval_at_step(args, global_step):
                best_wer = run_eval_and_maybe_save_best(
                    model=model,
                    eval_loader=eval_loader,
                    processor=processor,
                    device=device,
                    args=args,
                    output_dir=output_dir,
                    best_wer=best_wer,
                    writer=writer,
                    global_step=global_step,
                    epoch=epoch,
                )

            # Stop if max_steps reached
            if args.max_steps is not None and global_step >= args.max_steps:
                logger.info(f"Reached max_steps={args.max_steps}.")
                stop_training = True
                break

        mean_loss = epoch_loss / max(epoch_batches, 1)

        logger.info(
            f"── Epoch {epoch:03d} done — "
            f"mean_loss={mean_loss:.4f}  steps_this_epoch={epoch_batches}  "
            f"global_step={global_step}"
        )

        if writer:
            writer.add_scalar("Train/mean_loss", mean_loss, epoch)

        # Epoch-based checkpointing, kept for backward compatibility
        if args.save_every_steps is None:
            if epoch % args.save_every == 0 or epoch == args.epochs or stop_training:
                save_training_checkpoint(
                    output_dir=output_dir,
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    name=f"checkpoint_epoch{epoch:03d}.pt",
                )

        # Epoch-based eval, kept for backward compatibility
        if args.eval_every_steps is None:
            if epoch % args.eval_every == 0 or stop_training:
                best_wer = run_eval_and_maybe_save_best(
                    model=model,
                    eval_loader=eval_loader,
                    processor=processor,
                    device=device,
                    args=args,
                    output_dir=output_dir,
                    best_wer=best_wer,
                    writer=writer,
                    global_step=global_step,
                    epoch=epoch,
                )

        if stop_training:
            break

    # Always save a predictable final checkpoint.
    save_training_checkpoint(
        output_dir=output_dir,
        epoch=epoch,
        global_step=global_step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        name="checkpoint_final.pt",
    )

    logger.info(f"Training done. Best WER = {best_wer:.4f}")
    logger.info(f"Final global_step = {global_step}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3 — CTC fine-tuning for downstream ASR."
    )

    # model
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name.",
    )

    parser.add_argument(
        "--stage2_checkpoint",
        type=Path,
        default=None,
        help="Stage 2 .pt checkpoint — backbone weights only.",
    )

    # data
    parser.add_argument(
        "--train_parquet",
        type=Path,
        required=True,
        help="Train parquet.",
    )

    parser.add_argument(
        "--eval_parquet",
        type=Path,
        required=True,
        help="Eval parquet.",
    )

    parser.add_argument(
        "--max_duration_s",
        type=float,
        default=20.0,
        help="Hard cap on audio duration in seconds.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["librispeech", "aesrc"],
        help="Dataset to use for Stage 3 fine-tuning.",
    )

    # training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help=(
            "Maximum number of optimizer updates. "
            "If provided, training stops after this many steps, regardless of epochs."
        ),
    )

    # checkpointing
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Epoch-based checkpoint frequency. Used only if --save_every_steps is not set.",
    )

    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=None,
        help="Step-based checkpoint frequency. If set, overrides epoch-based saving.",
    )

    # eval
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="Epoch-based eval frequency. Used only if --eval_every_steps is not set.",
    )

    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=None,
        help="Step-based eval frequency. If set, overrides epoch-based evaluation.",
    )

    parser.add_argument(
        "--eval_max_batches",
        type=int,
        default=None,
        help="Cap WER eval at N batches. None = full eval set.",
    )

    # logging / dirs
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        required=True,
        help="TensorBoard log directory.",
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        required=True,
        help="Logging directory.",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info(f"Device : {device}")
    logger.info(f"Args   : {vars(args)}")

    train_loader, eval_loader, processor = build_loaders(args)

    model = build_model(args, processor).to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        args,
        steps_per_epoch=len(train_loader),
    )

    writer = None

    if TENSORBOARD_AVAILABLE:
        tb_dir = Path(args.tensorboard_dir)
        writer = SummaryWriter(log_dir=str(tb_dir))
        logger.info(f"TensorBoard → {tb_dir}")

    train(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        processor=processor,
        device=device,
        args=args,
        writer=writer,
    )

    if writer:
        writer.close()


if __name__ == "__main__":
    main()