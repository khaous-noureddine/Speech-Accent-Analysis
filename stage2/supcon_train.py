"""
train.py

Supervised Contrastive Training of XLSR-53 for accent-invariant speech representations.

Usage:
    python train.py \
        --arctic_parquet_path data/processed/arctic/corpus.parquet \
        --l2_arctic_parquet_path data/processed/l2_arctic/corpus.parquet \
        --save_dir outputs/run_01
"""

import sys
import argparse
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from transformers import Wav2Vec2CTCTokenizer, get_linear_schedule_with_warmup
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from stage2.supcon_data import SupConSpeechDataset, SupConBatchSampler
from stage2.supcon_xlsr import SupConXLSR
from stage2.supcon_evaluate import run_eval, build_eval_loader


def collate_with_tokenizer(batch: list[dict], tokenizer: Wav2Vec2CTCTokenizer) -> dict:
    """
    Extends collate_supcon with CTC targets derived from transcriptions.

    Returns:
        audio                : [B, T]
        attention_mask       : [B, T]    1 for real samples, 0 for padding
        labels               : [B]       utterance labels for SupCon
        ctc_targets          : [S]       flat concatenation of tokenized transcripts
        ctc_target_lengths   : [B]       length of each transcript in tokens
    """
    def pad_sequence(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(t.shape[0] for t in tensors)
        padded  = torch.zeros(len(tensors), max_len)
        mask    = torch.zeros(len(tensors), max_len)
        for i, t in enumerate(tensors):
            padded[i, :t.shape[0]] = t
            mask[i,   :t.shape[0]] = 1
        return padded, mask

    audio_padded, attention_mask = pad_sequence([b["audio"] for b in batch])

    transcripts        = [b["transcript"].lower() for b in batch]
    encoded            = tokenizer(transcripts).input_ids
    ctc_targets        = torch.tensor([idx for seq in encoded for idx in seq], dtype=torch.long)
    ctc_target_lengths = torch.tensor([len(seq) for seq in encoded], dtype=torch.long)

    return {
        "audio":               audio_padded,
        "attention_mask":      attention_mask,
        "labels":              torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "utterance_id":        [b["utterance_id"] for b in batch],
        "speaker_id":          [b["speaker_id"]   for b in batch],
        "ctc_targets":         ctc_targets,
        "ctc_target_lengths":  ctc_target_lengths,
    }


# def train_one_epoch(
#     model:     SupConXLSR,
#     loader:    DataLoader,
#     optimizer: torch.optim.Optimizer,
#     scheduler,
#     device:    torch.device,
#     use_ctc:   bool,
# ) -> dict:
#     model.train()

#     total_loss        = 0.0
#     total_supcon_loss = 0.0
#     total_ctc_loss    = 0.0
#     n_batches         = 0

#     for batch in loader:
#         audio          = batch["audio"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels         = batch["labels"].to(device)

#         out = model(audio, attention_mask=attention_mask)

#         if use_ctc:
#             input_lengths = model.backbone._get_feat_extract_output_lengths(
#                 attention_mask.sum(dim=-1).long()
#             ).long()

#             losses = model.compute_loss(
#                 embeddings=out["embeddings"],
#                 labels=labels,
#                 ctc_logits=out["ctc_logits"],
#                 ctc_targets=batch["ctc_targets"].to(device),
#                 ctc_input_lengths=input_lengths,
#                 ctc_target_lengths=batch["ctc_target_lengths"].to(device),
#             )
#         else:
#             losses = model.compute_loss(
#                 embeddings=out["embeddings"],
#                 labels=labels,
#             )

#         optimizer.zero_grad()
#         losses["loss"].backward()
#         nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         scheduler.step()

#         total_loss        += losses["loss"].item()
#         total_supcon_loss += losses["supcon_loss"].item()
#         total_ctc_loss    += losses["ctc_loss"].item()
#         n_batches         += 1

#     return {
#         "loss":        total_loss        / n_batches,
#         "supcon_loss": total_supcon_loss / n_batches,
#         "ctc_loss":    total_ctc_loss    / n_batches,
#     }


def train_one_epoch(
    model:     SupConXLSR,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device:    torch.device,
    use_ctc:   bool,
    use_mixed_precision: bool = False,
) -> dict:
    model.train()
    use_amp = use_mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    total_loss        = 0.0
    total_supcon_loss = 0.0
    total_ctc_loss    = 0.0
    n_batches         = 0

    for batch in loader:
        audio          = batch["audio"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

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
                    f"loss={losses['loss']} "
                    f"supcon={losses['supcon_loss']} "
                    f"ctc={losses['ctc_loss']}"
                )
                raise RuntimeError("Stopping because loss is NaN/Inf")

        optimizer.zero_grad()

        if use_amp:
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        total_loss        += losses["loss"].item()
        total_supcon_loss += losses["supcon_loss"].item()
        total_ctc_loss    += losses["ctc_loss"].item()
        n_batches         += 1

    assert n_batches == len(loader)
    return {
        "loss":        total_loss        / n_batches,
        "supcon_loss": total_supcon_loss / n_batches,
        "ctc_loss":    total_ctc_loss    / n_batches,
    }


def save_checkpoint(model: SupConXLSR, save_dir: Path, epoch: int, metrics: dict) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"checkpoint_epoch{epoch:03d}.pt"
    torch.save({"epoch": epoch, "model": model.state_dict(), "metrics": metrics}, ckpt_path)
    logger.info(f"Checkpoint saved: {ckpt_path}")


def main():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
        raise argparse.ArgumentTypeError("Expected boolean value.")


    parser = argparse.ArgumentParser(description="Supervised Contrastive Training of XLSR model")

    # Data
    parser.add_argument("--arctic_parquet_path",    type=Path, required=True)
    parser.add_argument("--l2_arctic_parquet_path", type=Path, required=True)
    parser.add_argument("--k_utterances",    type=int,   default=20)
    parser.add_argument("--s_speakers",      type=int,   default=25)
    parser.add_argument("--n_batches",       type=int,   default=300)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--sample_rate",     type=int,   default=16000)
    parser.add_argument("--max_audio_len_s", type=float, default=10.0)
    parser.add_argument("--num_workers",     type=int,   default=2)

    # Model
    parser.add_argument("--model_name",       type=str,   default="facebook/wav2vec2-large-xlsr-53")
    parser.add_argument("--proj_hidden_dim",  type=int,   default=512)
    parser.add_argument("--proj_out_dim",     type=int,   default=256)
    parser.add_argument("--temperature",      type=float, default=0.1)
    parser.add_argument("--ctc_lambda",       type=float, default=0.1)
    parser.add_argument("--min_frozen_layer", type=int,   default=0)
    parser.add_argument("--max_frozen_layer", type=int,   default=18)
    parser.add_argument("--vocab_size",       type=int,   default=32)

    # Training
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int,   default=500)
    parser.add_argument("--use_ctc",      type=bool,  default=True)
    parser.add_argument("--tokenizer",    type=str,   default="facebook/wav2vec2-large-960h")
    parser.add_argument("--device",       type=str,   choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--save_dir",     type=Path,  default=Path("./outputs"))
    parser.add_argument("--save_every_n_epochs",   type=int,   default=1)
    parser.add_argument("--tensorboard_dir", type=Path, default=None)
    parser.add_argument("--use_mixed_precision", type=str2bool, default=False)

    # Evaluation
    parser.add_argument("--eval_every_n_epochs", type=int, default=1)
    parser.add_argument("--eval_metrics", type=str, nargs="+", default=["alignment", "uniformity",])
    parser.add_argument("--eval_n_neg_samples", type=int, default=1000)
    parser.add_argument("--retrieval_ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--eval_batch_size", type=int, default=64)

    args   = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Tokenizer:
    tokenizer = None
    if args.use_ctc:
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.tokenizer)
        logger.info(f"Tokenizer: {args.tokenizer} — vocab_size={len(tokenizer)}")
        assert tokenizer.pad_token_id == 0, f"Expected blank at index 0, got {tokenizer.pad_token_id}"

    collate_fn = (
        partial(collate_with_tokenizer, tokenizer=tokenizer)
    )

    # Train Set:
    train_dataset = SupConSpeechDataset(
        parquet_paths={
            "arctic":    args.arctic_parquet_path,
            "l2_arctic": args.l2_arctic_parquet_path,
        },
        split="train",
        sample_rate=args.sample_rate,
        max_audio_len_s=args.max_audio_len_s,
    )

    train_sampler = SupConBatchSampler(
        train_dataset,
        k_utterances=args.k_utterances,
        s_speakers=args.s_speakers,
        n_batches=args.n_batches,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Eval Set:
    eval_loader, eval_dataset = build_eval_loader(args)
    logger.info(f"Eval set: {len(eval_loader.dataset)} samples")

    # Model, Optimizer, Scheduler:
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
        weight_decay=1e-4,
    )

    total_steps = args.epochs * len(train_loader)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(f"Training: {args.epochs} epochs x {len(train_loader)} batches = {total_steps} steps")

    args.save_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard:
    tb_dir = args.tensorboard_dir
    writer = SummaryWriter(log_dir=str(tb_dir))
    logger.info(f"TensorBoard logs: {tb_dir}")

    logger.info(f"Mixed precision : {'enabled (fp16)' if args.use_mixed_precision is True else 'disabled (fp32)'}")

    # Training Loop:
    for epoch in range(1, args.epochs + 1):
        train_sampler.rng.seed(args.seed + epoch)

        metrics = train_one_epoch(
            model, 
            train_loader, 
            optimizer, scheduler, 
            device, 
            args.use_ctc,
            use_mixed_precision=args.use_mixed_precision,
        )

        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={metrics['loss']:.4f} "
            f"(supcon={metrics['supcon_loss']:.4f}, ctc={metrics['ctc_loss']:.4f})"
        )

        writer.add_scalar("Loss/total", metrics["loss"], epoch)
        writer.add_scalar("Loss/supcon", metrics["supcon_loss"], epoch)
        writer.add_scalar("Loss/ctc", metrics["ctc_loss"], epoch)
        writer.add_scalar("Optim/LR", scheduler.get_last_lr()[0], epoch)

        if epoch % args.save_every_n_epochs == 0:
            save_checkpoint(model, args.save_dir, epoch, metrics)

        if epoch % args.eval_every_n_epochs == 0:
            eval_result = run_eval(
                model=model,
                loader=eval_loader,
                device=device,
                retrieval_ks=args.retrieval_ks,
                metrics=args.eval_metrics,
                n_neg_samples=args.eval_n_neg_samples,
            )

            eval_result.log(prefix=f"Epoch {epoch:03d}")

            for tag, val in eval_result.tensorboard_scalars().items():
                if not torch.isnan(torch.tensor(val)):
                    writer.add_scalar(tag, val, epoch)

        

    logger.info("Training complete.")
    writer.close() 


if __name__ == "__main__":
    main()