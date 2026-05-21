"""
stage2/eval_l2cv_wer.py

Quick WER sanity-check for a Stage 2 SupCon backbone.

Strategy
--------
The SupCon Stage 2 trains a backbone via contrastive learning.  Its auxiliary
CTC head is only a weak regulariser — it cannot transcribe.

To measure the impact of the backbone *without* full Stage 3 fine-tuning,
we graft a **pre-trained CTC head** (from wav2vec2-large-960h) onto the
SupCon backbone.  Both models share the same hidden dimension (1024), so
the head is a drop-in replacement.

This gives us a quick proxy WER:
  - If the SupCon backbone produces *better* representations than the
    vanilla XLSR-53, the 960h head should still decode reasonably.
  - We also run the same head on the vanilla XLSR-53 backbone as a
    baseline for comparison.

Outputs
-------
  {split}_transcriptions.csv   — per-utterance predictions + references
  {split}_wer_summary.csv      — overall WER
  {split}_wer_by_l1.csv        — WER per L1 group
  {split}_wer_by_speaker.csv   — WER per speaker
  {split}_wer_summary.json     — full details
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from jiwer import wer as jiwer_wer
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_audio
from stage2.supcon_xlsr import SupConXLSR


# ---------------------------------------------------------------------
# Text utils
# ---------------------------------------------------------------------

def normalize_text(text: str) -> str:
    return " ".join(str(text).upper().strip().split())


def safe_wer(refs: list[str], preds: list[str]) -> float:
    clean_refs, clean_preds = [], []
    for r, p in zip(refs, preds):
        if not r or not p:
            continue
        clean_refs.append(r)
        clean_preds.append(p)
    if not clean_refs:
        return float("nan")
    return jiwer_wer(clean_refs, clean_preds)


def summarize(df: pd.DataFrame) -> dict:
    wer = safe_wer(df["reference"].tolist(), df["prediction"].tolist())
    return {
        "wer": wer,
        "wer_percent": wer * 100 if not np.isnan(wer) else None,
        "n_utterances": int(len(df)),
        "n_empty_predictions": int((df["prediction"].str.len() == 0).sum()),
    }


def grouped_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group, g in df.groupby(group_col):
        row = summarize(g)
        row[group_col] = group
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def to_jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    return x


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class L2ArcticEvalDataset(Dataset):
    def __init__(
        self,
        parquet_path: Path,
        split: str = "test",
        sample_rate: int = 16_000,
        max_audio_len_s: float = 10.0,
        validate_audio: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.max_len_samples = int(sample_rate * max_audio_len_s)

        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        df = df[df["split"] == split].reset_index(drop=True)

        if df.empty:
            raise ValueError(f"No rows for split='{split}' in {parquet_path}")

        required = {"audio_path", "transcript", "speaker_id", "utterance_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.n_before_audio_filter = len(df)
        self.bad_audio_paths: list[str] = []

        if validate_audio:
            mask = []
            for _, row in df.iterrows():
                p = Path(row["audio_path"])
                try:
                    info = sf.info(str(p))
                    ok = p.exists() and info.frames > 0
                except Exception:
                    ok = False
                mask.append(ok)
                if not ok:
                    self.bad_audio_paths.append(str(p))

            if self.bad_audio_paths:
                logger.warning(f"Dropping {len(self.bad_audio_paths)} unreadable audio file(s)")
                for p in self.bad_audio_paths[:20]:
                    logger.warning(f"  bad audio: {p}")

            df = df[mask].reset_index(drop=True)

        if df.empty:
            raise ValueError(f"No readable audio for split='{split}'")

        self.df = df
        self.n_after_audio_filter = len(df)

        logger.info(f"Dataset loaded:")
        logger.info(f"  parquet  : {parquet_path}")
        logger.info(f"  split    : {split}")
        logger.info(f"  rows     : {len(self.df):,}")
        logger.info(f"  speakers : {self.df['speaker_id'].nunique()}")
        if "native_language" in self.df.columns:
            logger.info(f"  L1 groups: {self.df['native_language'].value_counts().to_dict()}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        audio = load_audio(
            str(row["audio_path"]),
            target_sr=self.sample_rate,
            max_len_samples=self.max_len_samples,
        )
        return {
            "audio": audio,
            "reference": str(row["transcript"]),
            "speaker_id": str(row["speaker_id"]),
            "utterance_id": str(row["utterance_id"]),
            "native_language": str(row.get("native_language", "unknown")),
            "audio_path": str(row["audio_path"]),
            "fold": int(row["fold"]) if "fold" in row and pd.notna(row["fold"]) else None,
        }


def collate_eval(batch: list[dict]) -> dict:
    return {
        "audio": [
            x["audio"].numpy() if torch.is_tensor(x["audio"]) else np.asarray(x["audio"])
            for x in batch
        ],
        "reference": [x["reference"] for x in batch],
        "speaker_id": [x["speaker_id"] for x in batch],
        "utterance_id": [x["utterance_id"] for x in batch],
        "native_language": [x["native_language"] for x in batch],
        "audio_path": [x["audio_path"] for x in batch],
        "fold": [x["fold"] for x in batch],
    }


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def get_ckpt_arg(ckpt: dict, key: str, default: Any) -> Any:
    args = ckpt.get("args", {}) or {}
    return args.get(key, default)


def load_supcon_backbone_with_pretrained_ctc_head(
    checkpoint_path: Path,
    ctc_donor_model: str,
    device: torch.device,
) -> tuple[SupConXLSR, Wav2Vec2Processor, dict, dict]:
    """
    Load the SupCon Stage 2 model but REPLACE its CTC head with the
    pre-trained lm_head from `ctc_donor_model` (e.g. wav2vec2-large-960h).

    Returns (model, processor, ckpt_dict, model_cfg).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # ── Load checkpoint ──────────────────────────────────────────────────
    logger.info(f"Loading SupCon checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_cfg = {
        "model_name": get_ckpt_arg(ckpt, "model_name", "facebook/wav2vec2-large-xlsr-53"),
        "proj_hidden_dim": int(get_ckpt_arg(ckpt, "proj_hidden_dim", 512)),
        "proj_out_dim": int(get_ckpt_arg(ckpt, "proj_out_dim", 256)),
        "vocab_size": int(get_ckpt_arg(ckpt, "vocab_size", 32)),
        "min_frozen_layer": int(get_ckpt_arg(ckpt, "min_frozen_layer", 0)),
        "max_frozen_layer": int(get_ckpt_arg(ckpt, "max_frozen_layer", 18)),
        "ctc_lambda": float(get_ckpt_arg(ckpt, "ctc_lambda", 0.1)),
        "temperature": float(get_ckpt_arg(ckpt, "temperature", 0.1)),
    }
    logger.info(f"SupCon model config: {model_cfg}")

    # ── Load the donor CTC model (wav2vec2-large-960h) ───────────────────
    logger.info(f"Loading CTC donor model: {ctc_donor_model}")
    donor = Wav2Vec2ForCTC.from_pretrained(ctc_donor_model).to(device)
    processor = Wav2Vec2Processor.from_pretrained(ctc_donor_model)

    donor_vocab_size = donor.lm_head.out_features
    donor_hidden_dim = donor.lm_head.in_features
    logger.info(f"  Donor lm_head: {donor_hidden_dim} → {donor_vocab_size}")

    # ── Build SupCon model with the DONOR vocab size ─────────────────────
    model = SupConXLSR(
        model_name=model_cfg["model_name"],
        proj_hidden_dim=model_cfg["proj_hidden_dim"],
        proj_out_dim=model_cfg["proj_out_dim"],
        vocab_size=donor_vocab_size,          # <── use donor vocab, not 32
        ctc_lambda=model_cfg["ctc_lambda"],
        temperature=model_cfg["temperature"],
        min_frozen_layer=model_cfg["min_frozen_layer"],
        max_frozen_layer=model_cfg["max_frozen_layer"],
    ).to(device)

    # ── Load SupCon backbone weights (skip the ctc_head mismatch) ────────
    state = ckpt["model"] if "model" in ckpt else ckpt

    # Filter out the old ctc_head weights (wrong shape if vocab differs)
    backbone_state = {
        k: v for k, v in state.items()
        if not k.startswith("ctc_head")
    }
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    logger.info(f"  Loaded backbone — missing keys: {missing}")
    logger.info(f"  Loaded backbone — unexpected keys: {unexpected}")

    # ── Graft the donor CTC head ─────────────────────────────────────────
    # model.ctc_head.weight.data.copy_(donor.lm_head.weight.data)
    # model.ctc_head.bias.data.copy_(donor.lm_head.bias.data)

    model.ctc_head.linear.weight.data.copy_(donor.lm_head.weight.data)
    model.ctc_head.linear.bias.data.copy_(donor.lm_head.bias.data)

    logger.info(f"  ✓ Grafted CTC head from {ctc_donor_model} "
                f"({donor_hidden_dim}→{donor_vocab_size})")

    # Sanity check dimensions
    supcon_hidden = model.backbone.config.hidden_size
    if supcon_hidden != donor_hidden_dim:
        raise ValueError(
            f"Hidden dim mismatch: SupCon backbone={supcon_hidden} vs "
            f"donor lm_head in_features={donor_hidden_dim}. Cannot graft."
        )

    # Free the donor
    del donor

    model.eval()
    logger.info(
        f"Loaded SupCon epoch={ckpt.get('epoch')} "
        f"global_step={ckpt.get('global_step')}"
    )

    return model, processor, ckpt, model_cfg


# ---------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------

@torch.inference_mode()
def transcribe(
    model: SupConXLSR,
    processor: Wav2Vec2Processor,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    model.eval()

    for batch_idx, batch in enumerate(loader):
        inputs = processor(
            batch["audio"],
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device)

        out = model(input_values, attention_mask=attention_mask)
        if "ctc_logits" not in out:
            raise RuntimeError("Model output has no 'ctc_logits'.")

        logits = out["ctc_logits"]
        pred_ids = torch.argmax(logits, dim=-1)
        predictions = processor.batch_decode(pred_ids)

        if batch_idx == 0:
            logger.info(f"  logits shape: {tuple(logits.shape)}")
            logger.info(f"  sample argmax[:30]: {pred_ids[0, :30].tolist()}")
            logger.info(f"  sample prediction: {predictions[0]!r}")
            logger.info(f"  sample reference : {batch['reference'][0]!r}")

        for i, pred in enumerate(predictions):
            rows.append({
                "prediction": normalize_text(pred),
                "reference": normalize_text(batch["reference"][i]),
                "speaker_id": batch["speaker_id"][i],
                "utterance_id": batch["utterance_id"][i],
                "native_language": batch["native_language"][i],
                "audio_path": batch["audio_path"][i],
                "fold": batch["fold"][i],
            })

        if (batch_idx + 1) % 20 == 0:
            logger.info(f"  Transcribed {len(rows):,} utterances")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Stage 2 SupCon backbone with a grafted pre-trained CTC head."
    )
    p.add_argument("--parquet_path", type=Path, required=True)
    p.add_argument("--checkpoint_path", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    p.add_argument("--sample_rate", type=int, default=16_000)
    p.add_argument("--max_audio_len_s", type=float, default=10.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--validate_audio", type=str, default="true")
    p.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    p.add_argument(
        "--ctc_donor",
        type=str,
        default="facebook/wav2vec2-large-960h",
        help="HF model whose CTC head (lm_head) will be grafted onto the SupCon backbone.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    validate_audio = str(args.validate_audio).lower() == "true"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info("=" * 80)
    logger.info("Stage 2 SupCon backbone eval — grafted CTC head")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load backbone + grafted head ─────────────────────────────────────
    model, processor, ckpt, model_cfg = load_supcon_backbone_with_pretrained_ctc_head(
        checkpoint_path=args.checkpoint_path,
        ctc_donor_model=args.ctc_donor,
        device=device,
    )

    logger.info(
        f"  Processor vocab_size={processor.tokenizer.vocab_size}, "
        f"pad_id={processor.tokenizer.pad_token_id}"
    )

    # ── Dataset / loader ─────────────────────────────────────────────────
    dataset = L2ArcticEvalDataset(
        parquet_path=args.parquet_path,
        split=args.split,
        sample_rate=args.sample_rate,
        max_audio_len_s=args.max_audio_len_s,
        validate_audio=validate_audio,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_eval,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Transcribe ───────────────────────────────────────────────────────
    df = transcribe(model, processor, loader, device)

    # ── Summaries ────────────────────────────────────────────────────────
    overall = summarize(df)
    by_l1 = grouped_summary(df, "native_language")
    by_speaker = grouped_summary(df, "speaker_id")

    fold = None
    if "fold" in df.columns and df["fold"].notna().any():
        fold = int(df["fold"].dropna().iloc[0])

    summary_row = {
        "fold": fold,
        "split": args.split,
        "checkpoint_path": str(args.checkpoint_path),
        "ctc_donor": args.ctc_donor,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_global_step": ckpt.get("global_step"),
        "n_samples": len(dataset),
        "n_before_audio_filter": dataset.n_before_audio_filter,
        "n_after_audio_filter": dataset.n_after_audio_filter,
        "n_dropped_audio": len(dataset.bad_audio_paths),
        **overall,
    }

    # ── Save ─────────────────────────────────────────────────────────────
    out = args.output_dir
    paths = {
        "transcriptions": out / f"{args.split}_transcriptions.csv",
        "summary":        out / f"{args.split}_wer_summary.csv",
        "by_l1":          out / f"{args.split}_wer_by_l1.csv",
        "by_speaker":     out / f"{args.split}_wer_by_speaker.csv",
        "json":           out / f"{args.split}_wer_summary.json",
    }

    df.to_csv(paths["transcriptions"], index=False)
    pd.DataFrame([summary_row]).to_csv(paths["summary"], index=False)
    by_l1.to_csv(paths["by_l1"], index=False)
    by_speaker.to_csv(paths["by_speaker"], index=False)
    paths["json"].write_text(
        json.dumps(
            to_jsonable({
                "summary": summary_row,
                "model_cfg": model_cfg,
                "checkpoint_args": ckpt.get("args", {}),
                "bad_audio_paths": dataset.bad_audio_paths,
            }),
            indent=2,
        )
    )

    for k, p in paths.items():
        logger.info(f"Saved {k:<14} → {p}")

    wer_pct = summary_row["wer_percent"]
    logger.info(f"WER = {wer_pct:.2f}%" if wer_pct is not None else "WER = NaN")


if __name__ == "__main__":
    main()