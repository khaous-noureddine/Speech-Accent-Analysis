"""
Data pipeline for Stage 3 CTC fine-tuning on LibriSpeech.

Reads from the local parquets produced by import_librispeech.py —
no HuggingFace datasets download, no streaming.

Processor
---------
Loaded from "facebook/wav2vec2-base-960h" (trained on LibriSpeech,
vocab_size=32, character-level English). The feature extractor and
tokenizer are reused as-is — only the backbone changes in Stage 3.

Parquet schema expected (produced by import_librispeech.py)
-----------------------------------------------------------
audio_path    str    absolute path to a 16 kHz WAV file
transcript    str    upper-case text, no punctuation
speaker_id    str    e.g. "LS_1272"
split         str    "train" | "eval"

API usage (from stage3_train.py)
---------------------------------
    from stage3_data import build_loaders, build_processor
    train_loader, eval_loader, processor = build_loaders(args)

Standalone smoke test
---------------------
    python stage3_data.py \\
        --train_parquet data/processed/librispeech_train/corpus.parquet \\
        --eval_parquet  data/processed/librispeech_eval/corpus.parquet \\
        --smoke_test
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor


PROCESSOR_ID = "facebook/wav2vec2-base-960h"
SAMPLE_RATE  = 16_000


def build_processor() -> Wav2Vec2Processor:
    """
    Load the Wav2Vec2Processor from facebook/wav2vec2-base-960h.

    This gives us:
    - Wav2Vec2FeatureExtractor  : normalises + pads raw waveforms
    - Wav2Vec2CTCTokenizer      : 32-token English char vocab
                                    (A-Z + apostrophe + | + PAD + UNK)

    The processor is backbone-agnostic — we reuse it unchanged for XLSR.
    """
    logger.info(f"Loading processor from '{PROCESSOR_ID}' …")
    processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_ID)
    logger.info(f"  vocab_size = {len(processor.tokenizer)}")
    return processor



class AESRCDataset(Dataset):
    """
    PyTorch Dataset over the AESRC2020 parquet.

    Supports filtering by split column ("train" | "eval").

    Parquet schema expected (produced by import_aesrc.py)
    ------------------------------------------------------
    audio_path    str    absolute path to a 16 kHz WAV file
    transcript    str    normalised text
    speaker_id    str    e.g. "G51624"
    country       str    e.g. "British"
    accent        str    e.g. "Britain"
    split         str    "train" | "eval"
    """

    def __init__(
        self,
        parquet_path:   Path,
        max_duration_s: float      = 20.0,
        split:          str | None = None,
    ) -> None:
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")

        self.max_samples = int(max_duration_s * SAMPLE_RATE)

        df = pd.read_parquet(parquet_path)

        # ── Filter by split ───────────────────────────────────────────────
        if split is not None:
            if "split" not in df.columns:
                raise ValueError(f"Column 'split' not found in {parquet_path}")
            df = df[df["split"] == split].reset_index(drop=True)
            logger.info(f"  Filtered split='{split}' → {len(df):,} rows")

        logger.info(
            f"AESRCDataset: {len(df):,} utterances "
            f"| {df['country'].nunique()} countries "
            f"| {df['speaker_id'].nunique()} speakers"
        )
        logger.info(f"\n{df.groupby('country').size().to_string()}")

        self.records = df[["audio_path", "transcript", "speaker_id", "country", "accent"]].to_dict("records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row   = self.records[idx]
        audio, sr = sf.read(row["audio_path"], dtype="float32")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio = audio[: self.max_samples]

        return {
            "audio":      audio,
            "text":       str(row["transcript"]).upper().strip(),
            "speaker_id": str(row["speaker_id"]),
            "country":    str(row["country"]),
            "accent":     str(row["accent"]),
        }

        
class LibriSpeechDataset(Dataset):
    """
    PyTorch Dataset over a local LibriSpeech parquet.

    Each item returns:
        {
            "audio"      : np.ndarray [T]  float32, 16 kHz
            "text"       : str             upper-case transcript
            "speaker_id" : str
        }
    """

    def __init__(
        self,
        parquet_path:   Path,
        max_duration_s: float = 20.0,
    ) -> None:
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")

        self.max_samples = int(max_duration_s * SAMPLE_RATE)

        df = pd.read_parquet(parquet_path, columns=["audio_path", "transcript", "speaker_id"])
        self.records = df.to_dict("records")

        logger.info(
            f"LibriSpeechDataset: {len(self.records):,} utterances "
            f"from {parquet_path.name}"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row   = self.records[idx]
        audio, sr = sf.read(row["audio_path"], dtype="float32")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)              # stereo → mono (shouldn't happen)

        audio = audio[: self.max_samples]           # hard cap on duration

        return {
            "audio":      audio,
            "text":       str(row["transcript"]).upper().strip(),
            "speaker_id": str(row["speaker_id"]),
        }


@dataclass
class CTCCollator:
    """
    Pads audio and encodes text labels for Wav2Vec2ForCTC.

    - Audio  : padded by the feature extractor, returned as input_values
    - Labels : tokenised + padded, padding positions set to -100
            so CTC loss ignores them (HuggingFace convention)
    """
    processor: Wav2Vec2Processor

    def __call__(self, batch: list[dict]) -> dict:
        audios = [b["audio"] for b in batch]
        texts  = [b["text"]  for b in batch]

        # ── Audio ──────────────────────────────────────────────────────────
        inputs = self.processor(
            audios,
            sampling_rate       = SAMPLE_RATE,
            return_tensors      = "pt",
            padding             = True,
            return_attention_mask = True,
        )

        # ── Labels ─────────────────────────────────────────────────────────
        labels_enc = self.processor.tokenizer(
            texts,
            return_tensors = "pt",
            padding        = True,
        )
        labels = labels_enc.input_ids.masked_fill(
            labels_enc.attention_mask.ne(1), -100
        )

        return {
            "input_values":   inputs.input_values,    # [B, T_audio]
            "attention_mask": inputs.attention_mask,   # [B, T_audio]
            "labels":         labels,                  # [B, T_text]
        }


def build_loaders(args) -> tuple[DataLoader, DataLoader, Wav2Vec2Processor]:
    """
    Build train + eval DataLoaders and the shared processor.

    args.dataset : "librispeech" | "aesrc"
    """
    processor = build_processor()
    collator  = CTCCollator(processor=processor)

    dataset = getattr(args, "dataset")

    if dataset == "aesrc":
        train_ds = AESRCDataset(
            parquet_path   = args.train_parquet,
            max_duration_s = args.max_duration_s,
            split          = "train",
        )
        eval_ds = AESRCDataset(
            parquet_path   = args.train_parquet,
            max_duration_s = args.max_duration_s,
            split          = "dev",
        )
    elif dataset == "librispeech":
        train_ds = LibriSpeechDataset(
            parquet_path   = args.train_parquet,
            max_duration_s = args.max_duration_s,
        )
        eval_ds = LibriSpeechDataset(
            parquet_path   = args.eval_parquet,
            max_duration_s = args.max_duration_s,
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Valid: 'librispeech', 'aesrc'")

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        collate_fn  = collator,
        pin_memory  = True,
        drop_last   = True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        collate_fn  = collator,
        pin_memory  = True,
    )

    logger.info(
        f"DataLoaders ready [{dataset}] — "
        f"train: {len(train_ds):,} samples ({len(train_loader):,} batches) | "
        f"eval: {len(eval_ds):,} samples ({len(eval_loader):,} batches)"
    )

    return train_loader, eval_loader, processor