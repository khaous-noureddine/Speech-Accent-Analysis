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

    Reads from args:
        args.train_parquet   : Path
        args.eval_parquet    : Path
        args.max_duration_s  : float
        args.batch_size      : int
        args.num_workers     : int

    Returns
    -------
    (train_loader, eval_loader, processor)
    """
    processor = build_processor()
    collator  = CTCCollator(processor=processor)

    train_ds = LibriSpeechDataset(
        parquet_path   = args.train_parquet,
        max_duration_s = args.max_duration_s,
    )
    eval_ds = LibriSpeechDataset(
        parquet_path   = args.eval_parquet,
        max_duration_s = args.max_duration_s,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        collate_fn  = collator,
        pin_memory  = True,
        drop_last   = True,     # avoids batch-size-1 edge cases with CTC
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
        f"DataLoaders ready — "
        f"train: {len(train_ds):,} samples ({len(train_loader):,} batches) | "
        f"eval: {len(eval_ds):,} samples ({len(eval_loader):,} batches)"
    )

    return train_loader, eval_loader, processor


# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Stage 3 data pipeline — smoke test and inspection."
#     )
#     parser.add_argument("--train_parquet", type=Path, required=True,
#                         help="Path to the train parquet (from import_librispeech.py).")
#     parser.add_argument("--eval_parquet",  type=Path, required=True,
#                         help="Path to the eval parquet (from import_librispeech.py).")
#     parser.add_argument("--max_duration_s", type=float, default=20.0)
#     parser.add_argument("--batch_size",    type=int,   default=4)
#     parser.add_argument("--num_workers",   type=int,   default=2)
#     parser.add_argument("--smoke_test",    action="store_true",
#                         help="Load one batch and print shapes + decoded reference.")

#     args = parser.parse_args()

#     train_loader, eval_loader, processor = build_loaders(args)

#     if args.smoke_test:
#         batch = next(iter(train_loader))

#         logger.info("── Smoke test ───────────────────────────────────────────")
#         logger.info(f"  input_values   : {tuple(batch['input_values'].shape)}")
#         logger.info(f"  attention_mask : {tuple(batch['attention_mask'].shape)}")
#         logger.info(f"  labels         : {tuple(batch['labels'].shape)}")

#         # Decode first example to verify tokenisation round-trip
#         label_ids = batch["labels"][0].clone()
#         label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
#         ref = processor.decode(label_ids)
#         logger.info(f"  reference[0]   : {ref!r}")

#         # Duration stats
#         df_train = pd.read_parquet(args.train_parquet, columns=["duration_s"])
#         logger.info(f"  train duration : {df_train['duration_s'].sum() / 3600:.1f} h")
#         logger.info(f"  train mean dur : {df_train['duration_s'].mean():.2f} s")
#         logger.info(f"  train max dur  : {df_train['duration_s'].max():.2f} s")

#         logger.info("  Smoke test passed ✓")


# if __name__ == "__main__":
#     main()