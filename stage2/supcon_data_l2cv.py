"""
stage2/supcon_data_l2cv.py

Data utilities for Stage 2 supervised contrastive training on L2-ARCTIC CV only.

Robustness:
  - filters missing / corrupted audio files at dataset construction
  - logs dropped files
  - sampler only uses prompts that still have enough speakers after filtering

Expected parquet schema:
  audio_path
  transcript
  speaker_id
  utterance_id
  prompt_id
  native_language
  split          train | dev | test
  fold
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from loguru import logger
from torch.utils.data import Dataset, Sampler

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_audio


VALID_SPLITS = {"train", "dev", "test"}


def is_readable_audio(path: str) -> bool:
    try:
        p = Path(path)
        if not p.exists():
            return False

        info = sf.info(str(p))
        if info.frames <= 0:
            return False

        return True
    except Exception:
        return False


@dataclass
class AudioFilterReport:
    n_before: int
    n_after: int
    n_dropped: int
    bad_paths: list[str]


class SupConL2ArcticCVDataset(Dataset):
    def __init__(
        self,
        parquet_path: Path,
        split: str = "train",
        sample_rate: int = 16_000,
        max_audio_len_s: float = 10.0,
        label_col: str = "prompt_id",
        validate_audio: bool = True,
        max_bad_audio_log: int = 20,
    ) -> None:
        parquet_path = Path(parquet_path)

        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")

        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split='{split}'. Expected one of {sorted(VALID_SPLITS)}")

        self.parquet_path = parquet_path
        self.split = split
        self.sample_rate = sample_rate
        self.max_len_samples = int(max_audio_len_s * sample_rate)
        self.label_col = label_col
        self.validate_audio = validate_audio

        df = pd.read_parquet(parquet_path)

        required = {
            "audio_path",
            "transcript",
            "speaker_id",
            "utterance_id",
            "split",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {parquet_path}: {missing}")

        if label_col not in df.columns:
            logger.warning(
                f"label_col='{label_col}' not found in {parquet_path}. "
                f"Falling back to utterance_id."
            )
            self.label_col = "utterance_id"

        df = df[df["split"] == split].reset_index(drop=True)

        if df.empty:
            raise ValueError(f"No rows found for split='{split}' in {parquet_path}")

        df = df.dropna(subset=["audio_path", "transcript", "speaker_id", self.label_col])
        df = df.reset_index(drop=True)

        self.audio_filter_report = AudioFilterReport(
            n_before=len(df),
            n_after=len(df),
            n_dropped=0,
            bad_paths=[],
        )

        if validate_audio:
            n_before = len(df)

            readable_mask = df["audio_path"].astype(str).map(is_readable_audio)
            bad_df = df[~readable_mask].copy()

            if len(bad_df) > 0:
                logger.warning(
                    f"Dropping {len(bad_df)} unreadable audio file(s) "
                    f"from split='{split}' in {parquet_path}"
                )

                for _, bad_row in bad_df.head(max_bad_audio_log).iterrows():
                    logger.warning(
                        f"  bad audio | speaker={bad_row.get('speaker_id')} | "
                        f"utterance={bad_row.get('utterance_id')} | "
                        f"path={bad_row.get('audio_path')}"
                    )

                if len(bad_df) > max_bad_audio_log:
                    logger.warning(
                        f"  ... {len(bad_df) - max_bad_audio_log} more bad files not shown"
                    )

            df = df[readable_mask].reset_index(drop=True)

            self.audio_filter_report = AudioFilterReport(
                n_before=n_before,
                n_after=len(df),
                n_dropped=n_before - len(df),
                bad_paths=bad_df["audio_path"].astype(str).tolist(),
            )

            logger.info(
                f"Audio readability filter [{split}]: "
                f"kept {len(df):,}/{n_before:,} rows"
            )

            if df.empty:
                raise ValueError(
                    f"No readable audio rows left for split='{split}' in {parquet_path}"
                )

        df["corpus"] = "l2_arctic"

        self.df = df.reset_index(drop=True)

        self.utt2indices: dict[str, list[int]] = defaultdict(list)

        for idx, row in self.df.iterrows():
            label_key = str(row[self.label_col])
            self.utt2indices[label_key].append(idx)

        self.valid_utts = [
            utt_id
            for utt_id, indices in self.utt2indices.items()
            if self.df.loc[indices, "speaker_id"].nunique() >= 2
        ]

        self.df = self.df[
            self.df[self.label_col].astype(str).isin(self.valid_utts)
        ].reset_index(drop=True)

        self.utt2indices = defaultdict(list)

        for idx, row in self.df.iterrows():
            label_key = str(row[self.label_col])
            self.utt2indices[label_key].append(idx)

        self.valid_utts = sorted(self.utt2indices.keys())

        self.utt2label = {
            utt_id: i for i, utt_id in enumerate(self.valid_utts)
        }

        logger.info("SupConL2ArcticCVDataset loaded:")
        logger.info(f"  Parquet          : {parquet_path}")
        logger.info(f"  Split            : {self.split}")
        logger.info(f"  Rows             : {len(self.df):,}")
        logger.info(f"  Valid prompts    : {len(self.valid_utts):,}")
        logger.info(f"  Speakers         : {self.df['speaker_id'].nunique()}")

        if "native_language" in self.df.columns:
            logger.info(f"  L1 groups        : {self.df['native_language'].nunique()}")
            logger.info(
                f"  Rows by L1       : {self.df['native_language'].value_counts().to_dict()}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.loc[idx]

        label_key = str(row[self.label_col])
        audio_path = str(row["audio_path"])

        try:
            audio = load_audio(
                audio_path,
                target_sr=self.sample_rate,
                max_len_samples=self.max_len_samples,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load audio at idx={idx} | "
                f"split={self.split} | "
                f"speaker={row.get('speaker_id')} | "
                f"utterance={row.get('utterance_id')} | "
                f"path={audio_path} | "
                f"error={repr(e)}"
            ) from e

        return {
            "audio": audio,
            "utterance_id": label_key,
            "speaker_id": str(row["speaker_id"]),
            "transcript": str(row["transcript"]),
            "label": self.utt2label[label_key],
            "corpus": "l2_arctic",
            "native_language": str(row["native_language"]) if "native_language" in row else "unknown",
        }


class SupConL2CVBatchSampler(Sampler[list[int]]):
    """
    Batch structure:
      K prompts × S speakers per prompt

    Example:
      k_utterances = 15
      s_speakers   = 12
      batch size   = 180
    """

    def __init__(
        self,
        dataset: SupConL2ArcticCVDataset,
        k_utterances: int = 15,
        s_speakers: int = 12,
        n_batches: int = 100,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.k = k_utterances
        self.s = s_speakers
        self.n_batches = n_batches
        self.rng = random.Random(seed)

        self.eligible_utts = []

        for utt_id, indices in dataset.utt2indices.items():
            n_speakers = dataset.df.loc[indices, "speaker_id"].nunique()
            if n_speakers >= s_speakers:
                self.eligible_utts.append(utt_id)

        if len(self.eligible_utts) < k_utterances:
            raise ValueError(
                f"Only {len(self.eligible_utts)} prompts with ≥ {s_speakers} speakers, "
                f"but k_utterances={k_utterances} required."
            )

        logger.info("SupConL2CVBatchSampler:")
        logger.info(f"  Eligible prompts : {len(self.eligible_utts):,}")
        logger.info(f"  K prompts        : {self.k}")
        logger.info(f"  S speakers       : {self.s}")
        logger.info(f"  Batch size       : {self.k * self.s}")
        logger.info(f"  Batches/epoch    : {self.n_batches}")

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            selected_utts = self.rng.sample(self.eligible_utts, self.k)

            batch_indices = []

            for utt_id in selected_utts:
                indices = self.dataset.utt2indices[utt_id]

                speaker_to_indices: dict[str, list[int]] = defaultdict(list)

                for idx in indices:
                    speaker_id = self.dataset.df.loc[idx, "speaker_id"]
                    speaker_to_indices[speaker_id].append(idx)

                speakers = list(speaker_to_indices.keys())
                selected_speakers = self.rng.sample(speakers, self.s)

                for speaker_id in selected_speakers:
                    selected_idx = self.rng.choice(speaker_to_indices[speaker_id])
                    batch_indices.append(selected_idx)

            yield batch_indices


def collate_supcon_l2cv(batch: list[dict]) -> dict:
    max_len = max(b["audio"].shape[0] for b in batch)

    audio = torch.zeros(len(batch), max_len)
    attention_mask = torch.zeros(len(batch), max_len)

    for i, b in enumerate(batch):
        length = b["audio"].shape[0]
        audio[i, :length] = b["audio"]
        attention_mask[i, :length] = 1.0

    return {
        "audio": audio,
        "attention_mask": attention_mask,
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "utterance_id": [b["utterance_id"] for b in batch],
        "speaker_id": [b["speaker_id"] for b in batch],
        "corpus": [b["corpus"] for b in batch],
        "native_language": [b["native_language"] for b in batch],
        "transcript": [b["transcript"] for b in batch],
    }


def collate_supcon_l2cv_with_tokenizer(batch: list[dict], tokenizer) -> dict:
    out = collate_supcon_l2cv(batch)

    transcripts = [t.lower() for t in out["transcript"]]
    encoded = tokenizer(transcripts).input_ids

    ctc_targets = torch.tensor(
        [idx for seq in encoded for idx in seq],
        dtype=torch.long,
    )

    ctc_target_lengths = torch.tensor(
        [len(seq) for seq in encoded],
        dtype=torch.long,
    )

    out["ctc_targets"] = ctc_targets
    out["ctc_target_lengths"] = ctc_target_lengths

    return out