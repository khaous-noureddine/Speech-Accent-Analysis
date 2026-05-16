"""
Data utilities for supervised contrastive speech training. Includes dataset, batch sampler, and collate function.

Usage:
    python supcon_data.py
"""

import sys
import random
# import pyarrow
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_audio


class SupConSpeechDataset(Dataset):
    def __init__(
        self,
        parquet_paths: dict[str, str],
        sample_rate: int = 16000,
        max_audio_len_s: float = 10.0,
        split: str = "train",
    ):
        self.sample_rate = sample_rate
        self.max_len_samples = int(max_audio_len_s * sample_rate)

        if split not in {"train", "eval"}:
            raise ValueError(f"Invalid split: {split}. Expected 'train' or 'eval'.")

        self.split = split

        frames = []

        for corpus_name, path in parquet_paths.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"Parquet not found: {path}")

            df = pd.read_parquet(path, engine="pyarrow")
            df["corpus"] = corpus_name
            frames.append(df)

        self.df = pd.concat(frames, ignore_index=True)

        required = {"speaker_id", "utterance_id", "transcript", "audio_path", "split"}
        missing = required - set(self.df.columns)

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Keep only rows for the selected split
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if self.df.empty:
            raise ValueError(f"No rows found for split='{split}'")

        self.df = self.df.dropna(subset=["audio_path"]).reset_index(drop=True)

        # Keep only shared corpora
        self.df = self.df[self.df["corpus"].isin(["arctic", "l2_arctic"])].reset_index(drop=True)

        # Remove problematic utterances (pronounced by very little speakers)
        utterances_to_remove = [
            "arctic_b0013",
            "arctic_b0540",
            "arctic_a0597",
            "arctic_a0596",
            "arctic_a0595",
            "arctic_a0594",
            "arctic_b0541",
        ]

        self.df = self.df[
            ~self.df["utterance_id"].isin(utterances_to_remove)
        ].reset_index(drop=True)

        # Build utterance_id -> row indices
        self.utt2indices: dict[str, list[int]] = defaultdict(list)

        for idx, row in self.df.iterrows():
            self.utt2indices[row["utterance_id"]].append(idx)

        # Keep only utterances with at least 2 different speakers
        self.valid_utts = [
            utt_id for utt_id, indices in self.utt2indices.items()
            if self.df.loc[indices, "speaker_id"].nunique() >= 2
        ]

        self.utt2label: dict[str, int] = {
            utt_id: i for i, utt_id in enumerate(self.valid_utts)
        }
        
        # Keep only valid utterances
        self.df = self.df[
            self.df["utterance_id"].isin(self.valid_utts)
        ].reset_index(drop=True)

        # Rebuild index after filtering
        self.utt2indices = defaultdict(list)

        for idx, row in self.df.iterrows():
            self.utt2indices[row["utterance_id"]].append(idx)

        logger.info("SupConSpeechDataset loaded:")
        logger.info(f"  Split           : {self.split}")
        logger.info(f"  Total rows      : {len(self.df)}")
        logger.info(f"  Valid utterances: {len(self.valid_utts)}")
        logger.info(f"  Speakers        : {self.df['speaker_id'].nunique()}")
        logger.info(f"  Corpora         : {self.df['corpus'].value_counts().to_dict()}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.loc[idx]
        
        audio = load_audio(
            row["audio_path"],
            target_sr=self.sample_rate,
            max_len_samples=self.max_len_samples,
        )
        # We take for label the utterance_id, so that all samples with the same utterance_id are positives for SupCon.
        return {
            "audio": audio,
            "utterance_id": row["utterance_id"],
            "speaker_id": row["speaker_id"],
            "transcript": row["transcript"],
            "label": self.utt2label[row["utterance_id"]],
        }


class SupConBatchSampler(Sampler):
    """
    We ensure using this sampler that each batch contains K different utterances, and for each utterance, S different speakers. 
    This guarantees that the SupCon loss has S-1 positive pairs available in every batch.
    """

    def __init__(
        self,
        dataset: SupConSpeechDataset,
        k_utterances: int = 20,
        s_speakers: int = 25,
        n_batches: int = 1000,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.k = k_utterances
        self.s = s_speakers
        self.n_batches = n_batches
        self.rng = random.Random(seed)

        # Pre-filter utterances with enough available speakers
        self.eligible_utts = [
            utt_id for utt_id, indices in dataset.utt2indices.items()
            if dataset.df.loc[indices, "speaker_id"].nunique() >= s_speakers
        ]

        if len(self.eligible_utts) < k_utterances:
            raise ValueError(
                f"Only {len(self.eligible_utts)} utterances with ≥ {s_speakers} speakers, "
                f"but K={k_utterances} required."
            )

        logger.info(f"SupConBatchSampler: {len(self.eligible_utts)} eligible utterances")
        logger.info(f"  Batch size = {k_utterances} × {s_speakers} = {k_utterances * s_speakers}")

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            # Randomly sample K utterances
            selected_utts = self.rng.sample(self.eligible_utts, self.k)

            batch_indices = []
            for utt_id in selected_utts:
                indices = self.dataset.utt2indices[utt_id]

                # Randomly sample S speakers for this utterance
                selected_indices = self.rng.sample(indices, self.s)
                batch_indices.extend(selected_indices)

            yield batch_indices