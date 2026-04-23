"""
dataset.py

Triplet dataset for accent-invariant speech representation learning.

Triplet definition:
    Anchor   : speaker A, utterance X
    Positive : speaker B (≠ A), utterance X  ← same transcript, different speaker
    Negative : any speaker,     utterance Y  ← different transcript

Sources:
    - ARCTIC native   (6 speakers,  ~1000 utterances each)
    - L2-ARCTIC       (24 speakers, ~1000 utterances each)
    - Speech Accent Archive (many speakers, different sentences → negatives only)

Usage:
    dataset = TripletSpeechDataset(
        parquet_paths={
            "arctic":   "data/processed/arctic/corpus.parquet",
            "l2_arctic": "data/processed/l2_arctic/corpus.parquet",
        },
        sample_rate=16000,
        max_audio_len_s=10.0,
        n_negatives=1,
    )
    anchor, positive, negative = dataset[0]
"""

import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset





def keep_only_fully_shared_utterances(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"speaker_id", "utterance_id", "audio_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Nettoyage de base
    df = df.copy()
    df = df.dropna(subset=["speaker_id", "utterance_id", "audio_path"])

    # Normalisation minimale
    df["speaker_id"] = df["speaker_id"].astype(str).str.strip()
    df["utterance_id"] = df["utterance_id"].astype(str).str.strip()
    df["audio_path"] = df["audio_path"].astype(str).str.strip()

    df = df[
        (df["speaker_id"] != "") &
        (df["utterance_id"] != "") &
        (df["audio_path"] != "")
    ]

    # Garder uniquement les wavs qui existent réellement
    df = df[df["audio_path"].apply(lambda p: Path(p).exists())]

    if df.empty:
        raise ValueError("No valid rows left after checking existing audio files.")

    # Supprimer les doublons speaker/utterance au cas où
    df = df.drop_duplicates(subset=["speaker_id", "utterance_id"])

    # Nombre total de speakers valides
    n_speakers = df["speaker_id"].nunique()
    if n_speakers == 0:
        raise ValueError("No speakers found after cleaning.")

    # Utterances présentes chez tous les speakers
    speaker_counts = df.groupby("utterance_id")["speaker_id"].nunique()
    common_utt_ids = speaker_counts[speaker_counts == n_speakers].index

    filtered_df = df[df["utterance_id"].isin(common_utt_ids)].copy()

    if filtered_df.empty:
        raise ValueError("No utterance is shared across all speakers with existing wav files.")

    filtered_df = filtered_df.sort_values(["utterance_id", "speaker_id"]).reset_index(drop=True)

    print(f"Speakers kept: {n_speakers}")
    print(f"Shared utterances kept: {len(common_utt_ids)}")
    print(f"Total rows kept: {len(filtered_df)}")

    return filtered_df


# ──────────────────────────────────────────────
# Audio loading
# ──────────────────────────────────────────────

def load_audio(path: str, target_sr: int = 16000, max_len_samples: int = None) -> torch.Tensor:
    """
    Load a wav file, resample if needed, convert to mono.
    Returns a 1D tensor [T].
    """
    waveform, sr = torchaudio.load(path)

    # Stereo → mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    waveform = waveform.squeeze(0)  # [T]

    # Truncate
    if max_len_samples is not None and waveform.shape[0] > max_len_samples:
        waveform = waveform[:max_len_samples]

    return waveform


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class TripletSpeechDataset(Dataset):
    """
    Builds triplets (anchor, positive, negative) from one or more parquet files.

    Positive mining  : same utterance_id, different speaker_id
    Negative mining  : different utterance_id (any speaker, any corpus)

    Speech Accent Archive rows are only used as negatives (no shared utterance_id).
    """

    def __init__(
        self,
        parquet_paths: dict[str, str],
        sample_rate: int = 16000,
        max_audio_len_s: float = 10.0,
        n_negatives: int = 1,
        seed: int = 42,
    ):
        """
        Args:
            parquet_paths   : dict corpus_name → path to parquet
            sample_rate     : target sample rate (wav2vec2 expects 16kHz)
            max_audio_len_s : truncate audio beyond this duration
            n_negatives     : number of negatives per anchor (default 1 = classic triplet)
            seed            : random seed for reproducibility
        """
        self.sample_rate      = sample_rate
        self.max_len_samples  = int(max_audio_len_s * sample_rate)
        self.n_negatives      = n_negatives
        self.rng              = random.Random(seed)

        # ── Load all parquets ──
        frames = []
        for corpus_name, path in parquet_paths.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"Parquet not found: {path}")
            df = pd.read_parquet(path)
            df["corpus"] = corpus_name
            frames.append(df)

        self.df = pd.concat(frames, ignore_index=True)

        # self.df = pd.concat(frames, ignore_index=True)
        # shared_mask = self.df["corpus"].isin(["arctic", "l2_arctic"])
        # shared_df = keep_only_fully_shared_utterances(self.df[shared_mask].copy())
        # other_df = self.df[~shared_mask].copy()
        # self.df = pd.concat([shared_df, other_df], ignore_index=True)



        # Validate required columns
        required = {"speaker_id", "utterance_id", "transcript", "audio_path"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in parquet: {missing}")

        # Drop rows with missing audio
        self.df = self.df.dropna(subset=["audio_path"]).reset_index(drop=True)

        # ── Index: utterance_id → list of row indices ──
        # Only ARCTIC + L2-ARCTIC share utterance_ids → used for positives
        self._utt2rows: dict[str, list[int]] = defaultdict(list)
        for idx, row in self.df.iterrows():
            if row["corpus"] in ("arctic", "l2_arctic"):
                self._utt2rows[row["utterance_id"]].append(idx)

        # ── Anchor pool: utterances that have at least 2 different speakers ──
        self._anchor_pool: list[int] = []
        for utt_id, indices in self._utt2rows.items():
            speakers = self.df.loc[indices, "speaker_id"].unique()
            if len(speakers) >= 2:
                self._anchor_pool.extend(indices)

        if len(self._anchor_pool) == 0:
            raise ValueError(
                "No valid anchor/positive pairs found. "
                "Check that ARCTIC and L2-ARCTIC share utterance_ids."
            )

        # ── Negative pool: all rows (including speech accents) ──
        self._all_indices = list(self.df.index)

        print(f"Dataset loaded:")
        print(f"  Total rows       : {len(self.df)}")
        print(f"  Anchor pool      : {len(self._anchor_pool)} rows")
        print(f"  Unique utt_ids   : {len(self._utt2rows)}")
        print(f"  Corpora          : {self.df['corpus'].value_counts().to_dict()}")

    # ──────────────────────────────────────────────
    # Triplet building
    # ──────────────────────────────────────────────

    def _get_positive(self, anchor_idx: int) -> int:
        """Return a row index with same utterance_id but different speaker."""
        anchor_row = self.df.loc[anchor_idx]
        utt_id     = anchor_row["utterance_id"]
        speaker_id = anchor_row["speaker_id"]

        candidates = [
            i for i in self._utt2rows[utt_id]
            if self.df.loc[i, "speaker_id"] != speaker_id
        ]
        return self.rng.choice(candidates)

    def _get_negative(self, anchor_idx: int) -> int:
        """Return a row index with a different utterance_id."""
        anchor_utt = self.df.loc[anchor_idx, "utterance_id"]
        while True:
            idx = self.rng.choice(self._all_indices)
            if self.df.loc[idx, "utterance_id"] != anchor_utt:
                return idx

    # ──────────────────────────────────────────────
    # Dataset interface
    # ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._anchor_pool)

    def __getitem__(self, idx: int) -> dict:
        anchor_idx   = self._anchor_pool[idx]
        positive_idx = self._get_positive(anchor_idx)
        negative_idx = self._get_negative(anchor_idx)

        anchor   = self._load_row(anchor_idx)
        positive = self._load_row(positive_idx)
        negative = self._load_row(negative_idx)

        return {
            "anchor":           anchor["audio"],
            "positive":         positive["audio"],
            "negative":         negative["audio"],
            "anchor_speaker":   anchor["speaker_id"],
            "positive_speaker": positive["speaker_id"],
            "negative_speaker": negative["speaker_id"],
            "utterance_id":     anchor["utterance_id"],
            "transcript":       anchor["transcript"],
        }

    def _load_row(self, idx: int) -> dict:
        row   = self.df.loc[idx]
        audio = load_audio(
            row["audio_path"],
            target_sr=self.sample_rate,
            max_len_samples=self.max_len_samples,
        )
        return {
            "audio":      audio,
            "speaker_id": row["speaker_id"],
            "utterance_id": row["utterance_id"],
            "transcript": row["transcript"],
        }


# ──────────────────────────────────────────────
# Collate function  (for DataLoader)
# ──────────────────────────────────────────────

def collate_triplets(batch: list[dict]) -> dict:
    """
    Pad audio tensors to the same length within each batch.
    Returns dict with keys: anchor, positive, negative  (each [B, T])
    """
    def pad_sequence(tensors: list[torch.Tensor]) -> torch.Tensor:
        max_len = max(t.shape[0] for t in tensors)
        padded  = torch.zeros(len(tensors), max_len)
        for i, t in enumerate(tensors):
            padded[i, :t.shape[0]] = t
        return padded

    return {
        "anchor":           pad_sequence([b["anchor"]   for b in batch]),
        "positive":         pad_sequence([b["positive"] for b in batch]),
        "negative":         pad_sequence([b["negative"] for b in batch]),
        "anchor_speaker":   [b["anchor_speaker"]   for b in batch],
        "positive_speaker": [b["positive_speaker"] for b in batch],
        "negative_speaker": [b["negative_speaker"] for b in batch],
        "utterance_id":     [b["utterance_id"]     for b in batch],
        "transcript":       [b["transcript"]       for b in batch],
    }


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    parquet_paths = {
        "arctic":         sys.argv[1],
        "l2_arctic":      sys.argv[2],
        # "speech_accents": sys.argv[3],
    }

    dataset = TripletSpeechDataset(parquet_paths)
    

    sample = dataset[0]
    print(f"\nSample triplet:")
    print(f"  anchor   : {sample['anchor_speaker']} | {sample['utterance_id']} | {sample['anchor'].shape}")
    print(f"  positive : {sample['positive_speaker']} | {sample['utterance_id']} | {sample['positive'].shape}")
    print(f"  negative : {sample['negative_speaker']} | {sample['transcript'][:40]} | {sample['negative'].shape}")


    sample = dataset[3]
    print(f"  anchor   : {sample['anchor_speaker']} | {sample['utterance_id']} | {sample['anchor'].shape}")
    print(f"  positive : {sample['positive_speaker']} | {sample['utterance_id']} | {sample['positive'].shape}")
    print(f"  negative : {sample['negative_speaker']} | {sample['utterance_id'][:40]} | {sample['negative'].shape}")