"""L2-ARCTIC datasets and structured prompt batch sampling for adaptation."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, Sampler


VALID_SPLITS = {"train", "dev"}


def _load_audio(path: Path, sample_rate: int, max_samples: int) -> torch.Tensor:
    waveform, source_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(waveform).mean(dim=1)
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    return audio[:max_samples]


class L2ArcticAdaptationDataset(Dataset):
    """One leakage-safe train or development view of a generated fold."""

    def __init__(
        self,
        parquet_path: Path,
        *,
        split: str,
        repository_root: Path,
        sample_rate: int = 16_000,
        max_audio_len_s: float = 10.0,
        manifest_sha256: str,
        validate_audio: bool = True,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"Adaptation may only load {sorted(VALID_SPLITS)}, got {split!r}.")
        frame = pd.read_parquet(parquet_path)
        required = {
            "audio_path", "transcript", "speaker_id", "prompt_id", "native_language",
            "split", "split_manifest_sha256",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing Parquet columns: {sorted(missing)}")

        hashes = set(frame["split_manifest_sha256"].astype(str))
        if hashes != {manifest_sha256}:
            raise ValueError(f"Parquet manifest hashes {sorted(hashes)} do not match {manifest_sha256}.")

        frame = frame[frame["split"] == split].copy().reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No {split!r} rows found in {parquet_path}.")

        root = repository_root.resolve()
        frame["resolved_audio_path"] = frame["audio_path"].astype(str).map(
            lambda value: str(Path(value) if Path(value).is_absolute() else root / value)
        )
        if validate_audio:
            bad = []
            for path in frame["resolved_audio_path"]:
                try:
                    info = sf.info(path)
                    if info.frames <= 0:
                        bad.append(path)
                except Exception:
                    bad.append(path)
            if bad:
                raise ValueError(f"Found {len(bad)} missing or unreadable audio files: {bad[:5]}")

        self.frame = frame
        self.split = split
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_audio_len_s)
        self.prompt_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, prompt in enumerate(frame["prompt_id"].astype(str)):
            self.prompt_to_indices[prompt].append(index)
        self.prompt_labels = {
            prompt: label for label, prompt in enumerate(sorted(self.prompt_to_indices))
        }

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        prompt = str(row["prompt_id"])
        return {
            "audio": _load_audio(
                Path(row["resolved_audio_path"]), self.sample_rate, self.max_samples
            ),
            "prompt_id": prompt,
            "speaker_id": str(row["speaker_id"]),
            "native_language": str(row["native_language"]),
            "transcript": str(row["transcript"]),
            "label": self.prompt_labels[prompt],
        }


class PromptBatchSampler(Sampler[list[int]]):
    """Sample K prompts and S distinct speakers per prompt."""

    def __init__(
        self,
        dataset: L2ArcticAdaptationDataset,
        *,
        prompts_per_batch: int,
        speakers_per_prompt: int,
        batches_per_epoch: int,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.k = prompts_per_batch
        self.s = speakers_per_prompt
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0
        self.eligible_prompts = [
            prompt
            for prompt, indices in dataset.prompt_to_indices.items()
            if dataset.frame.iloc[indices]["speaker_id"].nunique() >= self.s
        ]
        if len(self.eligible_prompts) < self.k:
            raise ValueError(
                f"Only {len(self.eligible_prompts)} prompts have {self.s} speakers; "
                f"{self.k} prompts per batch were requested."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            batch = []
            for prompt in rng.sample(self.eligible_prompts, self.k):
                by_speaker: dict[str, list[int]] = defaultdict(list)
                for index in self.dataset.prompt_to_indices[prompt]:
                    speaker = str(self.dataset.frame.iloc[index]["speaker_id"])
                    by_speaker[speaker].append(index)
                for speaker in rng.sample(sorted(by_speaker), self.s):
                    batch.append(rng.choice(by_speaker[speaker]))
            yield batch


def collate_adaptation(batch: list[dict], tokenizer=None) -> dict:
    max_length = max(item["audio"].numel() for item in batch)
    audio = torch.zeros(len(batch), max_length)
    attention_mask = torch.zeros(len(batch), max_length, dtype=torch.long)
    for index, item in enumerate(batch):
        length = item["audio"].numel()
        audio[index, :length] = item["audio"]
        attention_mask[index, :length] = 1

    result = {
        "audio": audio,
        "attention_mask": attention_mask,
        "labels": torch.tensor([item["label"] for item in batch]),
        "prompt_ids": [item["prompt_id"] for item in batch],
        "speaker_ids": [item["speaker_id"] for item in batch],
        "native_languages": [item["native_language"] for item in batch],
    }
    if tokenizer is not None:
        sequences = tokenizer([item["transcript"].upper() for item in batch]).input_ids
        result["ctc_targets"] = torch.tensor(
            [token for sequence in sequences for token in sequence], dtype=torch.long
        )
        result["ctc_target_lengths"] = torch.tensor(
            [len(sequence) for sequence in sequences], dtype=torch.long
        )
    return result
