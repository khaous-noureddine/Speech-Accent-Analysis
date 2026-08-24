"""LibriSpeech data pipeline for Stage 3 CTC fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


class LibriSpeechDataset(Dataset):
    def __init__(
        self,
        parquet_path: Path,
        *,
        repository_root: Path,
        sample_rate: int = 16_000,
        max_duration_s: float = 20.0,
        validate_audio: bool = True,
    ) -> None:
        frame = pd.read_parquet(parquet_path)
        required = {"audio_path", "transcript", "speaker_id"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing LibriSpeech columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"Empty LibriSpeech parquet: {parquet_path}")
        root = repository_root.resolve()
        frame = frame.copy().reset_index(drop=True)
        frame["resolved_audio_path"] = frame["audio_path"].astype(str).map(
            lambda value: str(Path(value) if Path(value).is_absolute() else root / value)
        )
        if validate_audio:
            bad = [path for path in frame["resolved_audio_path"] if not Path(path).is_file()]
            if bad:
                raise ValueError(f"Found {len(bad)} missing audio files: {bad[:5]}")
        self.frame = frame
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration_s)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        audio, source_rate = sf.read(
            row["resolved_audio_path"], dtype="float32", always_2d=True
        )
        waveform = torch.from_numpy(audio).mean(dim=1)
        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, source_rate, self.sample_rate
            )
        return {
            "audio": waveform[: self.max_samples],
            "text": str(row["transcript"]).upper().strip(),
        }


@dataclass
class CTCCollator:
    feature_extractor: object
    tokenizer: object
    sample_rate: int = 16_000

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        inputs = self.feature_extractor(
            [item["audio"].numpy() for item in batch],
            sampling_rate=self.sample_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            [item["text"] for item in batch], padding=True, return_tensors="pt"
        )
        label_ids = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        return {
            "input_values": inputs.input_values,
            "attention_mask": inputs.attention_mask,
            "labels": label_ids,
        }
