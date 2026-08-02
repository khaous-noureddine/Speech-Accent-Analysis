#!/usr/bin/env python3
"""
Stage 2 supervised contrastive training with masked mean pooling on L2-ARCTIC CV.

This is the non-DTW counterpart of the SupCon-DTW training script.

Training representation:
    audio
      -> speech backbone
      -> frame-level hidden states [B, T, D]
      -> masked temporal mean pooling [B, D]
      -> two-layer projection MLP + L2 normalization [B, d]
      -> supervised contrastive loss

Optional auxiliary objective:
    total_loss = supcon_loss + ctc_lambda * ctc_loss

The script is self-contained apart from its Python dependencies. It includes:
  - audio loading and resampling
  - L2-ARCTIC CV dataset and prompt/speaker batch sampler
  - masked mean-pooling SupCon model
  - optional auxiliary CTC loss
  - representation diagnostics (alignment, uniformity, retrieval@K)
  - TensorBoard logging
  - best/periodic/final checkpoints
  - YAML configuration with CLI overrides

It accepts either:
  1. a flat YAML file whose keys match CONFIG_SCHEMA, or
  2. the project-style nested block:
       supervised_contrastive_training_l2cv:
         data: ...
         sampler: ...
         model: ...
         training: ...
         evaluation: ...

CLI arguments override values loaded from YAML.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModel,
    Wav2Vec2CTCTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    import torchaudio.functional as AF
except Exception:
    AF = None


VALID_SPLITS = {"train", "dev", "test"}


# =============================================================================
# Utilities
# =============================================================================

def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(
        f"Expected a boolean value, received: {value!r}"
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_audio(
    audio_path: str | Path,
    target_sr: int,
    max_len_samples: int | None = None,
) -> torch.Tensor:
    """
    Load a waveform as mono float32 and resample when needed.

    Returns:
        Tensor shaped [num_samples].
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform_np, source_sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )

    if waveform_np.size == 0:
        raise ValueError(f"Empty audio file: {path}")

    # soundfile returns [time, channels].
    waveform_np = waveform_np.mean(axis=1)
    waveform = torch.from_numpy(waveform_np).contiguous()

    if source_sr != target_sr:
        if AF is None:
            raise RuntimeError(
                f"Audio {path} has sample rate {source_sr}, expected {target_sr}, "
                "and torchaudio is unavailable for resampling."
            )

        waveform = AF.resample(
            waveform,
            orig_freq=source_sr,
            new_freq=target_sr,
        )

    if max_len_samples is not None and waveform.numel() > max_len_samples:
        waveform = waveform[:max_len_samples]

    return waveform.float()


def is_readable_audio(path: str | Path) -> bool:
    try:
        audio_path = Path(path)
        if not audio_path.exists():
            return False

        info = sf.info(str(audio_path))
        return info.frames > 0
    except Exception:
        return False


def _parse_int_list(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return list(default)

    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]

    normalized = (
        str(value)
        .strip()
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "")
    )

    if not normalized:
        return list(default)

    return [int(item) for item in normalized.split(",") if item]


def _parse_str_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)

    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]

    normalized = str(value).strip()
    if not normalized:
        return list(default)

    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]

    return [
        item.strip().strip("'\"")
        for item in normalized.split(",")
        if item.strip()
    ]


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _gpu_mem_str(device: torch.device) -> str:
    if device.type != "cuda":
        return ""

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    return f" | mem={allocated:.2f}/{reserved:.2f} GiB"


@dataclass
class AudioFilterReport:
    n_before: int
    n_after: int
    n_dropped: int
    bad_paths: list[str]


# =============================================================================
# Dataset, sampler, collate
# =============================================================================

class SupConL2ArcticCVDataset(Dataset):
    """
    Dataset for prompt-level supervised contrastive learning.

    Positive examples share label_col (usually prompt_id) and come from
    different speakers. Only prompts spoken by at least two speakers are kept.
    """

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
        super().__init__()

        parquet_path = Path(parquet_path)

        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split={split!r}. Expected one of {sorted(VALID_SPLITS)}"
            )

        self.parquet_path = parquet_path
        self.split = split
        self.sample_rate = int(sample_rate)
        self.max_len_samples = int(max_audio_len_s * sample_rate)
        self.label_col = label_col

        df = pd.read_parquet(parquet_path)

        required_columns = {
            "audio_path",
            "transcript",
            "speaker_id",
            "utterance_id",
            "split",
        }
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in {parquet_path}: {sorted(missing)}"
            )

        if label_col not in df.columns:
            logger.warning(
                f"label_col={label_col!r} is missing; falling back to utterance_id."
            )
            self.label_col = "utterance_id"

        df = df[df["split"] == split].copy()
        df = df.dropna(
            subset=["audio_path", "transcript", "speaker_id", self.label_col]
        ).reset_index(drop=True)

        if df.empty:
            raise ValueError(
                f"No usable rows found for split={split!r} in {parquet_path}"
            )

        self.audio_filter_report = AudioFilterReport(
            n_before=len(df),
            n_after=len(df),
            n_dropped=0,
            bad_paths=[],
        )

        if validate_audio:
            n_before = len(df)
            readable_mask = df["audio_path"].astype(str).map(is_readable_audio)
            bad_df = df.loc[~readable_mask].copy()

            if not bad_df.empty:
                logger.warning(
                    f"Dropping {len(bad_df)} unreadable audio file(s) "
                    f"from split={split!r}."
                )
                for _, row in bad_df.head(max_bad_audio_log).iterrows():
                    logger.warning(
                        "bad audio | "
                        f"speaker={row.get('speaker_id')} | "
                        f"utterance={row.get('utterance_id')} | "
                        f"path={row.get('audio_path')}"
                    )

            df = df.loc[readable_mask].reset_index(drop=True)

            self.audio_filter_report = AudioFilterReport(
                n_before=n_before,
                n_after=len(df),
                n_dropped=n_before - len(df),
                bad_paths=bad_df["audio_path"].astype(str).tolist(),
            )

            logger.info(
                f"Audio readability [{split}]: kept {len(df):,}/{n_before:,}"
            )

            if df.empty:
                raise ValueError(
                    f"No readable audio remains for split={split!r}."
                )

        if "native_language" not in df.columns:
            df["native_language"] = "unknown"

        df["corpus"] = "l2_arctic"
        df[self.label_col] = df[self.label_col].astype(str)
        df["speaker_id"] = df["speaker_id"].astype(str)

        # Retain prompts represented by at least two distinct speakers.
        prompt_speaker_counts = (
            df.groupby(self.label_col)["speaker_id"]
            .nunique()
        )
        valid_prompts = set(
            prompt_speaker_counts[prompt_speaker_counts >= 2].index.astype(str)
        )

        df = df[df[self.label_col].isin(valid_prompts)].reset_index(drop=True)

        if df.empty:
            raise ValueError(
                f"No prompt has at least two speakers in split={split!r}."
            )

        self.df = df
        self.utt2indices: dict[str, list[int]] = defaultdict(list)

        for index, row in self.df.iterrows():
            self.utt2indices[str(row[self.label_col])].append(index)

        self.valid_utts = sorted(self.utt2indices)
        self.utt2label = {
            utterance_id: label
            for label, utterance_id in enumerate(self.valid_utts)
        }

        logger.info("SupConL2ArcticCVDataset loaded:")
        logger.info(f"  Parquet       : {parquet_path}")
        logger.info(f"  Split         : {split}")
        logger.info(f"  Rows          : {len(self.df):,}")
        logger.info(f"  Valid prompts : {len(self.valid_utts):,}")
        logger.info(f"  Speakers      : {self.df['speaker_id'].nunique():,}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        label_key = str(row[self.label_col])

        audio = load_audio(
            audio_path=row["audio_path"],
            target_sr=self.sample_rate,
            max_len_samples=self.max_len_samples,
        )

        return {
            "audio": audio,
            "utterance_id": label_key,
            "speaker_id": str(row["speaker_id"]),
            "transcript": str(row["transcript"]),
            "label": self.utt2label[label_key],
            "corpus": "l2_arctic",
            "native_language": str(row.get("native_language", "unknown")),
        }


class SupConL2CVBatchSampler(Sampler[list[int]]):
    """
    Sample K prompts and S distinct speakers per prompt.

    Batch size = K * S.
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
        self.k = int(k_utterances)
        self.s = int(s_speakers)
        self.n_batches = int(n_batches)
        self.rng = random.Random(seed)

        if self.k <= 0 or self.s <= 0 or self.n_batches <= 0:
            raise ValueError(
                "k_utterances, s_speakers, and n_batches must be positive."
            )

        self.eligible_utts: list[str] = []

        for utterance_id, indices in dataset.utt2indices.items():
            n_speakers = dataset.df.iloc[indices]["speaker_id"].nunique()
            if n_speakers >= self.s:
                self.eligible_utts.append(utterance_id)

        if len(self.eligible_utts) < self.k:
            raise ValueError(
                f"Only {len(self.eligible_utts)} prompts have at least "
                f"{self.s} speakers, but k_utterances={self.k}."
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
            selected_prompts = self.rng.sample(self.eligible_utts, self.k)
            batch_indices: list[int] = []

            for utterance_id in selected_prompts:
                indices = self.dataset.utt2indices[utterance_id]
                speaker_to_indices: dict[str, list[int]] = defaultdict(list)

                for index in indices:
                    speaker_id = str(
                        self.dataset.df.iloc[index]["speaker_id"]
                    )
                    speaker_to_indices[speaker_id].append(index)

                selected_speakers = self.rng.sample(
                    list(speaker_to_indices),
                    self.s,
                )

                for speaker_id in selected_speakers:
                    batch_indices.append(
                        self.rng.choice(speaker_to_indices[speaker_id])
                    )

            yield batch_indices


def collate_supcon_l2cv(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_length = max(item["audio"].numel() for item in batch)
    batch_size = len(batch)

    audio = torch.zeros(batch_size, max_length, dtype=torch.float32)
    attention_mask = torch.zeros(
        batch_size,
        max_length,
        dtype=torch.long,
    )

    for index, item in enumerate(batch):
        length = item["audio"].numel()
        audio[index, :length] = item["audio"]
        attention_mask[index, :length] = 1

    return {
        "audio": audio,
        "attention_mask": attention_mask,
        "labels": torch.tensor(
            [item["label"] for item in batch],
            dtype=torch.long,
        ),
        "utterance_id": [item["utterance_id"] for item in batch],
        "speaker_id": [item["speaker_id"] for item in batch],
        "corpus": [item["corpus"] for item in batch],
        "native_language": [item["native_language"] for item in batch],
        "transcript": [item["transcript"] for item in batch],
    }


def collate_supcon_l2cv_with_tokenizer(
    batch: list[dict[str, Any]],
    tokenizer: Wav2Vec2CTCTokenizer,
) -> dict[str, Any]:
    output = collate_supcon_l2cv(batch)

    transcripts = [text.lower() for text in output["transcript"]]
    encoded = tokenizer(transcripts).input_ids

    output["ctc_targets"] = torch.tensor(
        [token for sequence in encoded for token in sequence],
        dtype=torch.long,
    )
    output["ctc_target_lengths"] = torch.tensor(
        [len(sequence) for sequence in encoded],
        dtype=torch.long,
    )

    return output


# =============================================================================
# Model and losses
# =============================================================================

class ProjectionMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 256,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(inputs), dim=-1)


class CTCHead(nn.Module):
    def __init__(self, in_dim: int, vocab_size: int = 32) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states)


class SupConLoss(nn.Module):
    """
    Standard supervised contrastive loss over L2-normalized embeddings.

    For each valid anchor, all other examples with the same label are positives.
    The loss is averaged over positives and then over valid anchors.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be strictly positive.")

        self.temperature = float(temperature)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected embeddings [B, D], received {tuple(embeddings.shape)}"
            )

        batch_size = embeddings.shape[0]
        device = embeddings.device

        if batch_size < 2:
            raise ValueError("SupCon requires at least two examples per batch.")

        similarity = embeddings @ embeddings.T
        similarity = similarity / self.temperature

        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).to(similarity.dtype)

        self_mask = torch.eye(
            batch_size,
            device=device,
            dtype=similarity.dtype,
        )
        positive_mask = positive_mask * (1.0 - self_mask)
        denominator_mask = 1.0 - self_mask

        # Numerical stability.
        similarity = similarity - similarity.max(
            dim=1,
            keepdim=True,
        ).values.detach()

        exp_similarity = torch.exp(similarity) * denominator_mask
        log_probability = similarity - torch.log(
            exp_similarity.sum(dim=1, keepdim=True) + 1e-8
        )

        n_positives = positive_mask.sum(dim=1)
        valid_anchors = n_positives > 0

        if not valid_anchors.any():
            logger.warning(
                "No positive pairs found in the batch; returning zero SupCon loss."
            )
            return embeddings.sum() * 0.0

        loss_per_anchor = -(
            positive_mask * log_probability
        ).sum(dim=1) / n_positives.clamp_min(1.0)

        return loss_per_anchor[valid_anchors].mean()


class SupConMeanPoolModel(nn.Module):
    """
    Generic mean-pooling supervised contrastive model for speech encoders.

    Forward output:
      hidden_states : [B, T, hidden_size]
      frame_lengths : [B]
      pooled        : [B, hidden_size]
      embeddings    : [B, proj_out_dim]
      ctc_logits    : [B, T, vocab_size] or None
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-xlsr-53",
        proj_hidden_dim: int = 512,
        proj_out_dim: int = 256,
        vocab_size: int = 32,
        ctc_lambda: float = 0.1,
        temperature: float = 0.1,
        min_frozen_layer: int = 0,
        max_frozen_layer: int = 18,
        enable_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.ctc_lambda = float(ctc_lambda)
        self.min_frozen_layer = int(min_frozen_layer)
        self.max_frozen_layer = int(max_frozen_layer)

        logger.info(f"Loading speech backbone: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name)

        if (
            enable_gradient_checkpointing
            and hasattr(self.backbone, "gradient_checkpointing_enable")
        ):
            self.backbone.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                f"Backbone config for {model_name} has no hidden_size."
            )

        self._freeze_backbone_layers()

        self.projection = ProjectionMLP(
            in_dim=hidden_size,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_out_dim,
        )
        self.ctc_head = CTCHead(
            in_dim=hidden_size,
            vocab_size=vocab_size,
        )

        self.supcon_loss = SupConLoss(temperature=temperature)
        self.ctc_loss_fn = nn.CTCLoss(
            blank=0,
            reduction="mean",
            zero_infinity=True,
        )

        logger.info("SupConMeanPoolModel initialized:")
        logger.info(f"  Backbone        : {model_name}")
        logger.info(f"  Hidden size     : {hidden_size}")
        logger.info(
            f"  Frozen layers   : "
            f"{self.min_frozen_layer}-{self.max_frozen_layer - 1}"
        )
        logger.info(
            f"  Projection      : "
            f"{hidden_size} -> {proj_hidden_dim} -> {proj_out_dim}"
        )
        logger.info("  Pooling         : masked temporal mean")
        logger.info(f"  Vocab size      : {vocab_size}")
        logger.info(f"  lambda_CTC      : {ctc_lambda}")
        logger.info(f"  Temperature     : {temperature}")

    def _freeze_backbone_layers(self) -> None:
        if hasattr(self.backbone, "feature_extractor"):
            for parameter in self.backbone.feature_extractor.parameters():
                parameter.requires_grad = False
            logger.info("Frozen feature_extractor.")

        if hasattr(self.backbone, "feature_projection"):
            for parameter in self.backbone.feature_projection.parameters():
                parameter.requires_grad = False
            logger.info("Frozen feature_projection.")

        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layers", None)

        if layers is None:
            raise ValueError(
                f"Backbone {type(self.backbone)} does not expose encoder.layers."
            )

        n_layers = len(layers)

        if self.min_frozen_layer < 0:
            raise ValueError("min_frozen_layer must be >= 0.")

        if self.max_frozen_layer < self.min_frozen_layer:
            raise ValueError(
                "max_frozen_layer must be >= min_frozen_layer."
            )

        if self.max_frozen_layer > n_layers:
            logger.warning(
                f"max_frozen_layer={self.max_frozen_layer} exceeds "
                f"n_layers={n_layers}; clipping."
            )
            self.max_frozen_layer = n_layers

        for layer_index, layer in enumerate(layers):
            if self.min_frozen_layer <= layer_index < self.max_frozen_layer:
                for parameter in layer.parameters():
                    parameter.requires_grad = False

        n_trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        n_total = sum(parameter.numel() for parameter in self.parameters())

        logger.info(
            f"Trainable parameters: {n_trainable:,}/{n_total:,} "
            f"({100.0 * n_trainable / max(n_total, 1):.1f}%)"
        )

    def get_feat_extract_output_lengths(
        self,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self.backbone, "_get_feat_extract_output_lengths"):
            return self.backbone._get_feat_extract_output_lengths(
                input_lengths
            )

        raise AttributeError(
            f"Backbone {type(self.backbone)} does not expose "
            "_get_feat_extract_output_lengths()."
        )

    @staticmethod
    def masked_mean_pool(
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, time_steps, _ = hidden_states.shape

        if lengths.shape != (batch_size,):
            raise ValueError(
                f"Expected lengths [{batch_size}], received {tuple(lengths.shape)}"
            )

        lengths = lengths.clamp(min=1, max=time_steps)
        mask = (
            torch.arange(time_steps, device=hidden_states.device)
            .unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        mask = mask.to(hidden_states.dtype).unsqueeze(-1)

        summed = (hidden_states * mask).sum(dim=1)
        denominator = mask.sum(dim=1).clamp_min(1.0)

        return summed / denominator

    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        compute_ctc: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        outputs = self.backbone(
            input_values=audio,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        hidden_states = outputs.last_hidden_state

        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=-1).long()
            frame_lengths = self.get_feat_extract_output_lengths(
                input_lengths
            ).long()
        else:
            frame_lengths = torch.full(
                (hidden_states.shape[0],),
                hidden_states.shape[1],
                device=hidden_states.device,
                dtype=torch.long,
            )

        pooled = self.masked_mean_pool(hidden_states, frame_lengths)
        embeddings = self.projection(pooled)

        ctc_logits = (
            self.ctc_head(hidden_states)
            if compute_ctc
            else None
        )

        return {
            "hidden_states": hidden_states,
            "frame_lengths": frame_lengths,
            "pooled": pooled,
            "embeddings": embeddings,
            "ctc_logits": ctc_logits,
        }

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        use_ctc: bool,
        ctc_logits: torch.Tensor | None = None,
        ctc_targets: torch.Tensor | None = None,
        ctc_input_lengths: torch.Tensor | None = None,
        ctc_target_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        supcon_loss = self.supcon_loss(embeddings, labels)
        ctc_loss = embeddings.new_zeros(())

        if use_ctc:
            required = {
                "ctc_logits": ctc_logits,
                "ctc_targets": ctc_targets,
                "ctc_input_lengths": ctc_input_lengths,
                "ctc_target_lengths": ctc_target_lengths,
            }
            missing = [key for key, value in required.items() if value is None]

            if missing:
                raise ValueError(
                    "CTC is enabled but required inputs are missing: "
                    + ", ".join(missing)
                )

            log_probs = F.log_softmax(
                ctc_logits.float(),
                dim=-1,
            ).permute(1, 0, 2)

            ctc_loss = self.ctc_loss_fn(
                log_probs,
                ctc_targets,
                ctc_input_lengths,
                ctc_target_lengths,
            )

            total_loss = supcon_loss + self.ctc_lambda * ctc_loss
        else:
            total_loss = supcon_loss

        return {
            "loss": total_loss,
            "supcon_loss": supcon_loss,
            "ctc_loss": ctc_loss,
        }


# =============================================================================
# Representation diagnostics
# =============================================================================

@torch.no_grad()
def extract_eval_embeddings(
    model: SupConMeanPoolModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, Any]:
    model.eval()

    projected_embeddings: list[np.ndarray] = []
    backbone_embeddings: list[np.ndarray] = []
    utterance_ids: list[str] = []
    speaker_ids: list[str] = []
    corpus_tags: list[str] = []

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        audio = batch["audio"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(
            device,
            non_blocking=True,
        )

        output = model(
            audio,
            attention_mask=attention_mask,
            compute_ctc=False,
        )

        projected = output["embeddings"]
        backbone = F.normalize(output["pooled"], dim=-1)

        projected_embeddings.append(
            projected.detach().cpu().float().numpy()
        )
        backbone_embeddings.append(
            backbone.detach().cpu().float().numpy()
        )

        utterance_ids.extend(
            [str(item) for item in batch["utterance_id"]]
        )
        speaker_ids.extend(
            [str(item) for item in batch["speaker_id"]]
        )
        corpus_tags.extend(
            [str(item) for item in batch["corpus"]]
        )

    if not projected_embeddings:
        raise RuntimeError("No embeddings were extracted during evaluation.")

    return {
        "proj": np.concatenate(projected_embeddings, axis=0),
        "backbone": np.concatenate(backbone_embeddings, axis=0),
        "utterance_ids": utterance_ids,
        "speaker_ids": speaker_ids,
        "corpus_tags": corpus_tags,
    }


def compute_alignment_metrics(
    embeddings: np.ndarray,
    utterance_ids: list[str],
    speaker_ids: list[str],
    n_neg_samples: int,
    seed: int,
) -> dict[str, float | int]:
    """
    Positive pairs: same prompt, different speakers.
    Negative pairs: different prompts.
    """
    embeddings = embeddings.astype(np.float32, copy=False)
    utterances = np.asarray(utterance_ids)
    speakers = np.asarray(speaker_ids)
    rng = np.random.default_rng(seed)
    n_samples = len(embeddings)

    positive_pairs: list[tuple[int, int]] = []

    for utterance_id in sorted(set(utterance_ids)):
        indices = np.where(utterances == utterance_id)[0]

        for left_position in range(len(indices)):
            for right_position in range(left_position + 1, len(indices)):
                left = int(indices[left_position])
                right = int(indices[right_position])

                if speakers[left] != speakers[right]:
                    positive_pairs.append((left, right))

    if not positive_pairs:
        return {
            "pos_dist": float("nan"),
            "neg_dist": float("nan"),
            "ratio": float("nan"),
            "pos_cos": float("nan"),
            "n_pos_pairs": 0,
            "n_neg_pairs": 0,
        }

    pos_i = np.asarray(
        [pair[0] for pair in positive_pairs],
        dtype=np.int64,
    )
    pos_j = np.asarray(
        [pair[1] for pair in positive_pairs],
        dtype=np.int64,
    )

    positive_difference = embeddings[pos_i] - embeddings[pos_j]
    positive_distance = np.sum(
        positive_difference * positive_difference,
        axis=1,
    )
    positive_cosine = np.sum(
        embeddings[pos_i] * embeddings[pos_j],
        axis=1,
    )

    negative_i: list[int] = []
    negative_j: list[int] = []
    max_trials = max(n_neg_samples * 20, 1_000)

    for _ in range(max_trials):
        if len(negative_i) >= n_neg_samples:
            break

        left = int(rng.integers(0, n_samples))
        right = int(rng.integers(0, n_samples))

        if left == right or utterances[left] == utterances[right]:
            continue

        negative_i.append(left)
        negative_j.append(right)

    if negative_i:
        neg_i = np.asarray(negative_i, dtype=np.int64)
        neg_j = np.asarray(negative_j, dtype=np.int64)
        negative_difference = embeddings[neg_i] - embeddings[neg_j]
        negative_distance = np.sum(
            negative_difference * negative_difference,
            axis=1,
        )
        negative_mean = float(np.mean(negative_distance))
        ratio = float(
            np.mean(positive_distance) / (negative_mean + 1e-12)
        )
    else:
        negative_mean = float("nan")
        ratio = float("nan")

    return {
        "pos_dist": float(np.mean(positive_distance)),
        "neg_dist": negative_mean,
        "ratio": ratio,
        "pos_cos": float(np.mean(positive_cosine)),
        "n_pos_pairs": int(len(positive_pairs)),
        "n_neg_pairs": int(len(negative_i)),
    }


def compute_uniformity(
    embeddings: np.ndarray,
    n_pairs: int,
    seed: int,
    temperature: float = 2.0,
) -> dict[str, float | int]:
    """
    Approximate uniformity:
        log E[exp(-temperature * ||z_i - z_j||^2)]

    Lower values indicate a more uniformly spread embedding space.
    """
    embeddings = embeddings.astype(np.float32, copy=False)
    n_samples = len(embeddings)

    if n_samples < 2:
        return {
            "uniformity": float("nan"),
            "uniformity_n_pairs": 0,
        }

    rng = np.random.default_rng(seed)
    left = rng.integers(0, n_samples, size=max(n_pairs * 2, 100))
    right = rng.integers(0, n_samples, size=max(n_pairs * 2, 100))
    valid = left != right
    left = left[valid][:n_pairs]
    right = right[valid][:n_pairs]

    if len(left) == 0:
        return {
            "uniformity": float("nan"),
            "uniformity_n_pairs": 0,
        }

    difference = embeddings[left] - embeddings[right]
    squared_distance = np.sum(difference * difference, axis=1)
    values = -temperature * squared_distance

    # Stable log-mean-exp.
    maximum = float(np.max(values))
    uniformity = maximum + math.log(
        float(np.mean(np.exp(values - maximum)))
    )

    return {
        "uniformity": float(uniformity),
        "uniformity_n_pairs": int(len(left)),
    }


def compute_retrieval_at_k(
    embeddings: np.ndarray,
    utterance_ids: list[str],
    speaker_ids: list[str],
    ks: list[int],
    chunk_size: int = 512,
) -> dict[str, float | int]:
    """
    An anchor is correct at K when one of its top-K neighbours:
      - has the same prompt
      - comes from a different speaker
    """
    embeddings = embeddings.astype(np.float32, copy=False)
    utterances = np.asarray(utterance_ids)
    speakers = np.asarray(speaker_ids)
    n_samples = len(embeddings)

    ks = sorted(set(int(k) for k in ks if int(k) > 0))
    if not ks:
        raise ValueError("retrieval_ks must contain at least one positive integer.")

    max_k = min(max(ks), max(n_samples - 1, 1))

    valid_anchor = np.zeros(n_samples, dtype=bool)
    for index in range(n_samples):
        valid_anchor[index] = np.any(
            (utterances == utterances[index])
            & (speakers != speakers[index])
        )

    denominator = int(valid_anchor.sum())
    if denominator == 0:
        result: dict[str, float | int] = {
            f"retrieval_at_{k}": float("nan")
            for k in ks
        }
        result["retrieval_n_anchors"] = 0
        return result

    counts = {k: 0 for k in ks}

    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        similarities = embeddings[start:stop] @ embeddings.T

        local_rows = np.arange(stop - start)
        global_rows = np.arange(start, stop)
        similarities[local_rows, global_rows] = -np.inf

        top_indices = np.argpartition(
            -similarities,
            kth=max_k - 1,
            axis=1,
        )[:, :max_k]

        top_scores = np.take_along_axis(
            similarities,
            top_indices,
            axis=1,
        )
        order = np.argsort(-top_scores, axis=1)
        top_indices = np.take_along_axis(
            top_indices,
            order,
            axis=1,
        )

        for local_index, global_index in enumerate(range(start, stop)):
            if not valid_anchor[global_index]:
                continue

            neighbours = top_indices[local_index]
            hits = (
                (utterances[neighbours] == utterances[global_index])
                & (speakers[neighbours] != speakers[global_index])
            )

            for k in ks:
                effective_k = min(k, len(hits))
                if np.any(hits[:effective_k]):
                    counts[k] += 1

    result = {
        f"retrieval_at_{k}": float(counts[k] / denominator)
        for k in ks
    }
    result["retrieval_n_anchors"] = denominator
    return result


@torch.no_grad()
def run_simple_dev_ctc_eval(
    model: SupConMeanPoolModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()

    total_ctc_loss = 0.0
    n_batches = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        if "ctc_targets" not in batch:
            continue

        audio = batch["audio"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(
            device,
            non_blocking=True,
        )

        output = model(
            audio,
            attention_mask=attention_mask,
            compute_ctc=True,
        )

        log_probs = F.log_softmax(
            output["ctc_logits"].float(),
            dim=-1,
        ).permute(1, 0, 2)

        ctc_loss = model.ctc_loss_fn(
            log_probs,
            batch["ctc_targets"].to(device, non_blocking=True),
            output["frame_lengths"],
            batch["ctc_target_lengths"].to(
                device,
                non_blocking=True,
            ),
        )

        total_ctc_loss += float(ctc_loss.item())
        n_batches += 1

    return {
        "dev_ctc_loss": (
            total_ctc_loss / n_batches
            if n_batches > 0
            else float("nan")
        )
    }


def run_representation_eval(
    model: SupConMeanPoolModel,
    loader: DataLoader,
    device: torch.device,
    retrieval_ks: list[int],
    n_neg_samples: int,
    max_batches: int | None,
    seed: int,
    metrics: list[str],
) -> dict[str, float | int]:
    logger.info("Running Stage 2 mean-pooling representation diagnostics...")

    pack = extract_eval_embeddings(
        model=model,
        loader=loader,
        device=device,
        max_batches=max_batches,
    )

    utterance_ids = pack["utterance_ids"]
    speaker_ids = pack["speaker_ids"]

    result: dict[str, float | int] = {
        "n_samples": int(len(utterance_ids)),
        "n_utterances": int(len(set(utterance_ids))),
        "n_speakers": int(len(set(speaker_ids))),
    }

    enabled = set(metrics)

    # Backward compatibility with older configs that listed metrics as
    # retrieval_at_1 / retrieval_at_5 / retrieval_at_10.
    if any(metric.startswith("retrieval_at_") for metric in enabled):
        enabled.add("retrieval")

    for space in ("proj", "backbone"):
        embeddings = pack[space]

        if "alignment" in enabled:
            alignment = compute_alignment_metrics(
                embeddings=embeddings,
                utterance_ids=utterance_ids,
                speaker_ids=speaker_ids,
                n_neg_samples=n_neg_samples,
                seed=seed,
            )
            result[f"alignment_pos_{space}"] = alignment["pos_dist"]
            result[f"alignment_neg_{space}"] = alignment["neg_dist"]
            result[f"alignment_ratio_{space}"] = alignment["ratio"]
            result[f"alignment_cos_{space}"] = alignment["pos_cos"]
            result[f"n_pos_pairs_{space}"] = alignment["n_pos_pairs"]
            result[f"n_neg_pairs_{space}"] = alignment["n_neg_pairs"]

        if "uniformity" in enabled:
            uniformity = compute_uniformity(
                embeddings=embeddings,
                n_pairs=n_neg_samples,
                seed=seed,
            )
            result[f"uniformity_{space}"] = uniformity["uniformity"]
            result[f"uniformity_n_pairs_{space}"] = (
                uniformity["uniformity_n_pairs"]
            )

        if "retrieval" in enabled:
            retrieval = compute_retrieval_at_k(
                embeddings=embeddings,
                utterance_ids=utterance_ids,
                speaker_ids=speaker_ids,
                ks=retrieval_ks,
            )
            for key, value in retrieval.items():
                result[f"{key}_{space}"] = value

    logger.info(
        "Representation diagnostics | "
        f"n={result['n_samples']} | "
        f"prompts={result['n_utterances']} | "
        f"speakers={result['n_speakers']}"
    )

    for key in sorted(result):
        if key.startswith(
            ("alignment_", "uniformity_", "retrieval_at_")
        ):
            logger.info(f"  {key}={result[key]}")

    return result


# =============================================================================
# Output and plotting
# =============================================================================

def save_eval_outputs(
    eval_result: dict[str, Any],
    eval_output_dir: Path,
    epoch: int,
) -> None:
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    epoch_dir = eval_output_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    row = {"epoch": int(epoch), **to_jsonable(eval_result)}

    (epoch_dir / "metrics.json").write_text(
        json.dumps(row, indent=2),
        encoding="utf-8",
    )

    history_path = eval_output_dir / "metrics_history.csv"
    new_row = pd.DataFrame([row])

    if history_path.exists():
        old_history = pd.read_csv(history_path)
        old_history = old_history[old_history["epoch"] != epoch]
        history = pd.concat(
            [old_history, new_row],
            ignore_index=True,
        ).sort_values("epoch")
    else:
        history = new_row

    history.to_csv(history_path, index=False)
    plot_eval_history(history_path, eval_output_dir)

    logger.info(f"Evaluation metrics saved to {eval_output_dir}")


def plot_eval_history(history_path: Path, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.read_csv(history_path).sort_values("epoch")
    if frame.empty:
        return

    def make_plot(
        columns: list[str],
        filename: str,
        ylabel: str,
    ) -> None:
        existing = [column for column in columns if column in frame.columns]
        if not existing:
            return

        plt.figure(figsize=(8, 5))

        for column in existing:
            plt.plot(
                frame["epoch"],
                frame[column],
                marker="o",
                label=column,
            )

        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=200)
        plt.close()

    make_plot(
        [
            "alignment_pos_proj",
            "alignment_neg_proj",
            "alignment_pos_backbone",
            "alignment_neg_backbone",
        ],
        "alignment_pos_neg.png",
        "mean squared L2 distance",
    )
    make_plot(
        [
            "alignment_ratio_proj",
            "alignment_ratio_backbone",
        ],
        "alignment_ratio.png",
        "positive / negative distance",
    )
    make_plot(
        [
            "alignment_cos_proj",
            "alignment_cos_backbone",
        ],
        "alignment_positive_cosine.png",
        "positive cosine similarity",
    )
    make_plot(
        ["uniformity_proj", "uniformity_backbone"],
        "uniformity.png",
        "uniformity",
    )

    retrieval_columns = [
        column
        for column in frame.columns
        if column.startswith("retrieval_at_")
    ]
    make_plot(
        retrieval_columns,
        "retrieval_at_k.png",
        "Recall@K",
    )
    make_plot(
        ["dev_ctc_loss"],
        "dev_ctc_loss.png",
        "CTC loss",
    )


def save_checkpoint(
    model: SupConMeanPoolModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    save_dir: Path,
    epoch: int,
    global_step: int,
    train_metrics: dict[str, float],
    eval_metrics: dict[str, Any] | None,
    args: argparse.Namespace,
    name: str | None = None,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / (
        name if name is not None else f"checkpoint_epoch{epoch:03d}.pt"
    )

    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_metrics": to_jsonable(train_metrics),
            "eval_metrics": to_jsonable(eval_metrics),
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        checkpoint_path,
    )

    logger.info(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


# =============================================================================
# Training
# =============================================================================

def train_one_epoch(
    model: SupConMeanPoolModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    use_ctc: bool,
    use_mixed_precision: bool,
    grad_clip: float,
    verbose_timing: bool,
    log_every_n_batches: int,
) -> dict[str, float]:
    model.train()

    use_amp = use_mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    accumulated = {
        "loss": 0.0,
        "supcon_loss": 0.0,
        "ctc_loss": 0.0,
    }
    n_batches = 0
    n_total = len(loader)
    previous_end = time.perf_counter()

    for batch in loader:
        step_start = time.perf_counter()
        data_wait = step_start - previous_end

        audio = batch["audio"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(
            device,
            non_blocking=True,
        )
        labels = batch["labels"].to(device, non_blocking=True)

        ctc_targets = (
            batch["ctc_targets"].to(device, non_blocking=True)
            if use_ctc
            else None
        )
        ctc_target_lengths = (
            batch["ctc_target_lengths"].to(
                device,
                non_blocking=True,
            )
            if use_ctc
            else None
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=use_amp,
        ):
            output = model(
                audio,
                attention_mask=attention_mask,
                compute_ctc=use_ctc,
            )

            losses = model.compute_loss(
                embeddings=output["embeddings"],
                labels=labels,
                use_ctc=use_ctc,
                ctc_logits=output["ctc_logits"],
                ctc_targets=ctc_targets,
                ctc_input_lengths=output["frame_lengths"],
                ctc_target_lengths=ctc_target_lengths,
            )

        if not torch.isfinite(losses["loss"]):
            raise RuntimeError(
                "NaN/Inf loss detected: "
                f"loss={losses['loss']} "
                f"supcon={losses['supcon_loss']} "
                f"ctc={losses['ctc_loss']}"
            )

        if use_amp:
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            optimizer.step()

        scheduler.step()

        if verbose_timing:
            _sync_if_cuda(device)

        step_end = time.perf_counter()

        if (
            verbose_timing
            or n_batches % max(log_every_n_batches, 1) == 0
        ):
            logger.info(
                f"Batch {n_batches + 1}/{n_total} | "
                f"loss={losses['loss'].item():.4f} | "
                f"supcon={losses['supcon_loss'].item():.4f} | "
                f"ctc={losses['ctc_loss'].item():.4f} | "
                f"data={data_wait * 1000:.0f}ms | "
                f"step={(step_end - step_start) * 1000:.0f}ms | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
                f"{_gpu_mem_str(device)}"
            )

        for key in accumulated:
            accumulated[key] += float(losses[key].item())

        n_batches += 1
        previous_end = time.perf_counter()

    if n_batches == 0:
        raise RuntimeError("The training loader produced no batches.")

    return {
        key: value / n_batches
        for key, value in accumulated.items()
    }


# =============================================================================
# Configuration
# =============================================================================

CONFIG_SCHEMA: dict[str, tuple[str, Any]] = {
    # Paths
    "parquet_path": ("path", None),
    "save_dir": ("path", None),
    "tensorboard_dir": ("path", None),
    "eval_output_dir": ("path", None),

    # Data
    "sample_rate": ("int", 16_000),
    "max_audio_len_s": ("float", 10.0),
    "label_col": ("str", "prompt_id"),
    "num_workers": ("int", 2),
    "train_split": ("str", "train"),
    "dev_split": ("str", "dev"),
    "validate_audio": ("bool", True),

    # Sampler
    "k_utterances": ("int", 15),
    "s_speakers": ("int", 12),
    "n_batches": ("int", 100),
    "seed": ("int", 42),

    # Model
    "model_name": ("str", "facebook/wav2vec2-large-xlsr-53"),
    "proj_hidden_dim": ("int", 512),
    "proj_out_dim": ("int", 256),
    "vocab_size": ("int", 32),
    "min_frozen_layer": ("int", 0),
    "max_frozen_layer": ("int", 18),
    "ctc_lambda": ("float", 0.1),
    "temperature": ("float", 0.1),
    "enable_gradient_checkpointing": ("bool", True),

    # Optimization
    "epochs": ("int", 50),
    "lr": ("float", 2e-5),
    "weight_decay": ("float", 1e-4),
    "warmup_steps": ("int", 500),
    "grad_clip": ("float", 1.0),
    "use_ctc": ("bool", True),
    "tokenizer": ("str", "facebook/wav2vec2-large-960h"),
    "device": ("str", "cuda"),
    "use_mixed_precision": ("bool", False),

    # Checkpointing and evaluation
    "save_every_n_epochs": ("int", 1),
    "eval_every_n_epochs": ("int", 1),
    "eval_max_batches": ("optional_int", None),
    "eval_n_neg_samples": ("int", 5_000),
    "retrieval_ks": ("int_list", [1, 5, 10]),
    "eval_metrics": (
        "str_list",
        ["alignment", "uniformity", "retrieval"],
    ),
    "best_metric": ("str", "alignment_ratio_backbone"),
    "eval_k_utterances": ("int", 8),
    "eval_s_speakers": ("int", 3),
    "eval_n_batches": ("int", 20),

    # Logging
    "verbose_timing": ("bool", False),
    "log_every_n_batches": ("int", 1),
}


def _cast_config(value: Any, kind: str) -> Any:
    if value is None:
        return None
    if kind == "path":
        return Path(value)
    if kind == "int":
        return int(value)
    if kind == "optional_int":
        return None if value in {None, "null", "None", ""} else int(value)
    if kind == "float":
        return float(value)
    if kind == "bool":
        return str2bool(value)
    if kind == "str":
        return str(value)
    if kind == "int_list":
        return _parse_int_list(value, [1, 5, 10])
    if kind == "str_list":
        return _parse_str_list(
            value,
            ["alignment", "uniformity", "retrieval"],
        )

    raise ValueError(f"Unknown config kind: {kind}")


def _flatten_nested_project_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Convert the repository's nested Stage 2 config block to the flat names
    accepted by this script.
    """
    section = config.get("supervised_contrastive_training_l2cv")
    if not isinstance(section, dict):
        return config

    flat: dict[str, Any] = {}

    for group_name in (
        "data",
        "sampler",
        "model",
        "training",
        "evaluation",
    ):
        group = section.get(group_name, {})
        if isinstance(group, dict):
            flat.update(group)

    aliases = {
        "learning_rate": "lr",
        "checkpoint_dir": "save_dir",
    }

    for old_key, new_key in aliases.items():
        if old_key in flat and new_key not in flat:
            flat[new_key] = flat[old_key]

    return flat


def load_config(
    config_path: Path | None,
    cli_overrides: dict[str, Any],
) -> argparse.Namespace:
    loaded_config: dict[str, Any] = {}

    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        if not isinstance(raw, dict):
            raise ValueError(
                f"Config {config_path} must contain a YAML mapping."
            )

        loaded_config = _flatten_nested_project_config(raw)
        logger.info(f"Loaded config from {config_path}")

    for key, value in cli_overrides.items():
        if value is not None:
            loaded_config[key] = value

    final: dict[str, Any] = {}

    for key, (kind, default) in CONFIG_SCHEMA.items():
        raw_value = loaded_config.get(key, default)
        final[key] = _cast_config(raw_value, kind)

    known_aliases = {"learning_rate", "checkpoint_dir"}
    unknown = (
        set(loaded_config)
        - set(CONFIG_SCHEMA)
        - known_aliases
        - {"script"}
    )
    if unknown:
        logger.warning(
            f"Ignored unknown config keys: {sorted(unknown)}"
        )

    for required in ("parquet_path", "save_dir", "tensorboard_dir"):
        if final[required] is None:
            raise ValueError(
                f"Missing required config field: {required}"
            )

    if final["eval_output_dir"] is None:
        final["eval_output_dir"] = (
            final["save_dir"].parent / "eval" / "stage_2_representation"
        )

    if final["train_split"] not in VALID_SPLITS:
        raise ValueError(
            f"Invalid train_split={final['train_split']!r}"
        )
    if final["dev_split"] not in VALID_SPLITS:
        raise ValueError(
            f"Invalid dev_split={final['dev_split']!r}"
        )
    if final["dev_split"] == "test":
        raise ValueError(
            "Do not use the test split for monitoring or checkpoint selection."
        )

    return argparse.Namespace(**final)


def parse_args() -> tuple[Path | None, dict[str, Any]]:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2 mean-pooling supervised contrastive training "
            "on L2-ARCTIC CV."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML configuration file.",
    )

    for key, (kind, _) in CONFIG_SCHEMA.items():
        flag = f"--{key}"

        if kind == "bool":
            parser.add_argument(flag, type=str2bool, default=None)
        elif kind in {"int", "optional_int"}:
            parser.add_argument(flag, type=int, default=None)
        elif kind == "float":
            parser.add_argument(flag, type=float, default=None)
        elif kind == "path":
            parser.add_argument(flag, type=Path, default=None)
        elif kind in {"int_list", "str_list"}:
            # Accept comma-separated values so Snakemake can pass one argument.
            parser.add_argument(flag, type=str, default=None)
        else:
            parser.add_argument(flag, type=str, default=None)

    namespace = parser.parse_args()

    overrides = {
        key: getattr(namespace, key)
        for key in CONFIG_SCHEMA
    }

    return namespace.config, overrides


def metric_is_better(
    current: float,
    best: float,
    metric_name: str,
) -> bool:
    if np.isnan(current):
        return False
    if np.isnan(best):
        return True

    lower_is_better = {
        "loss",
        "supcon_loss",
        "ctc_loss",
        "dev_ctc_loss",
        "alignment_pos_proj",
        "alignment_pos_backbone",
        "alignment_ratio_proj",
        "alignment_ratio_backbone",
        "uniformity_proj",
        "uniformity_backbone",
    }

    return (
        current < best
        if metric_name in lower_is_better
        else current > best
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    config_path, cli_overrides = parse_args()
    args = load_config(config_path, cli_overrides)

    seed_everything(args.seed)

    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )

    logger.info("=" * 80)
    logger.info("Stage 2 SupCon mean-pooling training on L2-ARCTIC CV")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Arguments: {vars(args)}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    args.eval_output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if args.use_ctc:
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            args.tokenizer
        )

        if len(tokenizer) != args.vocab_size:
            raise ValueError(
                f"Tokenizer vocab size is {len(tokenizer)}, but "
                f"vocab_size={args.vocab_size}."
            )

        if tokenizer.pad_token_id != 0:
            raise ValueError(
                "This script configures CTCLoss(blank=0), but the tokenizer "
                f"pad_token_id is {tokenizer.pad_token_id}."
            )

        logger.info(
            f"Tokenizer: {args.tokenizer} | "
            f"vocab_size={len(tokenizer)} | "
            f"pad_token_id={tokenizer.pad_token_id}"
        )

    train_collate = (
        partial(
            collate_supcon_l2cv_with_tokenizer,
            tokenizer=tokenizer,
        )
        if args.use_ctc
        else collate_supcon_l2cv
    )
    eval_collate = train_collate

    train_dataset = SupConL2ArcticCVDataset(
        parquet_path=args.parquet_path,
        split=args.train_split,
        sample_rate=args.sample_rate,
        max_audio_len_s=args.max_audio_len_s,
        label_col=args.label_col,
        validate_audio=args.validate_audio,
    )
    dev_dataset = SupConL2ArcticCVDataset(
        parquet_path=args.parquet_path,
        split=args.dev_split,
        sample_rate=args.sample_rate,
        max_audio_len_s=args.max_audio_len_s,
        label_col=args.label_col,
        validate_audio=args.validate_audio,
    )

    train_sampler = SupConL2CVBatchSampler(
        dataset=train_dataset,
        k_utterances=args.k_utterances,
        s_speakers=args.s_speakers,
        n_batches=args.n_batches,
        seed=args.seed,
    )
    eval_sampler = SupConL2CVBatchSampler(
        dataset=dev_dataset,
        k_utterances=args.eval_k_utterances,
        s_speakers=args.eval_s_speakers,
        n_batches=args.eval_n_batches,
        seed=args.seed + 999,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=train_collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_sampler=eval_sampler,
        collate_fn=eval_collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = SupConMeanPoolModel(
        model_name=args.model_name,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_out_dim=args.proj_out_dim,
        vocab_size=args.vocab_size,
        ctc_lambda=args.ctc_lambda,
        temperature=args.temperature,
        min_frozen_layer=args.min_frozen_layer,
        max_frozen_layer=args.max_frozen_layer,
        enable_gradient_checkpointing=args.enable_gradient_checkpointing,
    ).to(device)

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, total_steps),
        num_training_steps=total_steps,
    )

    logger.info(
        f"Training schedule: {args.epochs} epochs x "
        f"{len(train_loader)} batches = {total_steps} steps"
    )

    writer = SummaryWriter(log_dir=str(args.tensorboard_dir))
    writer.add_scalar(
        "Data/train_dropped_audio",
        train_dataset.audio_filter_report.n_dropped,
        0,
    )
    writer.add_scalar(
        "Data/dev_dropped_audio",
        dev_dataset.audio_filter_report.n_dropped,
        0,
    )
    writer.add_scalar("Data/train_rows", len(train_dataset), 0)
    writer.add_scalar("Data/dev_rows", len(dev_dataset), 0)

    best_score = float("nan")
    best_epoch: int | None = None
    global_step = 0
    last_train_metrics: dict[str, float] | None = None
    last_eval_metrics: dict[str, Any] | None = None

    try:
        for epoch in range(1, args.epochs + 1):
            train_sampler.rng.seed(args.seed + epoch)
            eval_sampler.rng.seed(args.seed + 999 + epoch)

            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                use_ctc=args.use_ctc,
                use_mixed_precision=args.use_mixed_precision,
                grad_clip=args.grad_clip,
                verbose_timing=args.verbose_timing,
                log_every_n_batches=args.log_every_n_batches,
            )

            last_train_metrics = train_metrics
            global_step += len(train_loader)

            logger.info(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"loss={train_metrics['loss']:.4f} | "
                f"supcon={train_metrics['supcon_loss']:.4f} | "
                f"ctc={train_metrics['ctc_loss']:.4f}"
            )

            for key, value in train_metrics.items():
                writer.add_scalar(f"Train/{key}", value, epoch)
            writer.add_scalar(
                "Optim/lr",
                scheduler.get_last_lr()[0],
                epoch,
            )

            eval_metrics: dict[str, Any] | None = None

            if epoch % args.eval_every_n_epochs == 0:
                eval_metrics = run_representation_eval(
                    model=model,
                    loader=dev_loader,
                    device=device,
                    retrieval_ks=args.retrieval_ks,
                    n_neg_samples=args.eval_n_neg_samples,
                    max_batches=args.eval_max_batches,
                    seed=args.seed,
                    metrics=args.eval_metrics,
                )

                if args.use_ctc:
                    eval_metrics.update(
                        run_simple_dev_ctc_eval(
                            model=model,
                            loader=dev_loader,
                            device=device,
                            max_batches=args.eval_max_batches,
                        )
                    )

                last_eval_metrics = eval_metrics
                save_eval_outputs(
                    eval_result=eval_metrics,
                    eval_output_dir=args.eval_output_dir,
                    epoch=epoch,
                )

                for key, value in eval_metrics.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        writer.add_scalar(f"Eval/{key}", value, epoch)

                if args.best_metric not in eval_metrics:
                    raise KeyError(
                        f"best_metric={args.best_metric!r} was not produced. "
                        f"Available metrics: {sorted(eval_metrics)}"
                    )

                current_score = float(
                    eval_metrics[args.best_metric]
                )

                logger.info(
                    f"Best metric check | "
                    f"{args.best_metric}={current_score:.6f} | "
                    f"previous={best_score}"
                )

                if metric_is_better(
                    current=current_score,
                    best=best_score,
                    metric_name=args.best_metric,
                ):
                    best_score = current_score
                    best_epoch = epoch

                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        save_dir=args.save_dir,
                        epoch=epoch,
                        global_step=global_step,
                        train_metrics=train_metrics,
                        eval_metrics=eval_metrics,
                        args=args,
                        name="checkpoint_best.pt",
                    )

            if epoch % args.save_every_n_epochs == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    save_dir=args.save_dir,
                    epoch=epoch,
                    global_step=global_step,
                    train_metrics=train_metrics,
                    eval_metrics=eval_metrics,
                    args=args,
                )

        if last_train_metrics is None:
            raise RuntimeError("No training epoch completed.")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=args.save_dir,
            epoch=args.epochs,
            global_step=global_step,
            train_metrics=last_train_metrics,
            eval_metrics=last_eval_metrics,
            args=args,
            name="checkpoint_final.pt",
        )

        summary = {
            "pooling": "masked_mean",
            "use_ctc": args.use_ctc,
            "best_epoch": best_epoch,
            "best_metric": args.best_metric,
            "best_score": (
                None if np.isnan(best_score) else float(best_score)
            ),
            "final_epoch": args.epochs,
            "global_step": global_step,
            "train_split": args.train_split,
            "dev_split": args.dev_split,
            "train_rows": len(train_dataset),
            "dev_rows": len(dev_dataset),
            "train_dropped_audio": (
                train_dataset.audio_filter_report.n_dropped
            ),
            "dev_dropped_audio": (
                dev_dataset.audio_filter_report.n_dropped
            ),
            "model_name": args.model_name,
        }

        (args.save_dir / "training_summary.json").write_text(
            json.dumps(to_jsonable(summary), indent=2),
            encoding="utf-8",
        )

        logger.info("Training complete.")

    finally:
        writer.close()


if __name__ == "__main__":
    main()