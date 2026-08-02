#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
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
from loguru import logger
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, Wav2Vec2CTCTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_audio


VALID_SPLITS = {"train", "dev", "test"}


# =============================================================================
# Utils
# =============================================================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() == "true":
        return True
    if str(v).lower() == "false":
        return False
    raise argparse.ArgumentTypeError("Expected boolean value: true or false.")


def to_jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float):
        if np.isnan(x):
            return None
        return x
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    return x


def is_readable_audio(path: str) -> bool:
    try:
        p = Path(path)
        if not p.exists():
            return False
        info = sf.info(str(p))
        return info.frames > 0
    except Exception:
        return False


@dataclass
class AudioFilterReport:
    n_before: int
    n_after: int
    n_dropped: int
    bad_paths: list[str]


# =============================================================================
# Dataset / Sampler / Collate
# =============================================================================

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

        required = {"audio_path", "transcript", "speaker_id", "utterance_id", "split"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {parquet_path}: {missing}")

        if label_col not in df.columns:
            logger.warning(
                f"label_col='{label_col}' not found in {parquet_path}. Falling back to utterance_id."
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
                        f"bad audio | speaker={bad_row.get('speaker_id')} | "
                        f"utterance={bad_row.get('utterance_id')} | "
                        f"path={bad_row.get('audio_path')}"
                    )

            df = df[readable_mask].reset_index(drop=True)

            self.audio_filter_report = AudioFilterReport(
                n_before=n_before,
                n_after=len(df),
                n_dropped=n_before - len(df),
                bad_paths=bad_df["audio_path"].astype(str).tolist(),
            )

            logger.info(
                f"Audio readability filter [{split}]: kept {len(df):,}/{n_before:,} rows"
            )

            if df.empty:
                raise ValueError(f"No readable audio rows left for split='{split}'")

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
        self.utt2label = {utt_id: i for i, utt_id in enumerate(self.valid_utts)}

        logger.info("SupConL2ArcticCVDataset loaded:")
        logger.info(f"  Parquet       : {parquet_path}")
        logger.info(f"  Split         : {self.split}")
        logger.info(f"  Rows          : {len(self.df):,}")
        logger.info(f"  Valid prompts : {len(self.valid_utts):,}")
        logger.info(f"  Speakers      : {self.df['speaker_id'].nunique()}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.loc[idx]

        label_key = str(row[self.label_col])
        audio_path = str(row["audio_path"])

        audio = load_audio(
            audio_path,
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
            "native_language": str(row["native_language"]) if "native_language" in row else "unknown",
        }


class SupConL2CVBatchSampler(Sampler[list[int]]):
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

        if self.k * self.s > 64:
            logger.warning(
                "DTW SupCon is expensive. Recommended first run: "
                "k_utterances <= 8 and s_speakers <= 4."
            )

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


# =============================================================================
# Model + DTW SupCon
# =============================================================================

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


class CTCHead(nn.Module):
    def __init__(self, in_dim: int, vocab_size: int = 32) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states)


def cosine_cost_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sim = torch.matmul(x, y.T)
    return (1.0 - sim).clamp(min=0.0, max=2.0)


def softmin3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float) -> torch.Tensor:
    vals = torch.stack([a, b, c], dim=0)
    return -gamma * torch.logsumexp(-vals / gamma, dim=0)


def soft_dtw_distance(x: torch.Tensor, y: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """
    Differentiable Soft-DTW.
    x: [Tx, d]
    y: [Ty, d]
    """
    x = x.float()
    y = y.float()

    tx = x.shape[0]
    ty = y.shape[0]

    cost = cosine_cost_matrix(x, y)

    inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    R = torch.full((tx + 1, ty + 1), inf, device=x.device, dtype=x.dtype)
    R[0, 0] = 0.0

    for i in range(1, tx + 1):
        for j in range(1, ty + 1):
            R[i, j] = cost[i - 1, j - 1] + softmin3(
                R[i - 1, j],
                R[i, j - 1],
                R[i - 1, j - 1],
                gamma,
            )

    return R[tx, ty] / max(tx + ty, 1)


class SupConDTWLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        dtw_gamma: float = 0.1,
        dtw_max_frames: int = 160,
        dtw_downsample_stride: int = 2,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.dtw_gamma = dtw_gamma
        self.dtw_max_frames = dtw_max_frames
        self.dtw_downsample_stride = dtw_downsample_stride

    def _crop_sequence(self, z: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        n = int(length.item())
        z = z[:n]

        if self.dtw_downsample_stride > 1:
            z = z[:: self.dtw_downsample_stride]

        if z.shape[0] > self.dtw_max_frames:
            z = z[: self.dtw_max_frames]

        return z

    def _pairwise_similarity(self, frame_embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size = frame_embeddings.shape[0]
        device = frame_embeddings.device

        sim = torch.zeros(batch_size, batch_size, device=device, dtype=torch.float32)

        seqs = [
            self._crop_sequence(frame_embeddings[i], lengths[i])
            for i in range(batch_size)
        ]

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                dist = soft_dtw_distance(seqs[i], seqs[j], gamma=self.dtw_gamma)
                score = -dist
                sim[i, j] = score
                sim[j, i] = score

        return sim

    def forward(
        self,
        frame_embeddings: torch.Tensor,
        lengths: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = frame_embeddings.shape[0]
        device = frame_embeddings.device

        sim = self._pairwise_similarity(frame_embeddings, lengths)
        sim = sim / self.temperature

        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.T).float()

        self_mask = torch.eye(batch_size, device=device)
        pos_mask = pos_mask - self_mask
        all_mask = 1.0 - self_mask

        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim) * all_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        n_positives = pos_mask.sum(dim=1)
        valid = n_positives > 0

        if not valid.any():
            logger.warning("No positive pairs found in batch — SupCon DTW loss is zero.")
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss_per_anchor = -(pos_mask * log_prob).sum(dim=1) / (n_positives + 1e-8)
        return loss_per_anchor[valid].mean()


class SupConModel(nn.Module):
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
        dtw_gamma: float = 0.1,
        dtw_max_frames: int = 160,
        dtw_downsample_stride: int = 2,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.ctc_lambda = ctc_lambda
        self.min_frozen_layer = min_frozen_layer
        self.max_frozen_layer = max_frozen_layer

        logger.info(f"Loading speech backbone: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name)

        if enable_gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")

        if not hasattr(self.backbone.config, "hidden_size"):
            raise ValueError(f"Backbone config for {model_name} does not expose hidden_size.")

        hidden_size = self.backbone.config.hidden_size

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

        self.supcon_loss = SupConDTWLoss(
            temperature=temperature,
            dtw_gamma=dtw_gamma,
            dtw_max_frames=dtw_max_frames,
            dtw_downsample_stride=dtw_downsample_stride,
        )

        self.ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

        logger.info("SupConModel DTW initialized:")
        logger.info(f"  Backbone       : {model_name}")
        logger.info(f"  Hidden size    : {hidden_size}")
        logger.info(f"  Frozen layers  : {self.min_frozen_layer}-{self.max_frozen_layer - 1}")
        logger.info(f"  Projection     : {hidden_size} → {proj_hidden_dim} → {proj_out_dim}")
        logger.info(f"  Similarity     : Soft-DTW over frame embeddings")
        logger.info(f"  DTW gamma      : {dtw_gamma}")
        logger.info(f"  DTW max frames : {dtw_max_frames}")
        logger.info(f"  DTW stride     : {dtw_downsample_stride}")
        logger.info(f"  λ_CTC          : {ctc_lambda}")
        logger.info(f"  Temperature τ  : {temperature}")

    def _freeze_backbone_layers(self) -> None:
        if hasattr(self.backbone, "feature_extractor"):
            for param in self.backbone.feature_extractor.parameters():
                param.requires_grad = False
            logger.info("Frozen feature_extractor.")

        if hasattr(self.backbone, "feature_projection"):
            for param in self.backbone.feature_projection.parameters():
                param.requires_grad = False
            logger.info("Frozen feature_projection.")

        if not hasattr(self.backbone, "encoder") or not hasattr(self.backbone.encoder, "layers"):
            raise ValueError(f"Backbone {type(self.backbone)} does not expose encoder.layers.")

        layers = self.backbone.encoder.layers
        n_layers = len(layers)

        if self.max_frozen_layer > n_layers:
            logger.warning(
                f"max_frozen_layer={self.max_frozen_layer} > n_layers={n_layers}; clipping."
            )
            self.max_frozen_layer = n_layers

        for i, layer in enumerate(layers):
            if self.min_frozen_layer <= i < self.max_frozen_layer:
                for param in layer.parameters():
                    param.requires_grad = False

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())

        logger.info(
            f"Trainable params: {n_trainable:,} / {n_total:,} "
            f"({100 * n_trainable / n_total:.1f}%)"
        )

    def get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "_get_feat_extract_output_lengths"):
            return self.backbone._get_feat_extract_output_lengths(input_lengths)

        raise AttributeError(
            f"Backbone {type(self.backbone)} has no _get_feat_extract_output_lengths()."
        )

    def forward(self, audio: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict:
        outputs = self.backbone(
            input_values=audio,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        hidden_states = outputs.last_hidden_state
        ctc_logits = self.ctc_head(hidden_states)

        if attention_mask is not None:
            frame_lengths = self.get_feat_extract_output_lengths(
                attention_mask.sum(dim=-1).long()
            ).long()
        else:
            frame_lengths = torch.full(
                size=(hidden_states.shape[0],),
                fill_value=hidden_states.shape[1],
                device=hidden_states.device,
                dtype=torch.long,
            )

        frame_embeddings = self.projection(hidden_states)

        return {
            "frame_embeddings": frame_embeddings,
            "frame_lengths": frame_lengths,
            "ctc_logits": ctc_logits,
            "hidden_states": hidden_states,
        }

    def compute_loss(
        self,
        frame_embeddings: torch.Tensor,
        frame_lengths: torch.Tensor,
        labels: torch.Tensor,
        ctc_logits: torch.Tensor | None = None,
        ctc_targets: torch.Tensor | None = None,
        ctc_input_lengths: torch.Tensor | None = None,
        ctc_target_lengths: torch.Tensor | None = None,
    ) -> dict:
        l_supcon = self.supcon_loss(
            frame_embeddings=frame_embeddings,
            lengths=frame_lengths,
            labels=labels,
        )

        l_ctc = torch.tensor(0.0, device=frame_embeddings.device)

        if (
            ctc_logits is not None
            and ctc_targets is not None
            and ctc_input_lengths is not None
            and ctc_target_lengths is not None
        ):
            log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)

            l_ctc = self.ctc_loss_fn(
                log_probs,
                ctc_targets,
                ctc_input_lengths,
                ctc_target_lengths,
            )

        total = l_supcon + self.ctc_lambda * l_ctc

        return {
            "loss": total,
            "supcon_loss": l_supcon,
            "ctc_loss": l_ctc,
        }


# =============================================================================
# Eval
# =============================================================================

@torch.no_grad()
def run_simple_dev_eval(
    model: SupConModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 4,
) -> dict:
    """
    Lightweight dev monitoring only.
    DTW retrieval on full dev can be very slow, so this tracks only loss-like diagnostics.
    """
    model.eval()

    total_ctc = 0.0
    n = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        audio = batch["audio"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        out = model(audio, attention_mask=attention_mask)

        ctc_input_lengths = model.get_feat_extract_output_lengths(
            attention_mask.sum(dim=-1).long()
        ).long()

        if "ctc_targets" not in batch:
            continue

        losses = model.compute_loss(
            frame_embeddings=out["frame_embeddings"],
            frame_lengths=out["frame_lengths"],
            labels=batch["labels"].to(device),
            ctc_logits=out["ctc_logits"],
            ctc_targets=batch["ctc_targets"].to(device),
            ctc_input_lengths=ctc_input_lengths,
            ctc_target_lengths=batch["ctc_target_lengths"].to(device),
        )

        total_ctc += float(losses["ctc_loss"].item())
        n += 1

    return {
        "dev_ctc_loss": total_ctc / max(n, 1),
    }


# =============================================================================
# Training
# =============================================================================

def save_checkpoint(
    model: SupConModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    save_dir: Path,
    epoch: int,
    global_step: int,
    train_metrics: dict,
    eval_metrics: dict | None,
    args,
    name: str | None = None,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = save_dir / (name if name is not None else f"checkpoint_epoch{epoch:03d}.pt")

    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_metrics": to_jsonable(train_metrics),
            "eval_metrics": to_jsonable(eval_metrics),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        ckpt_path,
    )

    logger.info(f"Checkpoint saved → {ckpt_path}")
    return ckpt_path


def train_one_epoch(
    model: SupConModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    use_ctc: bool,
    use_mixed_precision: bool,
    grad_clip: float,
) -> dict:
    model.train()

    use_amp = use_mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    total_loss = 0.0
    total_supcon_loss = 0.0
    total_ctc_loss = 0.0
    n_batches = 0

    for batch in loader:
        audio = batch["audio"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(audio, attention_mask=attention_mask)

            if use_ctc:
                input_lengths = model.get_feat_extract_output_lengths(
                    attention_mask.sum(dim=-1).long()
                ).long()

                losses = model.compute_loss(
                    frame_embeddings=out["frame_embeddings"],
                    frame_lengths=out["frame_lengths"],
                    labels=labels,
                    ctc_logits=out["ctc_logits"],
                    ctc_targets=batch["ctc_targets"].to(device),
                    ctc_input_lengths=input_lengths,
                    ctc_target_lengths=batch["ctc_target_lengths"].to(device),
                )
            else:
                losses = model.compute_loss(
                    frame_embeddings=out["frame_embeddings"],
                    frame_lengths=out["frame_lengths"],
                    labels=labels,
                )

            if not torch.isfinite(losses["loss"]):
                logger.error(
                    f"NaN/Inf loss detected | "
                    f"loss={losses['loss']} | "
                    f"supcon={losses['supcon_loss']} | "
                    f"ctc={losses['ctc_loss']}"
                )
                raise RuntimeError("Stopping because loss is NaN/Inf.")

        if use_amp:
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        scheduler.step()

        total_loss += float(losses["loss"].item())
        total_supcon_loss += float(losses["supcon_loss"].item())
        total_ctc_loss += float(losses["ctc_loss"].item())
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "supcon_loss": total_supcon_loss / max(n_batches, 1),
        "ctc_loss": total_ctc_loss / max(n_batches, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 SupCon-DTW training on L2-ARCTIC CV."
    )

    parser.add_argument("--parquet_path", type=Path, required=True)
    parser.add_argument("--sample_rate", type=int, default=16_000)
    parser.add_argument("--max_audio_len_s", type=float, default=10.0)
    parser.add_argument("--label_col", type=str, default="prompt_id")
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--train_split", type=str, default="train", choices=["train", "dev", "test"])
    parser.add_argument("--dev_split", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--validate_audio", type=str2bool, default=True)

    parser.add_argument("--k_utterances", type=int, default=15)
    parser.add_argument("--s_speakers", type=int, default=12)
    parser.add_argument("--n_batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-large-xlsr-53")
    parser.add_argument("--proj_hidden_dim", type=int, default=512)
    parser.add_argument("--proj_out_dim", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=32)
    parser.add_argument("--min_frozen_layer", type=int, default=0)
    parser.add_argument("--max_frozen_layer", type=int, default=18)
    parser.add_argument("--ctc_lambda", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_ctc", type=str2bool, default=True)
    parser.add_argument("--tokenizer", type=str, default="facebook/wav2vec2-large-960h")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--use_mixed_precision", type=str2bool, default=False)

    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--tensorboard_dir", type=Path, required=True)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)

    parser.add_argument("--eval_every_n_epochs", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_n_neg_samples", type=int, default=1000)
    parser.add_argument("--retrieval_ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--eval_metrics", type=str, nargs="+", default=["ctc"])
    parser.add_argument("--best_metric", type=str, default="dev_ctc_loss")

    # Extra DTW args. Optional; old configs can ignore them.
    parser.add_argument("--dtw_gamma", type=float, default=0.1)
    parser.add_argument("--dtw_max_frames", type=int, default=160)
    parser.add_argument("--dtw_downsample_stride", type=int, default=2)
    parser.add_argument("--eval_max_batches", type=int, default=4)

    return parser.parse_args()


def is_better_metric(current: float, best: float, metric_name: str) -> bool:
    if np.isnan(current):
        return False
    if np.isnan(best):
        return True

    lower_is_better = {"dev_ctc_loss", "loss", "ctc_loss", "supcon_loss"}
    if metric_name in lower_is_better:
        return current < best

    return current > best


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info("=" * 80)
    logger.info("Stage 2 SupCon-DTW L2-ARCTIC CV training")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    if args.dev_split == "test":
        raise ValueError("Do not use split='test' for monitoring / checkpoint selection.")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = None

    if args.use_ctc:
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.tokenizer)
        logger.info(f"Tokenizer: {args.tokenizer} | vocab_size={len(tokenizer)}")
        logger.info(f"Tokenizer pad_token_id={tokenizer.pad_token_id}")

    train_collate = (
        partial(collate_supcon_l2cv_with_tokenizer, tokenizer=tokenizer)
        if args.use_ctc
        else collate_supcon_l2cv
    )

    eval_collate = train_collate if args.use_ctc else collate_supcon_l2cv

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

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=train_collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=eval_collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logger.info(f"Monitoring eval uses split='{args.dev_split}' from {args.parquet_path}")
    logger.info(
        f"Audio drops: train={train_dataset.audio_filter_report.n_dropped}, "
        f"dev={dev_dataset.audio_filter_report.n_dropped}"
    )

    model = SupConModel(
        model_name=args.model_name,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_out_dim=args.proj_out_dim,
        vocab_size=args.vocab_size,
        ctc_lambda=args.ctc_lambda,
        temperature=args.temperature,
        min_frozen_layer=args.min_frozen_layer,
        max_frozen_layer=args.max_frozen_layer,
        dtw_gamma=args.dtw_gamma,
        dtw_max_frames=args.dtw_max_frames,
        dtw_downsample_stride=args.dtw_downsample_stride,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(
        f"Training schedule: {args.epochs} epochs × {len(train_loader)} batches = {total_steps} steps"
    )

    writer = SummaryWriter(log_dir=str(args.tensorboard_dir))
    logger.info(f"TensorBoard → {args.tensorboard_dir}")

    best_score = float("nan")
    best_epoch = None
    global_step = 0

    last_train_metrics = None
    last_eval_metrics_dict = None

    writer.add_scalar("Data/train_dropped_audio", train_dataset.audio_filter_report.n_dropped, 0)
    writer.add_scalar("Data/dev_dropped_audio", dev_dataset.audio_filter_report.n_dropped, 0)
    writer.add_scalar("Data/train_rows", len(train_dataset), 0)
    writer.add_scalar("Data/dev_rows", len(dev_dataset), 0)

    for epoch in range(1, args.epochs + 1):
        train_sampler.rng.seed(args.seed + epoch)

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_ctc=args.use_ctc,
            use_mixed_precision=args.use_mixed_precision,
            grad_clip=args.grad_clip,
        )

        last_train_metrics = train_metrics
        global_step += len(train_loader)

        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={train_metrics['loss']:.4f} | "
            f"supcon={train_metrics['supcon_loss']:.4f} | "
            f"ctc={train_metrics['ctc_loss']:.4f}"
        )

        writer.add_scalar("Train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("Train/supcon_loss", train_metrics["supcon_loss"], epoch)
        writer.add_scalar("Train/ctc_loss", train_metrics["ctc_loss"], epoch)
        writer.add_scalar("Optim/lr", scheduler.get_last_lr()[0], epoch)

        eval_metrics_dict = None

        if epoch % args.eval_every_n_epochs == 0:
            eval_metrics_dict = run_simple_dev_eval(
                model=model,
                loader=dev_loader,
                device=device,
                max_batches=args.eval_max_batches,
            )
            last_eval_metrics_dict = eval_metrics_dict

            for k, v in eval_metrics_dict.items():
                writer.add_scalar(f"Eval/{k}", v, epoch)

            current_score = float(eval_metrics_dict.get(args.best_metric, np.nan))

            logger.info(
                f"Best metric check: {args.best_metric}={current_score:.6f} "
                f"| previous_best={best_score}"
            )

            if is_better_metric(current_score, best_score, args.best_metric):
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
                    eval_metrics=eval_metrics_dict,
                    args=args,
                    name="checkpoint_best.pt",
                )

                logger.info(
                    f"✓ New best checkpoint at epoch {epoch:03d}: "
                    f"{args.best_metric}={best_score:.6f}"
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
                eval_metrics=eval_metrics_dict,
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
        eval_metrics=last_eval_metrics_dict,
        args=args,
        name="checkpoint_final.pt",
    )

    metadata = {
        "best_epoch": best_epoch,
        "best_metric": args.best_metric,
        "best_score": None if np.isnan(best_score) else float(best_score),
        "final_epoch": args.epochs,
        "global_step": global_step,
        "train_split": args.train_split,
        "dev_split": args.dev_split,
        "train_rows": len(train_dataset),
        "dev_rows": len(dev_dataset),
        "train_dropped_audio": train_dataset.audio_filter_report.n_dropped,
        "dev_dropped_audio": dev_dataset.audio_filter_report.n_dropped,
        "train_bad_audio_paths": train_dataset.audio_filter_report.bad_paths,
        "dev_bad_audio_paths": dev_dataset.audio_filter_report.bad_paths,
        "model_name": args.model_name,
        "dtw_gamma": args.dtw_gamma,
        "dtw_max_frames": args.dtw_max_frames,
        "dtw_downsample_stride": args.dtw_downsample_stride,
    }

    metadata_path = args.save_dir / "training_summary.json"
    metadata_path.write_text(json.dumps(to_jsonable(metadata), indent=2))
    logger.info(f"Training summary → {metadata_path}")

    writer.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()