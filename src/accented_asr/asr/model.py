"""Stage 3 ASR model initialization and Stage 2 backbone transfer."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from transformers import Wav2Vec2ForCTC


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage2_backbone(model, checkpoint_path: Path) -> dict:
    """Load only ``backbone.*`` tensors; discard Stage 2 projection/CTC heads."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Stage 2 checkpoint has no model_state_dict.")
    source = checkpoint["model_state_dict"]
    target = model.state_dict()
    mapped = {}
    unexpected = []
    shape_mismatches = []
    for key, value in source.items():
        if not key.startswith("backbone."):
            continue
        target_key = "wav2vec2." + key.removeprefix("backbone.")
        if target_key not in target:
            unexpected.append(target_key)
        elif target[target_key].shape != value.shape:
            shape_mismatches.append(target_key)
        else:
            mapped[target_key] = value
    if not mapped or unexpected or shape_mismatches:
        raise ValueError(
            "Invalid Stage 2 backbone transfer: "
            f"mapped={len(mapped)}, absent={unexpected[:5]}, "
            f"shape_mismatches={shape_mismatches[:5]}"
        )
    model.load_state_dict(mapped, strict=False)
    metadata = checkpoint.get("metadata", {})
    return {
        "mapped_tensors": len(mapped),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "stage2_epoch": metadata.get("epoch"),
        "stage2_loss_mode": metadata.get("loss_mode"),
        "stage2_manifest_sha256": metadata.get("split_manifest_sha256"),
        "stage2_seed": metadata.get("seed"),
        "stage2_fold": metadata.get("fold"),
        "stage2_backbone_name": metadata.get("backbone_name"),
    }


def build_asr_model(
    *, backbone_name: str, vocab_size: int, pad_token_id: int,
    stage2_checkpoint: Path, gradient_checkpointing: bool,
    mask_time_prob: float, mask_time_length: int,
    mask_feature_prob: float, mask_feature_length: int,
    layerdrop: float, activation_dropout: float,
):
    model = Wav2Vec2ForCTC.from_pretrained(
        backbone_name,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        mask_time_prob=mask_time_prob,
        mask_time_length=mask_time_length,
        mask_feature_prob=mask_feature_prob,
        mask_feature_length=mask_feature_length,
        layerdrop=layerdrop,
        activation_dropout=activation_dropout,
        ignore_mismatched_sizes=True,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    transfer = load_stage2_backbone(model, stage2_checkpoint)
    return model, transfer
