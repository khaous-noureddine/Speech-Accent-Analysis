"""Model helpers for joint CTC and utterance-level SupCon training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from accented_asr.adaptation.model import SupConLoss


class ProjectionHead(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, output_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, output_size),
        )

    def forward(self, hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
        mask = (steps < lengths[:, None]).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return F.normalize(self.layers(pooled), dim=-1)


def configure_backbone_trainability(
    model,
    *,
    head_only: bool,
    frozen_transformer_layers: int,
    freeze_feature_encoder: bool,
) -> None:
    """Apply either head-only warm-up or the configured post-warm-up policy."""
    layers = model.wav2vec2.encoder.layers
    if not 0 <= frozen_transformer_layers <= len(layers):
        raise ValueError(
            f"frozen_transformer_layers={frozen_transformer_layers} "
            f"for {len(layers)} layers."
        )
    for parameter in model.wav2vec2.parameters():
        parameter.requires_grad = not head_only
    if not head_only:
        for layer in layers[:frozen_transformer_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
        if freeze_feature_encoder:
            for parameter in model.wav2vec2.feature_extractor.parameters():
                parameter.requires_grad = False
    for parameter in model.lm_head.parameters():
        parameter.requires_grad = True


def contrastive_loss(
    model,
    projection: ProjectionHead,
    criterion: SupConLoss,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    outputs = model.wav2vec2(
        input_values=batch["audio"], attention_mask=batch["attention_mask"]
    )
    lengths = model._get_feat_extract_output_lengths(
        batch["attention_mask"].sum(dim=1)
    ).long()
    embeddings = projection(outputs.last_hidden_state, lengths)
    return criterion(embeddings, batch["labels"])
