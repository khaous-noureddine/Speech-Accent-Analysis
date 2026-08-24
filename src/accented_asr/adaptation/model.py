"""Speech encoder, projection head, and explicit Stage 2 ablation losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


LOSS_MODES = frozenset({"supcon_ctc", "supcon_only", "ctc_only"})


class SupConLoss(nn.Module):
    """Mean supervised contrastive loss over anchors with positive examples."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        similarities = embeddings @ embeddings.T / self.temperature
        self_mask = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
        positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
        valid = positive_mask.any(dim=1)
        if not valid.all():
            raise ValueError("Every contrastive anchor must have at least one positive.")
        logits = similarities - similarities.max(dim=1, keepdim=True).values.detach()
        logits = logits.masked_fill(self_mask, float("-inf"))
        log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        per_anchor = -(
            log_probabilities.masked_fill(~positive_mask, 0.0).sum(dim=1)
            / positive_mask.sum(dim=1)
        )
        return per_anchor.mean()


class AdaptationModel(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str,
        vocab_size: int,
        projection_hidden_size: int = 512,
        projection_size: int = 256,
        temperature: float = 0.1,
        frozen_transformer_layers: int = 18,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = AutoModel.from_pretrained(backbone_name)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        hidden_size = self.backbone.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_hidden_size),
            nn.GELU(),
            nn.Linear(projection_hidden_size, projection_size),
        )
        self.ctc_head = nn.Linear(hidden_size, vocab_size)
        self.supcon_loss = SupConLoss(temperature)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self._freeze_backbone(frozen_transformer_layers)

    def _freeze_backbone(self, frozen_layers: int) -> None:
        if hasattr(self.backbone, "feature_extractor"):
            for parameter in self.backbone.feature_extractor.parameters():
                parameter.requires_grad = False
        layers = self.backbone.encoder.layers
        if not 0 <= frozen_layers <= len(layers):
            raise ValueError(
                f"frozen_transformer_layers={frozen_layers} for {len(layers)} layers."
            )
        for layer in layers[:frozen_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    def feature_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return self.backbone._get_feat_extract_output_lengths(input_lengths).long()

    @staticmethod
    def masked_mean(hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
        mask = (steps < lengths[:, None]).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def forward(self, audio: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        hidden = self.backbone(
            input_values=audio, attention_mask=attention_mask
        ).last_hidden_state
        lengths = self.feature_lengths(attention_mask.sum(dim=1))
        pooled = self.masked_mean(hidden, lengths)
        embeddings = F.normalize(self.projection(pooled), dim=-1)
        return {
            "embeddings": embeddings,
            "pooled": pooled,
            "ctc_logits": self.ctc_head(hidden),
            "feature_lengths": lengths,
        }

    def compute_losses(self, batch: dict, outputs: dict, *, mode: str, ctc_weight: float) -> dict:
        if mode not in LOSS_MODES:
            raise ValueError(f"Unknown loss mode {mode!r}; expected {sorted(LOSS_MODES)}.")
        zero = outputs["embeddings"].sum() * 0.0
        supcon = (
            self.supcon_loss(outputs["embeddings"], batch["labels"])
            if mode != "ctc_only" else zero
        )
        ctc = zero
        if mode != "supcon_only":
            required = {"ctc_targets", "ctc_target_lengths"}
            if not required <= batch.keys():
                raise ValueError(f"CTC mode requires batch keys {sorted(required)}.")
            log_probs = F.log_softmax(outputs["ctc_logits"], dim=-1).transpose(0, 1)
            ctc = self.ctc_loss(
                log_probs,
                batch["ctc_targets"],
                outputs["feature_lengths"],
                batch["ctc_target_lengths"],
            )
        total = supcon if mode == "supcon_only" else ctc if mode == "ctc_only" else supcon + ctc_weight * ctc
        return {"loss": total, "supcon_loss": supcon, "ctc_loss": ctc}
