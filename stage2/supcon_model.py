"""
stage2/supcon_model.py

Generic supervised contrastive speech model.

Supports encoder-only HuggingFace speech backbones such as:
  - facebook/wav2vec2-large-xlsr-53
  - facebook/wav2vec2-large-960h
  - facebook/hubert-large-ls960-ft
  - microsoft/wavlm-large

The model contains:
  - a generic speech backbone loaded with AutoModel
  - a projection head for supervised contrastive learning
  - an auxiliary CTC head used as a regularizer
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModel


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


class CTCHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        vocab_size: int = 32,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states)


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = embeddings.shape[0]
        device = embeddings.device

        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

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
            logger.warning("No positive pairs found in batch — SupCon loss is zero.")
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss_per_anchor = -(pos_mask * log_prob).sum(dim=1) / (n_positives + 1e-8)
        return loss_per_anchor[valid].mean()


class SupConModel(nn.Module):
    """
    Generic supervised contrastive model for speech encoders.

    Forward returns:
      embeddings    : [B, proj_out_dim]
      pooled        : [B, hidden_size]
      ctc_logits    : [B, T', vocab_size]
      hidden_states : [B, T', hidden_size]
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
        self.ctc_lambda = ctc_lambda
        self.min_frozen_layer = min_frozen_layer
        self.max_frozen_layer = max_frozen_layer

        logger.info(f"Loading speech backbone: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name)

        if enable_gradient_checkpointing and hasattr(
            self.backbone,
            "gradient_checkpointing_enable",
        ):
            self.backbone.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")

        if not hasattr(self.backbone.config, "hidden_size"):
            raise ValueError(
                f"Backbone config for {model_name} does not expose hidden_size."
            )

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

        self.supcon_loss = SupConLoss(temperature=temperature)
        self.ctc_loss_fn = nn.CTCLoss(
            blank=0,
            reduction="mean",
            zero_infinity=True,
        )

        logger.info("SupConModel initialized:")
        logger.info(f"  Backbone       : {model_name}")
        logger.info(f"  Hidden size    : {hidden_size}")
        logger.info(f"  Frozen layers  : {self.min_frozen_layer}-{self.max_frozen_layer - 1}")
        logger.info(f"  Projection     : {hidden_size} → {proj_hidden_dim} → {proj_out_dim}")
        logger.info(f"  Vocab size     : {vocab_size}")
        logger.info(f"  λ_CTC          : {ctc_lambda}")
        logger.info(f"  Temperature τ  : {temperature}")

    def _freeze_backbone_layers(self) -> None:
        """
        Freeze CNN frontend / feature projection when present,
        then freeze transformer layers in [min_frozen_layer, max_frozen_layer).
        """

        if hasattr(self.backbone, "feature_extractor"):
            for param in self.backbone.feature_extractor.parameters():
                param.requires_grad = False
            logger.info("Frozen feature_extractor.")

        if hasattr(self.backbone, "feature_projection"):
            for param in self.backbone.feature_projection.parameters():
                param.requires_grad = False
            logger.info("Frozen feature_projection.")

        if not hasattr(self.backbone, "encoder") or not hasattr(
            self.backbone.encoder,
            "layers",
        ):
            raise ValueError(
                f"Backbone {type(self.backbone)} does not expose encoder.layers. "
                "Cannot apply transformer layer freezing."
            )

        layers = self.backbone.encoder.layers
        n_layers = len(layers)

        if self.max_frozen_layer > n_layers:
            logger.warning(
                f"max_frozen_layer={self.max_frozen_layer} > n_layers={n_layers}; "
                f"clipping max_frozen_layer to {n_layers}."
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

    def get_feat_extract_output_lengths(
        self,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generic wrapper around HF speech models' feature-length function.
        Wav2Vec2, HuBERT, and WavLM usually expose this method.
        """

        if hasattr(self.backbone, "_get_feat_extract_output_lengths"):
            return self.backbone._get_feat_extract_output_lengths(input_lengths)

        raise AttributeError(
            f"Backbone {type(self.backbone)} has no "
            "_get_feat_extract_output_lengths() method."
        )

    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict:
        outputs = self.backbone(
            input_values=audio,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        hidden_states = outputs.last_hidden_state
        ctc_logits = self.ctc_head(hidden_states)

        if attention_mask is not None:
            feat_lengths = self.get_feat_extract_output_lengths(
                attention_mask.sum(dim=-1).long()
            )
            pooled = self._masked_mean_pool(hidden_states, feat_lengths)
        else:
            pooled = hidden_states.mean(dim=1)

        embeddings = self.projection(pooled)

        return {
            "embeddings": embeddings,
            "pooled": pooled,
            "ctc_logits": ctc_logits,
            "hidden_states": hidden_states,
        }

    @staticmethod
    def _masked_mean_pool(
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, time_steps, hidden_dim = hidden_states.shape

        mask = (
            torch.arange(time_steps, device=hidden_states.device)
            .unsqueeze(0)
            < lengths.unsqueeze(1)
        )

        mask = mask.unsqueeze(-1).float()

        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        ctc_logits: torch.Tensor | None = None,
        ctc_targets: torch.Tensor | None = None,
        ctc_input_lengths: torch.Tensor | None = None,
        ctc_target_lengths: torch.Tensor | None = None,
    ) -> dict:
        l_supcon = self.supcon_loss(embeddings, labels)

        l_ctc = torch.tensor(0.0, device=embeddings.device)

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


# Backward-compatible alias.
# This lets old code/checkpoints still refer to SupConXLSR if needed.
SupConXLSR = SupConModel