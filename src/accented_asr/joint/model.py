"""Model helpers for joint CTC and utterance-level SupCon training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

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


def masked_mean(hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    steps = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
    mask = (steps < lengths[:, None]).unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.scale * gradient, None


def gradient_reverse(inputs: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Identity in the forward pass and sign reversal for encoder gradients."""
    return _GradientReversal.apply(inputs, scale)


class AccentClassifier(nn.Module):
    """Utterance-level accent head shared by MTL and adversarial training."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_accents: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, num_accents),
        )

    def forward(self, hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.layers(masked_mean(hidden, lengths))


def encoder_hidden(model, batch: dict[str, torch.Tensor]):
    outputs = model.wav2vec2(
        input_values=batch["audio"], attention_mask=batch["attention_mask"]
    )
    lengths = model._get_feat_extract_output_lengths(
        batch["attention_mask"].sum(dim=1)
    ).long()
    return outputs.last_hidden_state, lengths


def accent_classification_loss(
    model,
    classifier: AccentClassifier,
    batch: dict[str, torch.Tensor],
    *,
    adversarial_scale: float | None = None,
) -> torch.Tensor:
    hidden, lengths = encoder_hidden(model, batch)
    if adversarial_scale is not None:
        hidden = gradient_reverse(hidden, adversarial_scale)
    return F.cross_entropy(classifier(hidden, lengths), batch["accent_labels"])


def augmented_view_supcon_loss(
    model,
    projection: ProjectionHead,
    criterion: SupConLoss,
    batch: dict[str, torch.Tensor],
    *,
    noise_std: float,
    time_mask_ratio: float,
) -> torch.Tensor:
    """Contrast each utterance with a stochastic waveform-level view of itself."""
    audio = batch["audio"]
    mask = batch["attention_mask"]
    augmented = audio + torch.randn_like(audio) * noise_std * mask
    if time_mask_ratio > 0:
        for index, length in enumerate(mask.sum(dim=1).tolist()):
            width = max(1, int(length * time_mask_ratio))
            if length > width:
                start = int(
                    torch.randint(length - width + 1, (1,), device=audio.device)
                )
                augmented[index, start:start + width] = 0
    paired = {
        "audio": torch.cat([audio, augmented]),
        "attention_mask": torch.cat([mask, mask]),
        "labels": torch.arange(audio.shape[0], device=audio.device).repeat(2),
    }
    return contrastive_loss(model, projection, criterion, paired)


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
    hidden, lengths = encoder_hidden(model, batch)
    embeddings = projection(hidden, lengths)
    return criterion(embeddings, batch["labels"])


def balanced_shuffled_labels(labels: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Create balanced pseudo-groups that break every true positive group.

    Each true class must occur equally often and fewer times than the number of
    classes. Every occurrence of a true class receives a different pseudo-label,
    while every pseudo-label retains the original class size.
    """
    unique, inverse, counts = torch.unique(
        labels, sorted=True, return_inverse=True, return_counts=True
    )
    if len(unique) < 2 or not torch.all(counts.eq(counts[0])):
        raise ValueError("Shuffled-label SupCon requires balanced content groups.")
    group_size = int(counts[0])
    if group_size >= len(unique):
        raise ValueError(
            "Shuffled-label SupCon requires fewer positives per group than classes."
        )
    occurrence = torch.empty_like(inverse)
    for class_index in range(len(unique)):
        positions = torch.nonzero(inverse.eq(class_index), as_tuple=True)[0]
        occurrence[positions] = torch.arange(group_size, device=labels.device)
    generator = torch.Generator(device=labels.device).manual_seed(seed)
    shifts = torch.randperm(
        len(unique), generator=generator, device=labels.device
    )[:group_size]
    shuffled = (inverse + shifts[occurrence]) % len(unique)
    return shuffled


def multidomain_ctc_supcon_losses(
    model,
    projection: ProjectionHead,
    criterion: SupConLoss,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute accented-speech CTC and content SupCon from one encoder pass."""
    hidden, lengths = encoder_hidden(model, batch)
    embeddings = projection(hidden, lengths)
    supcon = criterion(embeddings, batch["labels"])

    labels = batch["ctc_labels"]
    label_mask = labels.ge(0)
    target_lengths = label_mask.sum(dim=-1)
    flattened_targets = labels.masked_select(label_mask)
    logits = model.lm_head(model.dropout(hidden))
    log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
    with torch.backends.cudnn.flags(enabled=False):
        accented_ctc = F.ctc_loss(
            log_probs,
            flattened_targets,
            lengths,
            target_lengths,
            blank=model.config.pad_token_id,
            reduction=model.config.ctc_loss_reduction,
            zero_infinity=model.config.ctc_zero_infinity,
        )
    return accented_ctc, supcon
