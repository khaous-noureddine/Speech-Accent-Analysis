"""
supcon_model.py

Supervised Contrastive fine-tuning of XLSR-53 for accent-invariant speech representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config
from loguru import logger


class ProjectionMLP(nn.Module):
    """
    2-layer MLP projecting transformer output to the contrastive space.
    Output is L2-normalized (required by SupCon loss).

    in_dim  : XLSR-53 hidden size (1024)
    hidden_dim : intermediate dimension
    out_dim : contrastive embedding dimension (256 as per the project spec)
    """

    def __init__(self, in_dim: int = 1024, hidden_dim: int = 512, out_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, in_dim] — mean-pooled transformer output
        Returns:
            z : [B, out_dim] — L2-normalized contrastive embedding
        """
        z = self.net(x)
        return F.normalize(z, dim=-1)


class CTCHead(nn.Module):
    """
    Linear CTC head attached directly to the transformer output (before mean-pooling).
    Serves as an auxiliary regularizer to prevent phonemic collapse.

    in_dim  : XLSR-53 hidden size (1024)
    vocab_size : number of output tokens (characters + blank)
    """

    def __init__(self, in_dim: int = 1024, vocab_size: int = 32):
        super().__init__()
        self.linear = nn.Linear(in_dim, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states : [B, T, in_dim] — full sequence from transformer
        Returns:
            logits : [B, T, vocab_size]
        """
        return self.linear(hidden_states)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    For each anchor i, all samples j in the batch with the same label
    (same utterance_id) and j ≠ i are treated as positives.
    All other samples are negatives.

    L = Σ_i [ -1/|P(i)| · Σ_{p ∈ P(i)} log( exp(z_i·z_p/τ) / Σ_{a ∈ A(i)} exp(z_i·z_a/τ) ) ]

    Args:
        temperature : τ in the loss formula (default 0.1 as per project spec)
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings : [B, D] — L2-normalized contrastive embeddings
            labels     : [B]    — integer label; same label = positive pair
        Returns:
            loss : scalar
        """
        B = embeddings.shape[0]
        device = embeddings.device
        
        # Similarity matrix [B, B] — dot product (= cosine sim since embeddings are L2-normalized)
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature  # [B, B]

        # Mask: positive pairs (same label, different index)
        labels = labels.unsqueeze(1)                          # [B, 1]
        pos_mask = (labels == labels.T).float()               # [B, B] , 1 if same label
        self_mask = torch.eye(B, device=device)               # [B, B] , 1 on diagonal
        pos_mask = pos_mask - self_mask                       # exclude self

        # A(i) = all indices except self
        all_mask = 1.0 - self_mask                            # [B, B]

        # For numerical stability: subtract row max before exp
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim) * all_mask                   # zero out self [B, B]
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)  # [B, B]

        # Mean log-prob over positives for each anchor
        n_positives = pos_mask.sum(dim=1)                     # [B]

        # Avoid division by zero for anchors with no positives in batch
        valid = n_positives > 0
        if not valid.any():
            logger.warning("No positive pairs found in batch — SupCon loss is zero.")
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss_per_anchor = -(pos_mask * log_prob).sum(dim=1) / (n_positives + 1e-8)  # [B]
        loss = loss_per_anchor[valid].mean()

        return loss


# Main model
class SupConXLSR(nn.Module):
    """
    XLSR-53 fine-tuned with Supervised Contrastive Learning for accent invariance.

    Layers 1-18  : frozen (preserve multilingual pre-training)
    Layers 19-24 : fine-tuned (learn accent-invariant representations)

    Forward pass returns:
        embeddings : [B, proj_dim] — L2-normalized, used for SupCon loss
        ctc_logits : [B, T, vocab_size] — used for CTC loss (auxiliary)
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
    ):
        """
        Args:
            model_name        : HuggingFace model identifier
            proj_hidden_dim   : hidden dim of projection MLP
            proj_out_dim      : output dim of projection MLP (contrastive space)
            vocab_size        : CTC vocabulary size (characters + blank token)
            ctc_lambda        : λ weighting the CTC auxiliary loss
            temperature       : τ for SupCon loss
            min_frozen_layer  : first transformer layer to freeze (0-indexed, inclusive)
            max_frozen_layer  : last transformer layer to freeze (0-indexed, exclusive)
        """
        super().__init__()

        self.ctc_lambda       = ctc_lambda
        self.min_frozen_layer = min_frozen_layer
        self.max_frozen_layer = max_frozen_layer

        # ── Backbone ──
        logger.info(f"Loading {model_name}...")
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        #
        self.backbone.gradient_checkpointing_enable()
        #
        hidden_size = self.backbone.config.hidden_size  # 1024 for large

        # ── Freeze layers min_frozen_layer to max_frozen_layer-1 ──
        self._freeze_backbone_layers()

        # ── Projection MLP → contrastive space ──
        self.projection = ProjectionMLP(
            in_dim=hidden_size,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_out_dim,
        )

        # ── CTC head (auxiliary) — operates on full sequence before pooling ──
        self.ctc_head = CTCHead(in_dim=hidden_size, vocab_size=vocab_size)

        # ── Losses ──
        self.supcon_loss = SupConLoss(temperature=temperature)
        self.ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

        logger.info(f"SupConXLSR initialized:")
        logger.info(f"  Frozen layers   : {min_frozen_layer}-{max_frozen_layer - 1} (= layers {min_frozen_layer + 1}-{max_frozen_layer})")
        logger.info(f"  Fine-tuned      : layers {max_frozen_layer}-23 (= layers {max_frozen_layer + 1}-24)")
        logger.info(f"  Proj dim        : {hidden_size} → {proj_hidden_dim} → {proj_out_dim}")
        logger.info(f"  Vocab size      : {vocab_size}")
        logger.info(f"  λ_CTC           : {ctc_lambda}")
        logger.info(f"  Temperature τ   : {temperature}")


    def _freeze_backbone_layers(self) -> None:
        """Freeze feature extractor + transformer layers min_frozen_layer to max_frozen_layer-1."""

        # Always freeze the CNN feature extractor
        for param in self.backbone.feature_extractor.parameters():
            param.requires_grad = False

        # Freeze the feature projection
        for param in self.backbone.feature_projection.parameters():
            param.requires_grad = False

        # Freeze transformer layers in [min_frozen_layer, max_frozen_layer)
        for i, layer in enumerate(self.backbone.encoder.layers):
            if self.min_frozen_layer <= i < self.max_frozen_layer:
                for param in layer.parameters():
                    param.requires_grad = False

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(f"  Trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.1f}%)")


    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            audio          : [B, T] — raw waveform at 16kHz, with B: Batch size, T: Signal Length (sample_rate x audio lenght in seconds)
            attention_mask : [B, T] — 1 for real frames, 0 for padding (optional)

        Returns dict with:
            embeddings  : [B, proj_out_dim] — L2-normalized contrastive embeddings
            ctc_logits  : [B, T', vocab_size] — CTC log-probs over time
            hidden_states : [B, T', hidden_size] — raw transformer output (for probing)
        """
        # (B, T) -> (B, T', H) through XLSR-53 backbone, T: number audio point, T': number of acoustic frames, we have 49 frame per second.
        outputs = self.backbone(
            input_values=audio,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        hidden_states = outputs.last_hidden_state  # [B, T', H]

        # [B, T', H] -> [B, T', vocab_size] CTC head operates on full sequence (before pooling)
        ctc_logits = self.ctc_head(hidden_states)

        # [B, T', H] → [B, H] Mean-pool over time
        if attention_mask is not None:
            # Compute mask in the feature extractor's output time dimension
            feat_lengths = self.backbone._get_feat_extract_output_lengths(
                attention_mask.sum(dim=-1)
            )  # [B]
            pooled = self._masked_mean_pool(hidden_states, feat_lengths)
        else:
            pooled = hidden_states.mean(dim=1)

        # [B, H] -> [B, proj_out_dim=256] 
        embeddings = self.projection(pooled)

        return {
            "embeddings":   embeddings,     # [B, 256]
            "pooled":       pooled,         # [B, 1024]
            "ctc_logits":   ctc_logits,     # [B, T', vocab_size]
            "hidden_states": hidden_states, # [B, T', 1024]
        }

    @staticmethod
    def _masked_mean_pool(hidden_states: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool hidden_states respecting variable sequence lengths.

        Args:
            hidden_states : [B, T, H]
            lengths       : [B] — number of valid frames per example
        Returns:
            pooled : [B, H]
        """
        B, T, H = hidden_states.shape
        mask = torch.arange(T, device=hidden_states.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()               # [B, T, 1]
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled  # [B, H]


    def compute_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        ctc_logits: torch.Tensor | None = None,
        ctc_targets: torch.Tensor | None = None,
        ctc_input_lengths: torch.Tensor | None = None,
        ctc_target_lengths: torch.Tensor | None = None,
    ) -> dict:
        """
        Compute combined SupCon + CTC loss.

        Args:
            embeddings          : [B, D] — L2-normalized embeddings
            labels              : [B]    — utterance label (same = positive)
            ctc_logits          : [B, T, vocab_size] — raw logits (optional)
            ctc_targets         : [S]   — concatenated target sequences
            ctc_input_lengths   : [B]   — length of each ctc_logits sequence
            ctc_target_lengths  : [B]   — length of each target sequence

        Returns:
            dict with keys: loss, supcon_loss, ctc_loss
        """
        # SupCon loss (always computed)
        l_supcon = self.supcon_loss(embeddings, labels)

        # CTC loss (only if targets are provided)
        l_ctc = torch.tensor(0.0, device=embeddings.device)

        if (
            ctc_logits is not None
            and ctc_targets is not None
            and ctc_input_lengths is not None
            and ctc_target_lengths is not None
        ):
            # CTCLoss expects [T, B, C] log-probs
            log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)
            l_ctc = self.ctc_loss_fn(
                log_probs, ctc_targets, ctc_input_lengths, ctc_target_lengths
            )

        total = l_supcon + self.ctc_lambda * l_ctc

        return {
            "loss":        total,
            "supcon_loss": l_supcon,
            "ctc_loss":    l_ctc,
        }