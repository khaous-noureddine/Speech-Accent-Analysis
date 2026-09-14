from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch import nn

from accented_asr.joint.model import ProjectionHead, configure_backbone_trainability
from accented_asr.joint.train import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = (
    ROOT
    / "experiments/joint-training/wav2vec2-large-lv60/utterance-supcon/arabic"
)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Linear(2, 2)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(24)])


class TinyCTC(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = TinyBackbone()
        self.lm_head = nn.Linear(2, 3)


def test_joint_configs_differ_only_by_freezing_policy_and_paths():
    frozen = load_config(CONFIG_ROOT / "freeze-18/config.yaml")
    unfrozen = load_config(CONFIG_ROOT / "full-transformer/config.yaml")
    assert frozen.fold == unfrozen.fold == "arabic"
    assert frozen.frozen_transformer_layers == 18
    assert frozen.freeze_feature_encoder is True
    assert unfrozen.frozen_transformer_layers == 0
    assert unfrozen.freeze_feature_encoder is True

    ignored = {
        "name", "output_dir", "frozen_transformer_layers", "freeze_feature_encoder"
    }
    frozen_values = {k: v for k, v in asdict(frozen).items() if k not in ignored}
    unfrozen_values = {k: v for k, v in asdict(unfrozen).items() if k not in ignored}
    assert frozen_values == unfrozen_values


def test_head_only_then_freeze_18_policy():
    model = TinyCTC()
    configure_backbone_trainability(
        model, head_only=True, frozen_transformer_layers=18,
        freeze_feature_encoder=True,
    )
    assert not any(p.requires_grad for p in model.wav2vec2.parameters())
    assert all(p.requires_grad for p in model.lm_head.parameters())

    configure_backbone_trainability(
        model, head_only=False, frozen_transformer_layers=18,
        freeze_feature_encoder=True,
    )
    assert not any(p.requires_grad for p in model.wav2vec2.feature_extractor.parameters())
    assert not any(
        p.requires_grad for layer in model.wav2vec2.encoder.layers[:18]
        for p in layer.parameters()
    )
    assert all(
        p.requires_grad for layer in model.wav2vec2.encoder.layers[18:]
        for p in layer.parameters()
    )


def test_full_transformer_policy_keeps_feature_encoder_frozen():
    model = TinyCTC()
    configure_backbone_trainability(
        model, head_only=False, frozen_transformer_layers=0,
        freeze_feature_encoder=True,
    )
    assert not any(p.requires_grad for p in model.wav2vec2.feature_extractor.parameters())
    assert all(
        p.requires_grad for layer in model.wav2vec2.encoder.layers
        for p in layer.parameters()
    )
    assert all(p.requires_grad for p in model.lm_head.parameters())


def test_projection_head_masks_padding_and_normalizes():
    projection = ProjectionHead(4, 8, 3)
    hidden = torch.randn(2, 5, 4)
    output = projection(hidden, torch.tensor([5, 3]))
    assert output.shape == (2, 3)
    assert torch.linalg.vector_norm(output, dim=-1).tolist() == pytest.approx([1.0, 1.0])
