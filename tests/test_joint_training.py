from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from accented_asr.adaptation.model import SupConLoss
from accented_asr.joint.model import (
    ProjectionHead,
    balanced_shuffled_labels,
    configure_backbone_trainability,
    multidomain_ctc_supcon_losses,
)
from accented_asr.joint.train import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = (
    ROOT
    / "experiments/joint/librispeech-100h/wav2vec2-large-lv60/utterance-supcon/arabic"
)
ACCENTS = ("arabic", "chinese", "hindi", "korean", "spanish", "vietnamese")


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


class TinySpeechEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 4)

    def forward(self, input_values, attention_mask):
        return SimpleNamespace(
            last_hidden_state=self.projection(input_values.unsqueeze(-1))
        )


class TinyCombinedCTC(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = TinySpeechEncoder()
        self.dropout = nn.Identity()
        self.lm_head = nn.Linear(4, 5)
        self.config = SimpleNamespace(
            pad_token_id=0,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,
        )

    @staticmethod
    def _get_feat_extract_output_lengths(lengths):
        return lengths


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


@pytest.mark.parametrize("accent", ACCENTS)
@pytest.mark.parametrize("variant", ("freeze-18", "full-transformer"))
def test_every_joint_fold_has_a_matching_training_config(accent, variant):
    path = CONFIG_ROOT.parent / accent / variant / "config.yaml"
    config = load_config(path)
    assert config.fold == config.heldout_accent == accent
    assert f"/{accent}/corpus.parquet" in config.l2_parquet
    assert f"/{accent}/{variant}/outputs" in config.output_dir


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


def test_multidomain_ctc_supcon_uses_one_batch_for_both_losses():
    model = TinyCombinedCTC()
    projection = ProjectionHead(4, 6, 3)
    batch = {
        "audio": torch.randn(4, 8),
        "attention_mask": torch.ones(4, 8, dtype=torch.long),
        "labels": torch.tensor([0, 0, 1, 1]),
        "ctc_labels": torch.tensor(
            [[1, 2, -100], [1, 2, -100], [2, 3, 4], [2, 3, 4]]
        ),
    }
    accented_ctc, supcon = multidomain_ctc_supcon_losses(
        model, projection, SupConLoss(temperature=0.1), batch
    )
    assert accented_ctc.ndim == supcon.ndim == 0
    assert torch.isfinite(accented_ctc)
    assert torch.isfinite(supcon)
    (0.1 * accented_ctc + 0.1 * supcon).backward()
    assert model.wav2vec2.projection.weight.grad is not None


def test_balanced_shuffled_labels_preserve_counts_and_break_true_groups():
    labels = torch.arange(8).repeat_interleave(5)
    shuffled = balanced_shuffled_labels(labels, seed=13)
    assert torch.bincount(shuffled).tolist() == [5] * 8
    for label in labels.unique():
        assigned = shuffled[labels.eq(label)]
        assert assigned.unique().numel() == 5


def test_balanced_shuffled_labels_reject_invalid_group_geometry():
    with pytest.raises(ValueError, match="balanced"):
        balanced_shuffled_labels(torch.tensor([0, 0, 1]), seed=13)
    with pytest.raises(ValueError, match="fewer positives"):
        balanced_shuffled_labels(torch.tensor([0, 0, 1, 1]), seed=13)


@pytest.mark.parametrize("accent", ACCENTS)
def test_md_ft_cp_supcon_utterance_configs(accent):
    path = (
        ROOT
        / "experiments/joint/librispeech-100h/wav2vec2-large-lv60/"
        "md-ft-cp-supcon/utterance"
        / accent
        / "full-transformer/config.yaml"
    )
    config = load_config(path)
    assert config.contrastive_unit == "prompt"
    assert config.auxiliary_objective == "multidomain_ctc_supcon"
    assert config.auxiliary_weight == pytest.approx(0.1)
    assert config.supcon_weight == pytest.approx(0.1)
    assert config.fold == config.heldout_accent == accent


def test_md_ft_cp_supcon_word_config():
    path = (
        ROOT
        / "experiments/joint/librispeech-100h/wav2vec2-large-lv60/"
        "md-ft-cp-supcon/word/mswc-common-voice-50h/full-transformer/config.yaml"
    )
    config = load_config(path)
    assert config.contrastive_unit == "word"
    assert config.auxiliary_objective == "multidomain_ctc_supcon"
    assert config.auxiliary_weight == pytest.approx(0.1)
    assert config.supcon_weight == pytest.approx(0.1)


@pytest.mark.parametrize("accent", ("arabic", "chinese"))
def test_md_ft_cp_supcon_weight_pilot_configs(accent):
    path = (
        ROOT
        / "experiments/ablations/md-ft-cp-supcon-weight/librispeech-100h/"
        "wav2vec2-large-lv60/utterance"
        / accent
        / "lambda-0p03/full-transformer/config.yaml"
    )
    config = load_config(path)
    assert config.contrastive_unit == "prompt"
    assert config.auxiliary_objective == "multidomain_ctc_supcon"
    assert config.auxiliary_weight == pytest.approx(0.1)
    assert config.supcon_weight == pytest.approx(0.03)
    assert config.fold == config.heldout_accent == accent


def test_md_ft_cp_supcon_lambda_0p01_arabic_config():
    path = (
        ROOT
        / "experiments/ablations/md-ft-cp-supcon-weight/librispeech-100h/"
        "wav2vec2-large-lv60/utterance/arabic/lambda-0p01/"
        "full-transformer/config.yaml"
    )
    config = load_config(path)
    assert config.contrastive_unit == "prompt"
    assert config.auxiliary_objective == "multidomain_ctc_supcon"
    assert config.auxiliary_weight == pytest.approx(0.1)
    assert config.supcon_weight == pytest.approx(0.01)
    assert config.fold == config.heldout_accent == "arabic"


@pytest.mark.parametrize("accent", ACCENTS)
def test_shuffled_label_supcon_configs(accent):
    path = (
        ROOT
        / "experiments/joint/librispeech-100h/wav2vec2-large-lv60/"
        "controls/shuffled-label-supcon"
        / accent
        / "full-transformer/config.yaml"
    )
    config = load_config(path)
    assert config.auxiliary_objective == "shuffled_parallel_supcon"
    assert config.supcon_weight == pytest.approx(0.1)
    assert config.auxiliary_weight == 0.0


def test_librispeech_960_pilot_is_a_matched_objective_ablation():
    joint_root = ROOT / "experiments/joint/librispeech-960h/wav2vec2-large-lv60/utterance-supcon/arabic"
    baseline_root = ROOT / "experiments/baselines/internal/librispeech-960h/wav2vec2-large-lv60/ctc-only"
    ctc = load_config(baseline_root / "full-transformer/config.yaml")
    joint = load_config(joint_root / "full-transformer/config.yaml")
    assert ctc.supcon_weight == 0.0
    assert joint.supcon_weight == 0.1
    assert ctc.max_steps == joint.max_steps == 320_000
    assert ctc.head_warmup_steps == joint.head_warmup_steps == 10_000
    assert ctc.head_warmup_epochs is joint.head_warmup_epochs is None
    assert ctc.librispeech_train_parquet == joint.librispeech_train_parquet
    assert ctc.freeze_feature_encoder is joint.freeze_feature_encoder is True
    ignored = {"name", "output_dir", "supcon_weight"}
    assert (
        {k: v for k, v in asdict(ctc).items() if k not in ignored}
        == {k: v for k, v in asdict(joint).items() if k not in ignored}
    )


@pytest.mark.parametrize(
    "accent", ("arabic", "chinese", "hindi", "korean", "spanish", "vietnamese")
)
def test_librispeech_960_joint_fold_config(accent):
    path = (
        ROOT
        / "experiments/joint/librispeech-960h/wav2vec2-large-lv60/utterance-supcon"
        / accent
        / "full-transformer/config.yaml"
    )
    config = load_config(path)
    assert config.heldout_accent == accent
    assert config.supcon_weight == 0.1
    assert config.max_steps == 320_000
    assert config.head_warmup_steps == 10_000
    assert config.frozen_transformer_layers == 0
    assert config.freeze_feature_encoder is True
    assert f"/{accent}/corpus.parquet" in config.contrastive_parquet
    assert f"/{accent}/full-transformer/outputs" in config.output_dir


@pytest.mark.parametrize(
    ("variant", "frozen_layers"),
    (("freeze-18", 18), ("full-transformer", 0)),
)
def test_mswc_word_joint_config(variant, frozen_layers):
    path = (
        ROOT
        / "experiments/joint/librispeech-100h/wav2vec2-large-lv60/word-supcon/"
        f"mswc-common-voice-50h/{variant}/config.yaml"
    )
    config = load_config(path)
    assert config.contrastive_unit == "word"
    assert config.heldout_accent == "from_report"
    assert config.contrastive_parquet.endswith(
        "mswc_common_voice_words/en_50h_heldout_accent/corpus.parquet"
    )
    assert config.frozen_transformer_layers == frozen_layers
    assert config.freeze_feature_encoder is True


@pytest.mark.parametrize(
    ("variant", "frozen_layers"),
    (("freeze-18", 18), ("full-transformer", 0)),
)
def test_mswc_word_joint_librispeech_960_config(variant, frozen_layers):
    path = (
        ROOT
        / "experiments/joint/librispeech-960h/wav2vec2-large-lv60/word-supcon/"
        f"mswc-common-voice-50h/{variant}/config.yaml"
    )
    config = load_config(path)
    assert config.contrastive_unit == "word"
    assert config.heldout_accent == "from_report"
    assert config.librispeech_train_parquet.endswith("librispeech_960/corpus.parquet")
    assert config.max_steps == 320_000
    assert config.head_warmup_steps == 10_000
    assert config.head_warmup_epochs is None
    assert config.frozen_transformer_layers == frozen_layers
    assert config.freeze_feature_encoder is True


def test_mswc_500h_word_joint_librispeech_960_config():
    path = (
        ROOT
        / "experiments/joint/librispeech-960h/wav2vec2-large-lv60/word-supcon/"
        "mswc-common-voice-500h/full-transformer/config.yaml"
    )
    config = load_config(path)
    assert config.fold == "mswc-common-voice-500h"
    assert config.contrastive_parquet.endswith(
        "mswc_common_voice_words/en_500h_heldout_accent/corpus.parquet"
    )
    assert config.librispeech_train_parquet.endswith("librispeech_960/corpus.parquet")
    assert config.max_steps == 320_000
    assert config.head_warmup_steps == 10_000
    assert config.frozen_transformer_layers == 0
