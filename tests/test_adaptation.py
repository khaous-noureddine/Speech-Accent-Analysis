from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn

from accented_asr.adaptation.data import (
    L2ArcticAdaptationDataset,
    PromptBatchSampler,
    collate_adaptation,
)
from accented_asr.adaptation.model import AdaptationModel, SupConLoss
from accented_asr.adaptation.train import CONDITION_TO_MODE, load_config


def _write_test_fold(root: Path) -> Path:
    rows = []
    for prompt in ("p1", "p2", "p3"):
        for speaker in ("s1", "s2", "s3"):
            audio_path = root / f"{prompt}-{speaker}.wav"
            sf.write(audio_path, torch.zeros(800).numpy(), 16_000)
            rows.append(
                {
                    "audio_path": str(audio_path),
                    "transcript": "A TEST",
                    "speaker_id": speaker,
                    "prompt_id": prompt,
                    "native_language": "test",
                    "split": "train",
                    "split_manifest_sha256": "fixed-hash",
                }
            )
    parquet = root / "corpus.parquet"
    pd.DataFrame(rows).to_parquet(parquet, index=False)
    return parquet


def test_prompt_sampler_is_deterministic_and_keeps_positive_pairs(tmp_path):
    dataset = L2ArcticAdaptationDataset(
        _write_test_fold(tmp_path), split="train", repository_root=tmp_path,
        manifest_sha256="fixed-hash",
    )
    sampler = PromptBatchSampler(
        dataset, prompts_per_batch=2, speakers_per_prompt=3,
        batches_per_epoch=2, seed=13,
    )
    first = list(iter(sampler))
    second = list(iter(sampler))
    assert first == second
    assert all(len(batch) == 6 for batch in first)
    for batch in first:
        items = [dataset[index] for index in batch]
        collated = collate_adaptation(items)
        labels = collated["labels"]
        assert all(int((labels == label).sum()) == 3 for label in labels.unique())


def test_supcon_loss_is_a_finite_mean_over_anchors():
    embeddings = torch.nn.functional.normalize(torch.randn(6, 4), dim=-1)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    loss = SupConLoss(temperature=0.1)(embeddings, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_explicit_ablation_loss_modes():
    model = AdaptationModel.__new__(AdaptationModel)
    nn.Module.__init__(model)
    model.supcon_loss = SupConLoss(temperature=0.1)
    model.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    outputs = {
        "embeddings": torch.nn.functional.normalize(torch.randn(4, 3), dim=-1),
        "ctc_logits": torch.randn(4, 5, 6),
        "feature_lengths": torch.tensor([5, 5, 5, 5]),
    }
    batch = {
        "labels": torch.tensor([0, 0, 1, 1]),
        "ctc_targets": torch.tensor([1, 2, 3, 4]),
        "ctc_target_lengths": torch.tensor([1, 1, 1, 1]),
    }
    combined = model.compute_losses(batch, outputs, mode="supcon_ctc", ctc_weight=0.1)
    supcon = model.compute_losses(batch, outputs, mode="supcon_only", ctc_weight=0.1)
    ctc = model.compute_losses(batch, outputs, mode="ctc_only", ctc_weight=0.1)
    assert torch.allclose(ctc["loss"], 0.1 * ctc["ctc_loss"])
    assert torch.allclose(combined["loss"], supcon["loss"] + ctc["loss"])
    assert ctc["supcon_loss"].item() == 0.0
    assert supcon["ctc_loss"].item() == 0.0


def test_masked_mean_ignores_padding():
    hidden = torch.tensor([[[1.0], [3.0], [100.0]], [[2.0], [4.0], [6.0]]])
    result = AdaptationModel.masked_mean(hidden, torch.tensor([2, 3]))
    assert torch.equal(result, torch.tensor([[2.0], [4.0]]))


def test_all_stage2_experiment_configs_are_self_consistent():
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "experiments" / "stage2" / "wav2vec2-large-lv60"
    configs = sorted(model_dir.glob("*/*/config.yaml"))
    assert len(configs) == 18
    for path in configs:
        config = load_config(path)
        assert config.loss_mode == CONDITION_TO_MODE[config.condition]
        assert config.fold == config.heldout_accent == path.parent.name
        assert config.backbone_name == "facebook/wav2vec2-large-lv60"
        assert config.frozen_transformer_layers == 18
        assert config.tensorboard is True
        assert config.tensorboard_subdir == "tensorboard"
        assert config.experiment_name == f"wav2vec2-large-lv60_{path.parents[1].name}"
        assert config.output_dir == str(path.parent.relative_to(root) / "outputs")
