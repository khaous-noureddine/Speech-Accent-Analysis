from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn

from accented_asr.asr.data import LibriSpeechDataset
from accented_asr.asr.model import load_stage2_backbone
from accented_asr.asr.train import (
    load_config,
    should_evaluate_step,
    word_error_counts,
)


class DummyCTCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = nn.Linear(2, 2)
        self.lm_head = nn.Linear(2, 3)


def test_stage2_transfer_loads_backbone_but_not_heads(tmp_path):
    model = DummyCTCModel()
    original_head = model.lm_head.weight.detach().clone()
    checkpoint = tmp_path / "stage2.pt"
    torch.save({
        "model_state_dict": {
            "backbone.weight": torch.full((2, 2), 7.0),
            "backbone.bias": torch.full((2,), 3.0),
            "projection.0.weight": torch.zeros(4, 2),
            "ctc_head.weight": torch.zeros(3, 2),
        },
        "metadata": {
            "epoch": 49, "loss_mode": "supcon_only", "seed": 13,
            "fold": "arabic", "backbone_name": "facebook/wav2vec2-large-lv60",
        },
    }, checkpoint)
    report = load_stage2_backbone(model, checkpoint)
    assert report["mapped_tensors"] == 2
    assert report["stage2_epoch"] == 49
    assert torch.equal(model.wav2vec2.weight, torch.full((2, 2), 7.0))
    assert torch.equal(model.lm_head.weight, original_head)


def test_librispeech_dataset_resolves_portable_audio_paths(tmp_path):
    audio = tmp_path / "audio.wav"
    sf.write(audio, torch.zeros(160).numpy(), 16_000)
    parquet = tmp_path / "corpus.parquet"
    pd.DataFrame([{
        "audio_path": "audio.wav", "transcript": "hello world", "speaker_id": "1"
    }]).to_parquet(parquet, index=False)
    dataset = LibriSpeechDataset(parquet, repository_root=tmp_path)
    assert dataset[0]["text"] == "HELLO WORLD"
    assert dataset[0]["audio"].shape == (160,)


def test_word_error_counts():
    assert word_error_counts("THE CAT SAT", "THE CAT") == (1, 2)
    assert word_error_counts("THE DOG", "THE CAT") == (1, 2)


def test_periodic_evaluation_hits_exact_steps_and_final_step():
    kwargs = {"interval": 5_000, "target_steps": 35_680, "smoke": False}
    assert not should_evaluate_step(4_999, **kwargs)
    assert should_evaluate_step(5_000, **kwargs)
    assert should_evaluate_step(10_000, **kwargs)
    assert not should_evaluate_step(10_001, **kwargs)
    assert should_evaluate_step(35_680, **kwargs)
    assert not should_evaluate_step(
        1, interval=5_000, target_steps=2, smoke=True
    )
    assert should_evaluate_step(
        2, interval=5_000, target_steps=2, smoke=True
    )


def test_all_stage3_configs_point_to_matching_stage2_runs():
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "experiments/stage3/wav2vec2-large-lv60"
    paths = sorted(model_dir.glob("*/*/config.yaml"))
    assert len(paths) == 18
    for path in paths:
        config = load_config(path)
        objective, accent = path.parents[1].name, path.parent.name
        assert config.objective == objective
        assert config.fold == accent
        assert config.backbone_name == "facebook/wav2vec2-large-lv60"
        assert config.stage2_output_dir.endswith(
            f"{objective}/{accent}/outputs"
        )
        assert config.output_dir.endswith(f"{objective}/{accent}/outputs")
