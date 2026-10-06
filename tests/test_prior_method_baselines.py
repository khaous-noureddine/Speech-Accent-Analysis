from pathlib import Path

import torch

from accented_asr.joint.model import gradient_reverse
from accented_asr.joint.train import load_config
from accented_asr.evaluation.run import load_config as load_evaluation_config


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/baselines/prior-methods/librispeech-100h/wav2vec2-large-lv60"


def test_gradient_reversal_changes_only_backward_sign():
    inputs = torch.tensor([1.0, -2.0], requires_grad=True)
    outputs = gradient_reverse(inputs, 0.25)
    assert torch.equal(outputs, inputs)
    outputs.sum().backward()
    assert torch.allclose(inputs.grad, torch.full_like(inputs, -0.25))


def test_all_prior_method_fold_configs_load_and_are_leakage_safe():
    expected = {
        "accent-dat": "accent_dat",
        "accent-mtl": "accent_mtl",
        "multidomain-ctc": "multidomain_ctc",
        "augmented-view-supcon": "augmented_view_supcon",
    }
    accents = ("arabic", "chinese", "hindi", "korean", "spanish", "vietnamese")
    for method, objective in expected.items():
        for accent in accents:
            path = BASE / method / accent / "full-transformer/config.yaml"
            config = load_config(path)
            assert config.auxiliary_objective == objective
            assert config.fold == config.heldout_accent == accent
            assert f"/{accent}/corpus.parquet" in config.l2_parquet
            assert config.max_steps == 50_000
            assert config.seeds == (13,)
            evaluation_path = path.with_name("evaluation.yaml")
            for dataset in (
                "l2_arctic", "librispeech_test_clean", "aesrc",
                "speech_accent_archive", "edacc",
            ):
                evaluation = load_evaluation_config(evaluation_path, dataset)
                assert evaluation.fold == accent
                assert evaluation.objective == config.name
