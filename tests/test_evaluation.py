from pathlib import Path

import pytest
import yaml
import json

from accented_asr.evaluation.metrics import (
    NORMALIZATION_VERSION,
    aggregate_dataset_metrics,
    aggregate_edit_counts,
    normalize_for_wer,
    score_utterance,
)
from accented_asr.evaluation.run import load_config
from accented_asr.evaluation.run import git_commit


def test_normalization_matches_character_tokenizer_contract():
    assert NORMALIZATION_VERSION == "english_char_v1"
    assert normalize_for_wer("  Don’t, stop!  ") == "DON'T STOP"


def test_normalization_rejects_unexpanded_numbers():
    with pytest.raises(ValueError, match="must be expanded"):
        normalize_for_wer("chapter 42")


def test_utterance_edit_counts_cover_substitution():
    result = score_utterance("THE CAT SAT", "THE DOG SAT")
    assert result["substitutions"] == 1
    assert result["deletions"] == 0
    assert result["insertions"] == 0
    assert result["hits"] == 2
    assert result["reference_words"] == 3
    assert result["wer"] == pytest.approx(1 / 3)


def test_empty_hypothesis_counts_all_reference_words_as_deletions():
    result = score_utterance("THE CAT SAT", "")
    assert result["deletions"] == 3
    assert result["errors"] == 3
    assert result["reference_words"] == 3
    assert result["wer"] == 1.0


def test_corpus_wer_sums_counts_instead_of_averaging_utterance_wer():
    records = [
        score_utterance("WRONG", "OTHER"),
        score_utterance("ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE", "ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE"),
    ]
    result = aggregate_edit_counts(records)
    assert result["utterances"] == 2
    assert result["errors"] == 1
    assert result["reference_words"] == 10
    assert result["wer"] == 0.1


def test_empty_reference_is_a_data_contract_error():
    with pytest.raises(ValueError, match="reference is empty"):
        score_utterance("...", "HELLO")


def test_greedy_evaluation_config_is_fold_matched(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"evaluation": {
        "name": "test", "model": "model", "objective": "supcon-only",
        "fold": "arabic", "seed": 13, "checkpoint": "checkpoint.pt",
        "stage3_config": "stage3.yaml",
        "decoder": "greedy", "output_dir": "outputs", "batch_size": 2,
        "num_workers": 0, "device": "cpu", "datasets": {"l2_arctic": {
            "split": "test", "parquet": "folds/arabic/corpus.parquet",
            "raw_dir": "raw/l2_arctic",
        }},
    }}), encoding="utf-8")
    config = load_config(path, "l2_arctic")
    assert config.fold == "arabic"
    assert config.decoder == "greedy"


def test_evaluation_decoder_can_be_overridden_for_4gram(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"evaluation": {
        "name": "test", "model": "model", "objective": "supcon-only",
        "fold": "arabic", "seed": 13, "checkpoint": "checkpoint.pt",
        "stage3_config": "stage3.yaml",
        "decoder": "greedy", "output_dir": "outputs", "batch_size": 2,
        "num_workers": 0, "device": "cpu", "datasets": {"l2_arctic": {
            "split": "test", "parquet": "folds/arabic/corpus.parquet",
            "raw_dir": "raw/l2_arctic",
        }},
    }}), encoding="utf-8")
    config = load_config(path, "l2_arctic", decoder_override="beam_4gram")
    assert config.decoder == "beam_4gram"


def test_evaluation_rejects_mismatched_fold(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"evaluation": {
        "name": "test", "model": "model", "objective": "supcon-only",
        "fold": "arabic", "seed": 13, "checkpoint": "checkpoint.pt",
        "stage3_config": "stage3.yaml",
        "decoder": "greedy", "output_dir": "outputs", "batch_size": 2,
        "num_workers": 0, "device": "cpu", "datasets": {"l2_arctic": {
            "split": "test", "parquet": "folds/spanish/corpus.parquet",
            "raw_dir": "raw/l2_arctic",
        }},
    }}), encoding="utf-8")
    with pytest.raises(ValueError, match="held-out accent"):
        load_config(path, "l2_arctic")


def test_campaign_config_declares_all_five_evaluation_datasets():
    path = Path(
        "experiments/eval/wav2vec2-large-lv60/supcon-only/arabic/config.yaml"
    )
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert set(document["evaluation"]["datasets"]) == {
        "l2_arctic",
        "librispeech_test_clean",
        "aesrc",
        "speech_accent_archive",
        "edacc",
    }
    for dataset in document["evaluation"]["datasets"]:
        assert load_config(path, dataset).dataset == dataset


@pytest.mark.parametrize(
    "accent", ("arabic", "chinese", "hindi", "korean", "spanish", "vietnamese")
)
def test_supcon_campaign_uses_matching_stage3_and_l2_fold(accent):
    path = Path(
        f"experiments/eval/wav2vec2-large-lv60/supcon-only/{accent}/config.yaml"
    )
    config = load_config(path, "l2_arctic")
    assert config.fold == accent
    assert f"/{accent}/outputs/seed=13/checkpoint_best.pt" in config.checkpoint
    assert f"/{accent}/corpus.parquet" in config.parquet


@pytest.mark.parametrize("accent", ("arabic", "chinese"))
def test_supcon_ctc_campaign_uses_matching_stage3_and_l2_fold(accent):
    path = Path(
        f"experiments/eval/wav2vec2-large-lv60/supcon-ctc/{accent}/config.yaml"
    )
    config = load_config(path, "l2_arctic")
    assert config.objective == "supcon-ctc"
    assert config.fold == accent
    assert f"/supcon-ctc/{accent}/outputs/seed=13/checkpoint_best.pt" in config.checkpoint
    assert f"/{accent}/corpus.parquet" in config.parquet


def test_no_stage2_campaign_covers_all_l2_folds_and_external_datasets_once():
    path = Path(
        "experiments/eval/wav2vec2-large-lv60/no-stage2/config.yaml"
    )
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    datasets = document["evaluation"]["datasets"]
    assert set(datasets) == {
        "l2_arctic_arabic",
        "l2_arctic_chinese",
        "l2_arctic_hindi",
        "l2_arctic_korean",
        "l2_arctic_spanish",
        "l2_arctic_vietnamese",
        "librispeech_test_clean",
        "aesrc",
        "speech_accent_archive",
        "edacc",
    }
    for dataset in datasets:
        config = load_config(path, dataset)
        assert config.fold is None
        assert config.objective == "no-stage2"


def test_external_huggingface_baseline_is_revision_pinned():
    path = Path(
        "experiments/external-baselines/"
        "facebook-wav2vec2-large-960h-lv60/evaluation/config.yaml"
    )
    config = load_config(path, "l2_arctic_arabic")
    assert config.source == "huggingface"
    assert config.seed == "official"
    assert config.checkpoint is None
    assert config.stage3_config is None
    assert config.hf_model == "facebook/wav2vec2-large-960h-lv60"
    assert config.hf_revision == "8e7d14742e8f98c6bbb24e5231406af321a8f9ce"


def test_external_100h_baseline_and_processor_are_revision_pinned():
    path = Path(
        "experiments/external-baselines/"
        "patrickvonplaten-wav2vec2-large-lv60h-100h/evaluation/config.yaml"
    )
    config = load_config(path, "l2_arctic_arabic")
    assert config.source == "huggingface"
    assert config.seed == "published"
    assert config.hf_model == (
        "patrickvonplaten/wav2vec2-large-lv60h-100h-2nd-try"
    )
    assert config.hf_revision == "38dfd9c80bcbe9f43a36447e1aec30dd5d12415a"
    assert config.hf_processor == "facebook/wav2vec2-large-960h-lv60"
    assert config.hf_processor_revision == (
        "8e7d14742e8f98c6bbb24e5231406af321a8f9ce"
    )


@pytest.mark.parametrize(
    "accent", ("arabic", "chinese", "hindi", "korean", "spanish", "vietnamese")
)
@pytest.mark.parametrize("variant", ("freeze-18", "full-transformer"))
def test_joint_evaluation_uses_matching_heldout_checkpoint(accent, variant):
    root = Path(
        "experiments/joint-training/wav2vec2-large-lv60/"
        f"utterance-supcon/{accent}/{variant}"
    )
    config = load_config(root / "evaluation.yaml", "l2_arctic")
    assert config.source == "joint"
    assert config.fold == accent
    assert config.joint_config == str(root / "config.yaml")
    assert config.checkpoint == str(root / "outputs/seed=13/checkpoint_best.pt")
    document = yaml.safe_load((root / "evaluation.yaml").read_text(encoding="utf-8"))
    assert set(document["evaluation"]["datasets"]) == {
        "l2_arctic", "librispeech_test_clean", "aesrc",
        "speech_accent_archive", "edacc",
    }


def test_missing_git_does_not_abort_evaluation(monkeypatch, tmp_path):
    monkeypatch.setattr("accented_asr.evaluation.run.shutil.which", lambda _: None)
    assert git_commit(tmp_path) is None


def test_dataset_metrics_are_combined_without_hiding_individual_wer(tmp_path):
    paths = []
    for dataset, errors, words in (("l2_arctic", 2, 10), ("edacc", 6, 20)):
        path = tmp_path / f"{dataset}.json"
        path.write_text(json.dumps({
            "model": "wav2vec2", "objective": "supcon-only", "fold": "arabic",
            "seed": 13, "decoder": "greedy", "checkpoint_path": "best.pt",
            "checkpoint_sha256": "abc", "smoke": True, "dataset": dataset,
            "wer": errors / words, "utterances": 8, "reference_words": words,
            "errors": errors, "hits": words - errors, "substitutions": errors,
            "deletions": 0, "insertions": 0,
        }), encoding="utf-8")
        paths.append(path)
    summary = aggregate_dataset_metrics(paths)
    assert summary["dataset_count"] == 2
    assert summary["datasets"]["l2_arctic"]["wer_percent"] == 20
    assert summary["datasets"]["edacc"]["wer_percent"] == 30
