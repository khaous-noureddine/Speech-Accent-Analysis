from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from accented_asr.error_analysis.alignment import align_words, error_transitions
from accented_asr.error_analysis.analysis import sentence_category
from accented_asr.error_analysis.run import git_commit, main, markdown_examples
from accented_asr.evaluation.metrics import score_utterance


def prediction(speaker: str, utterance: str, reference: str, hypothesis: str) -> dict:
    return {
        "speaker_id": speaker,
        "utterance_id": utterance,
        "accent": "toy",
        **score_utterance(reference, hypothesis),
    }


def test_alignment_labels_corrected_and_introduced_errors():
    transitions = error_transitions("THE CAT SAT", "THE BAT", "THE CAT SAT NOW")
    names = {record["transition"] for record in transitions}
    assert "substitution_to_correct" in names
    assert "deletion_to_correct" in names
    assert "insertion_introduced" in names
    aligned = align_words("A B", "A X B")
    assert aligned.insertions == {1: ("X",)}


def test_sentence_categories_are_directional():
    assert sentence_category(2, 0, "A", "B") == "fully_corrected"
    assert sentence_category(3, 1, "A", "B") == "improved"
    assert sentence_category(0, 1, "A", "B") == "introduced_error"
    assert sentence_category(1, 2, "A", "B") == "degraded"
    assert sentence_category(1, 1, "A", "A") == "unchanged"


def test_git_commit_tolerates_missing_git(monkeypatch, tmp_path):
    monkeypatch.setattr("accented_asr.error_analysis.run.shutil.which", lambda _: None)
    assert git_commit(Path(tmp_path)) is None


def test_end_to_end_before_after_analysis(tmp_path, monkeypatch):
    before = pd.DataFrame(
        [
            prediction("s1", "u1", "THE CAT SAT", "THE BAT"),
            prediction("s2", "u2", "HELLO WORLD", "HELLO WORLD"),
        ]
    )
    after = pd.DataFrame(
        [
            prediction("s1", "u1", "THE CAT SAT", "THE CAT SAT"),
            prediction("s2", "u2", "HELLO WORLD", "HELLO WORD"),
        ]
    )
    for root, frame in (("before", before), ("after", after)):
        directory = tmp_path / root / "seed=13" / "greedy" / "toy"
        directory.mkdir(parents=True)
        frame.to_parquet(directory / "predictions.parquet", index=False)
    config = {
        "error_analysis": {
            "seed": 13,
            "baseline": {"output_dir": "before", "seed": 13},
            "comparisons": {"supcon": {"output_dir": "after", "seed": 13}},
            "decoders": ["greedy"],
            "matching_keys": ["speaker_id", "utterance_id"],
            "datasets": {"toy": {"baseline": "toy", "supcon": "toy"}},
            "bootstrap_replicates": 20,
            "examples_per_category": 2,
            "top_words": 10,
            "output_dir": "outputs",
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["error-analysis", "--config", str(config_path), "--repository-root", str(tmp_path)],
    )
    main()
    output = tmp_path / "outputs"
    assert (output / "_SUCCESS").is_file()
    assert (output / "corrected_errors.csv").is_file()
    assert (output / "introduced_errors.csv").is_file()
    assert (output / "persistent_errors.csv").is_file()
    assert (output / "substitution_confusions.csv").is_file()
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    summary = result["supcon"]["greedy"]["toy"]
    assert summary["before_errors"] == 2
    assert summary["after_errors"] == 1
    assert summary["resolved_error_events"] == 2
    assert summary["introduced_error_events"] == 1


def test_qualitative_report_distinguishes_speakers_sharing_a_prompt():
    records = []
    for speaker, category in [("ABA", "fully_corrected"), ("SKA", "degraded")]:
        records.append(dict(comparison="utterance_supcon", decoder="beam_4gram",
                            dataset_name="l2_arctic_arabic", sentence_category=category,
                            utterance_id="arctic_a0095", speaker_id=speaker,
                            audio_path=f"data/{speaker}_arctic_a0095.wav",
                            selection_rule="largest_change", before_errors=1, after_errors=0,
                            reference="THE CAT", before_hypothesis="THE BAT", after_hypothesis="THE CAT"))
    report = markdown_examples(pd.DataFrame(records))
    assert "Speaker: `ABA`" in report
    assert "Speaker: `SKA`" in report
    assert "Audio: `data/ABA_arctic_a0095.wav`" in report
    assert "Audio: `data/SKA_arctic_a0095.wav`" in report
