"""Deterministic text normalization and auditable JiWER edit counts."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path

import jiwer


NORMALIZATION_VERSION = "english_char_v1"
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})


def normalize_for_wer(text: str) -> str:
    """Match the fixed English character vocabulary without hiding digits."""
    normalized = unicodedata.normalize("NFKC", str(text)).translate(_APOSTROPHES)
    normalized = normalized.upper()
    if any(character.isdigit() for character in normalized):
        raise ValueError(
            "Numeric evaluation references/hypotheses must be expanded during "
            f"dataset preparation, got {text!r}."
        )
    normalized = re.sub(r"[^A-Z'\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def score_utterance(reference: str, hypothesis: str) -> dict[str, int | float | str]:
    """Return normalized text and word-edit counts for one utterance."""
    normalized_reference = normalize_for_wer(reference)
    normalized_hypothesis = normalize_for_wer(hypothesis)
    if not normalized_reference:
        raise ValueError("An evaluation reference is empty after normalization.")
    output = jiwer.process_words(normalized_reference, normalized_hypothesis)
    reference_words = output.hits + output.substitutions + output.deletions
    errors = output.substitutions + output.deletions + output.insertions
    return {
        "reference_normalized": normalized_reference,
        "hypothesis_normalized": normalized_hypothesis,
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "hits": output.hits,
        "reference_words": reference_words,
        "errors": errors,
        "wer": errors / reference_words,
    }


def aggregate_edit_counts(
    utterances: Iterable[Mapping[str, int | float | str]],
) -> dict[str, int | float]:
    """Compute corpus micro WER by summing edit counts before division."""
    records = list(utterances)
    if not records:
        raise ValueError("Cannot aggregate an empty evaluation corpus.")
    keys = ("substitutions", "deletions", "insertions", "hits", "reference_words")
    totals = {key: sum(int(record[key]) for record in records) for key in keys}
    if totals["reference_words"] <= 0:
        raise ValueError("Evaluation corpus has no reference words.")
    totals["errors"] = (
        totals["substitutions"] + totals["deletions"] + totals["insertions"]
    )
    totals["wer"] = totals["errors"] / totals["reference_words"]
    totals["utterances"] = len(records)
    return totals


def aggregate_dataset_metrics(paths: Iterable[Path]) -> dict:
    """Combine per-dataset metrics for one checkpoint without pooling their WERs."""
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not records:
        raise ValueError("No dataset metrics were provided.")
    identity_keys = (
        "model", "objective", "fold", "seed", "decoder",
        "checkpoint_path", "checkpoint_sha256", "smoke",
    )
    identity = {key: records[0].get(key) for key in identity_keys}
    mismatches = {
        key for record in records for key in identity_keys
        if record.get(key) != identity[key]
    }
    if mismatches:
        raise ValueError(f"Dataset metrics have inconsistent identity: {sorted(mismatches)}")
    datasets = {}
    metric_keys = (
        "wer", "utterances", "reference_words", "errors", "hits",
        "substitutions", "deletions", "insertions",
    )
    for record in records:
        dataset = record["dataset"]
        if dataset in datasets:
            raise ValueError(f"Duplicate dataset metrics: {dataset}")
        datasets[dataset] = {
            key: record[key] for key in metric_keys
        }
        datasets[dataset]["wer_percent"] = 100 * record["wer"]
    return {**identity, "dataset_count": len(datasets), "datasets": datasets}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate one model's dataset WER files.")
    parser.add_argument("--metrics", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = aggregate_dataset_metrics(args.metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
