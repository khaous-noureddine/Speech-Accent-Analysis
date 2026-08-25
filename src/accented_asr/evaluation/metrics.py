"""Deterministic text normalization and auditable JiWER edit counts."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping

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

