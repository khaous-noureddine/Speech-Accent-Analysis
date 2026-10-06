"""Aggregation helpers for paired before/after prediction comparisons."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd


RESOLVED_TRANSITIONS = {
    "substitution_to_correct", "deletion_to_correct", "insertion_removed"
}
INTRODUCED_TRANSITIONS = {
    "correct_to_substitution", "correct_to_deletion", "insertion_introduced"
}


def sentence_category(before_errors: int, after_errors: int, before: str, after: str) -> str:
    if before_errors > 0 and after_errors == 0:
        return "fully_corrected"
    if after_errors < before_errors:
        return "improved"
    if after_errors > before_errors:
        return "introduced_error" if before_errors == 0 else "degraded"
    if before == after:
        return "unchanged"
    return "changed_equal_error_count"


def paired_bootstrap_wer_delta(
    frame: pd.DataFrame, *, seed: int, replicates: int
) -> dict[str, float]:
    """Paired utterance bootstrap for after-WER minus before-WER."""

    rng = np.random.default_rng(seed)
    before_errors = frame["before_errors"].to_numpy(int)
    after_errors = frame["after_errors"].to_numpy(int)
    words = frame["reference_words"].to_numpy(int)
    deltas = []
    for _ in range(replicates):
        selected = rng.integers(0, len(frame), size=len(frame))
        denominator = words[selected].sum()
        if denominator:
            deltas.append(
                100.0
                * (after_errors[selected].sum() - before_errors[selected].sum())
                / denominator
            )
    return {
        "lower_95": float(np.quantile(deltas, 0.025)),
        "upper_95": float(np.quantile(deltas, 0.975)),
    }


def summarize_comparison(
    sentences: pd.DataFrame,
    transitions: pd.DataFrame,
    *,
    seed: int,
    bootstrap_replicates: int,
) -> dict[str, object]:
    reference_words = int(sentences["reference_words"].sum())
    before_errors = int(sentences["before_errors"].sum())
    after_errors = int(sentences["after_errors"].sum())
    transition_counts = Counter(transitions["transition"].astype(str))
    resolved = sum(transition_counts[name] for name in RESOLVED_TRANSITIONS)
    introduced = sum(transition_counts[name] for name in INTRODUCED_TRANSITIONS)
    categories = Counter(sentences["sentence_category"].astype(str))
    return {
        "utterances": len(sentences),
        "reference_words": reference_words,
        "before_errors": before_errors,
        "after_errors": after_errors,
        "before_wer_percent": 100.0 * before_errors / reference_words,
        "after_wer_percent": 100.0 * after_errors / reference_words,
        "wer_delta_points": 100.0 * (after_errors - before_errors) / reference_words,
        "relative_error_reduction_percent": (
            100.0 * (before_errors - after_errors) / before_errors
            if before_errors else 0.0
        ),
        "resolved_error_events": resolved,
        "introduced_error_events": introduced,
        "net_resolved_events": resolved - introduced,
        "correction_rate_percent": 100.0 * resolved / before_errors if before_errors else 0.0,
        "regression_rate_per_100_reference_words": 100.0 * introduced / reference_words,
        "sentence_categories": dict(sorted(categories.items())),
        "word_transitions": dict(sorted(transition_counts.items())),
        "paired_bootstrap_wer_delta_points_95": paired_bootstrap_wer_delta(
            sentences, seed=seed, replicates=bootstrap_replicates
        ),
    }


def top_word_counts(
    transitions: pd.DataFrame, names: Iterable[str], *, limit: int
) -> list[dict[str, object]]:
    selected = transitions.loc[transitions["transition"].isin(names)].copy()
    selected["word"] = selected["reference_word"].fillna(
        selected["before_word"].fillna(selected["after_word"])
    )
    counts = selected.groupby(["transition", "word"], dropna=False).size()
    records = []
    for transition in sorted(counts.index.get_level_values(0).unique()):
        current = counts.loc[transition].sort_values(ascending=False).head(limit)
        records.extend(
            {"transition": transition, "word": str(word), "count": int(count)}
            for word, count in current.items()
        )
    return records
