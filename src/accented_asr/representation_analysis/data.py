"""Deterministic balanced sampling for word-level representation analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "accent", "audio_path", "end_s", "normalized_word", "speaker_id",
    "split", "start_s",
}


def balanced_word_sample(
    parquet: Path,
    *,
    split: str,
    min_accents_per_word: int,
    examples_per_word_accent: int,
    max_words: int,
    seed: int,
) -> pd.DataFrame:
    frame = pd.read_parquet(parquet)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing word-analysis columns: {sorted(missing)}")
    frame = frame.loc[frame["split"] == split].copy()
    if frame.empty:
        raise ValueError(f"No rows for analysis split {split!r}.")
    # Apply speaker coverage before choosing words: otherwise random selection
    # can discard all words with enough eligible accents.
    speaker_counts = frame.groupby(["normalized_word", "accent"])["speaker_id"].transform("nunique")
    frame = frame.loc[speaker_counts >= examples_per_word_accent].copy()
    coverage = frame.groupby("normalized_word")["accent"].nunique()
    eligible = sorted(coverage.loc[coverage >= min_accents_per_word].index.astype(str))
    if not eligible:
        raise ValueError(
            "No word has the requested accent coverage with "
            f"{examples_per_word_accent} distinct speakers per accent."
        )
    if len(eligible) > max_words:
        eligible = (
            pd.Series(eligible).sample(max_words, random_state=seed).sort_values().tolist()
        )
    frame = frame.loc[frame["normalized_word"].astype(str).isin(eligible)].copy()
    sampled = []
    for (_word, _accent), group in frame.groupby(
        ["normalized_word", "accent"], sort=True
    ):
        group = group.sort_values(["speaker_id", "audio_path", "start_s"])
        group = group.drop_duplicates("speaker_id")
        if len(group) >= examples_per_word_accent:
            sampled.append(group.sample(examples_per_word_accent, random_state=seed))
    if not sampled:
        raise ValueError("No word/accent group has enough distinct speakers.")
    result = pd.concat(sampled, ignore_index=True)
    valid_words = result.groupby("normalized_word")["accent"].nunique()
    valid_words = set(valid_words.loc[valid_words >= min_accents_per_word].index)
    result = result.loc[result["normalized_word"].isin(valid_words)].copy()
    return result.sort_values(
        ["normalized_word", "accent", "speaker_id", "audio_path", "start_s"]
    ).reset_index(drop=True)
