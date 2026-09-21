"""Build a size-controlled MSWC word corpus with one unseen accent."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SplitPlan:
    heldout_accent: str
    seen_accents: tuple[str, ...]
    train_words: frozenset[str]
    dev_words: frozenset[str]
    test_words: frozenset[str]
    score: tuple[int, int, int]


def slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return result or "unknown"


def _word_sets_with_speaker_floor(
    frame: pd.DataFrame, minimum_speakers: int
) -> dict[str, set[str]]:
    coverage = (
        frame.groupby(["accent", "normalized_word"])["speaker_id"]
        .nunique()
        .rename("speakers")
        .reset_index()
    )
    coverage = coverage.loc[coverage["speakers"] >= minimum_speakers]
    return {
        str(accent): set(group["normalized_word"].astype(str))
        for accent, group in coverage.groupby("accent")
    }


def _greedy_seen_accents(
    word_sets: dict[str, set[str]], heldout: str, count: int
) -> tuple[tuple[str, ...], set[str]]:
    remaining = sorted(accent for accent in word_sets if accent != heldout)
    selected: list[str] = []
    intersection: set[str] | None = None
    for _ in range(count):
        if not remaining:
            break
        ranked = []
        for accent in remaining:
            candidate = word_sets[accent]
            common = candidate if intersection is None else intersection & candidate
            ranked.append((len(common), len(candidate), accent, common))
        _, _, best, common = max(ranked, key=lambda item: (item[0], item[1], item[2]))
        selected.append(best)
        remaining.remove(best)
        intersection = set(common)
    if len(selected) != count or not intersection:
        raise ValueError(f"Could not find {count} seen accents with shared words.")
    return tuple(selected), intersection


def choose_plan(
    frame: pd.DataFrame,
    *,
    heldout_accent: str,
    seen_accent_count: int,
    min_train_speakers: int,
    min_dev_accents: int,
    targets: dict[str, int],
) -> SplitPlan:
    train = frame.loc[frame["source_split"].eq("train")]
    dev = frame.loc[frame["source_split"].eq("dev")]
    train_word_sets = _word_sets_with_speaker_floor(train, min_train_speakers)
    all_accents = sorted(train_word_sets)
    candidates = all_accents if heldout_accent == "auto" else [heldout_accent]
    plans: list[SplitPlan] = []

    all_coverage = (
        frame.groupby(["accent", "normalized_word"])
        .agg(examples=("audio_path", "size"), speakers=("speaker_id", "nunique"))
        .reset_index()
    )
    dev_coverage = (
        dev.groupby(["accent", "normalized_word"])["speaker_id"]
        .nunique()
        .rename("speakers")
        .reset_index()
    )

    for heldout in candidates:
        if heldout not in set(frame["accent"]):
            if heldout_accent != "auto":
                raise ValueError(f"Unknown held-out accent: {heldout!r}")
            continue
        try:
            seen, train_words = _greedy_seen_accents(
                train_word_sets, heldout, seen_accent_count
            )
        except ValueError:
            continue

        heldout_stats = all_coverage.loc[
            all_coverage["accent"].eq(heldout)
            & all_coverage["normalized_word"].isin(train_words)
            & all_coverage["speakers"].ge(2)
        ]
        test_words = set(heldout_stats["normalized_word"].astype(str))

        dev_seen = dev_coverage.loc[
            dev_coverage["accent"].isin(seen)
            & dev_coverage["normalized_word"].isin(train_words)
        ]
        dev_word_counts = dev_seen.groupby("normalized_word")["accent"].nunique()
        dev_words = set(dev_word_counts.loc[dev_word_counts >= min_dev_accents].index.astype(str))

        train_capacity = len(
            train.loc[
                train["accent"].isin(seen)
                & train["normalized_word"].isin(train_words)
            ]
        )
        dev_capacity = len(
            dev.loc[
                dev["accent"].isin(seen)
                & dev["normalized_word"].isin(dev_words)
            ]
        )
        test_capacity = int(heldout_stats["examples"].sum())
        if (
            train_capacity < targets["train"]
            or dev_capacity < targets["dev"]
            or test_capacity < targets["test"]
        ):
            continue

        plans.append(
            SplitPlan(
                heldout_accent=heldout,
                seen_accents=seen,
                train_words=frozenset(train_words),
                dev_words=frozenset(dev_words),
                test_words=frozenset(test_words),
                score=(len(test_words), len(dev_words), test_capacity),
            )
        )

    if not plans:
        requested = "an automatically selected accent" if heldout_accent == "auto" else repr(heldout_accent)
        raise ValueError(
            f"No valid 50-hour split could be built with held-out accent {requested}. "
            "Inspect accent coverage or reduce the requested size."
        )
    return max(plans, key=lambda plan: (plan.score, plan.heldout_accent))


def _speaker_diverse_order(group: pd.DataFrame, rng: random.Random) -> list[int]:
    by_speaker: dict[str, list[int]] = defaultdict(list)
    for index, speaker in zip(group.index, group["speaker_id"].astype(str)):
        by_speaker[speaker].append(int(index))
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    ordered: list[int] = []
    remainder: list[int] = []
    for speaker in speakers:
        indices = by_speaker[speaker]
        rng.shuffle(indices)
        ordered.append(indices[0])
        remainder.extend(indices[1:])
    rng.shuffle(remainder)
    return ordered + remainder


def balanced_sample(
    frame: pd.DataFrame,
    *,
    target: int,
    group_columns: list[str],
    minimum_speakers: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    queues: dict[tuple, deque[int]] = {}
    selected: list[int] = []
    for key, group in frame.groupby(group_columns, sort=True):
        if group["speaker_id"].nunique() < minimum_speakers:
            continue
        key_tuple = key if isinstance(key, tuple) else (key,)
        order = _speaker_diverse_order(group, rng)
        selected.extend(order[:minimum_speakers])
        queues[key_tuple] = deque(order[minimum_speakers:])

    if len(selected) > target:
        raise ValueError(
            f"Coverage floor requires {len(selected):,} rows, exceeding target {target:,}."
        )
    keys = sorted(queues)
    while len(selected) < target:
        rng.shuffle(keys)
        progress = False
        for key in keys:
            if queues[key]:
                selected.append(queues[key].popleft())
                progress = True
                if len(selected) == target:
                    break
        if not progress:
            raise ValueError(
                f"Only {len(selected):,} balanced rows are available; target is {target:,}."
            )
    return frame.loc[selected].copy()


def _limit_words_for_floor(
    frame: pd.DataFrame,
    words: set[str] | frozenset[str],
    *,
    target: int,
    floor_per_word: int,
    seed: int,
) -> set[str]:
    candidates = sorted(words)
    random.Random(seed).shuffle(candidates)
    limit = target // floor_per_word
    if limit < 1:
        raise ValueError("Target is too small for even one contrastive word class.")
    return set(candidates[:limit])


def build_subset(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    required = {
        "audio_path", "duration_s", "speaker_id", "accent", "normalized_word", "split"
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input corpus lacks columns: {sorted(missing)}")
    frame = frame.copy()
    frame["accent"] = frame["accent"].astype(str).str.strip().str.casefold()
    frame["normalized_word"] = frame["normalized_word"].astype(str)
    frame["source_split"] = frame["split"].astype(str)
    if not frame["duration_s"].eq(1.0).all():
        raise ValueError("This size-controlled MSWC builder expects one-second clips.")

    speaker_accent_counts = frame.groupby("speaker_id")["accent"].nunique()
    ambiguous_speakers = set(speaker_accent_counts.loc[speaker_accent_counts > 1].index)
    frame = frame.loc[~frame["speaker_id"].isin(ambiguous_speakers)].copy()

    total = round(args.target_hours * 3600)
    test_target = round(total * args.test_fraction)
    dev_target = round(total * args.dev_fraction)
    train_target = total - dev_target - test_target
    targets = {"train": train_target, "dev": dev_target, "test": test_target}

    plan = choose_plan(
        frame,
        heldout_accent=args.heldout_accent.casefold(),
        seen_accent_count=args.seen_accents,
        min_train_speakers=args.min_train_speakers_per_accent,
        min_dev_accents=args.min_dev_accents,
        targets=targets,
    )

    train_words = _limit_words_for_floor(
        frame, plan.train_words, target=train_target,
        floor_per_word=args.seen_accents * args.min_train_speakers_per_accent,
        seed=args.seed,
    )
    train_pool = frame.loc[
        frame["source_split"].eq("train")
        & frame["accent"].isin(plan.seen_accents)
        & frame["normalized_word"].isin(train_words)
    ]
    train = balanced_sample(
        train_pool, target=train_target,
        group_columns=["normalized_word", "accent"],
        minimum_speakers=args.min_train_speakers_per_accent, seed=args.seed,
    )

    dev_words = set(plan.dev_words) & set(train["normalized_word"])
    dev_words = _limit_words_for_floor(
        frame, dev_words, target=dev_target,
        floor_per_word=args.seen_accents, seed=args.seed + 1,
    )
    dev_pool = frame.loc[
        frame["source_split"].eq("dev")
        & frame["accent"].isin(plan.seen_accents)
        & frame["normalized_word"].isin(dev_words)
    ]
    dev = balanced_sample(
        dev_pool, target=dev_target,
        group_columns=["normalized_word", "accent"],
        minimum_speakers=1, seed=args.seed + 1,
    )

    test_words = set(plan.test_words) & set(train["normalized_word"])
    test_words = _limit_words_for_floor(
        frame, test_words, target=test_target,
        floor_per_word=2, seed=args.seed + 2,
    )
    test_pool = frame.loc[
        frame["accent"].eq(plan.heldout_accent)
        & frame["normalized_word"].isin(test_words)
    ]
    test = balanced_sample(
        test_pool, target=test_target, group_columns=["normalized_word"],
        minimum_speakers=2, seed=args.seed + 2,
    )

    train["split"] = "train"
    dev["split"] = "dev"
    test["split"] = "test"
    output = pd.concat([train, dev, test], ignore_index=True)
    speaker_sets = {
        split: set(output.loc[output["split"].eq(split), "speaker_id"])
        for split in ("train", "dev", "test")
    }
    overlaps = {
        "train_dev": len(speaker_sets["train"] & speaker_sets["dev"]),
        "train_test": len(speaker_sets["train"] & speaker_sets["test"]),
        "dev_test": len(speaker_sets["dev"] & speaker_sets["test"]),
    }
    train_accents = set(output.loc[output["split"].eq("train"), "accent"])
    dev_accents = set(output.loc[output["split"].eq("dev"), "accent"])
    test_accents = set(output.loc[output["split"].eq("test"), "accent"])
    passed = (
        not any(overlaps.values())
        and test_accents == {plan.heldout_accent}
        and plan.heldout_accent not in train_accents | dev_accents
        and len(output) == total
    )
    report = {
        "status": "passed" if passed else "failed",
        "target_hours": args.target_hours,
        "actual_hours": float(output["duration_s"].sum() / 3600),
        "target_examples": total,
        "heldout_accent": plan.heldout_accent,
        "heldout_accent_slug": slug(plan.heldout_accent),
        "seen_accents": list(plan.seen_accents),
        "split_targets": targets,
        "speaker_overlap": overlaps,
        "ambiguous_speakers_removed": len(ambiguous_speakers),
        "min_train_speakers_per_accent": args.min_train_speakers_per_accent,
        "min_dev_accents": args.min_dev_accents,
        "seed": args.seed,
        "selection_note": (
            "Vocabulary is fitted on seen-accent train rows; held-out coverage is used only "
            "to construct a matched lexical evaluation set."
        ),
    }
    if not passed:
        raise ValueError(f"Held-out-accent validation failed: {report}")
    return output, report


def save(output: pd.DataFrame, report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_dir / "corpus.parquet", index=False)
    split_stats = (
        output.groupby("split")
        .agg(
            examples=("audio_path", "size"),
            duration_s=("duration_s", "sum"),
            speakers=("speaker_id", "nunique"),
            accents=("accent", "nunique"),
            words=("normalized_word", "nunique"),
        )
        .reset_index()
    )
    split_stats["hours"] = split_stats["duration_s"] / 3600
    split_stats.to_csv(output_dir / "split_stats.csv", index=False)
    (
        output.groupby(["split", "accent"])
        .agg(
            examples=("audio_path", "size"),
            speakers=("speaker_id", "nunique"),
            words=("normalized_word", "nunique"),
        )
        .reset_index()
        .to_csv(output_dir / "accent_stats.csv", index=False)
    )
    train = output.loc[output["split"].eq("train")]
    vocabulary = (
        train.groupby("normalized_word")
        .agg(
            examples=("audio_path", "size"),
            speakers=("speaker_id", "nunique"),
            accents=("accent", "nunique"),
        )
        .reset_index()
        .sort_values("normalized_word")
    )
    vocabulary.to_csv(output_dir / "vocabulary.csv", index=False)
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "_SUCCESS").write_text("MSWC 50-hour held-out-accent corpus ready\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-hours", type=float, default=50.0)
    parser.add_argument("--heldout-accent", default="auto")
    parser.add_argument("--seen-accents", type=int, default=6)
    parser.add_argument("--min-train-speakers-per-accent", type=int, default=3)
    parser.add_argument("--min-dev-accents", type=int, default=5)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dev_fraction <= 0 or args.test_fraction <= 0:
        raise ValueError("Development and test fractions must be positive.")
    if args.dev_fraction + args.test_fraction >= 1:
        raise ValueError("Development and test fractions must sum to less than one.")
    frame = pd.read_parquet(args.input_parquet)
    output, report = build_subset(frame, args)
    save(output, report, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
