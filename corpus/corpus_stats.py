#!/usr/bin/env python3

# python corpus/corpus_stats.py data/processed/aesrc/corpus.parquet
# python corpus/corpus_stats.py data/processed/librispeech_train/corpus.parquet

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute number of train examples and total duration from a parquet corpus."
    )
    parser.add_argument(
        "parquet",
        type=Path,
        help="Path to the parquet file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to filter. Default: train.",
    )
    parser.add_argument(
        "--split_col",
        type=str,
        default="split",
        help="Name of the split column. Default: split.",
    )
    parser.add_argument(
        "--duration_col",
        type=str,
        default="duration_s",
        help="Name of the duration column in seconds. Default: duration_s.",
    )

    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)

    if args.split_col not in df.columns:
        raise ValueError(
            f"Split column '{args.split_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if args.duration_col not in df.columns:
        raise ValueError(
            f"Duration column '{args.duration_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    split_df = df[
        df[args.split_col].astype(str).str.lower()
        == args.split.lower()
    ].copy()

    durations = pd.to_numeric(split_df[args.duration_col], errors="coerce")

    n_examples = len(split_df)
    total_seconds = durations.sum()
    total_hours = total_seconds / 3600
    mean_seconds = durations.mean()
    median_seconds = durations.median()
    min_seconds = durations.min()
    max_seconds = durations.max()

    print(f"Parquet: {args.parquet}")
    print(f"Split: {args.split}")
    print(f"Examples: {n_examples:,}")
    print(f"Total duration: {total_seconds:,.2f} s")
    print(f"Total duration: {total_hours:,.2f} h")
    print(f"Mean duration: {mean_seconds:.2f} s")
    print(f"Median duration: {median_seconds:.2f} s")
    print(f"Min duration: {min_seconds:.2f} s")
    print(f"Max duration: {max_seconds:.2f} s")


if __name__ == "__main__":
    main()