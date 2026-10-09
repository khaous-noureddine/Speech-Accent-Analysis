#!/usr/bin/env python3
"""Aggregate completed cross-accent geometry and accent-probe folds."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def aggregate(root: Path, filename: str, output_name: str) -> None:
    files = sorted(root.glob(f"*/{filename}"))
    if not files:
        raise FileNotFoundError(f"No {filename} files below {root}")
    frame = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    frame.to_csv(root / output_name, index=False)
    group = [column for column in ("model", "space", "condition") if column in frame]
    frame.groupby(group, as_index=False).mean(numeric_only=True).to_csv(
        root / output_name.replace(".csv", "_macro.csv"), index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    args = parser.parse_args()
    aggregate(args.input_root, "geometry.csv", "geometry_all_folds.csv")
    aggregate(args.input_root, "accent_probe.csv", "accent_probe_all_folds.csv")


if __name__ == "__main__":
    main()
