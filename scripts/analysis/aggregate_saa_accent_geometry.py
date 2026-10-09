"""Combine completed SAA accent-geometry folds and compute fold macros."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    args = parser.parse_args()
    paths = sorted(args.input_root.glob("*/saa_accent_geometry.csv"))
    if not paths:
        raise FileNotFoundError(f"No completed SAA results under {args.input_root}")
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    frame.to_csv(args.input_root / "saa_accent_geometry_all_folds.csv", index=False)
    groups = ["model", "space", "condition"]
    numeric = frame.select_dtypes(include="number").columns.tolist()
    macro = frame.groupby(groups, as_index=False)[numeric].mean()
    fold_counts = frame.groupby(groups)["fold"].nunique().rename("folds").reset_index()
    macro = macro.merge(fold_counts, on=groups, validate="one_to_one")
    macro.to_csv(args.input_root / "saa_accent_geometry_macro.csv", index=False)


if __name__ == "__main__":
    main()
