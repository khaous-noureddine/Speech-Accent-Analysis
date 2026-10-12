#!/usr/bin/env python3
"""Aggregate Arabic sequential word-SupCon weight-ablation evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "experiments/ablations/sequential-l2-word-supcon-weight/"
            "librispeech-100h/wav2vec2-large-lv60/arabic"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sequential-l2-word-lambda-ablation/results.csv"),
    )
    args = parser.parse_args()

    rows = []
    for path in sorted(args.input_root.glob("lambda-*/evaluation/*/seed=*/*/*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "lambda": path.parents[5].name,
                "objective": metrics["objective"],
                "decoder": metrics["decoder"],
                "dataset": metrics["dataset"],
                "wer_percent": metrics.get("wer_percent", 100 * metrics["wer"]),
                "checkpoint_sha256": metrics.get("checkpoint_sha256"),
                "metrics_path": str(path),
            }
        )

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    if frame.empty:
        print(f"No completed evaluations found. Wrote empty file: {args.output}")
        return

    summary = frame.pivot(index=["lambda", "decoder"], columns="dataset", values="wer_percent")
    print(summary.round(2).to_string())
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
