#!/usr/bin/env python3
"""Generate leakage-safe L2-ARCTIC Parquet splits and manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from accented_asr.data.l2_arctic_splits import (  # noqa: E402
    PROTOCOL_LOO,
    PROTOCOL_MAIN,
    SPLITS,
    SplitRatios,
    assign_records,
    build_all_manifests,
    build_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build strict main and leave-one-L1-out L2-ARCTIC splits."
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        required=True,
        help="One-row-per-record L2-ARCTIC inventory with no split filtering.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260817)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def _directory_name(manifest: dict) -> str:
    if manifest["protocol"] == PROTOCOL_MAIN:
        return "main"
    if manifest["protocol"] == PROTOCOL_LOO:
        return f"loo_{manifest['heldout_l1'].lower()}"
    raise ValueError(f"Unsupported protocol: {manifest['protocol']}")


def _write_stats(frame: pd.DataFrame, output_path: Path) -> None:
    aggregations = {
        "n_examples": ("speaker_id", "size"),
        "n_speakers": ("speaker_id", "nunique"),
        "n_prompts": ("prompt_id", "nunique"),
        "n_l1s": ("native_language", "nunique"),
    }
    if "duration_s" in frame.columns:
        aggregations["duration_s"] = ("duration_s", "sum")
    stats = frame.groupby("split").agg(**aggregations).reindex(SPLITS).reset_index()
    if "duration_s" in stats.columns:
        stats["duration_h"] = stats["duration_s"] / 3600.0
    stats.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    ratios = SplitRatios(args.train_ratio, args.dev_ratio, args.test_ratio)
    frame = pd.read_parquet(args.input_parquet)
    records = frame.to_dict("records")
    manifests = build_all_manifests(
        records, split_seed=args.split_seed, ratios=ratios
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for manifest in manifests:
        run_dir = args.output_dir / _directory_name(manifest)
        run_dir.mkdir(parents=True, exist_ok=True)
        assigned = pd.DataFrame(assign_records(records, manifest))

        assigned.to_parquet(run_dir / "corpus.parquet", index=False)
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (run_dir / "manifest.content.sha256").write_text(
            f"{manifest['sha256']}\n", encoding="utf-8"
        )
        report = build_validation_report(assigned.to_dict("records"), manifest)
        (run_dir / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_stats(assigned, run_dir / "split_stats.csv")
        summary.append(
            {
                "protocol": manifest["protocol"],
                "heldout_l1": manifest["heldout_l1"],
                "manifest_sha256": manifest["sha256"],
                "n_examples": len(assigned),
            }
        )
        print(f"Wrote {run_dir} ({len(assigned):,} examples)")

    pd.DataFrame(summary).to_csv(args.output_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
