"""Build the canonical processed L2-ARCTIC inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from accented_asr.data.l2_arctic_inventory import build_processed_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-parquet",
        type=Path,
        nargs="+",
        required=True,
        help="Legacy processed Parquets used only as per-audio metadata sources.",
    )
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--expected-sample-rate",
        type=int,
        default=None,
        help="Optional strict check; omit when training resamples audio on load.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = [pd.read_parquet(path) for path in args.reference_parquet]
    reference = pd.concat(frames, ignore_index=True).to_dict("records")
    inventory, report = build_processed_inventory(
        reference,
        wav_dir=args.wav_dir,
        repository_root=args.repository_root,
        expected_sample_rate=args.expected_sample_rate,
    )

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inventory).to_parquet(args.output_parquet, index=False)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output_parquet} ({len(inventory):,} examples)")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
