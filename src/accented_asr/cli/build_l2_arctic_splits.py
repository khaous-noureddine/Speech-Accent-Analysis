"""Generate leakage-safe L2-ARCTIC Parquet splits and manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from accented_asr.data.l2_arctic_splits import (
    PROTOCOL_LOO,
    PROTOCOL_MAIN,
    SPLITS,
    SplitRatios,
    assign_records,
    build_all_manifests,
    build_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Ignored data directory receiving split Parquets.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Versioned directory receiving manifests and validation reports.",
    )
    parser.add_argument("--split-seed", type=int, default=20260817)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def _relative_directory(manifest: dict) -> Path:
    if manifest["protocol"] == PROTOCOL_MAIN:
        return Path("main")
    if manifest["protocol"] == PROTOCOL_LOO:
        return Path("leave_one_l1_out") / manifest["heldout_l1"].lower()
    raise ValueError(f"Unsupported protocol: {manifest['protocol']}")


def _stats(frame: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "n_examples": ("speaker_id", "size"),
        "n_speakers": ("speaker_id", "nunique"),
        "n_prompts": ("prompt_id", "nunique"),
        "n_l1s": ("native_language", "nunique"),
    }
    if "duration_s" in frame.columns:
        aggregations["duration_s"] = ("duration_s", "sum")
    result = frame.groupby("split").agg(**aggregations).reindex(SPLITS).reset_index()
    if "duration_s" in result.columns:
        result["duration_h"] = result["duration_s"] / 3600.0
    return result


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    ratios = SplitRatios(args.train_ratio, args.dev_ratio, args.test_ratio)
    records = pd.read_parquet(args.input_parquet).to_dict("records")
    manifests = build_all_manifests(records, split_seed=args.split_seed, ratios=ratios)

    summary = []
    for manifest in manifests:
        relative_dir = _relative_directory(manifest)
        data_dir = args.output_dir / relative_dir
        metadata_dir = args.manifest_dir / relative_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        assigned_records = assign_records(records, manifest)
        assigned = pd.DataFrame(assigned_records)
        report = build_validation_report(assigned_records, manifest)
        stats = _stats(assigned)

        assigned.to_parquet(data_dir / "corpus.parquet", index=False)
        _write_json(metadata_dir / "manifest.json", manifest)
        _write_json(metadata_dir / "validation_report.json", report)
        (metadata_dir / "manifest.content.sha256").write_text(
            f"{manifest['sha256']}\n", encoding="utf-8"
        )
        stats.to_csv(metadata_dir / "split_stats.csv", index=False)

        summary.append(
            {
                "protocol": manifest["protocol"],
                "heldout_l1": manifest["heldout_l1"],
                "manifest_sha256": manifest["sha256"],
                "n_examples": len(assigned),
                "data_parquet": (relative_dir / "corpus.parquet").as_posix(),
            }
        )
        print(f"Wrote {data_dir / 'corpus.parquet'} ({len(assigned):,} examples)")
        print(f"Wrote {metadata_dir}")

    summary_frame = pd.DataFrame(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    summary_frame.to_csv(args.manifest_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
