"""Import raw L2-ARCTIC and create six leave-one-accent-out folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from accented_asr.data.l2_arctic_raw import build_raw_inventory
from accented_asr.data.l2_arctic_splits import (
    PROTOCOL_LOO,
    SPLITS,
    SplitRatios,
    assign_records,
    build_manifest,
    build_validation_report,
    inventory_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir", type=Path, required=True,
        help="Raw directory containing ABA/ABA/wav, ABA/ABA/transcript, etc.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--split-seed", type=int, default=20260817)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stats(frame: pd.DataFrame) -> pd.DataFrame:
    stats = (
        frame.groupby("split")
        .agg(
            n_examples=("speaker_id", "size"),
            n_speakers=("speaker_id", "nunique"),
            n_prompts=("prompt_id", "nunique"),
            n_l1s=("native_language", "nunique"),
            duration_s=("duration_s", "sum"),
        )
        .reindex(SPLITS)
        .reset_index()
    )
    stats["duration_s"] = stats["duration_s"].round(3)
    stats["duration_h"] = (stats["duration_s"] / 3600).round(3)
    return stats


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory, inventory_report = build_raw_inventory(
        corpus_dir=args.corpus_dir,
        wavs_dir=output_dir / "wavs",
        repository_root=args.repository_root,
    )
    pd.DataFrame(inventory).to_parquet(output_dir / "inventory.parquet", index=False)
    _write_json(output_dir / "inventory_report.json", inventory_report)

    speakers_by_l1, _ = inventory_index(inventory)
    ratios = SplitRatios(args.train_ratio, args.dev_ratio, args.test_ratio)
    summary = []
    for heldout_l1 in sorted(speakers_by_l1):
        manifest = build_manifest(
            inventory,
            protocol=PROTOCOL_LOO,
            split_seed=args.split_seed,
            ratios=ratios,
            heldout_l1=heldout_l1,
        )
        assigned_records = assign_records(inventory, manifest)
        frame = pd.DataFrame(assigned_records)
        fold_dir = output_dir / heldout_l1.lower()
        fold_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(fold_dir / "corpus.parquet", index=False)
        _stats(frame).to_csv(fold_dir / "split_stats.csv", index=False)
        _write_json(fold_dir / "manifest.json", manifest)
        _write_json(
            fold_dir / "validation_report.json",
            build_validation_report(assigned_records, manifest),
        )
        (fold_dir / "manifest.content.sha256").write_text(
            manifest["sha256"] + "\n", encoding="utf-8"
        )
        summary.append(
            {
                "heldout_accent": heldout_l1,
                "n_examples": len(frame),
                "manifest_sha256": manifest["sha256"],
                "parquet": f"{heldout_l1.lower()}/corpus.parquet",
            }
        )
        print(f"[{heldout_l1}] wrote {len(frame):,} rows to {fold_dir}")

    pd.DataFrame(summary).to_csv(output_dir / "fold_summary.csv", index=False)
    print(f"Inventory: {len(inventory):,} rows from raw data")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
