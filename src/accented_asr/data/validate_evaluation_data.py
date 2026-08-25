"""Validate a processed evaluation parquet and all referenced audio files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from accented_asr.data.prepare_evaluation_data import frame_fingerprint


def validate(parquet: Path, *, dataset: str, split: str, root: Path) -> dict:
    frame = pd.read_parquet(parquet)
    if "dataset" not in frame:
        frame = frame.copy()
        frame["dataset"] = frame["corpus"] if "corpus" in frame else dataset
    required = {
        "dataset", "speaker_id", "split", "utterance_id", "transcript", "audio_path"
    }
    missing_columns = sorted(required - set(frame.columns))
    selected = frame.loc[frame["split"] == split].copy() if not missing_columns else frame.iloc[:0]
    duplicate_ids = (
        int(selected.duplicated(["speaker_id", "utterance_id"]).sum())
        if not selected.empty else 0
    )
    duplicate_audio = int(selected["audio_path"].duplicated().sum()) if not selected.empty else 0
    empty_transcripts = (
        int(selected["transcript"].fillna("").astype(str).str.strip().eq("").sum())
        if not selected.empty else 0
    )
    missing_audio = []
    if not selected.empty:
        for value in selected["audio_path"].astype(str):
            path = Path(value) if Path(value).is_absolute() else root / value
            if not path.is_file():
                missing_audio.append(value)
    errors = []
    if missing_columns:
        errors.append(f"missing columns: {missing_columns}")
    if selected.empty:
        errors.append(f"no rows for split={split}")
    if duplicate_ids:
        errors.append(f"{duplicate_ids} duplicate speaker/utterance records")
    if duplicate_audio:
        errors.append(f"{duplicate_audio} duplicate audio paths")
    if empty_transcripts:
        errors.append(f"{empty_transcripts} empty transcripts")
    if missing_audio:
        errors.append(f"{len(missing_audio)} missing audio files")
    report = {
        "dataset": dataset,
        "split": split,
        "rows": len(selected),
        "speakers": int(selected["speaker_id"].nunique()) if not selected.empty else 0,
        "missing_columns": missing_columns,
        "duplicate_speaker_utterance_records": duplicate_ids,
        "duplicate_audio_paths": duplicate_audio,
        "empty_transcripts": empty_transcripts,
        "missing_audio_files": len(missing_audio),
        "missing_audio_examples": missing_audio[:10],
        "content_sha256": frame_fingerprint(selected) if not errors else None,
        "valid": not errors,
        "errors": errors,
    }
    if errors:
        raise ValueError(f"Invalid processed {dataset} data: {'; '.join(errors)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    parquet = args.parquet if args.parquet.is_absolute() else root / args.parquet
    report = validate(parquet, dataset=args.dataset, split=args.split, root=root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
