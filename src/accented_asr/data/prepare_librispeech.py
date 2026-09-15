"""Build a portable LibriSpeech parquet from one or more extracted subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import soundfile as sf


def normalize(text: str) -> str:
    return " ".join(text.upper().strip().split())


def build_inventory(corpus_dirs: list[Path], repository_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    seen: set[str] = set()
    for corpus_dir in corpus_dirs:
        if not corpus_dir.is_dir():
            raise FileNotFoundError(corpus_dir)
        subset = corpus_dir.name
        transcript_files = sorted(corpus_dir.glob("*/*/*.trans.txt"))
        if not transcript_files:
            raise ValueError(f"No LibriSpeech transcripts found in {corpus_dir}.")
        for transcript_file in transcript_files:
            for line in transcript_file.read_text(encoding="utf-8").splitlines():
                utterance_id, separator, transcript = line.partition(" ")
                if not separator or utterance_id in seen:
                    if utterance_id in seen:
                        raise ValueError(f"Duplicate LibriSpeech utterance: {utterance_id}")
                    raise ValueError(f"Malformed transcript line in {transcript_file}: {line}")
                audio = transcript_file.parent / f"{utterance_id}.flac"
                if not audio.is_file():
                    raise FileNotFoundError(audio)
                info = sf.info(audio)
                if info.samplerate != 16_000 or info.channels != 1:
                    raise ValueError(f"Unexpected audio format for {audio}: {info}")
                seen.add(utterance_id)
                rows.append({
                    "speaker_id": f"LS_{utterance_id.split('-', 1)[0]}",
                    "gender": "unknown",
                    "split": "train",
                    "subset": subset,
                    "utterance_id": utterance_id,
                    "transcript": normalize(transcript),
                    "audio_path": str(audio.resolve().relative_to(repository_root)),
                    "duration_s": round(info.duration, 6),
                })
    return pd.DataFrame(rows).sort_values("utterance_id").reset_index(drop=True)


def save_inventory(frame: pd.DataFrame, output: Path, root: Path) -> None:
    if frame.empty:
        raise ValueError("The LibriSpeech inventory is empty.")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    fingerprint_rows = frame[["utterance_id", "audio_path", "duration_s"]].to_dict("records")
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "dataset": "LibriSpeech",
        "subsets": {
            subset: {
                "utterances": int(len(part)),
                "hours": float(part["duration_s"].sum() / 3600),
                "speakers": int(part["speaker_id"].nunique()),
            }
            for subset, part in frame.groupby("subset", sort=True)
        },
        "total_utterances": int(len(frame)),
        "total_hours": float(frame["duration_s"].sum() / 3600),
        "source_fingerprint_sha256": fingerprint,
        "path_policy": "repository-relative raw FLAC",
    }
    (output.parent / "inventory_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument("--repository-root", default=Path.cwd(), type=Path)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    corpus_dirs = [path if path.is_absolute() else root / path for path in args.corpus_dirs]
    output = args.output_parquet if args.output_parquet.is_absolute() else root / args.output_parquet
    save_inventory(build_inventory(corpus_dirs, root), output, root)


if __name__ == "__main__":
    main()
