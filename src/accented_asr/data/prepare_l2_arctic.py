"""Import a canonical L2-ARCTIC inventory directly from the raw corpus."""

from __future__ import annotations

import re
import shutil
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf
import pandas as pd

from accented_asr.data.l2_arctic_splits import (
    EXPECTED_L1S,
    PROTOCOL_LOO,
    SPLITS,
    SplitRatios,
    SplitValidationError,
    assign_records,
    build_manifest,
    build_validation_report,
    inventory_index,
)


SPEAKER_METADATA = {
    "ABA": ("Arabic", "male"),
    "YBAA": ("Arabic", "male"),
    "ZHAA": ("Arabic", "female"),
    "SKA": ("Arabic", "male"),
    "BWC": ("Chinese", "male"),
    "LXC": ("Chinese", "female"),
    "NCC": ("Chinese", "female"),
    "TXHC": ("Chinese", "male"),
    "ASI": ("Hindi", "male"),
    "RRBI": ("Hindi", "male"),
    "SVBI": ("Hindi", "female"),
    "TNI": ("Hindi", "female"),
    "HJK": ("Korean", "female"),
    "HKK": ("Korean", "male"),
    "YDCK": ("Korean", "female"),
    "YKWK": ("Korean", "male"),
    "EBVS": ("Spanish", "male"),
    "ERMS": ("Spanish", "male"),
    "MBMPS": ("Spanish", "female"),
    "NJS": ("Spanish", "female"),
    "PNV": ("Vietnamese", "female"),
    "THV": ("Vietnamese", "female"),
    "TLV": ("Vietnamese", "male"),
    "HQTV": ("Vietnamese", "male"),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def inventory_sha256(records: list[dict[str, Any]]) -> str:
    payload = [
        {
            key: record.get(key)
            for key in (
                "speaker_id", "native_language", "prompt_id", "transcript",
                "audio_path", "duration_s", "sample_rate", "channels",
            )
        }
        for record in sorted(records, key=lambda row: (row["speaker_id"], row["prompt_id"]))
    ]
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def normalize_transcript(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s']", "", text)
    return re.sub(r"\s+", " ", text)


def _portable_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as error:
        raise SplitValidationError(
            f"Processed audio must be inside the repository: {path}"
        ) from error


def build_raw_inventory(
    *,
    corpus_dir: Path,
    wavs_dir: Path,
    repository_root: Path,
    copy_audio: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read raw speaker folders and retain prompts complete for all 24 speakers.

    Audio is copied once into ``wavs_dir``. Split folds subsequently reference
    those files and never make fold-specific audio copies.
    """

    corpus_dir = corpus_dir.resolve()
    wavs_dir.mkdir(parents=True, exist_ok=True)
    rows_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    prompts_by_speaker: dict[str, set[str]] = defaultdict(set)
    missing_transcripts: list[str] = []
    audio_formats: Counter[str] = Counter()
    raw_wav_count = 0

    for speaker, (l1, gender) in sorted(SPEAKER_METADATA.items()):
        speaker_root = corpus_dir / speaker / speaker
        raw_wav_dir = speaker_root / "wav"
        transcript_dir = speaker_root / "transcript"
        if not raw_wav_dir.is_dir() or not transcript_dir.is_dir():
            raise FileNotFoundError(
                f"Missing raw wav/transcript directories for {speaker}: {speaker_root}"
            )

        for source_wav in sorted(raw_wav_dir.glob("*.wav")):
            raw_wav_count += 1
            prompt = source_wav.stem
            transcript_path = transcript_dir / f"{prompt}.txt"
            if not transcript_path.is_file():
                missing_transcripts.append(f"{speaker}/{prompt}")
                continue

            destination = wavs_dir / f"{speaker}_{prompt}.wav"
            if copy_audio:
                if not destination.exists() or source_wav.stat().st_size != destination.stat().st_size:
                    shutil.copy2(source_wav, destination)
            else:
                destination = source_wav

            info = sf.info(str(source_wav))
            if info.frames <= 0:
                raise SplitValidationError(f"Empty raw audio file: {source_wav}")
            transcript = normalize_transcript(
                transcript_path.read_text(encoding="utf-8", errors="replace")
            )
            if not transcript:
                raise SplitValidationError(f"Empty transcript: {transcript_path}")

            pair = (speaker, prompt)
            if pair in rows_by_pair:
                raise SplitValidationError(f"Duplicate raw speaker/prompt pair: {pair}")
            rows_by_pair[pair] = {
                "corpus": "l2_arctic",
                "speaker_id": speaker,
                "gender": gender,
                "native_language": l1,
                "accent": l1,
                "utterance_id": prompt,
                "prompt_id": prompt,
                "transcript": transcript,
                "audio_path": _portable_path(destination, repository_root),
                "duration_s": round(float(info.duration), 6),
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
            }
            prompts_by_speaker[speaker].add(prompt)
            audio_formats[f"{info.samplerate}Hz/{info.channels}ch/{info.subtype}"] += 1

    found_l1s = {metadata[0] for metadata in SPEAKER_METADATA.values()}
    if found_l1s != EXPECTED_L1S:
        raise SplitValidationError("The built-in speaker map has invalid L1 groups.")

    common_prompts = sorted(
        set.intersection(*(prompts_by_speaker[s] for s in sorted(SPEAKER_METADATA)))
    )
    if not common_prompts:
        raise SplitValidationError("No prompt has raw audio and transcript for all speakers.")

    inventory = [
        rows_by_pair[(speaker, prompt)]
        for prompt in common_prompts
        for speaker in sorted(SPEAKER_METADATA)
    ]
    report = {
        "schema_version": 2,
        "dataset": "l2_arctic",
        "source": "raw_speaker_wav_and_transcript_directories",
        "corpus_dir": corpus_dir.relative_to(repository_root.resolve()).as_posix(),
        "policy": "prompts_with_raw_audio_and_transcript_for_all_24_speakers",
        "n_speakers": len(SPEAKER_METADATA),
        "n_raw_wav_files": raw_wav_count,
        "n_raw_pairs_with_transcript": len(rows_by_pair),
        "n_missing_transcripts": len(missing_transcripts),
        "missing_transcripts": missing_transcripts,
        "n_eligible_prompts": len(common_prompts),
        "n_inventory_examples": len(inventory),
        "n_excluded_raw_pairs": len(rows_by_pair) - len(inventory),
        "speakers_by_l1": {
            l1: sorted(s for s, (speaker_l1, _) in SPEAKER_METADATA.items() if speaker_l1 == l1)
            for l1 in sorted(EXPECTED_L1S)
        },
        "audio_formats": dict(sorted(audio_formats.items())),
        "inventory_sha256": inventory_sha256(inventory),
    }
    return inventory, report


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _split_stats(frame: pd.DataFrame) -> pd.DataFrame:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare raw L2-ARCTIC leave-one-accent-out folds."
    )
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--split-seed", type=int, default=20260817)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory, report = build_raw_inventory(
        corpus_dir=args.corpus_dir,
        wavs_dir=output_dir / "wavs",
        repository_root=args.repository_root,
    )
    pd.DataFrame(inventory).to_parquet(output_dir / "inventory.parquet", index=False)
    _write_json(output_dir / "inventory_report.json", report)

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
        records = assign_records(inventory, manifest)
        frame = pd.DataFrame(records)
        fold_dir = output_dir / heldout_l1.lower()
        fold_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(fold_dir / "corpus.parquet", index=False)
        _split_stats(frame).to_csv(fold_dir / "split_stats.csv", index=False)
        _write_json(fold_dir / "manifest.json", manifest)
        _write_json(
            fold_dir / "validation_report.json",
            build_validation_report(records, manifest),
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
