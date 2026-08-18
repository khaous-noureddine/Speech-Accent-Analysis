"""Build a canonical L2-ARCTIC inventory from existing processed artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import soundfile as sf

from accented_asr.data.l2_arctic_splits import EXPECTED_L1S, SplitValidationError


SEMANTIC_FIELDS = (
    "corpus",
    "speaker_id",
    "gender",
    "native_language",
    "accent",
    "utterance_id",
    "prompt_id",
    "transcript",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def inventory_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            key: record.get(key)
            for key in (
                "speaker_id",
                "native_language",
                "prompt_id",
                "transcript",
                "audio_path",
                "duration_s",
                "sample_rate",
                "channels",
            )
        }
        for record in sorted(
            records, key=lambda row: (str(row["speaker_id"]), str(row["prompt_id"]))
        )
    ]
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _deduplicate_reference_rows(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for index, record in enumerate(records):
        missing = {"speaker_id", "prompt_id", "native_language", "transcript"} - set(
            record
        )
        if missing:
            raise SplitValidationError(
                f"Reference row {index} is missing fields: {sorted(missing)}"
            )
        pair = (str(record["speaker_id"]), str(record["prompt_id"]))
        candidate = {field: record.get(field) for field in SEMANTIC_FIELDS}
        previous = by_pair.get(pair)
        if previous is not None and candidate != previous:
            differences = [
                field for field in SEMANTIC_FIELDS if previous[field] != candidate[field]
            ]
            raise SplitValidationError(
                f"Conflicting metadata for speaker/prompt {pair}: {differences}"
            )
        by_pair[pair] = candidate
    return by_pair


def build_processed_inventory(
    reference_records: Sequence[Mapping[str, Any]],
    *,
    wav_dir: Path,
    repository_root: Path,
    expected_sample_rate: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create an inventory for prompts fully documented for all 24 speakers.

    Existing fold assignments and legacy audio paths are ignored. Audio files
    are resolved from the single canonical ``wav_dir`` and are never copied.
    """

    wav_dir = wav_dir.resolve()
    repository_root = repository_root.resolve()
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"L2-ARCTIC WAV directory not found: {wav_dir}")

    by_pair = _deduplicate_reference_rows(reference_records)
    speakers_by_l1: dict[str, set[str]] = defaultdict(set)
    prompt_speakers: dict[str, set[str]] = defaultdict(set)
    for (speaker, prompt), record in by_pair.items():
        l1 = str(record["native_language"])
        speakers_by_l1[l1].add(speaker)
        prompt_speakers[prompt].add(speaker)

    if set(speakers_by_l1) != EXPECTED_L1S:
        raise SplitValidationError(
            f"Expected L1 groups {sorted(EXPECTED_L1S)}, "
            f"found {sorted(speakers_by_l1)}."
        )
    invalid_l1s = {
        l1: sorted(speakers)
        for l1, speakers in speakers_by_l1.items()
        if len(speakers) != 4
    }
    if invalid_l1s:
        raise SplitValidationError(
            f"Each L1 must contain exactly four speakers: {invalid_l1s}"
        )

    all_speakers = {speaker for speakers in speakers_by_l1.values() for speaker in speakers}
    eligible_prompts = sorted(
        prompt for prompt, speakers in prompt_speakers.items() if speakers == all_speakers
    )
    if not eligible_prompts:
        raise SplitValidationError("No prompt is documented for all 24 speakers.")

    audio_formats: Counter[str] = Counter()
    inventory: list[dict[str, Any]] = []
    for prompt in eligible_prompts:
        for speaker in sorted(all_speakers):
            record = dict(by_pair[(speaker, prompt)])
            wav_path = wav_dir / f"{speaker}_{prompt}.wav"
            if not wav_path.is_file():
                raise SplitValidationError(f"Missing canonical audio file: {wav_path}")
            try:
                info = sf.info(str(wav_path))
            except Exception as error:
                raise SplitValidationError(f"Unreadable audio file {wav_path}: {error}") from error
            if expected_sample_rate is not None and info.samplerate != expected_sample_rate:
                raise SplitValidationError(
                    f"Unexpected sample rate for {wav_path}: {info.samplerate}"
                )
            if info.frames <= 0:
                raise SplitValidationError(f"Empty audio file: {wav_path}")

            try:
                portable_audio_path = wav_path.relative_to(repository_root).as_posix()
            except ValueError as error:
                raise SplitValidationError(
                    f"Audio path must be inside repository root: {wav_path}"
                ) from error

            record.update(
                {
                    "audio_path": portable_audio_path,
                    "duration_s": round(info.duration, 6),
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                }
            )
            audio_formats[f"{info.samplerate}Hz/{info.channels}ch/{info.subtype}"] += 1
            inventory.append(record)

    wav_count = sum(1 for _ in wav_dir.glob("*.wav"))
    documented_prompts = len(prompt_speakers)
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "l2_arctic",
        "policy": "prompts_documented_for_all_24_speakers",
        "n_reference_rows": len(reference_records),
        "n_unique_reference_pairs": len(by_pair),
        "n_wav_files": wav_count,
        "n_documented_prompts": documented_prompts,
        "n_eligible_prompts": len(eligible_prompts),
        "n_inventory_examples": len(inventory),
        "n_excluded_wav_files": wav_count - len(inventory),
        "speakers_by_l1": {
            l1: sorted(speakers) for l1, speakers in sorted(speakers_by_l1.items())
        },
        "audio_formats": dict(sorted(audio_formats.items())),
        "inventory_sha256": inventory_sha256(inventory),
    }
    return inventory, report
