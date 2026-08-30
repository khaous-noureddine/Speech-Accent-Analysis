"""Build leakage-safe word-level contrastive corpora from accented speech.

The canonical input is one row per *word occurrence*.  This module currently
supports L2-ARCTIC word intervals from its supplied TextGrid files and isolated
single-word clips from Mozilla Common Voice TSV releases.  The fold builder is
dataset-agnostic once occurrences have been collected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import soundfile as sf

from accented_asr.data.prepare_l2_arctic import SPEAKER_METADATA


SCHEMA_VERSION = 1
REQUIRED_COLUMNS = {
    "dataset", "language", "accent", "speaker_id", "utterance_id",
    "word", "normalized_word", "audio_path", "start_s", "end_s",
    "duration_s", "alignment_source",
}


def normalize_word(value: str) -> str:
    """Unicode-aware lexical normalization without language-specific stemming."""

    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = "".join(
        character
        for character in value
        if unicodedata.category(character)[0] in {"L", "M"}
        or character in {"'", "’", "-"}
    )
    return value.replace("’", "'").strip("'-")


def portable_path(path: Path, repository_root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def occurrence_id(row: dict[str, Any]) -> str:
    identity = "|".join(
        str(row[key])
        for key in ("dataset", "speaker_id", "utterance_id", "start_s", "end_s")
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def parse_textgrid_words(path: Path) -> list[tuple[float, float, str]]:
    """Parse the ``words`` IntervalTier from a long Praat TextGrid."""

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    in_words = False
    current: dict[str, str] = {}
    intervals: list[tuple[float, float, str]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("item ["):
            in_words = False
            current = {}
        elif line.startswith("name ="):
            name = line.split("=", 1)[1].strip().strip('"')
            in_words = name == "words"
        elif in_words and line.startswith("intervals ["):
            current = {}
        elif in_words and line.startswith("xmin ="):
            current["xmin"] = line.split("=", 1)[1].strip()
        elif in_words and line.startswith("xmax ="):
            current["xmax"] = line.split("=", 1)[1].strip()
        elif in_words and line.startswith("text ="):
            text = line.split("=", 1)[1].strip().strip('"')
            if text and {"xmin", "xmax"} <= current.keys():
                intervals.append((float(current["xmin"]), float(current["xmax"]), text))
            current = {}
    if not intervals:
        raise ValueError(f"No non-empty word intervals in {path}")
    return intervals


def collect_l2_arctic(raw_dir: Path, repository_root: Path) -> pd.DataFrame:
    """Collect word occurrences using L2-ARCTIC's manually supplied TextGrids."""

    records: list[dict[str, Any]] = []
    missing_audio: list[Path] = []
    for speaker_id, (accent, _gender) in sorted(SPEAKER_METADATA.items()):
        speaker_root = raw_dir / speaker_id / speaker_id
        textgrid_dir = speaker_root / "textgrid"
        for textgrid_path in sorted(textgrid_dir.glob("*.TextGrid")):
            audio_path = speaker_root / "wav" / f"{textgrid_path.stem}.wav"
            if not audio_path.is_file():
                missing_audio.append(audio_path)
                continue
            audio_duration = float(sf.info(audio_path).duration)
            for position, (start_s, end_s, word) in enumerate(
                parse_textgrid_words(textgrid_path)
            ):
                normalized = normalize_word(word)
                if not normalized or end_s <= start_s or end_s > audio_duration + 0.05:
                    continue
                row: dict[str, Any] = {
                    "dataset": "l2_arctic",
                    "language": "en",
                    "accent": accent,
                    "speaker_id": speaker_id,
                    "utterance_id": textgrid_path.stem,
                    "prompt_id": textgrid_path.stem,
                    "word_position": position,
                    "word": word,
                    "normalized_word": normalized,
                    "audio_path": portable_path(audio_path, repository_root),
                    "alignment_path": portable_path(textgrid_path, repository_root),
                    "start_s": round(start_s, 4),
                    "end_s": round(end_s, 4),
                    "duration_s": round(end_s - start_s, 4),
                    "alignment_source": "l2_arctic_textgrid",
                }
                row["occurrence_id"] = occurrence_id(row)
                records.append(row)
    if missing_audio:
        raise FileNotFoundError(f"Missing {len(missing_audio)} L2-ARCTIC WAV files")
    return pd.DataFrame.from_records(records)


def _find_common_voice_tsv(raw_dir: Path, tsv_name: str) -> Path:
    direct = raw_dir / tsv_name
    matches = [direct] if direct.is_file() else sorted(raw_dir.rglob(tsv_name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {tsv_name} below {raw_dir}; found {len(matches)}"
        )
    return matches[0]


def collect_common_voice(
    raw_dir: Path,
    repository_root: Path,
    *,
    language: str,
    tsv_name: str = "validated.tsv",
) -> pd.DataFrame:
    """Collect genuinely isolated one-word Common Voice clips.

    Multi-word scripted clips are deliberately rejected because Common Voice
    does not ship word timestamps.  A later forced-aligner can export the same
    canonical occurrence schema and feed the generic fold builder.
    """

    tsv_path = _find_common_voice_tsv(raw_dir, tsv_name)
    table = pd.read_csv(tsv_path, sep="\t", keep_default_na=False)
    required = {"client_id", "path", "sentence", "accent"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{tsv_path} lacks Common Voice columns: {sorted(missing)}")
    records: list[dict[str, Any]] = []
    clips_dir = tsv_path.parent / "clips"
    for row in table.to_dict("records"):
        accent = str(row["accent"]).strip()
        normalized = normalize_word(row["sentence"])
        lexical_tokens = re.findall(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", str(row["sentence"]), re.UNICODE)
        if not accent or len(lexical_tokens) != 1 or not normalized:
            continue
        audio_path = clips_dir / str(row["path"])
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        duration = float(sf.info(audio_path).duration)
        record: dict[str, Any] = {
            "dataset": "common_voice",
            "language": language,
            "accent": accent,
            "speaker_id": str(row["client_id"]),
            "utterance_id": Path(str(row["path"])).stem,
            "word_position": 0,
            "word": str(row["sentence"]),
            "normalized_word": normalized,
            "audio_path": portable_path(audio_path, repository_root),
            "alignment_path": None,
            "start_s": 0.0,
            "end_s": round(duration, 4),
            "duration_s": round(duration, 4),
            "alignment_source": "isolated_word_clip",
        }
        record["occurrence_id"] = occurrence_id(record)
        records.append(record)
    return pd.DataFrame.from_records(records)


def validate_occurrences(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Occurrence inventory lacks columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("No usable word occurrences were collected.")
    if frame["occurrence_id"].duplicated().any():
        raise ValueError("Duplicate occurrence_id values detected.")
    if (frame["duration_s"] <= 0).any():
        raise ValueError("Non-positive word duration detected.")
    if frame[["language", "accent", "speaker_id", "normalized_word"]].eq("").any().any():
        raise ValueError("Empty language/accent/speaker/word metadata detected.")


def eligible_words(
    frame: pd.DataFrame,
    *,
    min_accents: int,
    min_speakers_per_accent: int,
    min_duration_s: float,
    max_duration_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = frame.loc[
        frame["duration_s"].between(min_duration_s, max_duration_s, inclusive="both")
    ].copy()
    speakers = (
        filtered.groupby(["language", "normalized_word", "accent"])["speaker_id"]
        .nunique()
        .rename("n_speakers")
        .reset_index()
    )
    covered = speakers.loc[speakers["n_speakers"] >= min_speakers_per_accent]
    vocabulary = (
        covered.groupby(["language", "normalized_word"])
        .agg(
            n_accents=("accent", "nunique"),
            min_speakers_per_accent=("n_speakers", "min"),
            total_speakers=("n_speakers", "sum"),
        )
        .reset_index()
    )
    vocabulary = vocabulary.loc[vocabulary["n_accents"] >= min_accents].copy()
    keys = pd.MultiIndex.from_frame(vocabulary[["language", "normalized_word"]])
    row_keys = pd.MultiIndex.from_frame(filtered[["language", "normalized_word"]])
    return filtered.loc[row_keys.isin(keys)].copy(), vocabulary


def assign_speaker_splits(
    frame: pd.DataFrame,
    *,
    heldout_accent: str,
    seed: int,
    dev_speakers_per_accent: int,
) -> pd.DataFrame:
    """Hold one accent out for test and disjoint seen-accent speakers for dev."""

    assigned = frame.copy()
    assigned["split"] = "train"
    assigned.loc[assigned["accent"] == heldout_accent, "split"] = "test"
    for accent in sorted(set(assigned["accent"]) - {heldout_accent}):
        speakers = sorted(assigned.loc[assigned["accent"] == accent, "speaker_id"].unique())
        if len(speakers) <= dev_speakers_per_accent:
            raise ValueError(
                f"Accent {accent!r} has {len(speakers)} speakers; cannot reserve "
                f"{dev_speakers_per_accent} for dev and retain train speakers."
            )
        generator = random.Random(f"{seed}:{accent}")
        generator.shuffle(speakers)
        dev_speakers = set(speakers[:dev_speakers_per_accent])
        assigned.loc[
            (assigned["accent"] == accent) & assigned["speaker_id"].isin(dev_speakers),
            "split",
        ] = "dev"

    # Parallel corpora such as L2-ARCTIC expose a shared prompt identifier.
    # Retain only disjoint prompt partitions as well as disjoint speakers. This
    # prevents a complete sentence from crossing splits while intentionally
    # allowing the target lexical vocabulary to be shared.
    if "prompt_id" in assigned.columns and assigned["prompt_id"].notna().any():
        prompts = sorted(assigned["prompt_id"].dropna().unique())
        generator = random.Random(f"{seed}:prompts")
        generator.shuffle(prompts)
        n_prompts = len(prompts)
        n_dev = max(1, round(n_prompts * 0.1))
        n_test = max(1, round(n_prompts * 0.1))
        test_prompts = set(prompts[:n_test])
        dev_prompts = set(prompts[n_test:n_test + n_dev])
        train_prompts = set(prompts[n_test + n_dev:])
        keep = (
            ((assigned["accent"] == heldout_accent) & assigned["prompt_id"].isin(test_prompts))
            | ((assigned["accent"] != heldout_accent) & (assigned["split"] == "dev")
               & assigned["prompt_id"].isin(dev_prompts))
            | ((assigned["accent"] != heldout_accent) & (assigned["split"] == "train")
               & assigned["prompt_id"].isin(train_prompts))
        )
        assigned = assigned.loc[keep].copy()
    return assigned


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def build_dataset(
    occurrences: pd.DataFrame,
    output_dir: Path,
    *,
    min_accents: int,
    min_speakers_per_accent: int,
    min_duration_s: float,
    max_duration_s: float,
    dev_speakers_per_accent: int,
    seed: int,
    heldout_accents: Iterable[str] | None = None,
) -> None:
    validate_occurrences(occurrences)
    output_dir.mkdir(parents=True, exist_ok=True)
    occurrences = occurrences.sort_values("occurrence_id").reset_index(drop=True)
    occurrences.to_parquet(output_dir / "inventory.parquet", index=False)
    selected, vocabulary = eligible_words(
        occurrences,
        min_accents=min_accents,
        min_speakers_per_accent=min_speakers_per_accent,
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
    )
    if selected.empty:
        raise ValueError(
            "No word satisfies the requested coverage. Lower --min-accents or "
            "--min-speakers-per-accent after inspecting inventory.parquet."
        )
    vocabulary.sort_values(["language", "normalized_word"]).to_csv(
        output_dir / "vocabulary.csv", index=False
    )
    accent_counts = selected.groupby("accent").agg(
        occurrences=("occurrence_id", "size"),
        speakers=("speaker_id", "nunique"),
        words=("normalized_word", "nunique"),
    ).reset_index()
    accent_counts.to_csv(output_dir / "accent_stats.csv", index=False)
    accents = sorted(set(selected["accent"]))
    folds = list(heldout_accents) if heldout_accents is not None else accents
    unknown = set(folds) - set(accents)
    if unknown:
        raise ValueError(f"Unknown held-out accents: {sorted(unknown)}")
    summary: list[dict[str, Any]] = []
    for accent in folds:
        # Split before selecting the fold vocabulary. Eligibility is learned
        # exclusively from train rows, so held-out accent/dev metadata cannot
        # influence which lexical classes are retained.
        candidate_fold = assign_speaker_splits(
            occurrences.loc[
                occurrences["duration_s"].between(
                    min_duration_s, max_duration_s, inclusive="both"
                )
            ],
            heldout_accent=accent,
            seed=seed,
            dev_speakers_per_accent=dev_speakers_per_accent,
        )
        train_rows = candidate_fold.loc[candidate_fold["split"] == "train"]
        n_train_accents = train_rows["accent"].nunique()
        _, fold_vocabulary = eligible_words(
            train_rows,
            min_accents=min(min_accents, n_train_accents),
            min_speakers_per_accent=min_speakers_per_accent,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
        )
        fold_keys = pd.MultiIndex.from_frame(
            fold_vocabulary[["language", "normalized_word"]]
        )
        candidate_keys = pd.MultiIndex.from_frame(
            candidate_fold[["language", "normalized_word"]]
        )
        folded = candidate_fold.loc[candidate_keys.isin(fold_keys)].copy()
        fold_dir = output_dir / re.sub(r"[^a-z0-9]+", "_", accent.casefold()).strip("_")
        fold_dir.mkdir(parents=True, exist_ok=True)
        folded.to_parquet(fold_dir / "corpus.parquet", index=False)
        fold_vocabulary.to_csv(fold_dir / "vocabulary.csv", index=False)
        split_stats = folded.groupby("split").agg(
            occurrences=("occurrence_id", "size"),
            speakers=("speaker_id", "nunique"),
            accents=("accent", "nunique"),
            words=("normalized_word", "nunique"),
            duration_h=("duration_s", lambda values: round(values.sum() / 3600, 4)),
        ).reset_index()
        split_stats.to_csv(fold_dir / "split_stats.csv", index=False)
        speaker_sets = {
            split: set(folded.loc[folded["split"] == split, "speaker_id"])
            for split in ("train", "dev", "test")
        }
        overlap = {
            "train_dev": len(speaker_sets["train"] & speaker_sets["dev"]),
            "train_test": len(speaker_sets["train"] & speaker_sets["test"]),
            "dev_test": len(speaker_sets["dev"] & speaker_sets["test"]),
        }
        prompt_overlap: dict[str, int] = {}
        if "prompt_id" in folded.columns:
            prompt_sets = {
                split: set(folded.loc[folded["split"] == split, "prompt_id"])
                for split in ("train", "dev", "test")
            }
            prompt_overlap = {
                "train_dev": len(prompt_sets["train"] & prompt_sets["dev"]),
                "train_test": len(prompt_sets["train"] & prompt_sets["test"]),
                "dev_test": len(prompt_sets["dev"] & prompt_sets["test"]),
            }
        report = {
            "schema_version": SCHEMA_VERSION,
            "heldout_accent": accent,
            "seed": seed,
            "speaker_overlap": overlap,
            "prompt_overlap": prompt_overlap,
            "status": "passed" if not any((*overlap.values(), *prompt_overlap.values())) else "failed",
        }
        _json(fold_dir / "validation_report.json", report)
        if report["status"] != "passed":
            raise ValueError(f"Split leakage in fold {accent}: speakers={overlap}, prompts={prompt_overlap}")
        summary.append({"heldout_accent": accent, "rows": len(folded), "directory": fold_dir.name})
    pd.DataFrame(summary).to_csv(output_dir / "fold_summary.csv", index=False)
    _json(
        output_dir / "inventory_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "datasets": sorted(selected["dataset"].unique()),
            "languages": sorted(selected["language"].unique()),
            "raw_occurrences": len(occurrences),
            "selected_occurrences": len(selected),
            "selected_words": len(vocabulary),
            "accents": accents,
            "filters": {
                "min_accents": min_accents,
                "min_speakers_per_accent": min_speakers_per_accent,
                "min_duration_s": min_duration_s,
                "max_duration_s": max_duration_s,
            },
        },
    )
    (output_dir / "_SUCCESS").write_text("word contrastive dataset ready\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=("l2_arctic", "common_voice"))
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--language", default="en", help="BCP-47/ISO language code")
    parser.add_argument("--common-voice-tsv", default="validated.tsv")
    parser.add_argument("--min-accents", type=int, default=6)
    parser.add_argument("--min-speakers-per-accent", type=int, default=3)
    parser.add_argument("--dev-speakers-per-accent", type=int, default=1)
    parser.add_argument("--min-duration-s", type=float, default=0.12)
    parser.add_argument("--max-duration-s", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--heldout-accents", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "l2_arctic":
        occurrences = collect_l2_arctic(args.raw_dir, args.repository_root)
    else:
        occurrences = collect_common_voice(
            args.raw_dir,
            args.repository_root,
            language=args.language,
            tsv_name=args.common_voice_tsv,
        )
    build_dataset(
        occurrences,
        args.output_dir,
        min_accents=args.min_accents,
        min_speakers_per_accent=args.min_speakers_per_accent,
        min_duration_s=args.min_duration_s,
        max_duration_s=args.max_duration_s,
        dev_speakers_per_accent=args.dev_speakers_per_accent,
        seed=args.seed,
        heldout_accents=args.heldout_accents,
    )
    print(f"Prepared {len(occurrences):,} raw word occurrences in {args.output_dir}")


if __name__ == "__main__":
    main()
