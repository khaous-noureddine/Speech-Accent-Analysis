"""
corpus/import_aesrc.py

Import the AESRC2020 (200 Hours - English Speech Data from 10 Countries)
into the project format.

Directory structure expected:
    {corpus_dir}/
        {country} English Speech Data/   (e.g. "British English Speech Data")
            {speaker_id}/                (e.g. "G00001")
                {speaker_id}S{utt}.wav
                {speaker_id}S{utt}.txt
                {speaker_id}S{utt}.metadata

Parquet schema
--------------
speaker_id    str    e.g. "G51624"
gender        str    "male" | "female"
age           int    speaker age
accent        str    country/accent from ACT field (e.g. "Spain")
country       str    clean country label (e.g. "Spanish")
device        str    recording device (MIT field)
environment   str    recording environment (SCC field)
duration_s    float  recording duration in seconds (LBR field)
utterance_id  str    e.g. "G51624S5454"
transcript    str    normalised text
audio_path    str    absolute path to WAV file
split         str    "train" | "dev" | "test"

Splits
------
train/dev:
    Speaker-level random split inside each seen accent/country.

test:
    Canadian and Spanish speakers.
    These are unseen accents and are reserved for final evaluation only.

Usage
-----
    # Full import with 10% dev speakers per seen accent
    python corpus/import_aesrc.py \\
        --corpus_dir     data/raw/aesrc/data \\
        --output_parquet data/processed/aesrc/corpus.parquet \\
        --audio_dir      data/processed/aesrc/wavs \\
        --output_csv     data/processed/aesrc/corpus.csv \\
        --dev_ratio      0.10 \\
        --seed           42

    # Quick test — 2 speakers per country
    python corpus/import_aesrc.py \\
        --corpus_dir     data/raw/aesrc/data \\
        --output_parquet data/processed/aesrc/corpus.parquet \\
        --audio_dir      data/processed/aesrc/wavs \\
        --max_speakers   2 \\
        --dev_ratio      0.10 \\
        --seed           42
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import normalize_transcript


# ── Country folder name → clean accent label ──────────────────────────────
COUNTRY_FOLDER_MAP = {
    "british english speech data":    "British",
    "american english speech data":   "American",
    "russian english speech data":    "Russian",
    "korean english speech data":     "Korean",
    "canadian english speech data":   "Canadian",
    "portuguese english speech data": "Portuguese",
    "japanese english speech data":   "Japanese",
    "spanish english speech data":    "Spanish",
    "india english speech data":      "Indian",
    "chinese english speech data":    "Chinese",
}

# ── Held-out accents for final test only ──────────────────────────────────
TEST_COUNTRIES = {"Canadian", "Spanish"}


def parse_metadata(path: Path) -> dict:
    """Parse a .metadata file into a key→value dict."""
    meta = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("CMT"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            meta[parts[0]] = parts[1].strip()
        elif len(parts) == 1:
            meta[parts[0]] = ""
    return meta


def make_speaker_split_map(
    country_label: str,
    speaker_dirs: list[Path],
    dev_ratio: float = 0.10,
    seed: int = 42,
) -> dict[str, str]:
    """
    Create speaker-level split mapping for one accent/country.

    Returns:
        {
            speaker_id: "train" | "dev" | "test"
        }

    Rules:
      - Canadian and Spanish are fully assigned to test.
      - Other countries are split into train/dev by speaker.
      - The split is speaker-level, not utterance-level.
      - The seed controls the random train/dev assignment.
    """
    if not 0.0 <= dev_ratio < 1.0:
        raise ValueError(f"dev_ratio must be in [0.0, 1.0), got {dev_ratio}")

    speaker_ids = [d.name for d in speaker_dirs]
    n_total = len(speaker_ids)

    if n_total == 0:
        logger.warning(f"  [{country_label}] no speakers found.")
        return {}

    if country_label in TEST_COUNTRIES:
        logger.info(
            f"  [{country_label}] speakers: total={n_total}, "
            f"train=0, dev=0, test={n_total}"
        )
        return {speaker_id: "test" for speaker_id in speaker_ids}

    rng = random.Random(seed)
    shuffled = speaker_ids.copy()
    rng.shuffle(shuffled)

    n_dev = int(round(n_total * dev_ratio))

    # Safety for small subsets:
    # if there are at least 2 speakers and dev_ratio > 0,
    # keep at least 1 dev speaker and at least 1 train speaker.
    if n_total >= 2 and dev_ratio > 0.0:
        n_dev = max(1, n_dev)
        n_dev = min(n_dev, n_total - 1)
    else:
        n_dev = 0

    dev_speakers = set(shuffled[:n_dev])

    split_map = {
        speaker_id: "dev" if speaker_id in dev_speakers else "train"
        for speaker_id in speaker_ids
    }

    n_train = sum(1 for split in split_map.values() if split == "train")
    n_dev_actual = sum(1 for split in split_map.values() if split == "dev")

    logger.info(
        f"  [{country_label}] speakers: total={n_total}, "
        f"train={n_train}, dev={n_dev_actual}, test=0 "
        f"(dev_ratio={dev_ratio}, seed={seed})"
    )

    return split_map


def import_corpus(
    corpus_dir:     Path,
    output_parquet: Path,
    audio_dir:      Path,
    output_csv:     Path | None = None,
    max_speakers:   int | None  = None,
    dev_ratio:      float       = 0.10,
    seed:           int         = 42,
) -> pd.DataFrame:
    corpus_dir     = Path(corpus_dir)
    output_parquet = Path(output_parquet)
    audio_dir      = Path(audio_dir)

    audio_dir.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped = 0

    # ── Walk country folders ───────────────────────────────────────────────
    country_dirs = sorted([d for d in corpus_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(country_dirs)} country folder(s) in {corpus_dir}")
    logger.info(f"Speaker split config: dev_ratio={dev_ratio}, seed={seed}")
    logger.info(f"Test countries: {sorted(TEST_COUNTRIES)}")

    for country_dir in country_dirs:
        country_key   = country_dir.name.lower().strip()
        country_label = COUNTRY_FOLDER_MAP.get(country_key, country_dir.name)
        speaker_dirs  = sorted([d for d in country_dir.iterdir() if d.is_dir()])

        if max_speakers is not None:
            speaker_dirs = speaker_dirs[:max_speakers]

        split_map = make_speaker_split_map(
            country_label=country_label,
            speaker_dirs=speaker_dirs,
            dev_ratio=dev_ratio,
            seed=seed,
        )

        for speaker_dir in tqdm(speaker_dirs, desc=f"{country_label}"):
            speaker_id = speaker_dir.name

            if speaker_id not in split_map:
                logger.warning(f"No split assigned for speaker {speaker_id} — skipping")
                skipped += 1
                continue

            split = split_map[speaker_id]
            wav_files = sorted(speaker_dir.glob("*.wav"))

            for wav_path in wav_files:
                utt_id    = wav_path.stem
                txt_path  = wav_path.with_suffix(".txt")
                meta_path = wav_path.with_suffix(".metadata")

                # ── Transcript ────────────────────────────────────────────
                if not txt_path.exists():
                    logger.warning(f"No transcript: {txt_path} — skipping")
                    skipped += 1
                    continue

                transcript = normalize_transcript(
                    txt_path.read_text(encoding="utf-8", errors="replace").strip()
                )

                # ── Metadata ──────────────────────────────────────────────
                meta = {}
                if meta_path.exists():
                    meta = parse_metadata(meta_path)

                gender       = meta.get("SEX", "").strip().lower()
                age_str      = meta.get("AGE", "").strip()
                accent       = meta.get("ACT", country_label).strip() or country_label
                device       = meta.get("MIT", "").strip()
                environment  = meta.get("SCC", "").strip()
                duration_str = meta.get("LBR", "").strip()

                try:
                    age = int(age_str)
                except (ValueError, TypeError):
                    age = None

                try:
                    duration_s = round(float(duration_str), 3)
                except (ValueError, TypeError):
                    try:
                        duration_s = round(sf.info(str(wav_path)).duration, 3)
                    except Exception:
                        duration_s = None

                # ── Copy WAV ──────────────────────────────────────────────
                dest_path = audio_dir / f"{utt_id}.wav"
                if not dest_path.exists():
                    shutil.copy2(wav_path, dest_path)

                rows.append({
                    "speaker_id":   speaker_id,
                    "gender":       gender,
                    "age":          age,
                    "accent":       accent,
                    "country":      country_label,
                    "device":       device,
                    "environment":  environment,
                    "duration_s":   duration_s,
                    "utterance_id": utt_id,
                    "transcript":   transcript,
                    "audio_path":   str(dest_path.resolve()),
                    "split":        split,
                })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError("No utterances were imported. Check corpus_dir and file layout.")

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info(f"Total utterances : {len(df):,}")
    logger.info(f"Skipped          : {skipped:,}")
    logger.info(f"Speakers         : {df['speaker_id'].nunique()}")
    logger.info(f"Countries        : {df['country'].nunique()}")

    if "duration_s" in df.columns:
        logger.info(f"Duration total   : {df['duration_s'].sum() / 3600:.1f} h")

    logger.info(
        f"\nUtterances per country + split:\n"
        f"{df.groupby(['country', 'split']).size().to_string()}"
    )

    logger.info(
        f"\nUtterances per split:\n"
        f"{df.groupby('split').size().to_string()}"
    )

    # Speaker-level summary
    speaker_summary = (
        df[["country", "speaker_id", "split"]]
        .drop_duplicates()
        .groupby(["country", "split"])
        .size()
        .unstack(fill_value=0)
    )

    for col in ["train", "dev", "test"]:
        if col not in speaker_summary.columns:
            speaker_summary[col] = 0

    speaker_summary = speaker_summary[["train", "dev", "test"]]
    speaker_summary["total"] = speaker_summary.sum(axis=1)

    logger.info(
        f"\nSpeakers per country + split:\n"
        f"{speaker_summary.to_string()}"
    )

    split_speaker_summary = (
        df[["speaker_id", "split"]]
        .drop_duplicates()
        .groupby("split")
        .size()
    )

    logger.info(
        f"\nSpeakers per split:\n"
        f"{split_speaker_summary.to_string()}"
    )

    # ── Save ───────────────────────────────────────────────────────────────
    df.to_parquet(output_parquet, index=False)
    logger.info(f"Parquet → {output_parquet}")

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"CSV    → {output_csv}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import AESRC2020 into the project format."
    )
    parser.add_argument("--corpus_dir",     type=Path, required=True)
    parser.add_argument("--output_parquet", type=Path, required=True)
    parser.add_argument("--audio_dir",      type=Path, required=True)
    parser.add_argument("--output_csv",     type=Path, default=None)
    parser.add_argument(
        "--max_speakers",
        type=int,
        default=None,
        help="Max speakers per country — for quick testing only.",
    )
    parser.add_argument(
        "--dev_ratio",
        type=float,
        default=0.10,
        help="Ratio of speakers per seen accent used for dev.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for speaker-level train/dev split.",
    )

    args = parser.parse_args()

    import_corpus(
        corpus_dir     = args.corpus_dir,
        output_parquet = args.output_parquet,
        audio_dir      = args.audio_dir,
        output_csv     = args.output_csv,
        max_speakers   = args.max_speakers,
        dev_ratio      = args.dev_ratio,
        seed           = args.seed,
    )