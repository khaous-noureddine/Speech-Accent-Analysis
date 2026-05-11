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
country       str    same as accent (ACT field)
device        str    recording device (MIT field)
environment   str    recording environment (SCC field)
duration_s    float  recording duration in seconds (LBR field)
utterance_id  str    e.g. "G51624S5454"
transcript    str    normalised text
audio_path    str    absolute path to WAV file

Usage
-----
    python corpus/import_aesrc.py \\
        --corpus_dir  data/raw/aesrc/data \\
        --output_parquet data/processed/aesrc/corpus.parquet \\
        --audio_dir      data/processed/aesrc/wavs
"""

from __future__ import annotations

import argparse
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


def import_corpus(
    corpus_dir:     Path,
    output_parquet: Path,
    audio_dir:      Path,
    output_csv:     Path | None = None,
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

    for country_dir in country_dirs:
        country_key    = country_dir.name.lower().strip()
        country_label  = COUNTRY_FOLDER_MAP.get(country_key, country_dir.name)
        speaker_dirs   = sorted([d for d in country_dir.iterdir() if d.is_dir()])
        logger.info(f"  [{country_label}] {len(speaker_dirs)} speakers")

        for speaker_dir in tqdm(speaker_dirs, desc=f"{country_label}"):
            speaker_id = speaker_dir.name

            # Collect all WAVs in this speaker folder
            wav_files = sorted(speaker_dir.glob("*.wav"))

            for wav_path in wav_files:
                utt_id   = wav_path.stem  # e.g. G51624S5454
                txt_path = wav_path.with_suffix(".txt")
                meta_path = wav_path.with_suffix(".metadata")

                # Transcript
                if not txt_path.exists():
                    logger.warning(f"No transcript: {txt_path} — skipping")
                    skipped += 1
                    continue

                transcript = normalize_transcript(
                    txt_path.read_text(encoding="utf-8", errors="replace").strip()
                )

                # Metadata
                meta = {}
                if meta_path.exists():
                    meta = parse_metadata(meta_path)

                gender      = meta.get("SEX", "").strip().lower()
                age_str     = meta.get("AGE", "").strip()
                accent      = meta.get("ACT", country_label).strip() or country_label
                device      = meta.get("MIT", "").strip()
                environment = meta.get("SCC", "").strip()
                duration_str = meta.get("LBR", "").strip()

                try:
                    age = int(age_str)
                except ValueError:
                    age = None

                try:
                    duration_s = round(float(duration_str), 3)
                except ValueError:
                    # Fall back to reading from file
                    try:
                        duration_s = round(sf.info(str(wav_path)).duration, 3)
                    except Exception:
                        duration_s = None

                # Copy WAV to output dir
                dest_name = f"{utt_id}.wav"
                dest_path = audio_dir / dest_name
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
                })

    df = pd.DataFrame(rows)

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info(f"Total utterances : {len(df):,}")
    logger.info(f"Skipped          : {skipped:,}")
    logger.info(f"Speakers         : {df['speaker_id'].nunique()}")
    logger.info(f"Countries        : {df['country'].nunique()}")
    logger.info(f"Duration total   : {df['duration_s'].sum() / 3600:.1f} h")
    logger.info(f"\nPer country:\n{df.groupby('country').size().to_string()}")

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
    parser.add_argument("--corpus_dir",      type=Path, required=True,
                        help="Root of the AESRC data folder (contains country subfolders).")
    parser.add_argument("--output_parquet",  type=Path, required=True)
    parser.add_argument("--audio_dir",       type=Path, required=True)
    parser.add_argument("--output_csv",      type=Path, default=None)
    args = parser.parse_args()

    import_corpus(
        corpus_dir     = args.corpus_dir,
        output_parquet = args.output_parquet,
        audio_dir      = args.audio_dir,
        output_csv     = args.output_csv,
    )