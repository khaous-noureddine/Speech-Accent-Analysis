"""
Import the George Mason University Speech Accent Archive into the project format.

One row per speaker (utterance-level — no phoneme alignment available).
All speakers read the same English elicitation passage.

Outputs
-------
- {output_parquet}   : parquet, one row per speaker
- {output_csv}       : CSV mirror for human inspection  (optional)
- {audio_dir}/       : WAV files converted from the original MP3s

Parquet schema
--------------
audio_path        str    path to the converted WAV file
speakerid         int    unique speaker id (from speakers_all.csv)
filename          str    original MP3 stem (e.g. "english3")
native_language   str    speaker's native language
sex               str    "male" | "female"
age               int    speaker's age at recording
age_onset         int    age at which English learning began
birthplace        str    free-text birthplace
country           str    ISO country code

Usage
-----
    python corpus/speech_accents/import_speech_accents.py \\
        --corpus_dir  speech_accents \\
        --output_parquet speech_accents/speech_accents_corpus/corpus.parquet \\
        --audio_dir   speech_accents/speech_accents_corpus/wavs
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from loguru import logger
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

_CSV_COLS = [
    "age", "age_onset", "birthplace", "filename",
    "native_language", "sex", "speakerid", "country", "file_missing?",
]


def load_metadata(corpus_dir: Path) -> pd.DataFrame:
    csv_path = corpus_dir / "speakers_all.csv"
    df = pd.read_csv(csv_path, usecols=lambda c: c in _CSV_COLS)

    # Normalise missing-file flag (can be bool or string "TRUE"/"FALSE")
    missing_col = "file_missing?"
    if missing_col in df.columns:
        df[missing_col] = (
            df[missing_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({"TRUE": True, "FALSE": False, "NAN": False})
            .fillna(False)
        )
        df = df[~df[missing_col]].drop(columns=[missing_col])
    else:
        logger.warning("Column '{}' not found — keeping all rows.", missing_col)

    df = df.reset_index(drop=True)
    logger.info("Metadata: {} speakers after filtering missing files.", len(df))
    return df


# ---------------------------------------------------------------------------
# MP3 → WAV conversion
# ---------------------------------------------------------------------------

def convert_mp3_to_wav(mp3_path: Path, wav_path: Path, sr: int = 16_000) -> bool:
    """Convert *mp3_path* to 16 kHz mono WAV at *wav_path*. Returns False on error."""
    try:
        audio, orig_sr = librosa.load(mp3_path, sr=sr, mono=True)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav_path, audio, sr)
        return True
    except Exception as exc:
        logger.warning("Conversion failed for {}: {}", mp3_path.name, exc)
        return False


def import_corpus(
    corpus_dir: Path,
    output_parquet: Path,
    audio_dir: Path,
    output_csv: Path | None = None,
    recordings_subdir: str = "recordings/recordings",
    target_sr: int = 16_000,
) -> pd.DataFrame:
    corpus_dir = Path(corpus_dir)
    recordings_dir = corpus_dir / recordings_subdir

    with open(corpus_dir / "reading-passage.txt", "r") as f:
        transcript = f.read().strip()

    df = load_metadata(corpus_dir)

    audio_paths: list[str] = []
    keep: list[bool] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="converting MP3→WAV"):
        stem    = str(row["filename"])
        mp3     = recordings_dir / f"{stem}.mp3"
        wav_out = audio_dir / f"{stem}.wav"

        if not mp3.exists():
            logger.warning("MP3 not found: {}", mp3)
            audio_paths.append("")
            keep.append(False)
            continue

        if wav_out.exists():
            # already converted — skip re-encoding
            audio_paths.append(str(wav_out))
            keep.append(True)
            continue

        ok = convert_mp3_to_wav(mp3, wav_out, sr=target_sr)
        audio_paths.append(str(wav_out) if ok else "")
        keep.append(ok)

    df["audio_path"] = audio_paths
    df = df[keep].reset_index(drop=True)
    df["transcript"] = transcript
    logger.info("Corpus ready: {} speakers with valid audio.", len(df))



    # Canonical column order
    col_order = [
        "audio_path", "speakerid", "filename",
        "native_language", "sex", "age", "age_onset", "birthplace", "country", "transcript"
    ]
    df = df[[c for c in col_order if c in df.columns]]

    # Save
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    logger.info("Parquet → {}", output_parquet)

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info("CSV    → {}", output_csv)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import the Speech Accent Archive into the project format."
    )
    parser.add_argument(
        "--corpus_dir", type=Path, required=True,
        help="Root of the speech_accents directory "
             "(must contain speakers_all.csv and recordings/).",
    )
    parser.add_argument(
        "--output_parquet", type=Path, required=True,
        help="Output parquet path.",
    )
    parser.add_argument(
        "--audio_dir", type=Path, required=True,
        help="Directory where converted WAV files will be written.",
    )
    parser.add_argument(
        "--output_csv", type=Path, default=None,
        help="Optional CSV mirror of the parquet (for human inspection).",
    )
    parser.add_argument(
        "--recordings_subdir", default="recordings/recordings",
        help="Subdirectory inside corpus_dir that contains the MP3 files "
             "(default: recordings/recordings).",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16_000,
        help="Target sample rate for WAV conversion (default: 16000 Hz).",
    )
    args = parser.parse_args()

    import_corpus(
        corpus_dir       = args.corpus_dir,
        output_parquet   = args.output_parquet,
        audio_dir        = args.audio_dir,
        output_csv       = args.output_csv,
        recordings_subdir = args.recordings_subdir,
        target_sr        = args.sample_rate,
    )
