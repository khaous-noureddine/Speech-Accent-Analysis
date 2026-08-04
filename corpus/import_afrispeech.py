"""
corpus/import_afrispeech.py

Import AfriSpeech-200 from a local HuggingFace cache into the project format.

Parquet schema
--------------
speaker_id   str    unique speaker identifier
gender       str    "male" | "female"
split        str    "train" | "dev" | "test"
accent       str    accent label (e.g. "Nigerian English")
country      str    speaker's country
domain       str    "clinical" | "general"
transcript   str    normalised text
audio_path   str    absolute path to the converted WAV file (16kHz)
duration_s   float  audio duration in seconds

Usage
-----
    python corpus/import_afrispeech.py \\
        --cache_dir  data/raw/afrispeech \\
        --output_parquet data/processed/afrispeech_train/corpus.parquet \\
        --audio_dir      data/processed/afrispeech_train/wavs \\
        --split          train

    python corpus/import_afrispeech.py \\
        --cache_dir  data/raw/afrispeech \\
        --output_parquet data/processed/afrispeech_dev/corpus.parquet \\
        --audio_dir      data/processed/afrispeech_dev/wavs \\
        --split          dev

    python corpus/import_afrispeech.py \\
        --cache_dir  data/raw/afrispeech \\
        --output_parquet data/processed/afrispeech_test/corpus.parquet \\
        --audio_dir      data/processed/afrispeech_test/wavs \\
        --split          test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import normalize_transcript

HF_DATASET = "tobiolatunji/afrispeech-200"
TARGET_SR  = 16_000


def import_corpus(
    cache_dir:      str,
    output_parquet: Path,
    audio_dir:      Path,
    split:          str = "train",
) -> pd.DataFrame:
    output_parquet = Path(output_parquet)
    audio_dir      = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading AfriSpeech-200 split='{split}' from cache: {cache_dir}")
    ds = load_dataset(
        HF_DATASET,
        "all",           # all accents combined
        split=split,
        cache_dir=cache_dir,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    logger.info(f"  {len(ds):,} examples loaded")

    rows: list[dict] = []

    for i, example in enumerate(tqdm(ds, desc=f"converting [{split}]")):
        audio_id   = example.get("audio_id", f"{split}_{i:06d}")
        audio_arr  = example["audio"]["array"].astype(np.float32)
        sr         = example["audio"]["sampling_rate"]

        safe_id  = str(audio_id).replace("/", "_")
        wav_name = f"{safe_id}.wav"
        wav_path = audio_dir / wav_name

        if not wav_path.exists():
            sf.write(wav_path, audio_arr, sr)

        rows.append({
            "speaker_id":  example.get("speaker_id", ""),
            "gender":      example.get("gender",     ""),
            "split":       split,
            "accent":      example.get("accent",     ""),
            "country":     example.get("country",    ""),
            "domain":      example.get("domain",     ""),
            "transcript":  normalize_transcript(example.get("transcript", "") or ""),
            "audio_path":  str(wav_path.resolve()),
            "duration_s":  round(example.get("duration", 0.0), 3),
        })

    df = pd.DataFrame(rows)

    logger.info(f"Total utterances : {len(df):,}")
    logger.info(f"Speakers         : {df['speaker_id'].nunique()}")
    logger.info(f"Accents          : {df['accent'].nunique()}")
    logger.info(f"Countries        : {df['country'].nunique()}")
    logger.info(f"\nTop accents:\n{df['accent'].value_counts().head(10).to_string()}")

    df.to_parquet(output_parquet, index=False)
    logger.info(f"Parquet → {output_parquet}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import AfriSpeech-200 from local HF cache into project format."
    )
    parser.add_argument("--cache_dir",      type=str,  required=True)
    parser.add_argument("--output_parquet", type=Path, required=True)
    parser.add_argument("--audio_dir",      type=Path, required=True)
    parser.add_argument("--split",          type=str,  default="train",
                        choices=["train", "dev", "test"])
    args = parser.parse_args()

    import_corpus(
        cache_dir      = args.cache_dir,
        output_parquet = args.output_parquet,
        audio_dir      = args.audio_dir,
        split          = args.split,
    )