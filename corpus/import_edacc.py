"""
import_edacc.py

Import the Edinburgh International Accentf of English Corpus (EdAcc)
into the project format.

Reads from a local HuggingFace cache directory (pre-downloaded).

Parquet schema
--------------
speaker_id      str    e.g. "EDACC-C06-A"
gender          str    "male" | "female"
split           str    "eval" | "test"
accent          str    standardised accent label (e.g. "Southern British English")
raw_accent      str    self-reported accent (free-form)
l1              str    speaker's first language
transcript      str    normalised text
audio_path      str    absolute path to the converted WAV file (16kHz)

Usage
-----
    # validation split → eval
    python corpus/import_edacc.py \\
        --cache_dir      data/raw/edacc \\
        --output_parquet data/processed/edacc/corpus.parquet \\
        --audio_dir      data/processed/edacc/wavs \\
        --split          validation

    # test split
    python corpus/import_edacc.py \\
        --cache_dir      data/raw/edacc \\
        --output_parquet data/processed/edacc_test/corpus.parquet \\
        --audio_dir      data/processed/edacc_test/wavs \\
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

HF_DATASET = "edinburghcstr/edacc"
TARGET_SR  = 16_000


def import_corpus(
    cache_dir:      str,
    output_parquet: Path,
    audio_dir:      Path,
    split:          str = "validation",
) -> pd.DataFrame:
    output_parquet = Path(output_parquet)
    audio_dir      = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading EdAcc split='{split}' from local cache: {cache_dir}")
    ds = load_dataset(
        HF_DATASET,
        split=split,
        cache_dir=cache_dir,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    logger.info(f"  {len(ds):,} examples loaded")

    rows: list[dict] = []

    for i, example in enumerate(tqdm(ds, desc="converting")):
        speaker_id  = example["speaker"]
        audio_arr   = example["audio"]["array"].astype(np.float32)
        sr          = example["audio"]["sampling_rate"]

        safe_speaker = speaker_id.replace("/", "_")
        wav_name = f"{safe_speaker}_{i:06d}.wav"
        wav_path = audio_dir / wav_name

        if not wav_path.exists():
            sf.write(wav_path, audio_arr, sr)

        rows.append({
            "speaker_id": speaker_id,
            "gender":     example["gender"],
            "split":      "eval" if split == "validation" else split,
            "accent":     example["accent"],
            "raw_accent": example["raw_accent"],
            "l1":         example["l1"],
            "transcript": normalize_transcript(example["text"]),
            "audio_path": str(wav_path.resolve()),
        })

    df = pd.DataFrame(rows)

    logger.info(f"Total utterances : {len(df):,}")
    logger.info(f"Speakers         : {df['speaker_id'].nunique()}")
    logger.info(f"Accents          : {df['accent'].nunique()}")
    logger.info(f"\n{df['accent'].value_counts().head(10).to_string()}")

    df.to_parquet(output_parquet, index=False)
    logger.info(f"Parquet → {output_parquet}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import EdAcc from local HF cache into the project format."
    )
    parser.add_argument("--cache_dir",      type=str,  required=True,
                        help="Local HuggingFace cache dir (e.g. data/raw/edacc).")
    parser.add_argument("--output_parquet", type=Path, required=True)
    parser.add_argument("--audio_dir",      type=Path, required=True)
    parser.add_argument("--split",          type=str,  default="validation",
                        choices=["validation", "test"])
    args = parser.parse_args()

    import_corpus(
        cache_dir      = args.cache_dir,
        output_parquet = args.output_parquet,
        audio_dir      = args.audio_dir,
        split          = args.split,
    )