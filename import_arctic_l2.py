"""
import_l2_arctic.py

Converts the L2-ARCTIC corpus into a single Parquet file,
with the same schema as the native ARCTIC parquet.

L2-ARCTIC structure expected:
    <corpus_dir>/
        <speaker_id>/               e.g. ABA, ZHAA, BWC ...
            wav/
                arctic_a0001.wav
                ...
            transcript/
                arctic_a0001.txt    ← one file per utterance, contains raw transcript

Output schema (identical to import_arctic.py):
    speaker_id | gender | utterance_id | transcript | audio_path | duration_s

Usage:
    python import_l2_arctic.py \
        --corpus_dir  /path/to/l2_arctic \
        --output_parquet /path/to/output/l2_arctic.parquet \
        --audio_dir   /path/to/output/wavs
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from utils import normalize_transcript


# ──────────────────────────────────────────────
# Speaker metadata
# Source: L2-ARCTIC paper (Zhao et al., 2018)
# 24 speakers, 4 per L1 background
# ──────────────────────────────────────────────

SPEAKER_META = {
    # Arabic L1
    "ABA":  "male",
    "YBAA": "female",
    "ZHAA": "female",
    "SKA":  "male",
    # Mandarin L1
    "BWC":  "male",
    "LXC":  "female",
    "NCC":  "female",
    "TXHC": "male",
    # Hindi L1
    "ASI":  "male",
    "RRBI": "male",
    "SVBI": "female",
    "TNI":  "female",
    # Korean L1
    "HJK":  "female",
    "HKK":  "male",
    "YDCK": "female",
    "YKWK": "male",
    # Spanish L1
    "EBVS": "male",
    "ERMS": "male",
    "MBMPS":"female",
    "NJS":  "female",
    # Vietnamese L1
    "PNV":  "female",
    "THV":  "female",
    "TLV":  "male",
    "HQTV": "male",
}


# ──────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────

def read_transcript_file(path: Path) -> str:
    """Read and normalize a single .txt transcript file."""
    return normalize_transcript(path.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────
# Build dataframe
# ──────────────────────────────────────────────

def build_l2_arctic_dataframe(corpus_dir: Path, audio_out_dir: Path) -> pd.DataFrame:
    rows = []

    speaker_dirs = sorted([d for d in corpus_dir.iterdir() if d.is_dir()])
    if not speaker_dirs:
        raise ValueError(f"No speaker directories found in {corpus_dir}")

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name

        wav_dir        = speaker_dir / "wav"
        transcript_dir = speaker_dir / "transcript"

        if not wav_dir.exists():
            print(f"  [WARN] No wav/ for {speaker_id}, skipping.")
            continue
        if not transcript_dir.exists():
            print(f"  [WARN] No transcript/ for {speaker_id}, skipping.")
            continue

        gender = SPEAKER_META.get(speaker_id, "unknown")
        if gender == "unknown":
            print(f"  [WARN] Unknown speaker {speaker_id}, gender set to 'unknown'.")

        n_matched = 0
        for wav_file in sorted(wav_dir.glob("*.wav")):
            utt_id = wav_file.stem  # e.g. arctic_a0001

            txt_file = transcript_dir / f"{utt_id}.txt"
            if not txt_file.exists():
                print(f"  [WARN] No transcript for {speaker_id}/{utt_id}, skipping.")
                continue

            transcript = read_transcript_file(txt_file)

            dest_name = f"{speaker_id}_{utt_id}.wav"
            dest_path = audio_out_dir / dest_name
            shutil.copy2(wav_file, dest_path)

            rows.append({
                "speaker_id":   speaker_id,
                "gender":       gender,
                "utterance_id": utt_id,
                "transcript":   transcript,
                "audio_path":   str(dest_path),
                "duration_s":   None,
            })
            n_matched += 1

        print(f"  [{speaker_id}] ({gender}) {n_matched} wavs matched.")

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Import L2-ARCTIC corpus to Parquet.")
    parser.add_argument("--corpus_dir",     required=True, help="Root of the L2-ARCTIC corpus")
    parser.add_argument("--output_parquet", required=True, help="Destination .parquet file")
    parser.add_argument("--audio_dir",      required=True, help="Directory to copy WAV files into")
    args = parser.parse_args()

    corpus_dir   = Path(args.corpus_dir)
    parquet_path = Path(args.output_parquet)
    audio_dir    = Path(args.audio_dir)

    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    audio_dir.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning speakers in: {corpus_dir}")
    df = build_l2_arctic_dataframe(corpus_dir, audio_dir)

    print(f"\nTotal rows : {len(df)}")
    print(df.head())

    df.to_parquet(parquet_path, index=False)
    print(f"\nSaved → {parquet_path}")


if __name__ == "__main__":
    main()