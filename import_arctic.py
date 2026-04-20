"""
import_arctic.py

Converts the CMU Arctic corpus into a single Parquet file.

Arctic corpus structure expected:
    <corpus_dir>/
        <speaker_id>/
            wav/
                arctic_a0001.wav
                arctic_a0002.wav
                ...
            etc/
                txt.done.data       ← transcription file

Usage:
    python import_arctic.py \
        --corpus_dir  /path/to/arctic \
        --output_parquet /path/to/output/arctic.parquet \
        --audio_dir   /path/to/output/wavs
"""

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd


# ──────────────────────────────────────────────
# Speaker metadata
# ──────────────────────────────────────────────

SPEAKER_META = {
    "awb": "male",
    "bdl": "male",
    "clb": "female",
    "jmk": "male",
    "rms": "male",
    "slt": "female",
}


# ──────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────

def parse_txt_done(path: Path) -> dict[str, str]:
    """
    Parse a txt.done.data file and return {utterance_id: transcript}.

    Each line looks like:
        ( arctic_a0001 "She had your dark eyes and this dark hair." )
    """
    transcripts = {}
    pattern = re.compile(r'\(\s*(\S+)\s+"(.+?)"\s*\)')
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line.strip())
        if m:
            transcripts[m.group(1)] = m.group(2)
    return transcripts


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def build_arctic_dataframe(corpus_dir: Path, audio_out_dir: Path) -> pd.DataFrame:
    rows = []

    speaker_dirs = sorted([d for d in corpus_dir.iterdir() if d.is_dir()])
    if not speaker_dirs:
        raise ValueError(f"No speaker directories found in {corpus_dir}")

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name
        # Extract short code from names like "cmu_us_awb_arctic" → "awb"
        # Falls back to full name if no match found
        parts = speaker_id.split("_")
        spk_code = next((p for p in parts if p in SPEAKER_META), speaker_id)

        txt_file = speaker_dir / "etc" / "txt.done.data"
        if not txt_file.exists():
            print(f"  [WARN] No txt.done.data for speaker {speaker_id} (code: {spk_code}), skipping.")
            continue

        transcripts = parse_txt_done(txt_file)

        wav_dir = speaker_dir / "wav"
        if not wav_dir.exists():
            print(f"  [WARN] No wav/ directory for speaker {speaker_id} (code: {spk_code}), skipping.")
            continue

        for wav_file in sorted(wav_dir.glob("*.wav")):
            utt_id = wav_file.stem  # e.g. arctic_a0001

            transcript = transcripts.get(utt_id)
            if transcript is None:
                print(f"  [WARN] No transcript for {speaker_id}/{utt_id}, skipping.")
                continue

            # Copy audio to flat output directory: <speaker_id>_<utt_id>.wav
            dest_name = f"{speaker_id}_{utt_id}.wav"
            dest_path = audio_out_dir / dest_name
            shutil.copy2(wav_file, dest_path)

            gender = SPEAKER_META.get(spk_code, "unknown")

            rows.append({
                "speaker_id":   speaker_id,
                "gender":       gender,
                "utterance_id": utt_id,
                "transcript":   transcript,
                "audio_path":   str(dest_path),
                "duration_s":   None,
            })

        print(f"  [{speaker_id}] {len(transcripts)} transcripts, "
              f"{sum(1 for r in rows if r['speaker_id'] == speaker_id)} wavs matched.")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Import CMU Arctic corpus to Parquet.")
    parser.add_argument("--corpus_dir",      required=True,  help="Root of the Arctic corpus")
    parser.add_argument("--output_parquet",  required=True,  help="Destination .parquet file")
    parser.add_argument("--audio_dir",       required=True,  help="Directory to copy WAV files into")
    args = parser.parse_args()

    corpus_dir   = Path(args.corpus_dir)
    parquet_path = Path(args.output_parquet)
    audio_dir    = Path(args.audio_dir)

    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    audio_dir.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning speakers in: {corpus_dir}")
    df = build_arctic_dataframe(corpus_dir, audio_dir)

    print(f"\nTotal rows: {len(df)}")
    print(df.head())

    df.to_parquet(parquet_path, index=False)
    print(f"\nSaved → {parquet_path}")


if __name__ == "__main__":
    main()