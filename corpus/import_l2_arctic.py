"""
import_l2_arctic.py

Usage:
python import_l2_arctic.py \
  --corpus_dir data/raw/l2_arctic/speakers \
  --output_parquet data/processed/l2_arctic/corpus.parquet \
  --audio_dir data/processed/l2_arctic/wavs \
  --speakers ABA ASI BWC EBVS ERMS HJK HKK HQTV LXC MBMPS NCC NJS PNV RRBI SKA SVBI THV TLV TNI TXHC YBAA YDCK YKWK ZHAA \
  --eval_speakers SKA EBVS BWC
"""

import sys
import argparse
import shutil
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import normalize_transcript


SPEAKER_META = {
    # Arabic L1
    "ABA":  "male",
    "YBAA": "male",
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
    "MBMPS": "female",
    "NJS":  "female",

    # Vietnamese L1
    "PNV":  "female",
    "THV":  "female",
    "TLV":  "male",
    "HQTV": "male",
}


def read_transcript_file(path: Path) -> str:
    return normalize_transcript(path.read_text(encoding="utf-8"))

    

def validate_speakers(
    requested_speakers: list[str],
    eval_speakers: list[str],
    existing_speakers: set[str],
):
    requested_speakers = set(requested_speakers)
    eval_speakers = set(eval_speakers)

    unknown_requested = requested_speakers - existing_speakers
    unknown_eval = eval_speakers - existing_speakers
    eval_not_selected = eval_speakers - requested_speakers

    if unknown_requested:
        raise ValueError(
            f"Requested speakers not found in corpus: {sorted(unknown_requested)}"
        )

    if unknown_eval:
        raise ValueError(
            f"Eval speakers not found in corpus: {sorted(unknown_eval)}"
        )

    if eval_not_selected:
        raise ValueError(
            f"Eval speakers must also be in --speakers: {sorted(eval_not_selected)}"
        )


def build_l2_arctic_dataframe(
    corpus_dir: Path,
    audio_out_dir: Path,
    speakers: list[str],
    eval_speakers: list[str],
) -> pd.DataFrame:
    rows = []

    speaker_dirs = sorted([d for d in corpus_dir.iterdir() if d.is_dir()])
    if not speaker_dirs:
        raise ValueError(f"No speaker directories found in {corpus_dir}")

    existing_speakers = {d.name for d in speaker_dirs}

    validate_speakers(
        requested_speakers=speakers,
        eval_speakers=eval_speakers,
        existing_speakers=existing_speakers,
    )

    selected_speakers = set(speakers)
    eval_speakers = set(eval_speakers)

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name

        if speaker_id not in selected_speakers:
            continue

        wav_dir = speaker_dir / speaker_id / "wav"
        transcript_dir = speaker_dir / speaker_id / "transcript"

        if not wav_dir.exists():
            print(f"  [WARN] No wav/ for {speaker_id}, skipping.")
            continue

        if not transcript_dir.exists():
            print(f"  [WARN] No transcript/ for {speaker_id}, skipping.")
            continue

        gender = SPEAKER_META.get(speaker_id, "unknown")
        split = "eval" if speaker_id in eval_speakers else "train"

        if gender == "unknown":
            print(f"  [WARN] Unknown speaker {speaker_id}, gender set to 'unknown'.")

        n_matched = 0

        for wav_file in sorted(wav_dir.glob("*.wav")):
            utt_id = wav_file.stem

            txt_file = transcript_dir / f"{utt_id}.txt"
            if not txt_file.exists():
                print(f"  [WARN] No transcript for {speaker_id}/{utt_id}, skipping.")
                continue

            transcript = read_transcript_file(txt_file)

            dest_name = f"{speaker_id}_{utt_id}.wav"
            dest_path = audio_out_dir / dest_name
            shutil.copy2(wav_file, dest_path)

            rows.append({
                "speaker_id": speaker_id,
                "gender": gender,
                "split": split,
                "utterance_id": utt_id,
                "transcript": transcript,
                "audio_path": str(dest_path),
                "duration_s": None,
            })

            n_matched += 1

        print(f"  [{speaker_id}] ({gender}, {split}) {n_matched} wavs matched.")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Import selected L2-ARCTIC speakers to Parquet.")

    parser.add_argument(
        "--corpus_dir",
        required=True,
        help="Root of the L2-ARCTIC corpus"
    )

    parser.add_argument(
        "--output_parquet",
        required=True,
        help="Destination parquet file"
    )

    parser.add_argument(
        "--audio_dir",
        required=True,
        help="Directory to copy WAV files into"
    )

    parser.add_argument(
        "--speakers",
        nargs="+",
        required=True,
        help="List of speakers to include"
    )

    parser.add_argument(
        "--eval_speakers",
        nargs="+",
        required=True,
        help="List of speakers assigned to eval split"
    )

    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    parquet_path = Path(args.output_parquet)
    audio_dir = Path(args.audio_dir)

    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    audio_dir.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning speakers in: {corpus_dir}")
    print(f"Selected speakers: {args.speakers}")
    print(f"Eval speakers: {args.eval_speakers}")

    df = build_l2_arctic_dataframe(
        corpus_dir=corpus_dir,
        audio_out_dir=audio_dir,
        speakers=args.speakers,
        eval_speakers=args.eval_speakers,
    )

    print(f"\nTotal rows: {len(df)}")
    print(df["split"].value_counts())
    print(df.groupby(["split", "speaker_id"]).size())

    df.to_parquet(parquet_path, index=False)

    print(f"\nSaved → {parquet_path}")


if __name__ == "__main__":
    main()