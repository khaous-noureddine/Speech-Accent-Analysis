"""
import_arctic.py

Usage:
python import_arctic.py \
    --corpus_dir data/raw/arctic \
    --output_parquet data/processed/arctic/corpus.parquet \
    --audio_dir data/processed/arctic/wavs \
    --speakers awb bdl clb jmk rms slt \
    --eval_speakers bdl jmk
"""

import sys
import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import normalize_transcript


SPEAKER_META = {
    "awb": "male",
    "bdl": "male",
    "clb": "female",
    "jmk": "male",
    "rms": "male",
    "slt": "female",
}


def parse_txt_done(path: Path) -> dict[str, str]:
    transcripts = {}
    pattern = re.compile(r'\(\s*(\S+)\s+"(.+?)"\s*\)')

    for line in path.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line.strip())
        if m:
            utt_id = m.group(1)
            transcript = normalize_transcript(m.group(2))
            transcripts[utt_id] = transcript

    return transcripts


def extract_speaker_code(speaker_dir_name: str) -> str:
    # cmu_us_awb_arctic -> awb
    parts = speaker_dir_name.split("_")
    return next((p for p in parts if p in SPEAKER_META), speaker_dir_name)


def format_speaker_id(spk_code: str) -> str:
    # awb -> US_AWB
    return f"US_{spk_code.upper()}"


def validate_speakers(
    requested_speakers: list[str],
    eval_speakers: list[str],
    existing_speakers: set[str],
):
    requested_speakers = {s.lower() for s in requested_speakers}
    eval_speakers = {s.lower() for s in eval_speakers}

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


def build_arctic_dataframe(
    corpus_dir: Path,
    audio_out_dir: Path,
    speakers: list[str],
    eval_speakers: list[str],
) -> pd.DataFrame:
    rows = []

    speaker_dirs = sorted([d for d in corpus_dir.iterdir() if d.is_dir()])
    if not speaker_dirs:
        raise ValueError(f"No speaker directories found in {corpus_dir}")

    speaker_dir_by_code = {
        extract_speaker_code(d.name): d
        for d in speaker_dirs
    }

    existing_speakers = set(speaker_dir_by_code.keys())

    validate_speakers(
        requested_speakers=speakers,
        eval_speakers=eval_speakers,
        existing_speakers=existing_speakers,
    )

    selected_speakers = {s.lower() for s in speakers}
    eval_speakers = {s.lower() for s in eval_speakers}

    for spk_code in sorted(selected_speakers):
        speaker_dir = speaker_dir_by_code[spk_code]

        original_speaker_id = speaker_dir.name
        speaker_id = format_speaker_id(spk_code)
        gender = SPEAKER_META.get(spk_code, "unknown")
        split = "eval" if spk_code in eval_speakers else "train"

        txt_file = speaker_dir / "etc" / "txt.done.data"
        wav_dir = speaker_dir / "wav"

        if not txt_file.exists():
            print(f"  [WARN] No txt.done.data for {original_speaker_id}, skipping.")
            continue

        if not wav_dir.exists():
            print(f"  [WARN] No wav/ directory for {original_speaker_id}, skipping.")
            continue

        transcripts = parse_txt_done(txt_file)

        n_matched = 0

        for wav_file in sorted(wav_dir.glob("*.wav")):
            utt_id = wav_file.stem

            transcript = transcripts.get(utt_id)
            if transcript is None:
                print(f"  [WARN] No transcript for {original_speaker_id}/{utt_id}, skipping.")
                continue

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

        print(
            f"  [{original_speaker_id} -> {speaker_id}] "
            f"({spk_code}, {gender}, {split}) "
            f"{len(transcripts)} transcripts, {n_matched} wavs matched."
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Import selected CMU ARCTIC speakers to Parquet.")

    parser.add_argument(
        "--corpus_dir",
        required=True,
        help="Root of the CMU ARCTIC corpus"
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
        help="Speaker codes to include, e.g. awb bdl clb jmk rms slt"
    )

    parser.add_argument(
        "--eval_speakers",
        nargs="+",
        required=True,
        help="Speaker codes assigned to eval split"
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

    df = build_arctic_dataframe(
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