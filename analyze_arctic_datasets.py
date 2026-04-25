"""
analyze_arctic_datasets.py

Analyze raw CMU Arctic or L2-ARCTIC datasets without using processed parquet files.
Used to verify corpus integrity and compare against processing scripts.

Expected CMU Arctic structure:
<corpus_dir>/
    cmu_us_awb_arctic/
        wav/
        etc/
            txt.done.data

Expected L2-ARCTIC structure:
<corpus_dir>/
    ABA/
        ABA/
            wav/
            transcript/

Usage:
CMU arctic:
python analyze_arctic_datasets.py --dataset arctic --corpus_dir data/raw/arctic

l2_arctic:
python analyze_arctic_datasets.py --dataset l2_arctic --corpus_dir data/raw/l2_arctic/speakers
""" 

import argparse
import re
from pathlib import Path

import pandas as pd


ARCTIC_SPEAKER_META = {
    "awb": "male",
    "bdl": "male",
    "clb": "female",
    "jmk": "male",
    "rms": "male",
    "slt": "female",
}


L2_ARCTIC_SPEAKER_META = {
    # Arabic L1
    "ABA":   {"gender": "male",   "l1_language": "Arabic"},
    "YBAA":  {"gender": "male",   "l1_language": "Arabic"},
    "ZHAA":  {"gender": "female", "l1_language": "Arabic"},
    "SKA":   {"gender": "male",   "l1_language": "Arabic"},

    # Mandarin L1
    "BWC":   {"gender": "male",   "l1_language": "Mandarin"},
    "LXC":   {"gender": "female", "l1_language": "Mandarin"},
    "NCC":   {"gender": "female", "l1_language": "Mandarin"},
    "TXHC":  {"gender": "male",   "l1_language": "Mandarin"},

    # Hindi L1
    "ASI":   {"gender": "male",   "l1_language": "Hindi"},
    "RRBI":  {"gender": "male",   "l1_language": "Hindi"},
    "SVBI":  {"gender": "female", "l1_language": "Hindi"},
    "TNI":   {"gender": "female", "l1_language": "Hindi"},

    # Korean L1
    "HJK":   {"gender": "female", "l1_language": "Korean"},
    "HKK":   {"gender": "male",   "l1_language": "Korean"},
    "YDCK":  {"gender": "female", "l1_language": "Korean"},
    "YKWK":  {"gender": "male",   "l1_language": "Korean"},

    # Spanish L1
    "EBVS":  {"gender": "male",   "l1_language": "Spanish"},
    "ERMS":  {"gender": "male",   "l1_language": "Spanish"},
    "MBMPS": {"gender": "female", "l1_language": "Spanish"},
    "NJS":   {"gender": "female", "l1_language": "Spanish"},

    # Vietnamese L1
    "PNV":   {"gender": "female", "l1_language": "Vietnamese"},
    "THV":   {"gender": "female", "l1_language": "Vietnamese"},
    "TLV":   {"gender": "male",   "l1_language": "Vietnamese"},
    "HQTV":  {"gender": "male",   "l1_language": "Vietnamese"},
}


def parse_txt_done(path: Path) -> dict[str, str]:
    """
    Parse txt.done.data and return {utterance_id: transcript}.

    Example line:
        ( arctic_a0001 "She had your dark eyes and this dark hair." )
    """
    transcripts = {}
    pattern = re.compile(r'\(\s*(\S+)\s+"(.+?)"\s*\)')

    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            utt_id = match.group(1)
            transcript = match.group(2)
            transcripts[utt_id] = transcript

    return transcripts


def validate_corpus_dir(corpus_dir: Path) -> list[Path]:
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Directory not found: {corpus_dir}")

    speaker_dirs = sorted([d for d in corpus_dir.iterdir() if d.is_dir()])

    if not speaker_dirs:
        raise ValueError(f"No speaker directories found in: {corpus_dir}")

    return speaker_dirs


def analyze_arctic_raw(corpus_dir: Path) -> pd.DataFrame:
    rows = []
    speaker_dirs = validate_corpus_dir(corpus_dir)

    print("=== Raw CMU Arctic Analysis ===\n")

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name

        parts = speaker_id.split("_")
        speaker_code = next(
            (part for part in parts if part in ARCTIC_SPEAKER_META),
            speaker_id
        )
        gender = ARCTIC_SPEAKER_META.get(speaker_code, "unknown")

        txt_file = speaker_dir / "etc" / "txt.done.data"
        wav_dir = speaker_dir / "wav"

        if not txt_file.exists() or not wav_dir.exists():
            print(f"[WARN] Invalid structure for {speaker_id}")
            continue

        transcripts = parse_txt_done(txt_file)

        wav_files = sorted(wav_dir.glob("*.wav"))
        wav_ids = {file.stem for file in wav_files}
        txt_ids = set(transcripts.keys())

        matched_ids = wav_ids & txt_ids
        wav_only = wav_ids - txt_ids
        txt_only = txt_ids - wav_ids

        rows.append({
            "speaker_id": speaker_id,
            "speaker_code": speaker_code,
            "gender": gender,
            "wav_count": len(wav_files),
            "txt_count": len(txt_ids),
            "matched_pairs": len(matched_ids),
            "missing_txt": len(wav_only),
            "missing_wav": len(txt_only),
        })

    return pd.DataFrame(rows)


def analyze_l2_arctic_raw(corpus_dir: Path) -> pd.DataFrame:
    rows = []
    speaker_dirs = validate_corpus_dir(corpus_dir)

    print("=== Raw L2-ARCTIC Analysis ===\n")

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name

        wav_dir = speaker_dir / speaker_id / "wav"
        txt_dir = speaker_dir / speaker_id / "transcript"

        if not wav_dir.exists() or not txt_dir.exists():
            print(f"[WARN] Invalid structure for {speaker_id}")
            continue

        wav_files = sorted(wav_dir.glob("*.wav"))
        txt_files = sorted(txt_dir.glob("*.txt"))

        wav_ids = {file.stem for file in wav_files}
        txt_ids = {file.stem for file in txt_files}

        matched_ids = wav_ids & txt_ids
        wav_only = wav_ids - txt_ids
        txt_only = txt_ids - wav_ids

        meta = L2_ARCTIC_SPEAKER_META.get(
            speaker_id,
            {"gender": "unknown", "l1_language": "unknown"}
        )

        rows.append({
            "speaker_id": speaker_id,
            "gender": meta["gender"],
            "l1_language": meta["l1_language"],
            "wav_count": len(wav_files),
            "txt_count": len(txt_files),
            "matched_pairs": len(matched_ids),
            "missing_txt": len(wav_only),
            "missing_wav": len(txt_only),
        })

    return pd.DataFrame(rows)


def print_results(df: pd.DataFrame, dataset: str) -> None:
    if df.empty:
        print("No valid speaker data found.")
        return

    print("=== Preview ===")
    print(df.head(), "\n")

    print("=== Columns ===")
    print(df.columns.tolist(), "\n")

    print("=== Total Speakers ===")
    print(len(df), "\n")

    print("=== Total Valid Examples (wav + transcript) ===")
    print(df["matched_pairs"].sum(), "\n")

    print("=== Examples Per Speaker ===")

    if dataset == "arctic":
        columns = [
            "speaker_id",
            "speaker_code",
            "gender",
            "wav_count",
            "txt_count",
            "matched_pairs",
            "missing_txt",
            "missing_wav",
        ]
    else:
        columns = [
            "speaker_id",
            "gender",
            "l1_language",
            "wav_count",
            "txt_count",
            "matched_pairs",
            "missing_txt",
            "missing_wav",
        ]

    print(
        df[columns]
        .sort_values(by="matched_pairs", ascending=False)
        .to_string(index=False),
        "\n"
    )

    print("=== Speakers With Issues ===")

    problems = df[
        (df["missing_txt"] > 0) |
        (df["missing_wav"] > 0)
    ]

    if problems.empty:
        print("No issues detected.")
    else:
        print(
            problems[columns]
            .sort_values(by="matched_pairs", ascending=False)
            .to_string(index=False)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["arctic", "l2_arctic"],
        help="Dataset to analyze: arctic or l2_arctic"
    )
    parser.add_argument(
        "--corpus_dir",
        required=True,
        help="Path to raw corpus directory"
    )

    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)

    if args.dataset == "arctic":
        df = analyze_arctic_raw(corpus_dir)
    elif args.dataset == "l2_arctic":
        df = analyze_l2_arctic_raw(corpus_dir)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print_results(df, args.dataset)


if __name__ == "__main__":
    main()