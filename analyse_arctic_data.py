"""
python analyse_arctic_data.py --corpus_dir data/raw/l2_arctic/speakers

Analyze the raw L2-ARCTIC dataset (without using the processed parquet)
to verify corpus integrity and compare against the processing script.

Expected structure:
<corpus_dir>/
    ABA/
        ABA/
            wav/
            transcript/

Usage:
python analyse_arctic_data.py --corpus_dir data/raw/l2_arctic/speakers
"""

import argparse
from pathlib import Path
import pandas as pd


# Speaker metadata from L2-ARCTIC
SPEAKER_META = {
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


def analyze_l2_arctic_raw(corpus_dir: Path):
    rows = []

    if not corpus_dir.exists():
        raise FileNotFoundError(f"Directory not found: {corpus_dir}")

    speaker_dirs = sorted([d for d in corpus_dir.iterdir() if d.is_dir()])

    if not speaker_dirs:
        raise ValueError("No speaker directories found.")

    print("=== Raw L2-ARCTIC Analysis ===\n")

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name

        # Same structure used in your import script
        wav_dir = speaker_dir / speaker_id / "wav"
        txt_dir = speaker_dir / speaker_id / "transcript"

        if not wav_dir.exists() or not txt_dir.exists():
            print(f"[WARN] Invalid structure for {speaker_id}")
            continue

        wav_files = sorted(wav_dir.glob("*.wav"))
        txt_files = sorted(txt_dir.glob("*.txt"))

        wav_ids = {f.stem for f in wav_files}
        txt_ids = {f.stem for f in txt_files}

        matched_ids = wav_ids & txt_ids
        wav_only = wav_ids - txt_ids
        txt_only = txt_ids - wav_ids

        meta = SPEAKER_META.get(
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

    df = pd.DataFrame(rows)

    # =========================
    # Results
    # =========================

    print("=== Preview ===")
    print(df.head(), "\n")

    print("=== Columns ===")
    print(df.columns.tolist(), "\n")

    print("=== Total Speakers ===")
    print(len(df), "\n")

    print("=== Total Valid Examples (wav + transcript) ===")
    print(df["matched_pairs"].sum(), "\n")

    print("=== Examples Per Speaker ===")
    print(
        df[
            [
                "speaker_id",
                "gender",
                "l1_language",
                "wav_count",
                "txt_count",
                "matched_pairs",
                "missing_txt",
                "missing_wav",
            ]
        ]
        # .sort_values(by=["l1_language", "speaker_id"], ascending=[True, True])
        .sort_values(by="matched_pairs", ascending=False)
        .to_string(index=False),
        "\n"
    )

    print("=== Speakers With Issues ===")
    problems = df[
        (df["missing_txt"] > 0) |
        (df["missing_wav"] > 0)
    ]

    if len(problems) == 0:
        print("No issues detected.")
    else:
        print(
            problems[
                [
                    "speaker_id",
                    "gender",
                    "l1_language",
                    "wav_count",
                    "txt_count",
                    "matched_pairs",
                    "missing_txt",
                    "missing_wav",
                ]
            ]
            # .sort_values(by=["l1_language", "speaker_id"], ascending=[True, True])
            .sort_values(by="matched_pairs", ascending=False)
            .to_string(index=False)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_dir",
        required=True,
        help="Path to raw L2-ARCTIC corpus"
    )
    args = parser.parse_args()

    analyze_l2_arctic_raw(Path(args.corpus_dir))


if __name__ == "__main__":
    main()