"""
corpus/import_l2_arctic_cv.py

Create 8-fold cross-validation splits for L2-ARCTIC.

Protocol:
  For each fold and each L1/accent:
    - train: 80% unique prompts spoken by 3 speakers
    - dev:   10% other prompts spoken by the same 3 speakers
    - test:  10% remaining prompts spoken by the held-out speaker

Properties:
  - 8 folds.
  - Each L1/accent has 4 speakers.
  - In each fold, 1 speaker per L1 is held out for test.
  - Each speaker appears in test in 2 folds.
  - Train/dev speakers and test speaker are disjoint.
  - Train/dev prompts and test prompts are disjoint.
  - Only prompts available for all 4 speakers of a given L1 are used.

Outputs:
  output_dir/
    wavs/
      <speaker>_<utterance_id>.wav

    fold_00/corpus.parquet
    fold_00/split_stats.csv
    ...
    fold_07/corpus.parquet
    fold_07/split_stats.csv

    test_all_folds.parquet
    fold_summary.csv

Usage:
  python corpus/import_l2_arctic_cv.py \
    --corpus_dir data/raw/l2_arctic/speakers \
    --output_dir data/processed/l2_arctic_cv \
    --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import normalize_transcript


SPEAKER_GENDER = {
    # Arabic L1
    "ABA":  "male",
    "YBAA": "male",
    "ZHAA": "female",
    "SKA":  "male",

    # Chinese / Mandarin L1
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
    "EBVS":  "male",
    "ERMS":  "male",
    "MBMPS": "female",
    "NJS":   "female",

    # Vietnamese L1
    "PNV":  "female",
    "THV":  "female",
    "TLV":  "male",
    "HQTV": "male",
}


SPEAKER_L1 = {
    "ABA":  "Arabic",
    "YBAA": "Arabic",
    "ZHAA": "Arabic",
    "SKA":  "Arabic",

    "BWC":  "Chinese",
    "LXC":  "Chinese",
    "NCC":  "Chinese",
    "TXHC": "Chinese",

    "ASI":  "Hindi",
    "RRBI": "Hindi",
    "SVBI": "Hindi",
    "TNI":  "Hindi",

    "HJK":  "Korean",
    "HKK":  "Korean",
    "YDCK": "Korean",
    "YKWK": "Korean",

    "EBVS":  "Spanish",
    "ERMS":  "Spanish",
    "MBMPS": "Spanish",
    "NJS":   "Spanish",

    "PNV":  "Vietnamese",
    "THV":  "Vietnamese",
    "TLV":  "Vietnamese",
    "HQTV": "Vietnamese",
}


EXPECTED_L1S = {
    "Arabic",
    "Chinese",
    "Hindi",
    "Korean",
    "Spanish",
    "Vietnamese",
}


def stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16)


def read_transcript(path: Path) -> str:
    return normalize_transcript(
        path.read_text(encoding="utf-8", errors="replace").strip()
    )


def get_duration_s(wav_path: Path) -> float | None:
    try:
        return round(float(sf.info(str(wav_path)).duration), 3)
    except Exception:
        return None


def split_into_10_blocks(items: list[str]) -> list[list[str]]:
    """
    Split a list into 10 approximately equal blocks.

    This gives:
      - 8 blocks for train
      - 1 block for dev
      - 1 block for test
    """
    n = len(items)
    blocks: list[list[str]] = []

    for i in range(10):
        start = round(i * n / 10)
        end = round((i + 1) * n / 10)
        blocks.append(items[start:end])

    return blocks


def load_l2_arctic_raw(
    corpus_dir: Path,
    wavs_dir: Path,
    speakers: list[str],
) -> pd.DataFrame:
    rows = []
    wavs_dir.mkdir(parents=True, exist_ok=True)

    for speaker_id in speakers:
        speaker_root = corpus_dir / speaker_id / speaker_id
        wav_dir = speaker_root / "wav"
        transcript_dir = speaker_root / "transcript"

        if not wav_dir.exists():
            raise FileNotFoundError(
                f"Missing wav directory for {speaker_id}: {wav_dir}"
            )

        if not transcript_dir.exists():
            raise FileNotFoundError(
                f"Missing transcript directory for {speaker_id}: {transcript_dir}"
            )

        gender = SPEAKER_GENDER.get(speaker_id, "unknown")
        native_language = SPEAKER_L1.get(speaker_id, "unknown")

        n_matched = 0

        for wav_path in sorted(wav_dir.glob("*.wav")):
            utterance_id = wav_path.stem
            prompt_id = utterance_id

            txt_path = transcript_dir / f"{utterance_id}.txt"

            if not txt_path.exists():
                print(f"[WARN] Missing transcript: {txt_path}")
                continue

            transcript = read_transcript(txt_path)

            dest_name = f"{speaker_id}_{utterance_id}.wav"
            dest_path = wavs_dir / dest_name

            if not dest_path.exists():
                shutil.copy2(wav_path, dest_path)

            rows.append(
                {
                    "corpus": "l2_arctic",
                    "speaker_id": speaker_id,
                    "gender": gender,
                    "native_language": native_language,
                    "accent": native_language,
                    "utterance_id": utterance_id,
                    "prompt_id": prompt_id,
                    "transcript": transcript,
                    "audio_path": str(dest_path.resolve()),
                    "duration_s": get_duration_s(dest_path),
                }
            )

            n_matched += 1

        print(
            f"[{speaker_id}] {native_language:10s} "
            f"{gender:6s} — {n_matched} utterances"
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No utterances loaded.")

    return df


def validate_speakers(df: pd.DataFrame) -> dict[str, list[str]]:
    speakers_by_l1 = {
        l1: sorted(group["speaker_id"].unique().tolist())
        for l1, group in df.groupby("native_language")
    }

    found_l1s = set(speakers_by_l1)

    if found_l1s != EXPECTED_L1S:
        raise ValueError(
            f"Expected L1 groups {sorted(EXPECTED_L1S)}, "
            f"found {sorted(found_l1s)}"
        )

    for l1, speakers in speakers_by_l1.items():
        if len(speakers) != 4:
            raise ValueError(
                f"L1 group '{l1}' must have exactly 4 speakers. "
                f"Found {len(speakers)}: {speakers}"
            )

    return speakers_by_l1


def get_fold_block_ids(fold: int) -> tuple[int, int]:
    """
    For 8 folds:
      - test uses blocks 0..7 exactly once.
      - dev uses block 8 for folds 0..3 and block 9 for folds 4..7.

    This keeps the protocol simple:
      train = 8 blocks
      dev   = 1 block
      test  = 1 block
    """
    test_block_id = fold
    dev_block_id = 8 if fold < 4 else 9

    return test_block_id, dev_block_id


def build_fold_dataframe(
    base_df: pd.DataFrame,
    fold: int,
    seed: int,
    speakers_by_l1: dict[str, list[str]],
) -> pd.DataFrame:
    fold_rows = []

    test_block_id, dev_block_id = get_fold_block_ids(fold)

    for native_language, speakers in sorted(speakers_by_l1.items()):
        speakers = sorted(speakers)

        # 4 speakers per L1.
        # fold % 4 means each speaker is test in exactly 2 folds.
        test_speaker = speakers[fold % 4]
        train_dev_speakers = [s for s in speakers if s != test_speaker]

        l1_df = base_df[base_df["native_language"] == native_language].copy()

        # Strict policy:
        # keep only prompts available for all 4 speakers of that L1.
        prompts_by_speaker = {
            speaker: set(
                l1_df[l1_df["speaker_id"] == speaker]["prompt_id"].unique()
            )
            for speaker in speakers
        }

        common_prompts = sorted(set.intersection(*prompts_by_speaker.values()))

        if len(common_prompts) < 10:
            raise ValueError(
                f"Not enough common prompts for {native_language}: "
                f"{len(common_prompts)}"
            )

        total_prompts = l1_df["prompt_id"].nunique()
        ignored_prompts = total_prompts - len(common_prompts)

        if ignored_prompts > 0:
            print(
                f"[fold {fold:02d}] [{native_language}] "
                f"using {len(common_prompts)} common prompts "
                f"({ignored_prompts} non-common prompts ignored)"
            )

        shuffled_prompts = (
            pd.Series(common_prompts)
            .sample(
                frac=1.0,
                random_state=stable_seed(seed, f"{native_language}_fold_{fold}"),
            )
            .tolist()
        )

        blocks = split_into_10_blocks(shuffled_prompts)

        test_prompts = set(blocks[test_block_id])
        dev_prompts = set(blocks[dev_block_id])

        train_prompts = set()
        for block_id, block in enumerate(blocks):
            if block_id not in {test_block_id, dev_block_id}:
                train_prompts.update(block)

        if not train_prompts:
            raise RuntimeError(f"Empty train prompts in fold {fold}, {native_language}")

        if not dev_prompts:
            raise RuntimeError(f"Empty dev prompts in fold {fold}, {native_language}")

        if not test_prompts:
            raise RuntimeError(f"Empty test prompts in fold {fold}, {native_language}")

        assert train_prompts.isdisjoint(dev_prompts)
        assert train_prompts.isdisjoint(test_prompts)
        assert dev_prompts.isdisjoint(test_prompts)

        train_part = l1_df[
            (l1_df["speaker_id"].isin(train_dev_speakers))
            & (l1_df["prompt_id"].isin(train_prompts))
        ].copy()
        train_part["split"] = "train"

        dev_part = l1_df[
            (l1_df["speaker_id"].isin(train_dev_speakers))
            & (l1_df["prompt_id"].isin(dev_prompts))
        ].copy()
        dev_part["split"] = "dev"

        test_part = l1_df[
            (l1_df["speaker_id"] == test_speaker)
            & (l1_df["prompt_id"].isin(test_prompts))
        ].copy()
        test_part["split"] = "test"

        fold_rows.extend([train_part, dev_part, test_part])

        print(
            f"[fold {fold:02d}] [{native_language}] "
            f"test_speaker={test_speaker} | "
            f"train={len(train_part)} dev={len(dev_part)} test={len(test_part)} | "
            f"prompt blocks train/dev/test="
            f"8/{dev_block_id}/{test_block_id} | "
            f"prompts train/dev/test="
            f"{len(train_prompts)}/{len(dev_prompts)}/{len(test_prompts)}"
        )

    fold_df = pd.concat(fold_rows, ignore_index=True)
    fold_df["fold"] = fold

    validate_fold_no_leakage(fold_df, fold)

    return fold_df


def validate_fold_no_leakage(fold_df: pd.DataFrame, fold: int) -> None:
    for native_language, group in fold_df.groupby("native_language"):
        train_dev_group = group[group["split"].isin(["train", "dev"])]
        test_group = group[group["split"] == "test"]

        train_dev_prompts = set(train_dev_group["prompt_id"])
        test_prompts = set(test_group["prompt_id"])

        prompt_overlap = train_dev_prompts & test_prompts

        if prompt_overlap:
            raise RuntimeError(
                f"Prompt leakage in fold {fold}, L1={native_language}: "
                f"{len(prompt_overlap)} overlapping prompts"
            )

        train_dev_speakers = set(train_dev_group["speaker_id"])
        test_speakers = set(test_group["speaker_id"])

        speaker_overlap = train_dev_speakers & test_speakers

        if speaker_overlap:
            raise RuntimeError(
                f"Speaker leakage in fold {fold}, L1={native_language}: "
                f"{speaker_overlap}"
            )


def write_fold_stats(fold_df: pd.DataFrame, output_csv: Path) -> pd.DataFrame:
    stats = (
        fold_df
        .groupby(["fold", "native_language", "split"])
        .agg(
            n_examples=("audio_path", "size"),
            n_speakers=("speaker_id", "nunique"),
            n_prompts=("prompt_id", "nunique"),
            duration_s=("duration_s", "sum"),
        )
        .reset_index()
    )

    stats["duration_h"] = stats["duration_s"] / 3600.0
    stats["duration_s"] = stats["duration_s"].round(2)
    stats["duration_h"] = stats["duration_h"].round(3)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_csv, index=False)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create 8-fold L2-ARCTIC CV parquets."
    )

    parser.add_argument(
        "--corpus_dir",
        type=Path,
        required=True,
        help="Raw L2-ARCTIC speakers directory, e.g. data/raw/l2_arctic/speakers",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory, e.g. data/processed/l2_arctic_cv",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed used for deterministic prompt shuffling.",
    )

    parser.add_argument(
        "--speakers",
        nargs="*",
        default=sorted(SPEAKER_L1.keys()),
        help="Speakers to include. Default: all 24 L2-ARCTIC speakers.",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    wavs_dir = output_dir / "wavs"

    output_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Loading raw L2-ARCTIC")
    print("=" * 80)
    print(f"corpus_dir : {args.corpus_dir}")
    print(f"output_dir : {output_dir}")
    print(f"wavs_dir   : {wavs_dir}")
    print(f"seed       : {args.seed}")
    print(f"speakers   : {args.speakers}")

    base_df = load_l2_arctic_raw(
        corpus_dir=args.corpus_dir,
        wavs_dir=wavs_dir,
        speakers=args.speakers,
    )

    speakers_by_l1 = validate_speakers(base_df)

    print("\nSpeakers by L1:")
    for native_language, speakers in sorted(speakers_by_l1.items()):
        print(f"  {native_language:10s}: {speakers}")

    all_test_parts = []
    all_stats = []

    for fold in range(8):
        print("\n" + "=" * 80)
        print(f"Building fold {fold:02d}")
        print("=" * 80)

        fold_df = build_fold_dataframe(
            base_df=base_df,
            fold=fold,
            seed=args.seed,
            speakers_by_l1=speakers_by_l1,
        )

        fold_dir = output_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = fold_dir / "corpus.parquet"
        stats_path = fold_dir / "split_stats.csv"

        fold_df.to_parquet(parquet_path, index=False)
        stats = write_fold_stats(fold_df, stats_path)

        all_stats.append(stats)
        all_test_parts.append(fold_df[fold_df["split"] == "test"].copy())

        print(f"Saved parquet → {parquet_path}")
        print(f"Saved stats   → {stats_path}")
        print(fold_df["split"].value_counts().to_string())

    test_all = pd.concat(all_test_parts, ignore_index=True)
    test_all_path = output_dir / "test_all_folds.parquet"
    test_all.to_parquet(test_all_path, index=False)

    summary = pd.concat(all_stats, ignore_index=True)
    summary_path = output_dir / "fold_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("Done")
    print("=" * 80)
    print(f"All-fold test parquet → {test_all_path}")
    print(f"Fold summary          → {summary_path}")

    print("\nGlobal split summary across all folds:")
    global_summary = (
        summary
        .groupby("split")[["n_examples", "duration_s"]]
        .sum()
        .assign(duration_h=lambda x: x["duration_s"] / 3600.0)
        .round(3)
    )
    print(global_summary.to_string())


if __name__ == "__main__":
    main()