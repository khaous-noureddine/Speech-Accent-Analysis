"""
corpus/import_aesrc.py

Import AESRC2020 into the project format.

Main rules
----------
- Seen accents are split into train/dev at speaker level.
- Canadian and Spanish are reserved for test.
- If a test transcript already appears in train/dev, the test example is removed.
  Train/dev examples are kept.

Outputs
-------
- Parquet corpus
- Optional CSV corpus
- Optional LaTeX table with split statistics per accent
- Optional TXT report checking phrase overlap between train/dev and test
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import normalize_transcript


COUNTRY_FOLDER_MAP = {
    # avec "speaking"
    "british english speech data":              "British",
    "american english speech data":             "American",
    "russian speaking english speech data":     "Russian",
    "korean speaking english speech data":      "Korean",
    "canadian speaking english speech data":    "Canadian",
    "portuguese speaking english speech data":  "Portuguese",
    "japanese speaking english speech data":    "Japanese",
    "spanish speaking english speech data":     "Spanish",
    "india english speech data":                "Indian",
    "chinese speaking english speech data":     "Chinese",

    # sans "speaking" au cas où
    "russian english speech data":              "Russian",
    "korean english speech data":               "Korean",
    "canadian english speech data":             "Canadian",
    "portuguese english speech data":           "Portuguese",
    "japanese english speech data":             "Japanese",
    "spanish english speech data":              "Spanish",
    "chinese english speech data":              "Chinese",
    "indian english speech data":               "Indian",
}

# Held-out accents for final test only
TEST_COUNTRIES = {"Canadian", "Spanish"}


def parse_metadata(path: Path) -> dict:
    """Parse a .metadata file into a key-value dict."""
    meta = {}

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()

        if not line or line.startswith("CMT"):
            continue

        parts = line.split(None, 1)

        if len(parts) == 2:
            meta[parts[0]] = parts[1].strip()
        elif len(parts) == 1:
            meta[parts[0]] = ""

    return meta


def make_speaker_split_map(
    country_label: str,
    speaker_dirs: list[Path],
    dev_ratio: float = 0.10,
    seed: int = 42,
) -> dict[str, str]:
    """
    Create speaker-level split mapping for one accent/country.

    Rules:
      - Canadian and Spanish are assigned to test.
      - Other countries are split into train/dev by speaker.
    """
    if not 0.0 <= dev_ratio < 1.0:
        raise ValueError(f"dev_ratio must be in [0.0, 1.0), got {dev_ratio}")

    speaker_ids = [d.name for d in speaker_dirs]
    n_total = len(speaker_ids)

    if n_total == 0:
        logger.warning(f"  [{country_label}] no speakers found.")
        return {}

    if country_label in TEST_COUNTRIES:
        logger.info(
            f"  [{country_label}] speakers: total={n_total}, "
            f"train=0, dev=0, test={n_total}"
        )
        return {speaker_id: "test" for speaker_id in speaker_ids}

    rng = random.Random(seed)
    shuffled = speaker_ids.copy()
    rng.shuffle(shuffled)

    n_dev = int(round(n_total * dev_ratio))

    # Safety for small subsets
    if n_total >= 2 and dev_ratio > 0.0:
        n_dev = max(1, n_dev)
        n_dev = min(n_dev, n_total - 1)
    else:
        n_dev = 0

    dev_speakers = set(shuffled[:n_dev])

    split_map = {
        speaker_id: "dev" if speaker_id in dev_speakers else "train"
        for speaker_id in speaker_ids
    }

    n_train = sum(1 for split in split_map.values() if split == "train")
    n_dev_actual = sum(1 for split in split_map.values() if split == "dev")

    logger.info(
        f"  [{country_label}] speakers: total={n_total}, "
        f"train={n_train}, dev={n_dev_actual}, test=0 "
        f"(dev_ratio={dev_ratio}, seed={seed})"
    )

    return split_map


def apply_phrase_leakage_lock(
    df: pd.DataFrame,
    phrase_col: str = "transcript",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove test examples whose phrase/transcript already appears in train/dev.

    Train/dev examples are kept.
    Only test rows are removed.
    """
    if phrase_col not in df.columns:
        raise ValueError(f"Missing phrase column: {phrase_col}")

    df_before = df.copy()

    seen_phrases = set(
        df.loc[df["split"].isin(["train", "dev"]), phrase_col]
        .dropna()
        .astype(str)
    )

    test_overlap_mask = (
        (df["split"] == "test")
        & (df[phrase_col].astype(str).isin(seen_phrases))
    )

    n_removed = int(test_overlap_mask.sum())

    if n_removed > 0:
        logger.warning(
            f"Removing {n_removed:,} test utterances whose transcript already "
            f"appears in train/dev."
        )

        logger.warning(
            "Removed test utterances by country:\n"
            f"{df.loc[test_overlap_mask].groupby('country').size().to_string()}"
        )
    else:
        logger.info("No test transcript overlap with train/dev found.")

    df_after = df.loc[~test_overlap_mask].copy()

    remaining_seen = set(
        df_after.loc[df_after["split"].isin(["train", "dev"]), phrase_col]
        .dropna()
        .astype(str)
    )

    remaining_test = set(
        df_after.loc[df_after["split"] == "test", phrase_col]
        .dropna()
        .astype(str)
    )

    remaining_overlap = remaining_seen & remaining_test

    if remaining_overlap:
        raise RuntimeError(
            f"Leakage lock failed: {len(remaining_overlap)} phrases still appear "
            f"in both train/dev and test."
        )

    logger.info(
        "Phrase leakage lock passed: no transcript overlap between train/dev and test."
    )

    return df_after, df_before


def build_accent_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build table:

        accent,total,train,dev,ratio_dev_train,test
    """
    stats = (
        df.groupby(["country", "split"])
        .size()
        .unstack(fill_value=0)
    )

    for col in ["train", "dev", "test"]:
        if col not in stats.columns:
            stats[col] = 0

    stats = stats[["train", "dev", "test"]]
    stats["total"] = stats[["train", "dev", "test"]].sum(axis=1)

    stats["ratio_dev_train"] = stats.apply(
        lambda r: r["dev"] / r["train"] if r["train"] > 0 else None,
        axis=1,
    )

    stats = stats.reset_index().rename(columns={"country": "accent"})

    stats = stats[
        ["accent", "total", "train", "dev", "ratio_dev_train", "test"]
    ]

    stats = stats.sort_values("accent").reset_index(drop=True)

    return stats


def write_accent_stats_latex(df: pd.DataFrame, output_tex: Path) -> None:
    """Write accent-level split statistics as a LaTeX table."""
    output_tex = Path(output_tex)
    output_tex.parent.mkdir(parents=True, exist_ok=True)

    stats = build_accent_stats(df).copy()

    stats_for_latex = stats.copy()
    stats_for_latex["ratio_dev_train"] = stats_for_latex["ratio_dev_train"].map(
        lambda x: "" if pd.isna(x) else f"{x:.4f}"
    )

    latex = stats_for_latex.to_latex(
        index=False,
        escape=True,
        caption="AESRC split statistics by accent after transcript-overlap filtering.",
        label="tab:aesrc_split_stats",
        column_format="lrrrrr",
    )

    output_tex.write_text(latex, encoding="utf-8")
    logger.info(f"LaTeX stats table → {output_tex}")


def write_phrase_overlap_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_txt: Path,
    phrase_col: str = "transcript",
) -> None:
    """
    Write a text report verifying that no phrase appears in both:
      - train/dev
      - test
    """
    output_txt = Path(output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    def get_seen_test_overlap(df: pd.DataFrame) -> set[str]:
        seen = set(
            df.loc[df["split"].isin(["train", "dev"]), phrase_col]
            .dropna()
            .astype(str)
        )
        test = set(
            df.loc[df["split"] == "test", phrase_col]
            .dropna()
            .astype(str)
        )
        return seen & test

    def phrase_split_country_counts(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.groupby([phrase_col, "split", "country"])
            .size()
            .reset_index(name="n")
            .sort_values([phrase_col, "split", "country"])
        )

    overlap_before = sorted(get_seen_test_overlap(df_before))
    overlap_after = sorted(get_seen_test_overlap(df_after))

    removed = df_before.loc[
        (df_before["split"] == "test")
        & (~df_before.index.isin(df_after.index))
    ].copy()

    with output_txt.open("w", encoding="utf-8") as f:
        f.write("AESRC phrase overlap report\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Phrase column used: {phrase_col}\n\n")

        f.write("Before filtering\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total rows:              {len(df_before):,}\n")
        f.write(
            f"Unique train/dev phrases: "
            f"{df_before.loc[df_before['split'].isin(['train', 'dev']), phrase_col].nunique():,}\n"
        )
        f.write(
            f"Unique test phrases:      "
            f"{df_before.loc[df_before['split'] == 'test', phrase_col].nunique():,}\n"
        )
        f.write(f"Overlapping phrases:      {len(overlap_before):,}\n\n")

        f.write("After filtering\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total rows:              {len(df_after):,}\n")
        f.write(
            f"Unique train/dev phrases: "
            f"{df_after.loc[df_after['split'].isin(['train', 'dev']), phrase_col].nunique():,}\n"
        )
        f.write(
            f"Unique test phrases:      "
            f"{df_after.loc[df_after['split'] == 'test', phrase_col].nunique():,}\n"
        )
        f.write(f"Overlapping phrases:      {len(overlap_after):,}\n\n")

        f.write("Removed test examples\n")
        f.write("-" * 100 + "\n")
        f.write(f"Removed rows:             {len(removed):,}\n\n")

        if len(removed) > 0:
            f.write("Removed by country:\n")
            f.write(removed.groupby("country").size().to_string())
            f.write("\n\n")

            f.write("Removed examples:\n")
            cols = ["country", "speaker_id", "utterance_id", phrase_col]
            f.write(removed[cols].to_string(index=False))
            f.write("\n\n")

        f.write("Final verification\n")
        f.write("-" * 100 + "\n")

        if len(overlap_after) == 0:
            f.write("OK: no phrase appears in both train/dev and test.\n")
        else:
            f.write("WARNING: some phrases still appear in both train/dev and test.\n")
            f.write(f"Remaining overlaps: {len(overlap_after)}\n\n")

            for phrase in overlap_after:
                f.write(f"- {phrase}\n")

        f.write("\n\nPhrase counts after filtering\n")
        f.write("=" * 100 + "\n")
        f.write(
            phrase_split_country_counts(df_after).to_string(index=False)
        )

    logger.info(f"Phrase overlap report → {output_txt}")


def write_accent_stats_txt(df: pd.DataFrame, output_txt: Path) -> None:
    """
    Optional plain text stats table with:

        accent,total,train,dev,ratio_dev_train,test
    """
    output_txt = Path(output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    stats = build_accent_stats(df)

    with output_txt.open("w", encoding="utf-8") as f:
        f.write(stats.to_csv(index=False))

    logger.info(f"TXT/CSV-style accent stats → {output_txt}")


def import_corpus(
    corpus_dir: Path,
    output_parquet: Path,
    audio_dir: Path,
    output_csv: Path | None = None,
    output_latex: Path | None = None,
    overlap_report: Path | None = None,
    accent_stats_txt: Path | None = None,
    max_speakers: int | None = None,
    dev_ratio: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    corpus_dir = Path(corpus_dir)
    output_parquet = Path(output_parquet)
    audio_dir = Path(audio_dir)

    audio_dir.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped = 0

    country_dirs = sorted([d for d in corpus_dir.iterdir() if d.is_dir()])

    logger.info(f"Found {len(country_dirs)} country folder(s) in {corpus_dir}")
    logger.info(f"Speaker split config: dev_ratio={dev_ratio}, seed={seed}")
    logger.info(f"Test countries: {sorted(TEST_COUNTRIES)}")

    for country_dir in country_dirs:
        country_key = country_dir.name.lower().strip()
        country_label = COUNTRY_FOLDER_MAP.get(country_key, country_dir.name)
        speaker_dirs = sorted([d for d in country_dir.iterdir() if d.is_dir()])

        if max_speakers is not None:
            speaker_dirs = speaker_dirs[:max_speakers]

        split_map = make_speaker_split_map(
            country_label=country_label,
            speaker_dirs=speaker_dirs,
            dev_ratio=dev_ratio,
            seed=seed,
        )

        for speaker_dir in tqdm(speaker_dirs, desc=f"{country_label}"):
            speaker_id = speaker_dir.name

            if speaker_id not in split_map:
                logger.warning(f"No split assigned for speaker {speaker_id} — skipping")
                skipped += 1
                continue

            split = split_map[speaker_id]
            wav_files = sorted(speaker_dir.glob("*.wav"))

            for wav_path in wav_files:
                utt_id = wav_path.stem
                txt_path = wav_path.with_suffix(".txt")
                meta_path = wav_path.with_suffix(".metadata")

                if not txt_path.exists():
                    logger.warning(f"No transcript: {txt_path} — skipping")
                    skipped += 1
                    continue

                transcript = normalize_transcript(
                    txt_path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    ).strip()
                )

                meta = {}
                if meta_path.exists():
                    meta = parse_metadata(meta_path)

                gender = meta.get("SEX", "").strip().lower()
                age_str = meta.get("AGE", "").strip()
                accent = meta.get("ACT", country_label).strip() or country_label
                device = meta.get("MIT", "").strip()
                environment = meta.get("SCC", "").strip()
                duration_str = meta.get("LBR", "").strip()

                try:
                    age = int(age_str)
                except (ValueError, TypeError):
                    age = None

                try:
                    duration_s = round(float(duration_str), 3)
                except (ValueError, TypeError):
                    try:
                        duration_s = round(sf.info(str(wav_path)).duration, 3)
                    except Exception:
                        duration_s = None

                dest_path = audio_dir / f"{utt_id}.wav"

                if not dest_path.exists():
                    shutil.copy2(wav_path, dest_path)

                rows.append(
                    {
                        "speaker_id": speaker_id,
                        "gender": gender,
                        "age": age,
                        "accent": accent,
                        "country": country_label,
                        "device": device,
                        "environment": environment,
                        "duration_s": duration_s,
                        "utterance_id": utt_id,
                        "transcript": transcript,
                        "audio_path": str(dest_path.resolve()),
                        "split": split,
                    }
                )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError("No utterances were imported. Check corpus_dir and file layout.")

    # Remove test examples whose transcript appears in train/dev
    df, df_before_filter = apply_phrase_leakage_lock(
        df=df,
        phrase_col="transcript",
    )

    # Summary
    logger.info(f"Total utterances : {len(df):,}")
    logger.info(f"Skipped          : {skipped:,}")
    logger.info(f"Speakers         : {df['speaker_id'].nunique()}")
    logger.info(f"Countries        : {df['country'].nunique()}")

    if "duration_s" in df.columns:
        logger.info(f"Duration total   : {df['duration_s'].sum() / 3600:.1f} h")

    logger.info(
        "\nUtterances per country + split:\n"
        f"{df.groupby(['country', 'split']).size().to_string()}"
    )

    logger.info(
        "\nUtterances per split:\n"
        f"{df.groupby('split').size().to_string()}"
    )

    speaker_summary = (
        df[["country", "speaker_id", "split"]]
        .drop_duplicates()
        .groupby(["country", "split"])
        .size()
        .unstack(fill_value=0)
    )

    for col in ["train", "dev", "test"]:
        if col not in speaker_summary.columns:
            speaker_summary[col] = 0

    speaker_summary = speaker_summary[["train", "dev", "test"]]
    speaker_summary["total"] = speaker_summary.sum(axis=1)

    logger.info(
        "\nSpeakers per country + split:\n"
        f"{speaker_summary.to_string()}"
    )

    split_speaker_summary = (
        df[["speaker_id", "split"]]
        .drop_duplicates()
        .groupby("split")
        .size()
    )

    logger.info(
        "\nSpeakers per split:\n"
        f"{split_speaker_summary.to_string()}"
    )

    # Extra reports
    if output_latex is not None:
        write_accent_stats_latex(df, output_latex)

    if overlap_report is not None:
        write_phrase_overlap_report(
            df_before=df_before_filter,
            df_after=df,
            output_txt=overlap_report,
            phrase_col="transcript",
        )

    if accent_stats_txt is not None:
        write_accent_stats_txt(df, accent_stats_txt)

    # Save corpus
    df.to_parquet(output_parquet, index=False)
    logger.info(f"Parquet → {output_parquet}")

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"CSV    → {output_csv}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import AESRC2020 into the project format."
    )

    parser.add_argument("--corpus_dir", type=Path, required=True)
    parser.add_argument("--output_parquet", type=Path, required=True)
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=None)

    parser.add_argument(
        "--output_latex",
        type=Path,
        default=None,
        help="Optional LaTeX file containing accent-level split statistics.",
    )

    parser.add_argument(
        "--overlap_report",
        type=Path,
        default=None,
        help="Optional TXT report verifying phrase overlap between train/dev and test.",
    )

    parser.add_argument(
        "--accent_stats_txt",
        type=Path,
        default=None,
        help="Optional TXT/CSV-style file with accent,total,train,dev,ratio_dev_train,test.",
    )

    parser.add_argument(
        "--max_speakers",
        type=int,
        default=None,
        help="Max speakers per country — for quick testing only.",
    )

    parser.add_argument(
        "--dev_ratio",
        type=float,
        default=0.10,
        help="Ratio of speakers per seen accent used for dev.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for speaker-level train/dev split.",
    )

    args = parser.parse_args()

    import_corpus(
        corpus_dir=args.corpus_dir,
        output_parquet=args.output_parquet,
        audio_dir=args.audio_dir,
        output_csv=args.output_csv,
        output_latex=args.output_latex,
        overlap_report=args.overlap_report,
        accent_stats_txt=args.accent_stats_txt,
        max_speakers=args.max_speakers,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )