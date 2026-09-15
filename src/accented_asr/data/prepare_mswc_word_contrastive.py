"""Join MSWC English words to pinned Common Voice accent metadata."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

from accented_asr.data.prepare_word_contrastive import normalize_word, portable_path


def source_filename(link: str) -> str:
    name = Path(str(link)).name
    stem = re.sub(r"__\d+$", "", Path(name).stem)
    return f"{stem}.mp3"


def load_mswc(mswc_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ("train", "dev", "test"):
        path = mswc_dir / f"en_{split}.csv"
        frame = pd.read_csv(path)
        required = {"LINK", "WORD", "VALID", "SPEAKER"}
        if not required <= set(frame.columns):
            raise ValueError(f"{path} lacks columns {sorted(required - set(frame.columns))}")
        frame = frame.loc[frame["VALID"].astype(str).str.casefold().eq("true")].copy()
        frame["mswc_original_split"] = split
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["source_filename"] = result["LINK"].map(source_filename)
    result["normalized_word"] = result["WORD"].map(normalize_word)
    return result.loc[result["normalized_word"].ne("")].copy()


def load_common_voice(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    accent_column = "accent" if "accent" in frame else "accents" if "accents" in frame else None
    required = {"client_id", "path", "sentence"}
    if accent_column is None or not required <= set(frame.columns):
        raise ValueError("Common Voice metadata requires path, client_id, sentence and accent(s).")
    frame = frame.rename(columns={accent_column: "accent"})
    frame["source_filename"] = frame["path"].map(lambda value: Path(str(value)).name)
    if frame["source_filename"].duplicated().any():
        raise ValueError("Common Voice validated.tsv contains duplicate clip filenames.")
    return frame[["source_filename", "client_id", "accent", "sentence"]]


def audio_index(mswc_dir: Path) -> dict[tuple[str, str], Path]:
    index = {}
    for path in mswc_dir.rglob("*.opus"):
        key = (normalize_word(path.parent.name), path.name)
        if key in index:
            raise ValueError(f"Ambiguous MSWC word audio key: {key}")
        index[key] = path
    if not index:
        raise FileNotFoundError(f"No OPUS files found below {mswc_dir}")
    return index


def build(args: argparse.Namespace) -> None:
    root = args.repository_root.resolve()
    mswc = load_mswc(args.mswc_dir)
    common_voice = load_common_voice(args.common_voice_tsv)
    merged = mswc.merge(common_voice, on="source_filename", how="left", validate="many_to_one")
    merged["accent"] = merged["accent"].fillna("").astype(str).str.strip().str.casefold()
    merged["client_id"] = merged["client_id"].fillna("").astype(str)
    matched = merged["client_id"].ne("")
    speaker_consistent = merged.loc[matched, "SPEAKER"].astype(str).eq(
        merged.loc[matched, "client_id"].astype(str)
    )
    if not speaker_consistent.all():
        raise ValueError(f"Speaker mismatch for {(~speaker_consistent).sum()} joined rows.")
    merged = merged.loc[matched & merged["accent"].ne("")].copy()

    speakers = sorted(merged["client_id"].unique())
    rng = random.Random(args.seed)
    rng.shuffle(speakers)
    dev_speakers = set(speakers[: max(1, round(len(speakers) * args.dev_fraction))])
    merged["split"] = merged["client_id"].map(
        lambda speaker: "dev" if speaker in dev_speakers else "train"
    )

    train = merged.loc[merged["split"] == "train"]
    coverage = (
        train.groupby(["normalized_word", "accent"])["client_id"].nunique()
        .rename("speakers").reset_index()
    )
    coverage = coverage.loc[coverage["speakers"] >= args.min_speakers_per_accent]
    vocabulary = (
        coverage.groupby("normalized_word")
        .agg(n_accents=("accent", "nunique"), min_speakers_per_accent=("speakers", "min"))
        .reset_index()
    )
    vocabulary = vocabulary.loc[vocabulary["n_accents"] >= args.min_accents].copy()
    merged = merged.loc[merged["normalized_word"].isin(vocabulary["normalized_word"])].copy()
    if merged.empty:
        raise ValueError("No word satisfies the requested accent/speaker coverage.")

    paths = audio_index(args.mswc_dir)
    merged["audio_path"] = [
        portable_path(paths[(word, Path(link).name)], root)
        for word, link in zip(merged["normalized_word"], merged["LINK"])
    ]
    output = pd.DataFrame({
        "dataset": "mswc_common_voice_en",
        "language": "en",
        "accent": merged["accent"],
        "speaker_id": merged["client_id"],
        "utterance_id": merged["source_filename"].map(lambda value: Path(value).stem),
        "source_transcript": merged["sentence"],
        "word": merged["WORD"],
        "normalized_word": merged["normalized_word"],
        "audio_path": merged["audio_path"],
        "start_s": 0.0,
        "end_s": 1.0,
        "duration_s": 1.0,
        "alignment_source": "mswc_mfa",
        "split": merged["split"],
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output_dir / "corpus.parquet", index=False)
    vocabulary.sort_values("normalized_word").to_csv(args.output_dir / "vocabulary.csv", index=False)
    stats = output.groupby("split").agg(
        occurrences=("audio_path", "size"), speakers=("speaker_id", "nunique"),
        accents=("accent", "nunique"), words=("normalized_word", "nunique"),
    ).reset_index()
    stats.to_csv(args.output_dir / "split_stats.csv", index=False)
    report = {
        "raw_mswc_rows": len(mswc), "joined_rows": int(matched.sum()),
        "join_rate": float(matched.mean()),
        "common_voice_metadata_release": "21.0",
        "rows_with_accent": len(merged), "selected_words": len(vocabulary),
        "speaker_overlap": len(
            set(output.loc[output.split == "train", "speaker_id"])
            & set(output.loc[output.split == "dev", "speaker_id"])
        ),
        "min_accents": args.min_accents,
        "min_speakers_per_accent": args.min_speakers_per_accent,
        "seed": args.seed,
    }
    report["status"] = "passed" if report["speaker_overlap"] == 0 else "failed"
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if report["status"] != "passed":
        raise ValueError("Speaker leakage detected.")
    (args.output_dir / "_SUCCESS").write_text("MSWC/Common Voice word corpus ready\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mswc-dir", type=Path, required=True)
    parser.add_argument("--common-voice-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--min-accents", type=int, default=6)
    parser.add_argument("--min-speakers-per-accent", type=int, default=3)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
