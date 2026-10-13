"""Summarize matched sequential CTC and CTC--SupCon evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from accented_asr.error_analysis.analysis import paired_bootstrap_wer_delta


REQUIRED_COLUMNS = {"utterance_id", "speaker_id", "reference_words", "errors"}


def read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    if frame.duplicated(["utterance_id", "speaker_id"]).any():
        raise ValueError(f"{path} has duplicate utterance/speaker pairs")
    return frame.loc[:, sorted(REQUIRED_COLUMNS)].copy()


def summarize_pair(
    baseline_path: Path, comparison_path: Path, *, replicates: int, seed: int
) -> tuple[dict[str, object], pd.DataFrame]:
    before = read_predictions(baseline_path).rename(
        columns={"reference_words": "before_words", "errors": "before_errors"}
    )
    after = read_predictions(comparison_path).rename(
        columns={"reference_words": "after_words", "errors": "after_errors"}
    )
    merged = before.merge(after, on=["utterance_id", "speaker_id"], validate="one_to_one")
    if len(merged) != len(before) or len(merged) != len(after):
        raise ValueError(f"Prediction sets differ: {baseline_path} vs {comparison_path}")
    if not (merged["before_words"] == merged["after_words"]).all():
        raise ValueError(f"Reference word counts differ: {baseline_path} vs {comparison_path}")

    paired = merged.rename(columns={"before_words": "reference_words"})
    words = int(paired["reference_words"].sum())
    before_errors = int(paired["before_errors"].sum())
    after_errors = int(paired["after_errors"].sum())
    interval = paired_bootstrap_wer_delta(paired, seed=seed, replicates=replicates)
    summary = {
        "utterances": len(paired),
        "reference_words": words,
        "ctc_wer_percent": 100.0 * before_errors / words,
        "ctc_supcon_wer_percent": 100.0 * after_errors / words,
        "delta_wer_points": 100.0 * (after_errors - before_errors) / words,
        "bootstrap_lower_95": interval["lower_95"],
        "bootstrap_upper_95": interval["upper_95"],
    }
    return summary, paired


def metadata(path: Path, root: Path) -> dict[str, object]:
    fold, marker, objective, seed, decoder, dataset, filename = path.relative_to(root).parts
    if (marker, objective, filename) != ("evaluation", "ctc", "predictions.parquet"):
        raise ValueError(f"Unexpected sequential prediction path: {path}")
    return {
        "fold": fold,
        "seed": int(seed.removeprefix("seed=")),
        "decoder": decoder,
        "dataset": dataset,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("experiments/sequential-l2-adaptation/librispeech-100h/wav2vec2-large-lv60"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/sequential-l2-statistics")
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    speaker_rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    pattern = "*/evaluation/ctc/seed=*/*/*/predictions.parquet"

    for baseline_path in sorted(args.input_root.glob(pattern)):
        comparison_path = Path(str(baseline_path).replace("/evaluation/ctc/", "/evaluation/ctc_supcon/"))
        if not comparison_path.is_file():
            skipped.append({"ctc_predictions": str(baseline_path), "reason": "missing ctc_supcon predictions"})
            continue
        context = metadata(baseline_path, args.input_root)
        try:
            summary, paired = summarize_pair(
                baseline_path, comparison_path,
                replicates=args.bootstrap_replicates,
                seed=args.seed + len(summaries),
            )
        except (OSError, ValueError) as error:
            skipped.append({"ctc_predictions": str(baseline_path), "reason": str(error)})
            continue
        summaries.append({**context, **summary})

        for speaker_id, group in paired.groupby("speaker_id", sort=True):
            words = int(group["reference_words"].sum())
            speaker_rows.append({
                **context,
                "speaker_id": speaker_id,
                "utterances": len(group),
                "reference_words": words,
                "ctc_wer_percent": 100.0 * group["before_errors"].sum() / words,
                "ctc_supcon_wer_percent": 100.0 * group["after_errors"].sum() / words,
                "delta_wer_points": 100.0 * (group["after_errors"].sum() - group["before_errors"].sum()) / words,
            })

    pd.DataFrame(summaries).to_csv(args.output_dir / "paired_bootstrap_summary.csv", index=False)
    pd.DataFrame(speaker_rows).to_csv(args.output_dir / "wer_by_speaker.csv", index=False)
    pd.DataFrame(skipped).to_csv(args.output_dir / "skipped_pairs.csv", index=False)
    print(f"Matched comparisons: {len(summaries)}")
    print(f"Speaker rows: {len(speaker_rows)}")
    print(f"Skipped comparisons: {len(skipped)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
