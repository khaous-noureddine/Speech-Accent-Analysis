"""Summarize matched sequential CTC and CTC--SupCon evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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


def aggregate_seeds(
    pairs: dict[tuple[str, int, str, str], pd.DataFrame],
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], pd.DataFrame]]:
    """Aggregate prediction errors across seeds within each held-out accent.

    Summing seed-specific numerator and denominator values is exactly equivalent
    to the arithmetic mean of seed WERs because every seed is scored on the
    same reference utterances. This is the aggregation used by the paper
    tables before their macro-average over accents.
    """

    grouped: dict[tuple[str, str, str], list[tuple[int, pd.DataFrame]]] = {}
    for (fold, model_seed, decoder, dataset), frame in pairs.items():
        grouped.setdefault((fold, decoder, dataset), []).append((model_seed, frame))

    rows: list[dict[str, object]] = []
    aggregated: dict[tuple[str, str, str], pd.DataFrame] = {}
    for index, ((fold, decoder, dataset), entries) in enumerate(sorted(grouped.items())):
        entries.sort(key=lambda item: item[0])
        seeds = [item[0] for item in entries]
        combined = pd.concat(
            [frame.assign(_seed=model_seed) for model_seed, frame in entries],
            ignore_index=True,
        )
        frame = (
            combined.groupby(["utterance_id", "speaker_id"], as_index=False, sort=True)[
                ["reference_words", "before_errors", "after_errors"]
            ]
            .sum()
        )
        interval = paired_bootstrap_wer_delta(
            frame, seed=seed + index, replicates=replicates
        )
        words = int(frame["reference_words"].sum())
        before_errors = int(frame["before_errors"].sum())
        after_errors = int(frame["after_errors"].sum())
        rows.append({
            "fold": fold,
            "decoder": decoder,
            "dataset": dataset,
            "seed_count": len(seeds),
            "seeds": ",".join(str(value) for value in seeds),
            "utterances": len(frame),
            "reference_words_across_seeds": words,
            "staged_md_ft_wer_percent": 100.0 * before_errors / words,
            "staged_md_ft_cp_supcon_wer_percent": 100.0 * after_errors / words,
            "delta_wer_points": 100.0 * (after_errors - before_errors) / words,
            "bootstrap_lower_95": interval["lower_95"],
            "bootstrap_upper_95": interval["upper_95"],
        })
        aggregated[(fold, decoder, dataset)] = frame
    return pd.DataFrame(rows), aggregated


def macro_summary(
    by_accent: pd.DataFrame,
    aggregated: dict[tuple[str, str, str], pd.DataFrame],
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Macro-average accent means, with a stratified paired utterance bootstrap."""

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for (decoder, dataset), group in by_accent.groupby(["decoder", "dataset"], sort=True):
        group = group.sort_values("fold")
        accent_deltas: list[float] = []
        frames = [aggregated[(row.fold, decoder, dataset)] for row in group.itertuples()]
        for frame in frames:
            words = frame["reference_words"].to_numpy(int)
            accent_deltas.append(
                100.0
                * (frame["after_errors"].sum() - frame["before_errors"].sum())
                / words.sum()
            )

        draws = np.zeros(replicates, dtype=float)
        for frame in frames:
            words = frame["reference_words"].to_numpy(int)
            before = frame["before_errors"].to_numpy(int)
            after = frame["after_errors"].to_numpy(int)
            selections = rng.integers(0, len(frame), size=(replicates, len(frame)))
            denominators = words[selections].sum(axis=1)
            draws += (
                100.0
                * (after[selections].sum(axis=1) - before[selections].sum(axis=1))
                / denominators
            )
        draws /= len(frames)
        rows.append({
            "decoder": decoder,
            "dataset": dataset,
            "accent_count": len(group),
            "accents": ",".join(group["fold"]),
            "staged_md_ft_wer_percent": group["staged_md_ft_wer_percent"].mean(),
            "staged_md_ft_cp_supcon_wer_percent": group[
                "staged_md_ft_cp_supcon_wer_percent"
            ].mean(),
            "delta_wer_points": float(np.mean(accent_deltas)),
            "bootstrap_lower_95": float(np.quantile(draws, 0.025)),
            "bootstrap_upper_95": float(np.quantile(draws, 0.975)),
        })
    return pd.DataFrame(rows)


def plot_l2_forest(by_accent: pd.DataFrame, output_dir: Path) -> None:
    """Plot paired WER deltas after seed aggregation for the six L2 folds."""

    import matplotlib.pyplot as plt

    data = by_accent.loc[by_accent["dataset"] == "l2_arctic"].copy()
    if data.empty:
        return
    decoders = [
        decoder
        for decoder in ("greedy", "beam_4gram")
        if decoder in set(data["decoder"])
    ]
    figure, axes = plt.subplots(
        1, len(decoders), figsize=(6.8 * len(decoders), 3.5), sharey=True
    )
    axes = np.atleast_1d(axes)
    for axis, decoder in zip(axes, decoders, strict=True):
        current = data.loc[data["decoder"] == decoder].sort_values("fold")
        y = np.arange(len(current))
        delta = current["delta_wer_points"].to_numpy(float)
        lower = current["bootstrap_lower_95"].to_numpy(float)
        upper = current["bootstrap_upper_95"].to_numpy(float)
        axis.errorbar(
            delta, y, xerr=np.vstack((delta - lower, upper - delta)),
            fmt="o", color="#c63d32", ecolor="#555555", capsize=3,
        )
        axis.axvline(0.0, color="#222222", linewidth=1)
        axis.set_yticks(y, [value.replace("_", " ").title() for value in current["fold"]])
        axis.invert_yaxis()
        axis.set_xlabel("CP-SupCon minus MD-FT WER (points)")
        axis.set_title("Greedy" if decoder == "greedy" else "4-gram beam")
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("Held-out L2-ARCTIC: paired seed-averaged WER differences")
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(output_dir / f"l2_arctic_paired_delta_forest.{extension}", dpi=200)
    plt.close(figure)


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
    pairs: dict[tuple[str, int, str, str], pd.DataFrame] = {}
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
        pairs[(context["fold"], context["seed"], context["decoder"], context["dataset"])] = paired

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
    by_accent, aggregated = aggregate_seeds(
        pairs, replicates=args.bootstrap_replicates, seed=args.seed + 10_000
    )
    macro = macro_summary(
        by_accent, aggregated, replicates=args.bootstrap_replicates, seed=args.seed + 20_000
    )
    by_accent.to_csv(args.output_dir / "paired_bootstrap_by_accent.csv", index=False)
    macro.to_csv(args.output_dir / "paired_bootstrap_macro.csv", index=False)
    plot_l2_forest(by_accent, args.output_dir)
    print(f"Matched comparisons: {len(summaries)}")
    print(f"Speaker rows: {len(speaker_rows)}")
    print(f"Skipped comparisons: {len(skipped)}")
    print(f"Seed-averaged accent comparisons: {len(by_accent)}")
    print(f"Macro summaries: {len(macro)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
