"""
eval/compute_wer.py

Read transcription CSVs produced by transcribe.py, compute WER per
(model, dataset) pair and per accent group, and generate:
  - results.csv         : flat table of all WER scores
  - results_summary.csv : pivot table (models × datasets)
  - results.tex         : LaTeX table ready for the paper

Usage
-----
  python eval/compute_wer.py \\
      --transcriptions_dir evaluation/transcriptions \\
      --output_dir evaluation/results

  # With per-group breakdown
  python eval/compute_wer.py \\
      --transcriptions_dir evaluation/transcriptions \\
      --output_dir evaluation/results \\
      --group_col native_language
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from jiwer import wer as compute_wer_score
from loguru import logger


def compute_wer_from_df(df: pd.DataFrame) -> float:
    """Compute WER from a DataFrame with 'reference' and 'prediction' columns."""
    refs  = df["reference"].tolist()
    preds = df["prediction"].tolist()

    # Filter out empty references
    valid = [(r, p) for r, p in zip(refs, preds) if r.strip()]
    if not valid:
        return float("nan")

    refs_clean  = [r for r, _ in valid]
    preds_clean = [p for _, p in valid]

    return compute_wer_score(refs_clean, preds_clean)


def parse_csv_filename(csv_path: Path) -> tuple[str, str]:
    """
    Extract model_label and dataset_name from filename.
    Format: {model_label}__{dataset_name}.csv
    """
    stem = csv_path.stem
    if "__" in stem:
        parts = stem.split("__", 1)
        return parts[0], parts[1]
    return stem, "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute WER from transcription CSVs and generate result tables."
    )
    parser.add_argument("--transcriptions_dir", type=Path, required=True,
                        help="Directory containing CSV files from transcribe.py.")
    parser.add_argument("--output_dir", type=Path, default=Path("evaluation/results"),
                        help="Where to save results.csv, results_summary.csv, results.tex.")
    parser.add_argument("--group_col", type=str, default=None,
                        help="Column for per-group WER breakdown (e.g. native_language).")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(args.transcriptions_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {args.transcriptions_dir}")
        return

    logger.info(f"Found {len(csv_files)} transcription CSV(s)")

    # ── Compute WER for each CSV ──────────────────────────────────────────
    all_results: list[dict] = []

    for csv_path in csv_files:
        model_label, dataset_name = parse_csv_filename(csv_path)
        df = pd.read_csv(csv_path)

        if "reference" not in df.columns or "prediction" not in df.columns:
            logger.warning(f"Skipping {csv_path.name} — missing reference/prediction columns")
            continue

        # Overall WER
        overall_wer = compute_wer_from_df(df)
        logger.info(f"  {model_label} × {dataset_name} → WER = {overall_wer:.4f}")

        all_results.append({
            "model":    model_label,
            "dataset":  dataset_name,
            "group":    "ALL",
            "wer":      overall_wer,
            "n_samples": len(df),
        })

        # Per-group WER
        group_col = args.group_col
        if group_col and group_col in df.columns:
            for group_name, group_df in df.groupby(group_col):
                group_wer = compute_wer_from_df(group_df)
                all_results.append({
                    "model":    model_label,
                    "dataset":  dataset_name,
                    "group":    str(group_name),
                    "wer":      group_wer,
                    "n_samples": len(group_df),
                })

    if not all_results:
        logger.error("No results computed.")
        return

    # ── Save detailed results ─────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_path = args.output_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Detailed results → {results_path}")

    # ── Pivot table: models × datasets (overall WER only) ─────────────────
    overall_df = results_df[results_df["group"] == "ALL"].copy()
    overall_df["wer_pct"] = (overall_df["wer"] * 100).round(2)

    pivot = overall_df.pivot_table(
        index="model",
        columns="dataset",
        values="wer_pct",
        aggfunc="first",
    )
    pivot = pivot.fillna(-1)

    summary_path = args.output_dir / "results_summary.csv"
    pivot.to_csv(summary_path)
    logger.info(f"Summary table → {summary_path}")

    # Print to console
    logger.info("\n" + pivot.to_string())

    # ── Generate LaTeX table ──────────────────────────────────────────────
    latex = generate_latex_table(pivot)
    tex_path = args.output_dir / "results.tex"
    tex_path.write_text(latex, encoding="utf-8")
    logger.info(f"LaTeX table → {tex_path}")
    logger.info(f"\n{latex}")


def generate_latex_table(pivot: pd.DataFrame) -> str:
    """
    Generate a LaTeX table from the pivot DataFrame.

    Output format:
        \\begin{table}[htbp]
        \\centering
        \\caption{Word Error Rate (\\%) across evaluation datasets.}
        \\begin{tabular}{l ccc}
        \\toprule
        Model & Dataset1 & Dataset2 & ... \\\\
        \\midrule
        model_A & 12.3 & 8.1 & ... \\\\
        ...
        \\bottomrule
        \\end{tabular}
        \\end{table}
    """
    datasets = list(pivot.columns)
    models   = list(pivot.index)
    n_cols   = len(datasets)

    # Header
    col_spec = "l " + "c " * n_cols
    header_cols = " & ".join([_latex_escape(d) for d in datasets])

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Word Error Rate (\%) across evaluation datasets.}",
        r"\label{tab:wer_results}",
        f"\\begin{{tabular}}{{{col_spec.strip()}}}",
        r"\toprule",
        f"Model & {header_cols} \\\\",
        r"\midrule",
    ]

    # Data rows
    for model in models:
        row_values = []
        for dataset in datasets:
            val = pivot.loc[model, dataset]
            if val < 0:
                row_values.append("--")
            else:
                row_values.append(f"{val:.1f}")
        row_str = " & ".join(row_values)
        lines.append(f"{_latex_escape(model)} & {row_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def _latex_escape(s: str) -> str:
    """Escape underscores and special chars for LaTeX."""
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


if __name__ == "__main__":
    main() 