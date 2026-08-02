#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import evaluate
import pandas as pd
from loguru import logger


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_refs_preds(
    df: pd.DataFrame,
    ref_col: str = "reference",
    pred_col: str = "prediction",
    normalize: bool = True,
) -> tuple[list[str], list[str], dict]:
    refs: list[str] = []
    preds: list[str] = []

    n_total = len(df)
    n_empty_ref = 0
    n_empty_pred = 0

    for r, p in zip(df[ref_col], df[pred_col]):
        if pd.isna(r) or not str(r).strip():
            n_empty_ref += 1
            continue

        ref = str(r).strip()

        if pd.isna(p):
            pred = ""
        else:
            pred = str(p).strip()

        if pred == "[EMPTY]":
            pred = ""

        if not pred:
            n_empty_pred += 1

        if normalize:
            ref = normalize_text(ref)
            pred = normalize_text(pred)

        refs.append(ref)
        preds.append(pred)

    stats = {
        "n_samples": n_total,
        "n_used": len(refs),
        "n_empty_ref_skipped": n_empty_ref,
        "n_empty_predictions": n_empty_pred,
    }

    return refs, preds, stats


def compute_wer_hf(
    df: pd.DataFrame,
    wer_metric,
    ref_col: str = "reference",
    pred_col: str = "prediction",
    normalize: bool = True,
) -> tuple[float, dict]:
    refs, preds, stats = prepare_refs_preds(
        df=df,
        ref_col=ref_col,
        pred_col=pred_col,
        normalize=normalize,
    )

    if not refs:
        return float("nan"), stats

    wer = wer_metric.compute(references=refs, predictions=preds)
    return float(wer), stats


def process_csv(
    csv_path: Path,
    wer_metric,
    group_col: Optional[str] = None,
    ref_col: str = "reference",
    pred_col: str = "prediction",
    normalize: bool = True,
) -> list[dict]:
    df = pd.read_csv(csv_path)

    if ref_col not in df.columns or pred_col not in df.columns:
        logger.warning(f"Skipping {csv_path} — missing columns")
        return []

    model = csv_path.stem
    dataset = csv_path.parent.name

    results: list[dict] = []

    overall_wer, stats = compute_wer_hf(
        df=df,
        wer_metric=wer_metric,
        ref_col=ref_col,
        pred_col=pred_col,
        normalize=normalize,
    )

    results.append({
        "model": model,
        "dataset": dataset,
        "group": "ALL",
        "wer": overall_wer,
        "n_samples": stats["n_samples"],
        "n_used": stats["n_used"],
        "n_empty_ref_skipped": stats["n_empty_ref_skipped"],
        "n_empty_predictions": stats["n_empty_predictions"],
    })

    logger.info(
        f"{model} × {dataset} → WER = {overall_wer * 100:.2f}% "
        f"({stats['n_used']} used, {stats['n_empty_predictions']} empty preds)"
    )

    if group_col and group_col in df.columns:
        for group_name, group_df in df.groupby(group_col):
            group_wer, group_stats = compute_wer_hf(
                df=group_df,
                wer_metric=wer_metric,
                ref_col=ref_col,
                pred_col=pred_col,
                normalize=normalize,
            )

            results.append({
                "model": model,
                "dataset": dataset,
                "group": str(group_name),
                "wer": group_wer,
                "n_samples": group_stats["n_samples"],
                "n_used": group_stats["n_used"],
                "n_empty_ref_skipped": group_stats["n_empty_ref_skipped"],
                "n_empty_predictions": group_stats["n_empty_predictions"],
            })

    return results


def generate_latex_table(pivot: pd.DataFrame) -> str:
    datasets = list(pivot.columns)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Word error rate (\%) across evaluation datasets. Lower is better.}",
        r"\label{tab:wer_results}",
        r"\small",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{l" + "c" * len(datasets) + "}",
        r"\toprule",
        "Model & " + " & ".join(datasets) + r" \\",
        r"\midrule",
    ]

    for model in pivot.index:
        vals = []
        for dataset in datasets:
            value = pivot.loc[model, dataset]
            vals.append("--" if pd.isna(value) else f"{value:.2f}")
        lines.append(f"{model} & " + " & ".join(vals) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcriptions_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--group_col", type=str, default=None)
    parser.add_argument("--ref_col", type=str, default="reference")
    parser.add_argument("--pred_col", type=str, default="prediction")
    parser.add_argument("--no_normalize", action="store_true")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(args.transcriptions_dir.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {args.transcriptions_dir}")

    logger.info(f"{len(csv_files)} CSV files found")
    logger.info("Loading HuggingFace WER metric...")
    wer_metric = evaluate.load("wer")

    all_results: list[dict] = []

    for csv_path in csv_files:
        all_results.extend(
            process_csv(
                csv_path=csv_path,
                wer_metric=wer_metric,
                group_col=args.group_col,
                ref_col=args.ref_col,
                pred_col=args.pred_col,
                normalize=not args.no_normalize,
            )
        )

    if not all_results:
        raise RuntimeError("No WER results computed.")

    df = pd.DataFrame(all_results)

    results_path = args.output_dir / "results.csv"
    df.to_csv(results_path, index=False)
    logger.info(f"Saved → {results_path}")

    overall = df[df["group"] == "ALL"].copy()
    overall["wer_pct"] = overall["wer"] * 100

    pivot = overall.pivot(index="model", columns="dataset", values="wer_pct")

    summary_path = args.output_dir / "results_summary.csv"
    pivot.to_csv(summary_path)
    logger.info(f"Saved → {summary_path}")

    tex_path = args.output_dir / "results.tex"
    tex_path.write_text(generate_latex_table(pivot))
    logger.info(f"Saved → {tex_path}")

    print("\n=== RESULTS SUMMARY WER (%) ===")
    print(pivot.round(2))


if __name__ == "__main__":
    main()