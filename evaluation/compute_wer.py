from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from jiwer import wer as compute_wer_score
from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════
# Core WER
# ═══════════════════════════════════════════════════════════════════════════

# def compute_wer_from_df(df: pd.DataFrame) -> float:
#     refs  = df["reference"].astype(str).tolist()
#     preds = df["prediction"].astype(str).tolist()

#     valid = [(r, p) for r, p in zip(refs, preds) if r.strip()]
#     if not valid:
#         return float("nan")

#     refs_clean  = [r for r, _ in valid]
#     preds_clean = [p for _, p in valid]

#     return compute_wer_score(refs_clean, preds_clean)


def compute_wer_from_df(df: pd.DataFrame) -> float:
    refs  = df["reference"]
    preds = df["prediction"]

    clean_refs = []
    clean_preds = []

    for r, p in zip(refs, preds):

        # skip NaN
        if pd.isna(r) or pd.isna(p):
            continue

        r = str(r).strip()
        p = str(p).strip()

        # skip empty reference
        if not r:
            continue

        # skip empty prediction
        if not p or p == "[EMPTY]":
            continue

        clean_refs.append(r)
        clean_preds.append(p)

    if not clean_refs:
        return float("nan")

    return compute_wer_score(clean_refs, clean_preds)

# ═══════════════════════════════════════════════════════════════════════════
# Processing one CSV
# ═══════════════════════════════════════════════════════════════════════════

def process_csv(
    csv_path: Path,
    group_col: Optional[str] = None,
) -> list[dict]:

    df = pd.read_csv(csv_path)

    if "reference" not in df.columns or "prediction" not in df.columns:
        logger.warning(f"Skipping {csv_path} — missing columns")
        return []

    model   = csv_path.stem
    dataset = csv_path.parent.name

    results = []

    # ── Global WER ───────────────────────────────────────────────────────
    overall_wer = compute_wer_from_df(df)

    results.append({
        "model": model,
        "dataset": dataset,
        "group": "ALL",
        "wer": overall_wer,
        "n_samples": len(df),
    })

    logger.info(f"{model} × {dataset} → WER = {overall_wer:.4f}")

    # ── Per-group WER ────────────────────────────────────────────────────
    if group_col and group_col in df.columns:
        for group_name, group_df in df.groupby(group_col):
            group_wer = compute_wer_from_df(group_df)

            results.append({
                "model": model,
                "dataset": dataset,
                "group": str(group_name),
                "wer": group_wer,
                "n_samples": len(group_df),
            })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# LaTeX
# ═══════════════════════════════════════════════════════════════════════════

def generate_latex_table(pivot: pd.DataFrame) -> str:
    datasets = list(pivot.columns)
    models   = list(pivot.index)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Word Error Rate (\%) across datasets.}",
        r"\label{tab:wer}",
        r"\begin{tabular}{l" + "c" * len(datasets) + "}",
        r"\toprule",
        "Model & " + " & ".join(datasets) + r" \\",
        r"\midrule",
    ]

    for model in models:
        vals = []
        for d in datasets:
            v = pivot.loc[model, d]
            vals.append("--" if pd.isna(v) else f"{v:.1f}")
        lines.append(f"{model} & " + " & ".join(vals) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcriptions_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--group_col", type=str, default=None)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 🔥 IMPORTANT: recursive search
    csv_files = sorted(args.transcriptions_dir.rglob("*.csv"))

    if not csv_files:
        logger.error(f"No CSV found in {args.transcriptions_dir}")
        return

    logger.info(f"{len(csv_files)} CSV files found")

    all_results = []

    for csv_path in csv_files:
        results = process_csv(csv_path, group_col=args.group_col)
        all_results.extend(results)

    if not all_results:
        logger.error("No results computed")
        return

    df = pd.DataFrame(all_results)

    # ── Save full results ────────────────────────────────────────────────
    results_path = args.output_dir / "results.csv"
    df.to_csv(results_path, index=False)
    logger.info(f"Saved → {results_path}")

    # ── Pivot (global only) ─────────────────────────────────────────────
    overall = df[df["group"] == "ALL"].copy()
    overall["wer_pct"] = overall["wer"] * 100

    pivot = overall.pivot(index="model", columns="dataset", values="wer_pct")

    summary_path = args.output_dir / "results_summary.csv"
    pivot.to_csv(summary_path)
    logger.info(f"Saved → {summary_path}")

    print("\n=== RESULTS ===")
    print(pivot.round(2))

    # ── LaTeX ───────────────────────────────────────────────────────────
    latex = generate_latex_table(pivot.round(1))
    tex_path = args.output_dir / "results.tex"
    tex_path.write_text(latex)

    logger.info(f"LaTeX saved → {tex_path}")


if __name__ == "__main__":
    main()