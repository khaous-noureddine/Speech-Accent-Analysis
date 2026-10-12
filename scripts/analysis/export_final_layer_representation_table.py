"""Export a compact final-layer representation table for the paper."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

MODEL_ORDER = (
    "ctc_only",
    "accent_dat",
    "accent_mtl",
    "multidomain_ctc",
    "augmented_view_supcon",
    "word_supcon",
    "utterance_supcon",
    "md_ft_cp_supcon",
)
MODEL_LABELS = {
    "ctc_only": "ASR-only",
    "accent_dat": "Accent-DAT",
    "accent_mtl": "Accent-MTL",
    "multidomain_ctc": "Multi-domain FT",
    "augmented_view_supcon": "Aug-SupCon",
    "word_supcon": r"CP-SupCon$_{\mathrm{word}}$",
    "utterance_supcon": r"CP-SupCon$_{\mathrm{utt}}$",
    "md_ft_cp_supcon": r"MD-FT + CP-SupCon$_{\mathrm{utt}}$",
}
METRICS = (
    "alignment_ratio",
    "recall_at_1",
    "recall_at_5",
    "map",
)


def encoder_layer(space: str) -> int:
    match = re.fullmatch(r"backbone_layer_(\d+)", str(space))
    if not match:
        raise ValueError(f"Unexpected representation space: {space}")
    return int(match.group(1))


def format_value(value: float, metric: str) -> str:
    if metric == "alignment_ratio":
        return f"{value:.3f}"
    return f"{100.0 * value:.2f}"


def render_table(frame: pd.DataFrame, condition: str, layer: int) -> str:
    data = frame.copy()
    data["layer"] = data["space"].map(encoder_layer)
    data = data.loc[(data["condition"] == condition) & (data["layer"] == layer)]
    summary = data.groupby("model", as_index=True)[list(METRICS)].mean()
    folds = data.groupby("model")["accent"].nunique()

    rows = []
    for model in MODEL_ORDER:
        if model not in summary.index:
            continue
        label = MODEL_LABELS[model]
        n = int(folds[model])
        values = " & ".join(
            format_value(float(summary.loc[model, metric]), metric)
            for metric in METRICS
        )
        rows.append(f"{label} & {n} & {values} \\\\")

    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{Mean-centered content geometry at encoder layer {layer} on
held-out-accent L2-ARCTIC. $n$ is the number of available accent folds.
Retrieval candidates are restricted to different speakers.}}
\label{{tab:final-layer-all-models}}
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrr}}
\toprule
Model & $n$ & Ratio $\downarrow$ & R@1 $\uparrow$ & R@5 $\uparrow$ & mAP $\uparrow$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-by-fold", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--condition", default="mean_centered")
    parser.add_argument("--layer", default=24, type=int)
    args = parser.parse_args()

    frame = pd.read_csv(args.metrics_by_fold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_table(frame, args.condition, args.layer), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
