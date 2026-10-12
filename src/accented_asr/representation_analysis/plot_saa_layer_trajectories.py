"""Plot layer-wise L1 structure measured on the fixed SAA passage."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


MODELS = {
    "asr_only": ("ASR-only", "#4C566A", "--"),
    "multidomain_ctc": ("MD-FT", "#5E81AC", "-."),
    "utterance_supcon": ("CP-SupCon", "#007C83", "-"),
    "md_ft_cp_supcon": ("MD-FT + CP-SupCon", "#BF3A30", "-"),
}
METRICS = {
    "l1_separation_gap": "L1 separation gap ↓",
    "same_l1_at_1": "Same-L1 NN@1 (%) ↓",
    "same_l1_at_5": "Same-L1 NN@5 (%) ↓",
    "probe_accuracy": "L1 probe accuracy (%) ↓",
    "probe_macro_f1": "L1 probe macro F1 (%) ↓",
}


def layer(space: str) -> int:
    match = re.fullmatch(r"backbone_layer_(\d+)", str(space))
    if not match:
        raise ValueError(f"Unexpected space: {space}")
    return int(match.group(1))


def render_condition(
    frame: pd.DataFrame,
    condition: str,
    output_dir: Path,
    layout: str,
) -> None:
    """Render three SAA metrics in one row at column or page width."""
    selected = (
        "l1_separation_gap",
        "same_l1_at_1",
        "probe_accuracy",
    )
    present = [model for model in MODELS if model in set(frame["model"])]
    column_layout = layout == "column"
    figsize = (3.45, 1.85) if column_layout else (9.0, 2.8)
    figure, axes = plt.subplots(1, len(selected), figsize=figsize, sharex=True)
    for axis, metric in zip(axes, selected, strict=True):
        title = METRICS[metric]
        scale = 100.0 if metric != "l1_separation_gap" else 1.0
        summary = (
            frame.groupby(["condition", "model", "layer"], as_index=False)[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        for model in present:
            values = summary.loc[
                (summary["condition"] == condition)
                & (summary["model"] == model)
            ].sort_values("layer")
            x = values["layer"].to_numpy()
            mean = values["mean"].to_numpy() * scale
            std = values["std"].fillna(0).to_numpy() * scale
            label, color, linestyle = MODELS[model]
            folds = frame.loc[frame["model"] == model, "fold"].nunique()
            axis.plot(
                x,
                mean,
                color=color,
                linestyle=linestyle,
                marker="o",
                markersize=2.3,
                linewidth=1.5,
                label=f"{label} (n={folds})",
            )
            axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)
        axis.set_title(
            title,
            fontsize=5.8 if column_layout else 9,
            fontweight="normal",
        )
        axis.set_xticks([1, 4, 8, 12, 16, 20, 24])
        axis.set_xlim(1, 24)
        axis.grid(color="#D8D8D8", linewidth=0.5, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=5.5 if column_layout else 8)
        axis.set_xlabel("Encoder layer", fontsize=6 if column_layout else 9)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2 if column_layout else 4,
        frameon=False,
        fontsize=5.2 if column_layout else 8,
    )
    figure.subplots_adjust(
        left=0.08 if column_layout else 0.055,
        right=0.995,
        top=0.88,
        bottom=0.32 if column_layout else 0.28,
        wspace=0.38 if column_layout else 0.28,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "centered" if condition == "mean_centered" else "raw"
    stem = output_dir / f"saa-l1-layer-trajectories-{suffix}-{layout}"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-folds", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.all_folds)
    frame["layer"] = frame["space"].map(layer)
    present = [model for model in MODELS if model in set(frame["model"])]
    figure, axes = plt.subplots(2, len(METRICS), figsize=(16, 6.2), sharex=True)
    for column, (metric, title) in enumerate(METRICS.items()):
        scale = 100.0 if metric != "l1_separation_gap" else 1.0
        summary = (
            frame.groupby(["condition", "model", "layer"], as_index=False)[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        for row, condition in enumerate(("raw", "mean_centered")):
            axis = axes[row, column]
            for model in present:
                values = summary.loc[
                    (summary["condition"] == condition) & (summary["model"] == model)
                ].sort_values("layer")
                x = values["layer"].to_numpy()
                mean = values["mean"].to_numpy() * scale
                std = values["std"].fillna(0).to_numpy() * scale
                label, color, linestyle = MODELS[model]
                folds = frame.loc[frame["model"] == model, "fold"].nunique()
                axis.plot(
                    x, mean, color=color, linestyle=linestyle, marker="o",
                    markersize=2.3, linewidth=1.5, label=f"{label} (n={folds})",
                )
                axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)
            if row == 0:
                axis.set_title(title, fontsize=9, fontweight="normal")
            if column == 0:
                axis.set_ylabel("Raw" if row == 0 else "Mean-centered")
            if row == 1:
                axis.set_xlabel("Encoder layer")
            axis.set_xticks([1, 4, 8, 12, 16, 20, 24])
            axis.set_xlim(1, 24)
            axis.grid(color="#D8D8D8", linewidth=0.5, alpha=0.7)
            axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    figure.subplots_adjust(left=0.055, right=0.995, top=0.94, bottom=0.16, wspace=0.28, hspace=0.28)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "saa_l1_structure_layer_trajectories"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    for condition in ("raw", "mean_centered"):
        for layout in ("column", "wide"):
            render_condition(frame, condition, args.output_dir, layout)


if __name__ == "__main__":
    main()
