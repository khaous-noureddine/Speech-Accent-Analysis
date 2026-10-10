"""Plot raw and mean-centered representation metrics across encoder layers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "ctc_only": "CTC only",
    "accent_dat": "Accent-DAT",
    "accent_mtl": "Accent-MTL",
    "multidomain_ctc": "MD-FT",
    "augmented_view_supcon": "Aug-SupCon",
    "word_supcon": "CTC + word SupCon",
    "utterance_supcon": "CTC + SupCon",
    "md_ft_cp_supcon": "MD-FT + CP-SupCon",
    "shuffled_supcon": "Shuffled-label SupCon",
}
MODEL_COLORS = {
    "ctc_only": "#4C566A",
    "accent_dat": "#D08770",
    "accent_mtl": "#B48EAD",
    "multidomain_ctc": "#5E81AC",
    "augmented_view_supcon": "#EBCB8B",
    "word_supcon": "#A3BE8C",
    "utterance_supcon": "#007C83",
    "md_ft_cp_supcon": "#BF3A30",
    "shuffled_supcon": "#8F8F8F",
}
MODEL_LINESTYLES = {
    "ctc_only": "--",
    "accent_dat": ":",
    "accent_mtl": ":",
    "multidomain_ctc": "-.",
    "augmented_view_supcon": ":",
    "word_supcon": "--",
    "utterance_supcon": "-",
    "md_ft_cp_supcon": "-",
    "shuffled_supcon": "--",
}
METRICS = {
    "recall_at_1": ("Recall@1 (%) ↑", 100.0),
    "recall_at_5": ("Recall@5 (%) ↑", 100.0),
    "map": ("mAP (%) ↑", 100.0),
    "alignment_ratio": ("Alignment ratio ↓", 1.0),
    "positive_cosine_distance": ("Same-prompt cosine distance ↓", 1.0),
    "negative_cosine_distance": ("Different-prompt cosine distance ↑", 1.0),
}
CONDITIONS = ("raw", "mean_centered")
CONDITION_LABELS = {
    "raw": "Raw representations",
    "mean_centered": "Mean-centered representations",
}


def encoder_layer(space: str) -> int:
    match = re.fullmatch(r"backbone_layer_(\d+)", str(space))
    if not match:
        raise ValueError(f"Unexpected representation space: {space}")
    return int(match.group(1))


def summarize(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    data = frame.copy()
    data["layer"] = data["space"].map(encoder_layer)
    return (
        data.groupby(["condition", "model", "layer"], as_index=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["condition", "model", "layer"])
    )


def available_models(frame: pd.DataFrame) -> list[str]:
    found = set(frame["model"].astype(str))
    return [model for model in MODEL_LABELS if model in found]


def legend_label(frame: pd.DataFrame, model: str) -> str:
    folds = frame.loc[frame["model"] == model, "accent"].nunique()
    return f"{MODEL_LABELS[model]} (n={folds})"


def render_metric(frame: pd.DataFrame, metric: str, output_dir: Path) -> None:
    ylabel, scale = METRICS[metric]
    summary = summarize(frame, metric)
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), sharex=True)
    for axis, condition in zip(axes, CONDITIONS, strict=True):
        for model in available_models(frame):
            values = summary.loc[
                (summary["condition"] == condition) & (summary["model"] == model)
            ]
            if values.empty:
                continue
            x = values["layer"].to_numpy()
            mean = values["mean"].to_numpy() * scale
            std = values["std"].fillna(0).to_numpy() * scale
            color = MODEL_COLORS[model]
            axis.plot(
                x, mean, color=color, marker="o", markersize=3.2,
                linewidth=1.8, linestyle=MODEL_LINESTYLES[model],
                label=legend_label(frame, model),
            )
            axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.14)
        axis.set_title(CONDITION_LABELS[condition], fontsize=10, fontweight="normal")
        axis.set_xlabel("Encoder layer")
        axis.set_xticks([1, 4, 8, 12, 16, 20, 24])
        axis.set_xlim(1, 24)
        axis.grid(color="#D8D8D8", linewidth=0.6, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    figure.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.30, wspace=0.18)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"layer_trajectory_{metric}"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def render_overview(frame: pd.DataFrame, output_dir: Path) -> None:
    selected = ("recall_at_1", "map", "alignment_ratio")
    figure, axes = plt.subplots(2, 3, figsize=(11.0, 6.1), sharex=True)
    for column, metric in enumerate(selected):
        ylabel, scale = METRICS[metric]
        summary = summarize(frame, metric)
        for row, condition in enumerate(CONDITIONS):
            axis = axes[row, column]
            for model in available_models(frame):
                values = summary.loc[
                    (summary["condition"] == condition)
                    & (summary["model"] == model)
                ]
                x = values["layer"].to_numpy()
                mean = values["mean"].to_numpy() * scale
                std = values["std"].fillna(0).to_numpy() * scale
                color = MODEL_COLORS[model]
                axis.plot(
                    x, mean, color=color, linestyle=MODEL_LINESTYLES[model],
                    marker="o", markersize=2.5, linewidth=1.5,
                )
                axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)
            if row == 0:
                axis.set_title(ylabel, fontsize=10, fontweight="normal")
            if column == 0:
                axis.set_ylabel(CONDITION_LABELS[condition])
            if row == 1:
                axis.set_xlabel("Encoder layer")
            axis.set_xticks([1, 4, 8, 12, 16, 20, 24])
            axis.set_xlim(1, 24)
            axis.grid(color="#D8D8D8", linewidth=0.5, alpha=0.7)
            axis.spines[["top", "right"]].set_visible(False)
    handles = [
        plt.Line2D(
            [0], [0], color=MODEL_COLORS[model],
            linestyle=MODEL_LINESTYLES[model], marker="o",
            label=legend_label(frame, model),
        )
        for model in available_models(frame)
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    figure.subplots_adjust(left=0.10, right=0.99, top=0.95, bottom=0.18, wspace=0.24, hspace=0.28)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "layer_trajectories_overview"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def render_model_grid(
    frame: pd.DataFrame, metric: str, condition: str, output_dir: Path
) -> None:
    """Render one readable panel per model against the ASR-only reference."""
    ylabel, scale = METRICS[metric]
    summary = summarize(frame, metric)
    models = available_models(frame)
    figure, axes = plt.subplots(2, 4, figsize=(11.2, 5.8), sharex=True, sharey=True)
    reference = summary.loc[
        (summary["condition"] == condition) & (summary["model"] == "ctc_only")
    ]

    for axis, model in zip(axes.flat, models, strict=False):
        values = summary.loc[
            (summary["condition"] == condition) & (summary["model"] == model)
        ]
        if model != "ctc_only" and not reference.empty:
            axis.plot(
                reference["layer"], reference["mean"] * scale,
                color="#A7ABB3", linestyle="--", linewidth=1.3,
                label="ASR-only reference",
            )
        axis.plot(
            values["layer"], values["mean"] * scale,
            color=MODEL_COLORS[model], marker="o", markersize=2.7,
            linewidth=1.8, label=MODEL_LABELS[model],
        )
        folds = frame.loc[frame["model"] == model, "accent"].nunique()
        axis.set_title(f"{MODEL_LABELS[model]} (n={folds})", fontsize=9)
        axis.set_xticks([1, 4, 8, 12, 16, 20, 24])
        axis.set_xlim(1, 24)
        axis.grid(color="#D8D8D8", linewidth=0.5, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    for axis in axes.flat[len(models):]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        axis.set_xlabel("Encoder layer")
    for axis in axes[:, 0]:
        axis.set_ylabel(ylabel)

    handles = [
        plt.Line2D([0], [0], color="#A7ABB3", linestyle="--", label="ASR-only reference"),
        plt.Line2D([0], [0], color="#333333", marker="o", label="Model in panel"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8)
    figure.subplots_adjust(
        left=0.07, right=0.995, top=0.94, bottom=0.13, wspace=0.12, hspace=0.30
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"layer_trajectory_grid_{metric}_{condition}"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-by-fold", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.metrics_by_fold)
    for metric in METRICS:
        render_metric(frame, metric, args.output_dir)
    render_overview(frame, args.output_dir)
    for metric in ("recall_at_1", "map", "alignment_ratio"):
        for condition in CONDITIONS:
            render_model_grid(frame, metric, condition, args.output_dir)


if __name__ == "__main__":
    main()
