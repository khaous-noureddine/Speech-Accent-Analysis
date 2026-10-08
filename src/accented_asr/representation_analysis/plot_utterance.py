"""Render publication figures from saved utterance-level t-SNE coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

MODEL_LABELS = {
    "ctc_only": "CTC only",
    "utterance_supcon": "CTC + SupCon",
}


def render_prompt_figure(
    coordinates_path: Path,
    output_stem: Path,
    models: list[str],
    accent_title: str | None = None,
) -> None:
    frame = pd.read_parquet(coordinates_path)
    required = {"normalized_word", "speaker_id", "model", "x", "y"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing t-SNE columns: {sorted(missing)}")

    frame = frame.loc[frame["model"].isin(models)].copy()
    found_models = set(frame["model"].astype(str))
    missing_models = set(models) - found_models
    if missing_models:
        raise ValueError(f"Missing models in {coordinates_path}: {sorted(missing_models)}")

    prompts = sorted(frame["normalized_word"].astype(str).unique())
    prompt_labels = {prompt: f"P{index + 1}" for index, prompt in enumerate(prompts)}
    colors = plt.get_cmap("tab20")([index / max(len(prompts), 1) for index in range(len(prompts))])
    prompt_colors = dict(zip(prompts, colors, strict=True))

    figure, axes = plt.subplots(
        1,
        len(models),
        figsize=(7.2, 4.35),
        sharex=False,
        sharey=False,
        constrained_layout=False,
    )
    if len(models) == 1:
        axes = [axes]
    if accent_title:
        figure.suptitle(
            f"Held-out accent: {accent_title.replace('_', ' ').title()}",
            fontsize=11,
            fontweight="bold",
            y=0.985,
        )

    for panel_index, (axis, model) in enumerate(zip(axes, models, strict=True)):
        model_frame = frame.loc[frame["model"] == model]
        for prompt in prompts:
            prompt_frame = model_frame.loc[
                model_frame["normalized_word"].astype(str) == prompt
            ]
            axis.scatter(
                prompt_frame["x"],
                prompt_frame["y"],
                color=prompt_colors[prompt],
                marker="o",
                s=38,
                alpha=0.86,
                edgecolors="black",
                linewidths=0.35,
            )
        axis.set_title(
            f"({chr(97 + panel_index)}) {MODEL_LABELS.get(model, model)}",
            fontsize=10,
            fontweight="bold",
            pad=7,
        )
        axis.set_xlabel("t-SNE dimension 1", fontsize=9)
        axis.set_ylabel("t-SNE dimension 2", fontsize=9)
        axis.margins(x=0.10, y=0.10)
        axis.grid(color="#dddddd", linewidth=0.5, alpha=0.65)
        axis.tick_params(labelsize=7, length=2.5)
        axis.spines[["top", "right"]].set_visible(False)
    prompt_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5,
            markerfacecolor=prompt_colors[prompt], markeredgecolor="black",
            markeredgewidth=0.35,
            label=prompt_labels[prompt],
        )
        for prompt in prompts
    ]
    figure.legend(
        handles=prompt_handles,
        title="Prompt (color)",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(6, len(prompt_handles)),
        frameon=False,
        fontsize=7,
        title_fontsize=8,
        columnspacing=0.8,
        handletextpad=0.25,
    )
    top = 0.86 if accent_title else 0.91
    figure.subplots_adjust(
        left=0.075, right=0.99, top=top, bottom=0.25, wspace=0.16
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--accents", nargs="+", required=True)
    parser.add_argument(
        "--models", nargs="+", default=["ctc_only", "utterance_supcon"]
    )
    parser.add_argument(
        "--show-accent-title",
        action="store_true",
        help="Add a held-out-accent title above the model panels.",
    )
    args = parser.parse_args()

    for accent in args.accents:
        accent_dir = args.input_root / accent
        filename = (
            "tsne_prompt_color_zoomed_accent_comparison"
            if args.show_accent_title
            else "tsne_prompt_color_zoomed_no_title_comparison"
        )
        render_prompt_figure(
            accent_dir / "tsne_coordinates.parquet",
            accent_dir / filename,
            args.models,
            accent_title=accent if args.show_accent_title else None,
        )
        print(f"Wrote publication figures to {accent_dir}")


if __name__ == "__main__":
    main()
