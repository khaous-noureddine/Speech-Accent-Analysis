"""Render fixed-passage SAA accent-geometry figures from saved embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from accented_asr.representation_analysis.metrics import mean_center


MODEL_LABELS = {
    "asr_only": "ASR-only",
    "multidomain_ctc": "Multi-domain FT",
    "utterance_supcon": r"CP-SupCon$_{utt}$",
    "md_ft_cp_supcon": r"MD-FT + CP-SupCon$_{utt}$",
}


def project(embeddings: np.ndarray, *, seed: int) -> np.ndarray:
    if len(embeddings) < 4:
        raise ValueError("t-SNE requires at least four SAA examples.")
    perplexity = min(30.0, max(2.0, (len(embeddings) - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    ).fit_transform(embeddings)


def render(
    fold_dir: Path,
    *,
    models: list[str],
    layer: int,
    condition: str,
    seed: int,
) -> Path:
    sample = pd.read_parquet(fold_dir / "analysis_sample.parquet")
    labels = sample["native_language"].astype(str)
    languages = sorted(labels.unique())
    palette = plt.get_cmap("tab20")(
        np.linspace(0.0, 1.0, len(languages), endpoint=False)
    )
    colors = dict(zip(languages, palette, strict=True))

    figure, axes = plt.subplots(2, 2, figsize=(8.2, 9.0), squeeze=False)
    axes = axes.ravel()
    for panel, (axis, model) in enumerate(zip(axes, models, strict=False)):
        path = fold_dir / f"{model}_backbone_layer_{layer}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        embeddings = np.load(path)["embeddings"]
        if condition == "mean_centered":
            train = sample["probe_split"].eq("train").to_numpy()
            embeddings = mean_center(embeddings, embeddings[train])
        elif condition != "raw":
            raise ValueError(f"Unknown condition: {condition}")
        coordinates = project(embeddings, seed=seed)
        for language in languages:
            selected = labels.eq(language).to_numpy()
            axis.scatter(
                coordinates[selected, 0],
                coordinates[selected, 1],
                color=colors[language],
                marker="o",
                s=22,
                alpha=0.82,
                edgecolors="black",
                linewidths=0.25,
            )
        axis.set_title(
            f"({chr(97 + panel)}) {MODEL_LABELS.get(model, model)}",
            fontsize=10,
            fontweight="normal",
            pad=7,
        )
        axis.set_xlabel("t-SNE dimension 1", fontsize=9)
        axis.set_ylabel("t-SNE dimension 2", fontsize=9)
        axis.set_box_aspect(1)
        axis.grid(color="#dddddd", linewidth=0.5, alpha=0.65)
        axis.tick_params(labelsize=7, length=2.5)
        for spine in axis.spines.values():
            spine.set_color("#444444")
            spine.set_linewidth(0.7)
    for axis in axes[len(models):]:
        axis.set_visible(False)

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=4.5,
            markerfacecolor=colors[language], markeredgecolor="black",
            markeredgewidth=0.25, label=language,
        )
        for language in languages
    ]
    figure.legend(
        handles=handles,
        title="First language (color)",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=min(5, len(handles)),
        frameon=False,
        fontsize=6.5,
        title_fontsize=8,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    legend_rows = int(np.ceil(len(handles) / min(5, len(handles))))
    figure.subplots_adjust(
        left=0.075,
        right=0.975,
        top=0.97,
        bottom=min(0.30, 0.08 + 0.027 * legend_rows),
        wspace=0.22,
        hspace=0.24,
    )
    stem = fold_dir / (
        f"tsne_saa_l1_color_2x2_layer_{layer}_{condition}_comparison"
    )
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--folds", nargs="+", required=True)
    parser.add_argument(
        "--models", nargs="+",
        default=list(MODEL_LABELS),
    )
    parser.add_argument("--layers", nargs="+", type=int, default=[24])
    parser.add_argument(
        "--conditions", nargs="+", choices=("raw", "mean_centered"),
        default=["mean_centered"],
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if len(args.models) > 4:
        raise ValueError("The SAA publication layout supports at most four models.")
    for fold in args.folds:
        fold_dir = args.input_root / fold
        for layer in args.layers:
            for condition in args.conditions:
                stem = render(
                    fold_dir, models=args.models, layer=layer,
                    condition=condition, seed=args.seed,
                )
                print(f"Wrote {stem}.{{pdf,svg,png}}")


if __name__ == "__main__":
    main()
