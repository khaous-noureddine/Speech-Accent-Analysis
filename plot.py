"""
Publication-Ready Figure for Interspeech 2026
==============================================

2-panel figure:
- Panel A: L2 mean + CI / Native LOO + CI (global signal)
- Panel B: Heatmap Δ(L1, layer) = dist(L1, layer) - dist(native, layer)

Usage:
    python plot_figure.py --results results/whisper_v2/results_whisper.pkl \
                          --output figures/figure1_whisper.pdf \
                          --model_name "Whisper large-v3"
"""

import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as ticker

from loguru import logger


# ============================================================================
# HELPERS
# ============================================================================

def compute_delta(results, languages):
    """
    Compute Δ(L1, layer) = dist(L1, layer) - dist(native, layer)
    
    Native baseline = 0 by definition.
    L2 languages = how much above native they are.
    
    Returns
    -------
    pd.DataFrame
        Shape (n_languages, n_layers) — rows=L1, cols=layers
    """
    layers = results["layers"]
    native_mean = np.array(results["mean_native"])
    
    delta_matrix = {}
    
    for lang in languages:
        if lang not in results.get("per_L1", {}):
            continue
        
        lang_mean = np.array(results["per_L1"][lang]["mean_L2"])
        
        # Handle NaN and length mismatch
        if len(lang_mean) != len(native_mean):
            logger.warning(f"Length mismatch for {lang}: {len(lang_mean)} vs {len(native_mean)}")
            continue
        
        # Delta = dist(L1) - dist(native)
        delta = lang_mean - native_mean
        delta_matrix[lang.capitalize()] = delta
    
    df = pd.DataFrame(delta_matrix, index=layers).T  # Shape: (n_languages, n_layers)
    
    return df


def plot_combined_figure(results, languages, model_name="Model", output_path=None):
    """
    Plot 2-panel publication-ready figure.
    
    Panel A: Global L2 vs Native with CI
    Panel B: Heatmap Δ(L1, layer)
    
    Parameters
    ----------
    results : dict
        Results from compute_distance_to_native.py
    languages : list
        List of L1 languages to include in heatmap
    model_name : str
        Model name for title
    output_path : Path, optional
        Where to save figure
    """
    layers = results["layers"]
    
    # ── Compute delta matrix ──────────────────────────────────────────────────
    delta_df = compute_delta(results, languages)
    
    if delta_df.empty:
        logger.error("No per-L1 data found in results! Did you run with --compute_per_l1?")
        return
    
    # Sort languages by mean delta (most different → top)
    delta_df = delta_df.loc[delta_df.mean(axis=1).sort_values(ascending=False).index]
    
    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    
    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[1, 1.6],
        wspace=0.08,
        left=0.07, right=0.97,
        top=0.88, bottom=0.12
    )
    
    ax_a = fig.add_subplot(gs[0])  # Panel A: Line plot
    ax_b = fig.add_subplot(gs[1])  # Panel B: Heatmap
    
    # ── PANEL A: Global L2 vs Native with CI ─────────────────────────────────
    
    # Native baseline
    ax_a.plot(
        layers, results["mean_native"],
        's-', color="#2ca02c", linewidth=2.5,
        markersize=5, label="Native (LOO)", zorder=5
    )
    ax_a.fill_between(
        layers,
        results["ci_native_lower"],
        results["ci_native_upper"],
        color="#2ca02c", alpha=0.15, zorder=3
    )
    
    # L2 overall
    ax_a.plot(
        layers, results["mean_L2"],
        'o-', color="#1f77b4", linewidth=2.5,
        markersize=5, label="L2 (all)", zorder=5
    )
    ax_a.fill_between(
        layers,
        results["ci_L2_lower"],
        results["ci_L2_upper"],
        color="#1f77b4", alpha=0.15, zorder=3
    )
    
    # Styling Panel A
    ax_a.set_xlabel("Layer Index", fontsize=12)
    ax_a.set_ylabel("Cosine Distance to Native Centroid", fontsize=12)
    ax_a.set_title("(A) Global", fontsize=13, fontweight="bold", pad=8)
    ax_a.legend(fontsize=11, framealpha=0.9, loc="upper right")
    ax_a.grid(True, linestyle="--", alpha=0.4)
    ax_a.set_xlim(layers[0] - 0.3, layers[-1] + 0.3)
    ax_a.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    
    # Add "A" label
    ax_a.text(-0.12, 1.02, "A", transform=ax_a.transAxes,
              fontsize=16, fontweight="bold", va="top")
    
    # ── PANEL B: Heatmap Δ(L1, layer) ────────────────────────────────────────
    
    delta_values = delta_df.values
    
    # Colormap centered on 0 (native = 0)
    vmax = np.nanpercentile(np.abs(delta_values), 95)
    vmin = -vmax * 0.3  # Allow some negative (some L1 closer than native)
    
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    im = ax_b.imshow(
        delta_values,
        aspect="auto",
        cmap="RdYlGn_r",  # Red = far from native, Green = close to native
        norm=norm,
        interpolation="nearest"
    )
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax_b, shrink=0.85, pad=0.02)
    cbar.set_label("Δ distance to native centroid", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    
    # Axes labels
    ax_b.set_xlabel("Layer Index", fontsize=12)
    ax_b.set_title("(B) Per-L1 Distance Delta  [Δ = dist(L1) − dist(native)]",
                   fontsize=13, fontweight="bold", pad=8)
    
    # Y-axis: language names
    ax_b.set_yticks(range(len(delta_df.index)))
    ax_b.set_yticklabels(delta_df.index, fontsize=11)
    
    # X-axis: layer indices
    n_layers = len(layers)
    step = max(1, n_layers // 8)
    tick_positions = list(range(0, n_layers, step))
    ax_b.set_xticks(tick_positions)
    ax_b.set_xticklabels([layers[i] for i in tick_positions], fontsize=10)
    
    # Add cell values
    for i in range(delta_values.shape[0]):
        for j in range(delta_values.shape[1]):
            val = delta_values[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.5 else "black"
                ax_b.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=6.5, color=color)
    
    # Add "B" label
    ax_b.text(-0.04, 1.02, "B", transform=ax_b.transAxes,
              fontsize=16, fontweight="bold", va="top")
    
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    
    # ── Global title ──────────────────────────────────────────────────────────
    fig.suptitle(
        f"Distance to Native Centroid Across Layers — {model_name}",
        fontsize=15, fontweight="bold", y=0.97
    )
    
    # ── Save ──────────────────────────────────────────────────────────────────
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as PDF (publication) + PNG (slides)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        
        # Also save PNG version
        png_path = output_path.with_suffix(".png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        
        logger.info(f"Saved PDF: {output_path}")
        logger.info(f"Saved PNG: {png_path}")
    
    plt.show()
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot publication-ready figure for Interspeech 2026"
    )
    parser.add_argument("--results", type=Path, required=True,
                        help="Pickle file with results from compute_distance_to_native.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path (PDF)")
    parser.add_argument("--model_name", type=str, default="Model",
                        help="Model name for title (e.g., 'Whisper large-v3')")
    parser.add_argument("--languages", nargs="+", default=None,
                        help="L1 languages to include in heatmap (default: all available)")
    parser.add_argument("--top_n", type=int, default=None,
                        help="Only show top N languages by mean delta (if --languages not specified)")
    
    args = parser.parse_args()
    
    # Load results
    logger.info(f"Loading results from {args.results}")
    with open(args.results, "rb") as f:
        results = pickle.load(f)
    
    logger.info(f"Layers available: {results['layers']}")
    
    # Get available languages
    if "per_L1" not in results or not results["per_L1"]:
        logger.error("No per-L1 data found! Re-run compute_distance_to_native.py with --compute_per_l1")
        exit(1)
    
    available_langs = list(results["per_L1"].keys())
    logger.info(f"Available L1 languages: {available_langs}")
    
    # Select languages
    if args.languages:
        languages = [l.lower() for l in args.languages]
    elif args.top_n:
        # Select top N by mean delta
        native_mean = np.array(results["mean_native"])
        lang_deltas = {}
        for lang in available_langs:
            lang_mean = np.array(results["per_L1"][lang]["mean_L2"])
            if len(lang_mean) == len(native_mean):
                lang_deltas[lang] = np.nanmean(lang_mean - native_mean)
        languages = sorted(lang_deltas, key=lang_deltas.get, reverse=True)[:args.top_n]
        logger.info(f"Top {args.top_n} languages by mean delta: {languages}")
    else:
        languages = available_langs
    
    # Plot
    plot_combined_figure(
        results=results,
        languages=languages,
        model_name=args.model_name,
        output_path=args.output
    )
    
    logger.info("✅ Done!")