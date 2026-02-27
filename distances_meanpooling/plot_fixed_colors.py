import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1) FIXED LANGUAGE ORDER (same as your Whisper figure / choice)
# ============================================================
LANGUAGES_TO_PLOT = [
    "spanish",
    "french",
    "mandarin",
    "arabic",
    "korean",
    "german",
    "vietnamese",
]

# ============================================================
# 2) FIXED COLORS PER LANGUAGE (constant across models)
#    Choose any palette you want ONCE and keep it forever.
#    Below: tab10-like stable colors (10 langs -> perfect).
# ============================================================
TAB10 = plt.cm.tab10(np.linspace(0, 1, 10))

LANG2COLOR = {
    "spanish":   TAB10[0],
    "french":    TAB10[1],
    "mandarin":  TAB10[2],
    "arabic":    TAB10[3],
    "korean":    TAB10[4],
    "dutch":     TAB10[5],
    "german":    TAB10[6],
    "italian":   TAB10[7],
    "turkish":   TAB10[8],
    "vietnamese":TAB10[9],
}

# If you later add more languages than 10, switch to tab20 or define manually.


def _as_float_list(xs):
    """Convert possible np.float32 etc. to python float list."""
    return [float(x) if x is not None else np.nan for x in xs]


def plot_model_results(
    results: dict,
    model_name: str,
    languages_to_plot=LANGUAGES_TO_PLOT,
    lang2color=LANG2COLOR,
    title_prefix="Distance to Native Centroid Across Layers",
    ylabel="Cosine Distance to Native Centroid",
    show_ci_per_l1=True,
    show_ci_overall=True,
    output_path=None,
):
    """
    Plot one model results dict with fixed colors per language.

    Expected keys:
      - layers, mean_L2, ci_L2_lower, ci_L2_upper
      - mean_native, ci_native_lower, ci_native_upper
      - per_L1[lang]: mean_L2, ci_L2_lower, ci_L2_upper (optional)
    """
    layers = results["layers"]

    fig, ax = plt.subplots(figsize=(14, 8))

    # ---- Plot per-L1 curves (same colors & order)
    if "per_L1" in results and results["per_L1"] is not None:
        per_L1 = results["per_L1"]

        for lang in languages_to_plot:
            key = lang.lower()
            if key not in per_L1:
                continue

            color = lang2color.get(key, "gray")

            mean = _as_float_list(per_L1[key]["mean_L2"])
            ax.plot(
                layers, mean,
                label=key.capitalize(),
                color=color,
                linewidth=2,
                alpha=0.85,
                zorder=3
            )

            if show_ci_per_l1 and "ci_L2_lower" in per_L1[key] and "ci_L2_upper" in per_L1[key]:
                lo = _as_float_list(per_L1[key]["ci_L2_lower"])
                hi = _as_float_list(per_L1[key]["ci_L2_upper"])
                ax.fill_between(layers, lo, hi, color=color, alpha=0.15, zorder=2)

    # ---- Overall L2 (blue)
    ax.plot(
        layers, _as_float_list(results["mean_L2"]),
        "o-",
        label="L2 speakers",
        color="blue",
        linewidth=3,
        markersize=6,
        zorder=10
    )
    if show_ci_overall:
        ax.fill_between(
            layers,
            _as_float_list(results["ci_L2_lower"]),
            _as_float_list(results["ci_L2_upper"]),
            alpha=0.2,
            color="blue",
            zorder=8
        )

    # ---- Native baseline (green)
    ax.plot(
        layers, _as_float_list(results["mean_native"]),
        "s-",
        label="Native baseline (LOO)",
        color="green",
        linewidth=3,
        markersize=6,
        zorder=11
    )
    if show_ci_overall:
        ax.fill_between(
            layers,
            _as_float_list(results["ci_native_lower"]),
            _as_float_list(results["ci_native_upper"]),
            alpha=0.2,
            color="green",
            zorder=8
        )

    # ---- Formatting
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{title_prefix} — {model_name}", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Legend: keep same order (languages first, then L2, then native)
    # Matplotlib keeps plotting order, so we plotted languages first => ok.
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def compare_two_models_side_by_side(
    results_model_a: dict,
    name_a: str,
    results_model_b: dict,
    name_b: str,
    languages_to_plot=LANGUAGES_TO_PLOT,
    lang2color=LANG2COLOR,
    output_path=None,
):
    """
    Side-by-side comparison with identical colors/legend ordering.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    for ax, results, model_name in [
        (axes[0], results_model_a, name_a),
        (axes[1], results_model_b, name_b),
    ]:
        layers = results["layers"]

        # per-L1
        if "per_L1" in results and results["per_L1"] is not None:
            per_L1 = results["per_L1"]
            for lang in languages_to_plot:
                key = lang.lower()
                if key not in per_L1:
                    continue
                color = lang2color.get(key, "gray")
                mean = _as_float_list(per_L1[key]["mean_L2"])
                ax.plot(layers, mean, color=color, linewidth=2, alpha=0.85, zorder=3)
                if "ci_L2_lower" in per_L1[key] and "ci_L2_upper" in per_L1[key]:
                    lo = _as_float_list(per_L1[key]["ci_L2_lower"])
                    hi = _as_float_list(per_L1[key]["ci_L2_upper"])
                    ax.fill_between(layers, lo, hi, color=color, alpha=0.15, zorder=2)

        # overall L2 + native
        ax.plot(layers, _as_float_list(results["mean_L2"]), "o-", color="blue", linewidth=3, markersize=6, zorder=10)
        ax.fill_between(layers, _as_float_list(results["ci_L2_lower"]), _as_float_list(results["ci_L2_upper"]),
                        alpha=0.2, color="blue", zorder=8)

        ax.plot(layers, _as_float_list(results["mean_native"]), "s-", color="green", linewidth=3, markersize=6, zorder=11)
        ax.fill_between(layers, _as_float_list(results["ci_native_lower"]), _as_float_list(results["ci_native_upper"]),
                        alpha=0.2, color="green", zorder=8)

        ax.set_title(model_name, fontsize=14, fontweight="bold")
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Cosine Distance to Native Centroid", fontsize=12)

    # Single legend on the right (fixed order)
    legend_handles = []
    legend_labels = []

    # language proxies
    for lang in languages_to_plot:
        key = lang.lower()
        color = lang2color.get(key, "gray")
        h, = axes[0].plot([], [], color=color, linewidth=2)
        legend_handles.append(h)
        legend_labels.append(key.capitalize())

    # L2 + native proxies
    h1, = axes[0].plot([], [], "o-", color="blue", linewidth=3)
    h2, = axes[0].plot([], [], "s-", color="green", linewidth=3)
    legend_handles += [h1, h2]
    legend_labels += ["L2 speakers", "Native baseline (LOO)"]

    fig.legend(legend_handles, legend_labels, loc="center right", bbox_to_anchor=(1.02, 0.5), fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.92, 1])

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()



def _as_float_list(xs):
    return [float(x) if x is not None else np.nan for x in xs]


def compare_two_models_side_by_side(
    results_model_a: dict,
    name_a: str,
    results_model_b: dict,
    name_b: str,
    languages_to_plot,
    lang2color,
    output_path=None,
    show_per_l1=True,
    show_ci_per_l1=True,
    show_ci_overall=True,
    show_markers=True,
):
    """
    Side-by-side comparison with identical colors/legend ordering.

    Parameters
    ----------
    show_per_l1 : bool
        If False, do not plot individual L1 curves.
    show_ci_per_l1 : bool
        If False, do not plot CI for individual L1 curves.
    show_ci_overall : bool
        If False, do not plot CI for overall L2 + native baseline.
    show_markers : bool
        If False, remove markers for overall curves.
    """

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    def plot_one(ax, results, model_name):
        layers = results["layers"]

        # -----------------------
        # 1) per-L1 curves
        # -----------------------
        if show_per_l1 and "per_L1" in results and results["per_L1"] is not None:
            per_L1 = results["per_L1"]

            for lang in languages_to_plot:
                key = lang.lower()
                if key not in per_L1:
                    continue

                color = lang2color.get(key, "gray")
                mean = _as_float_list(per_L1[key]["mean_L2"])

                ax.plot(
                    layers,
                    mean,
                    color=color,
                    linewidth=2,
                    alpha=0.85,
                    zorder=3
                )

                if show_ci_per_l1 and "ci_L2_lower" in per_L1[key] and "ci_L2_upper" in per_L1[key]:
                    lo = _as_float_list(per_L1[key]["ci_L2_lower"])
                    hi = _as_float_list(per_L1[key]["ci_L2_upper"])
                    ax.fill_between(layers, lo, hi, color=color, alpha=0.12, zorder=2)

        # -----------------------
        # 2) overall L2 + native
        # -----------------------
        l2_style = "o-" if show_markers else "-"
        nat_style = "s-" if show_markers else "-"

        ax.plot(
            layers,
            _as_float_list(results["mean_L2"]),
            l2_style,
            color="blue",
            linewidth=3,
            markersize=6,
            zorder=10
        )

        if show_ci_overall:
            ax.fill_between(
                layers,
                _as_float_list(results["ci_L2_lower"]),
                _as_float_list(results["ci_L2_upper"]),
                alpha=0.18,
                color="blue",
                zorder=8
            )

        ax.plot(
            layers,
            _as_float_list(results["mean_native"]),
            nat_style,
            color="green",
            linewidth=3,
            markersize=6,
            zorder=11
        )

        if show_ci_overall:
            ax.fill_between(
                layers,
                _as_float_list(results["ci_native_lower"]),
                _as_float_list(results["ci_native_upper"]),
                alpha=0.18,
                color="green",
                zorder=8
            )

        ax.set_title(model_name, fontsize=14, fontweight="bold")
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Plot both
    plot_one(axes[0], results_model_a, name_a)
    plot_one(axes[1], results_model_b, name_b)

    axes[0].set_ylabel("Cosine Distance to Native Centroid", fontsize=12)

    # -----------------------
    # Legend (single, fixed)
    # -----------------------
    legend_handles = []
    legend_labels = []

    if show_per_l1:
        for lang in languages_to_plot:
            key = lang.lower()
            color = lang2color.get(key, "gray")
            h, = axes[0].plot([], [], color=color, linewidth=2)
            legend_handles.append(h)
            legend_labels.append(key.capitalize())

    h1, = axes[0].plot([], [], "-" if not show_markers else "o-", color="blue", linewidth=3)
    h2, = axes[0].plot([], [], "-" if not show_markers else "s-", color="green", linewidth=3)

    legend_handles += [h1, h2]
    legend_labels += ["L2 speakers", "Native baseline (LOO)"]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10
    )

    plt.tight_layout(rect=[0, 0, 0.92, 1])

    if output_path is not None:
        from pathlib import Path
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved figure to: {out}")

    plt.show()
    
    

if __name__ == "__main__":
    # Example usage:
    # whisper = pd.read_pickle("results_whisper.pkl")
    # wav2vec = pd.read_pickle("results_wav2vec.pkl")

    # --- Replace with your paths
    whisper_path = "/home/nkhaous/myLLF/speech_accent/representations_distances/results/whisper_v2/results_whisper.pkl"
    wav2vec_path = "/home/nkhaous/myLLF/speech_accent/representations_distances/results/xlsr53/results_xlsr53.pkl"

    whisper = pd.read_pickle(whisper_path)
    wav2vec = pd.read_pickle(wav2vec_path)

    # Plot individually (same colors)
    plot_model_results(whisper, 
                       "Whisper", 
                       show_ci_overall=True, 
                       show_ci_per_l1=False, 
                       output_path="whisper_fixed.png")
    plot_model_results(wav2vec, 
                       "wav2vec", 
                       show_ci_overall=True, 
                       show_ci_per_l1=False, 
                       output_path="wav2vec_fixed.png")

    # Side-by-side comparison (same colors)
    # compare_two_models_side_by_side(
    #     whisper, "Whisper",
    #     wav2vec, "wav2vec",
    #     # show_ci_overall=False, 
    #     # show_ci_per_l1=False,
    #     output_path="compare_whisper_wav2vec.png"
    # )
    
    
    
    compare_two_models_side_by_side(
    whisper, "Whisper",
    wav2vec, "wav2vec",
    languages_to_plot=LANGUAGES_TO_PLOT,
    lang2color=LANG2COLOR,
    output_path="wav2vec-whisper_compare_clean.png",
    show_ci_per_l1=False,
    show_ci_overall=True,
    show_markers=True,
    )

