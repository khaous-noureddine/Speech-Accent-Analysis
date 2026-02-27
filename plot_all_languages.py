import os
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Stable color mapping (same language -> same color, independent of order/subset)
# -----------------------------------------------------------------------------
def stable_color(key: str, cmap_name: str = "tab20"):
    """
    Deterministically map a string key (language) to a color from a colormap.
    Same key will ALWAYS get the same color.
    """
    cmap = plt.get_cmap(cmap_name)
    h = int(hashlib.md5(key.lower().encode("utf-8")).hexdigest(), 16)
    return cmap(h % cmap.N)


def uniq_preserve_order(items):
    """Remove duplicates while preserving order."""
    return list(dict.fromkeys(items))


def plot_comprehensive_figure_combined(
    results_per_l1,
    results_full,
    languages_to_plot=None,
    output_path=None,
    cmap_name="tab20",
):
    """
    Plots selected L1s, L2 speakers, and Native Baseline (LOO), with stable colors.

    Parameters
    ----------
    results_per_l1 : dict
        Results from results_per_L1.pkl (contains per_L1 data)
    results_full : dict
        Results from distance_to_native_full.pkl (contains L2 speakers and native LOO)
    languages_to_plot : list, optional
        List of languages to plot. If None, plots all available languages.
    output_path : str, optional
        Where to save the figure
    cmap_name : str
        Matplotlib colormap name for stable language colors (default: tab20)
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    layers = results_per_l1["layers"]
    languages_data = results_per_l1["per_L1"]

    # If no languages specified, use all available languages (sorted for stability)
    if languages_to_plot is None:
        languages_to_plot = sorted(list(languages_data.keys()))
    else:
        languages_to_plot = uniq_preserve_order([l.lower() for l in languages_to_plot])

    # 1) Plot selected L1 curves (stable colors)
    for lang in languages_to_plot:
        if lang not in languages_data:
            print(f"[WARN] language '{lang}' not in results_per_l1['per_L1'], skipping")
            continue

        color = stable_color(lang, cmap_name=cmap_name)
        mean = np.array(languages_data[lang]["mean_L2"])

        ax.plot(
            layers,
            mean,
            label=lang.capitalize(),
            color=color,
            linewidth=2,
            alpha=0.75,
        )

    # 2) Plot L2 speakers (overall)
    ax.plot(
        layers,
        results_full["mean_L2"],
        "o-",
        label="L2 speakers",
        color="blue",
        linewidth=3,
        markersize=6,
        zorder=10,
    )
    ax.fill_between(
        layers,
        results_full["ci_L2_lower"],
        results_full["ci_L2_upper"],
        alpha=0.2,
        color="blue",
        zorder=8,
    )

    # 3) Plot Native baseline (LOO)
    ax.plot(
        layers,
        results_full["mean_native"],
        "s-",
        label="Native baseline (LOO)",
        color="green",
        linewidth=3,
        markersize=6,
        zorder=11,
    )
    ax.fill_between(
        layers,
        results_full["ci_native_lower"],
        results_full["ci_native_upper"],
        alpha=0.2,
        color="green",
        zorder=8,
    )

    # Formatting
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Cosine Distance to Native Centroid", fontsize=12)
    ax.set_title(
        "Distance to Native Centroid Across Layers",
        fontsize=14,
        fontweight="bold",
    )

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[OK] saved figure to: {output_path}")

    plt.show()


if __name__ == "__main__":
    # Load the two result files
    results_per_l1 = pd.read_pickle(
        "/home/nkhaous/myLLF/speech_accent/representations_distances/results/whisper/results_per_L1.pkl"
    )
    results_full = pd.read_pickle(
        "/home/nkhaous/myLLF/speech_accent/representations_distances/results/distance_to_native_full.pkl"
    )

    # Languages you want to display (duplicates OK; they'll be removed)
    languages_to_plot = [
        "spanish",
        "arabic",
        "french",
        "mandarin",
        "french",  # duplicate -> will be removed
        "korean",
        "portuguese",
        "russian",
        "dutch",
        "turkish",
        "german",
        "polish",
        "italian",
        "japanese",
        "macedonian",
        "cantonese",
        "farsi",
        "vietnamese",
        "swedish",
        "romanian",
        "amharic",
    ]

    output_dir = "/home/nkhaous/myLLF/speech_accent/representations_distances/plots/"
    output_path = os.path.join(output_dir, "combined_multilingual_fig2.png")

    plot_comprehensive_figure_combined(
        results_per_l1=results_per_l1,
        results_full=results_full,
        languages_to_plot=languages_to_plot,
        output_path=output_path,
        cmap_name="tab20",  # you can try "tab20b", "tab20c", "nipy_spectral", etc.
    )


# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import os

# def plot_comprehensive_figure_combined(results_per_l1, results_full, languages_to_plot=None, output_path=None):
#     """
#     Plots selected L1s, L2 speakers, and Native Baseline (LOO).
    
#     Parameters
#     ----------
#     results_per_l1 : dict
#         Results from results_per_L1.pkl (contains per_L1 data)
#     results_full : dict
#         Results from distance_to_native_full.pkl (contains L2 speakers and native LOO)
#     languages_to_plot : list, optional
#         List of languages to plot. If None, plots all available languages.
#     output_path : str, optional
#         Where to save the figure
#     """
#     fig, ax = plt.subplots(figsize=(14, 8))
#     layers = results_per_l1["layers"]
#     languages_data = results_per_l1["per_L1"]
    
#     # If no languages specified, use all available languages
#     if languages_to_plot is None:
#         languages_to_plot = list(languages_data.keys())
    
#     # 1. Plot Selected L1s
#     colors = plt.cm.tab20(np.linspace(0, 1, len(languages_to_plot)))

#     for lang, color in zip(languages_to_plot, colors):
#         if lang in languages_data:
#             mean = np.array(languages_data[lang]["mean_L2"])
#             ax.plot(layers, mean, label=lang.capitalize(), 
#                     color=color, linewidth=2, alpha=0.7)

#     # 2. Plot L2 SPEAKERS from the full results
#     ax.plot(layers, results_full["mean_L2"], 'o-', label='L2 speakers', 
#             color='blue', linewidth=3, markersize=6, zorder=10)
#     ax.fill_between(
#         layers,
#         results_full["ci_L2_lower"],
#         results_full["ci_L2_upper"],
#         alpha=0.2,
#         color='blue',
#         zorder=8
#     )

#     # 3. Plot NATIVE BASELINE (LOO)
#     ax.plot(layers, results_full["mean_native"], 's-', label='Native baseline (LOO)', 
#             color='green', linewidth=3, markersize=6, zorder=11)
#     ax.fill_between(
#         layers,
#         results_full["ci_native_lower"],
#         results_full["ci_native_upper"],
#         alpha=0.2,
#         color='green',
#         zorder=8
#     )

#     # Formatting
#     ax.set_xlabel("Layer Index", fontsize=12)
#     ax.set_ylabel("Cosine Distance to Native Centroid", fontsize=12)
#     ax.set_title("Distance to Native Centroid Across Layers", 
#                  fontsize=14, fontweight='bold')
    
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
#     ax.grid(True, linestyle='--', alpha=0.5)
    
#     plt.tight_layout()
    
#     if output_path:
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.show()


# if __name__ == "__main__":
#     # Charger les deux fichiers
#     results_per_l1 = pd.read_pickle("/home/nkhaous/myLLF/speech_accent/representations_distances/results/whisper/results_per_L1.pkl")
#     results_full = pd.read_pickle("/home/nkhaous/myLLF/speech_accent/representations_distances/results/distance_to_native_full.pkl")
    
#     # ========================================
#     # LISTE DES LANGUES À AFFICHER
#     # Modifiez cette liste pour changer les langues affichées
#     # ========================================
#     languages_to_plot = [
#         'spanish',
#         'arabic',
#         'french',
#         'mandarin',
#         'french',
#         'korean',
#         'portuguese',
#         'russian',
#         'dutch',
#         'turkish',
#         'german',
#         'polish',
#         'italian',
#         'japanese',
#         'macedonian',
#         'cantonese',
#         'farsi',
#         'vietnamese',
#         'swedish',
#         'romanian',
#         'amharic',
#     ]
    
#     # Pour afficher toutes les langues disponibles, utilisez None:
#     # languages_to_plot = None
    
#     # Définir le chemin de sortie
#     output_dir = "/home/nkhaous/myLLF/speech_accent/representations_distances/plots/"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # Créer la figure
#     plot_comprehensive_figure_combined(
#         results_per_l1, 
#         results_full,
#         languages_to_plot=languages_to_plot,
#         output_path=os.path.join(output_dir, 'combined_multilingual_fig.png')
#     )

















# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import os

# def plot_comprehensive_figure_combined(results_per_l1, results_full, languages_to_plot=None, output_path=None):
#     """
#     Plots selected L1s with confidence intervals, L2 speakers, and Native Baseline (LOO).
    
#     Parameters
#     ----------
#     results_per_l1 : dict
#         Results from results_per_L1.pkl (contains per_L1 data)
#     results_full : dict
#         Results from distance_to_native_full.pkl (contains L2 speakers and native LOO)
#     languages_to_plot : list, optional
#         List of languages to plot. If None, plots all available languages.
#     output_path : str, optional
#         Where to save the figure
#     """
#     fig, ax = plt.subplots(figsize=(14, 8))
#     layers = results_per_l1["layers"]
#     languages_data = results_per_l1["per_L1"]
    
#     # If no languages specified, use all available languages
#     if languages_to_plot is None:
#         languages_to_plot = list(languages_data.keys())
    
#     # 1. Plot Selected L1s with confidence intervals
#     colors = plt.cm.tab20(np.linspace(0, 1, len(languages_to_plot)))

#     for lang, color in zip(languages_to_plot, colors):
#         if lang in languages_data:
#             mean = np.array(languages_data[lang]["mean_L2"])
#             ax.plot(layers, mean, label=lang.capitalize(), 
#                     color=color, linewidth=2, alpha=0.7)
            
#             # Add confidence interval if available
#             if "ci_L2_lower" in languages_data[lang] and "ci_L2_upper" in languages_data[lang]:
#                 ci_lower = np.array(languages_data[lang]["ci_L2_lower"])
#                 ci_upper = np.array(languages_data[lang]["ci_L2_upper"])
#                 ax.fill_between(layers, ci_lower, ci_upper, 
#                                color=color, alpha=0.15)

#     # 2. Plot L2 SPEAKERS from the full results
#     ax.plot(layers, results_full["mean_L2"], 'o-', label='L2 speakers', 
#             color='blue', linewidth=3, markersize=6, zorder=10)
#     ax.fill_between(
#         layers,
#         results_full["ci_L2_lower"],
#         results_full["ci_L2_upper"],
#         alpha=0.2,
#         color='blue',
#         zorder=8
#     )

#     # 3. Plot NATIVE BASELINE (LOO)
#     ax.plot(layers, results_full["mean_native"], 's-', label='Native baseline (LOO)', 
#             color='green', linewidth=3, markersize=6, zorder=11)
#     ax.fill_between(
#         layers,
#         results_full["ci_native_lower"],
#         results_full["ci_native_upper"],
#         alpha=0.2,
#         color='green',
#         zorder=8
#     )

#     # Formatting
#     ax.set_xlabel("Layer Index", fontsize=12)
#     ax.set_ylabel("Cosine Distance to Native Centroid", fontsize=12)
#     ax.set_title("Distance to Native Centroid Across Layers", 
#                  fontsize=14, fontweight='bold')
    
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
#     ax.grid(True, linestyle='--', alpha=0.5)
    
#     plt.tight_layout()
    
#     if output_path:
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.show()


# if __name__ == "__main__":
#     # Charger les deux fichiers
#     results_per_l1 = pd.read_pickle("/home/nkhaous/myLLF/speech_accent/representations_distances/results/whisper/results_per_L1.pkl")
#     results_full = pd.read_pickle("/home/nkhaous/myLLF/speech_accent/representations_distances/results/distance_to_native_full.pkl")
    
#     # Vérifier la structure des données pour voir si les CI existent
#     print("Structure de results_per_l1:")
#     print(results_per_l1.keys())
#     print("\nStructure pour une langue (exemple 'spanish'):")
#     if 'spanish' in results_per_l1['per_L1']:
#         print(results_per_l1['per_L1']['spanish'].keys())
    
#     # ========================================
#     # LISTE DES LANGUES À AFFICHER
#     # ========================================
#     languages_to_plot = [
#         'spanish',
#         'arabic',
#         'french',
#         'mandarin',
#         'vietnamese',
#         'korean',
#         'portuguese',
#         'russian',
#         'polish',
#         'italian',
#         'dutch',
#         'turkish',
#         'japanese',
#         'german'
#     ]
    
#     # Définir le chemin de sortie
#     output_dir = "/home/nkhaous/myLLF/speech_accent/representations_distances/plots/"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # Créer la figure
#     plot_comprehensive_figure_combined(
#         results_per_l1, 
#         results_full,
#         languages_to_plot=languages_to_plot,
#         output_path=os.path.join(output_dir, 'combined_multilingual_fig.png')
#     )