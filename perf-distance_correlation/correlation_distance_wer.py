"""
Correlation Analysis: Distance to Native vs WER
================================================

Analyse la corrélation entre :
- Distance DTW/mean-pooling à l'anglais natif (par L1)
- WER moyen par L1 (Word Error Rate)

Pour un modèle donné (Whisper, wav2vec2, HuBERT).
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from loguru import logger


# ============================================================================
# STEP 1 : CHARGER LES DISTANCES (DTW ou mean-pooling)
# ============================================================================

def load_distance_results(results_pkl_path, target_layer=None):
    """
    Charge les résultats DTW ou mean-pooling.
    
    Parameters
    ----------
    results_pkl_path : Path
    target_layer : int, optional
        Si None, choisit la couche avec séparation L1 vs natifs maximale
    
    Returns
    -------
    dict, int
        {lang: distance_moyenne}, layer_used
    """
    logger.info(f"Loading distance results from {results_pkl_path}")
    
    with open(results_pkl_path, "rb") as f:
        results = pickle.load(f)
    
    # Auto-select layer si non spécifié
    if target_layer is None:
        layers = results["layers"]
        # Critère : maximise (mean_L2 - mean_native)
        separations = np.array(results["mean_L2"]) - np.array(results["mean_native"])
        target_layer = layers[np.argmax(separations)]
        logger.info(f"Auto-selected layer {target_layer} (max L2-native separation)")
    
    # Extraire distances par L1 à cette couche
    if target_layer not in results["layers"]:
        logger.error(f"Layer {target_layer} not in results!")
        return None, None
    
    layer_idx = results["layers"].index(target_layer)
    
    distances_by_l1 = {}
    for lang, data in results["per_L1"].items():
        dist = data["mean_L2"][layer_idx]
        if not np.isnan(dist):
            distances_by_l1[lang.lower()] = dist
    
    logger.info(f"Extracted distances for {len(distances_by_l1)} languages at layer {target_layer}")
    return distances_by_l1, target_layer


# ============================================================================
# STEP 2 : CALCULER WER MOYEN PAR L1
# ============================================================================

def compute_wer_by_l1(wer_df, wer_column):
    """
    Calcule le WER moyen par L1 language.
    Les natifs (native_language == 'english') sont exclus automatiquement.

    Parameters
    ----------
    wer_df : pd.DataFrame
        CSV avec colonnes: [speakerid, native_language, wer__model, ...]
        (le fichier speakers_with_metrics.csv contient déjà tout)
    wer_column : str
        Nom de la colonne WER à utiliser (ex: "wer__whisper-large-v3")

    Returns
    -------
    dict
        {lang: wer_moyenne} pour chaque L1
    """
    logger.info(f"Computing WER by L1 using column '{wer_column}'")

    # Vérifier que la colonne existe
    if wer_column not in wer_df.columns:
        available = [c for c in wer_df.columns if "wer" in c]
        logger.error(f"Column '{wer_column}' not found! Available WER columns: {available}")
        return {}

    # Filtrer L2 seulement (exclure les natifs anglais)
    l2_df = wer_df[wer_df["native_language"] != "english"].copy()

    logger.info(f"L2 speakers: {len(l2_df)} (excluded {len(wer_df) - len(l2_df)} native English)")

    # Moyenne par L1
    wer_by_l1 = (
        l2_df.groupby("native_language")[wer_column]
        .mean()
        .to_dict()
    )

    # Normaliser clés (lowercase)
    wer_by_l1 = {k.lower(): v for k, v in wer_by_l1.items()}

    logger.info(f"Computed WER for {len(wer_by_l1)} L1 languages")
    logger.info(f"Languages: {sorted(wer_by_l1.keys())}")

    return wer_by_l1


# ============================================================================
# STEP 3 : CORRÉLATION + BOOTSTRAP CI
# ============================================================================

def compute_correlation_with_bootstrap(distances, wers, n_bootstrap=1000, ci=95):
    """
    Corrélation Spearman + Pearson + CI par bootstrap.
    """
    # Aligner les langues communes
    common_langs = sorted(set(distances.keys()) & set(wers.keys()))

    if len(common_langs) < 3:
        logger.error(f"Not enough common languages ({len(common_langs)}), cannot compute correlation")
        logger.info(f"Languages in distances: {sorted(distances.keys())}")
        logger.info(f"Languages in WER: {sorted(wers.keys())}")
        return None

    x = np.array([distances[l] for l in common_langs])
    y = np.array([wers[l] for l in common_langs])

    logger.info(f"Computing correlation on {len(common_langs)} languages")

    # Spearman
    rho, p_spearman = spearmanr(x, y)

    # Pearson
    r, p_pearson = pearsonr(x, y)

    # Bootstrap CI pour Spearman
    bootstrap_rhos = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(x), size=len(x), replace=True)
        rho_boot, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(rho_boot):
            bootstrap_rhos.append(rho_boot)

    ci_lower = np.percentile(bootstrap_rhos, (100 - ci) / 2)
    ci_upper = np.percentile(bootstrap_rhos, 100 - (100 - ci) / 2)

    return {
        "spearman_rho": rho,
        "spearman_p": p_spearman,
        "pearson_r": r,
        "pearson_p": p_pearson,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "languages": common_langs,
        "distances": x,
        "wers": y,
    }


# ============================================================================
# STEP 4 : SCATTER PLOT
# ============================================================================

def plot_correlation(corr_result, model_name="Model", layer=None, output_path=None):
    """
    Scatter plot : Distance vs WER avec régression linéaire + stats.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    langs = corr_result["languages"]
    x = corr_result["distances"]
    y = corr_result["wers"]

    # Scatter
    ax.scatter(x, y, s=120, alpha=0.7, edgecolors="k", linewidths=1.5, zorder=3)

    # Labels par langue
    for lang, xi, yi in zip(langs, x, y):
        ax.annotate(
            lang.capitalize(),
            (xi, yi),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
            alpha=0.85,
            weight="bold"
        )

    # Régression linéaire (visualisation seulement)
    from scipy.stats import linregress
    slope, intercept, r_value, _, _ = linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, "r--", linewidth=2.5, alpha=0.7, label=f"Linear fit (R² = {r_value**2:.3f})")

    # Stats box
    rho = corr_result["spearman_rho"]
    p = corr_result["spearman_p"]
    ci_lo = corr_result["ci_lower"]
    ci_hi = corr_result["ci_upper"]

    significance = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

    stats_text = (
        f"Spearman ρ = {rho:.3f} {significance}\n"
        f"p-value = {p:.4f}\n"
        f"95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]\n"
        f"n = {len(langs)} languages"
    )

    ax.text(
        0.05, 0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6)
    )

    ax.set_xlabel("Distance to US Native (normalized)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Word Error Rate (WER)", fontsize=13, fontweight="bold")

    title = f"Correlation: Distance vs WER — {model_name}"
    if layer is not None:
        title += f" (Layer {layer})"
    ax.set_title(title, fontsize=15, fontweight="bold")

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved to {output_path}")

    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Correlation: Distance to native vs WER"
    )

    parser.add_argument(
        "--distance_results",
        type=Path,
        required=True,
        help="Path to DTW or mean-pooling results .pkl"
    )
    parser.add_argument(
        "--wer_data",
        type=Path,
        required=True,
        help="Path to speakers_with_metrics.csv (contient native_language + WER)"
    )
    parser.add_argument(
        "--wer_column",
        type=str,
        required=True,
        help="WER column name (e.g., 'wer__whisper-large-v3')"
    )
    parser.add_argument(
        "--target_layer",
        type=int,
        default=None,
        help="Layer to use. If not specified, auto-selects layer with max separation."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Model",
        help="Model name for plot title"
    )
    parser.add_argument(
        "--output_plot",
        type=Path,
        default=None,
        help="Path to save scatter plot"
    )
    parser.add_argument(
        "--output_stats",
        type=Path,
        default=None,
        help="Path to save correlation stats as JSON"
    )

    args = parser.parse_args()

    # 1. Charger distances
    distances_by_l1, layer_used = load_distance_results(
        args.distance_results,
        target_layer=args.target_layer
    )

    if distances_by_l1 is None:
        exit(1)

    # 2. Charger WER data
    wer_df = pd.read_csv(args.wer_data)

    # 3. Calculer WER par L1
    wer_by_l1 = compute_wer_by_l1(wer_df, args.wer_column)

    if not wer_by_l1:
        logger.error("No WER data computed, exiting")
        exit(1)

    # 4. Corrélation
    corr_result = compute_correlation_with_bootstrap(distances_by_l1, wer_by_l1)

    if corr_result is None:
        logger.error("Cannot compute correlation")
        exit(1)

    # 5. Afficher résultats
    logger.info("\n" + "="*70)
    logger.info("CORRELATION RESULTS")
    logger.info("="*70)
    logger.info(f"Distance layer used: {layer_used}")
    logger.info(f"WER column: {args.wer_column}")
    logger.info(f"Languages analyzed: {corr_result['languages']}")
    logger.info(f"Spearman ρ = {corr_result['spearman_rho']:.4f} (p = {corr_result['spearman_p']:.4f})")
    logger.info(f"Pearson r  = {corr_result['pearson_r']:.4f} (p = {corr_result['pearson_p']:.4f})")
    logger.info(f"95% CI: [{corr_result['ci_lower']:.4f}, {corr_result['ci_upper']:.4f}]")

    if corr_result['spearman_p'] < 0.001:
        logger.info("*** HIGHLY SIGNIFICANT ***")
    elif corr_result['spearman_p'] < 0.01:
        logger.info("** SIGNIFICANT **")
    elif corr_result['spearman_p'] < 0.05:
        logger.info("* MARGINALLY SIGNIFICANT *")
    else:
        logger.info("NOT SIGNIFICANT")

    logger.info("="*70)

    # 6. Plot
    plot_correlation(
        corr_result,
        model_name=args.model_name,
        layer=layer_used,
        output_path=args.output_plot
    )

    # 7. Sauvegarder stats (optionnel)
    if args.output_stats:
        import json

        stats = {
            "model": args.model_name,
            "wer_column": args.wer_column,
            "distance_layer": int(layer_used),
            "n_languages": len(corr_result["languages"]),
            "languages": corr_result["languages"],
            "spearman_rho": float(corr_result["spearman_rho"]),
            "spearman_p": float(corr_result["spearman_p"]),
            "pearson_r": float(corr_result["pearson_r"]),
            "pearson_p": float(corr_result["pearson_p"]),
            "ci_95_lower": float(corr_result["ci_lower"]),
            "ci_95_upper": float(corr_result["ci_upper"]),
            "distances": corr_result["distances"].tolist(),
            "wers": corr_result["wers"].tolist(),
        }

        args.output_stats.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_stats, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Stats saved to {args.output_stats}")

    logger.info("✅ Done!")