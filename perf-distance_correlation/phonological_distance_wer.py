"""
Correlation Analysis: Phonological Distance (PHOIBLE) vs WER
=============================================================

Pipeline :
1. Télécharger PHOIBLE depuis GitHub
2. Extraire les inventaires de phonèmes par langue
3. Calculer distance de Jaccard(L1, English)
4. Corréler avec WER moyen par L1

Distance de Jaccard :
    distance(L1, English) = 1 - |phonèmes(L1) ∩ phonèmes(English)|
                                 ─────────────────────────────────
                                 |phonèmes(L1) ∪ phonèmes(English)|

Plus la distance est grande → moins de phonèmes en commun → accent plus fort attendu.

Author: Noureddine Khaous
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, linregress
from loguru import logger


# ============================================================================
# MAPPING : noms de langues du corpus → noms ISO/PHOIBLE
# ============================================================================

# Certaines langues ont des noms différents entre ton corpus et PHOIBLE
# À compléter si besoin
LANGUAGE_NAME_TO_ISO = {
    "amharic":    "amh",
    "arabic":     "arb",   # Modern Standard Arabic
    "bengali":    "ben",
    "bulgarian":  "bul",
    "cantonese":  "yue",
    "dutch":      "nld",
    "farsi":      "pes",   # Persian
    "french":     "fra",
    "german":     "deu",
    "greek":      "ell",
    "hindi":      "hin",
    "italian":    "ita",
    "japanese":   "jpn",
    "korean":     "kor",
    "kurdish":    "kmr",   # Northern Kurdish
    "macedonian": "mkd",
    "mandarin":   "cmn",
    "miskito":    "miq",
    "nepali":     "nep",
    "pashto":     "pbu",
    "polish":     "pol",
    "portuguese": "por",
    "punjabi":    "pan",
    "romanian":   "ron",
    "russian":    "rus",
    "serbian":    "srp",
    "spanish":    "spa",
    "swedish":    "swe",
    "tagalog":    "tgl",
    "thai":       "tha",
    "turkish":    "tur",
    "ukrainian":  "ukr",
    "urdu":       "urd",
    "vietnamese": "vie",
    "english":    "eng",
}


# ============================================================================
# STEP 1 : CHARGER PHOIBLE
# ============================================================================

def load_phoible(phoible_path=None):
    """
    Charge PHOIBLE depuis un fichier local ou depuis GitHub.

    Parameters
    ----------
    phoible_path : Path, optional
        Chemin vers le CSV local. Si None, télécharge depuis GitHub.

    Returns
    -------
    pd.DataFrame
        PHOIBLE avec colonnes: [ISO6393, Phoneme, ...]
    """
    if phoible_path and Path(phoible_path).exists():
        logger.info(f"Loading PHOIBLE from local file: {phoible_path}")
        df = pd.read_csv(phoible_path, low_memory=False)
    else:
        url = "https://raw.githubusercontent.com/phoible/dev/master/data/phoible.csv"
        logger.info(f"Downloading PHOIBLE from GitHub: {url}")
        df = pd.read_csv(url, low_memory=False)
        if phoible_path:
            Path(phoible_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(phoible_path, index=False)
            logger.info(f"Saved PHOIBLE to {phoible_path}")

    logger.info(f"PHOIBLE loaded: {len(df)} rows, {df['ISO6393'].nunique()} languages")
    return df


# ============================================================================
# STEP 2 : EXTRAIRE INVENTAIRES DE PHONÈMES
# ============================================================================

def extract_phoneme_inventories(phoible_df, iso_codes):
    """
    Extrait l'inventaire de phonèmes pour chaque code ISO.

    Stratégie : si plusieurs inventaires existent pour une langue (plusieurs
    sources dans PHOIBLE), on prend l'union de tous les phonèmes → inventaire
    le plus complet possible.

    Parameters
    ----------
    phoible_df : pd.DataFrame
    iso_codes : list of str

    Returns
    -------
    dict
        {iso_code: set_of_phonemes}
    """
    inventories = {}

    for iso in iso_codes:
        lang_df = phoible_df[phoible_df["ISO6393"] == iso]

        if len(lang_df) == 0:
            logger.warning(f"ISO '{iso}' not found in PHOIBLE")
            inventories[iso] = set()
            continue

        phonemes = set(lang_df["Phoneme"].dropna().unique())
        inventories[iso] = phonemes
        logger.info(f"  {iso}: {len(phonemes)} phonemes")

    return inventories


# ============================================================================
# STEP 3 : DISTANCE DE JACCARD
# ============================================================================

def jaccard_distance(set_a, set_b):
    """
    Distance de Jaccard entre deux ensembles de phonèmes.
    = 1 - |A ∩ B| / |A ∪ B|

    Retourne NaN si les deux ensembles sont vides.
    """
    if len(set_a) == 0 or len(set_b) == 0:
        return np.nan

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return 1.0 - (intersection / union)


def compute_phonological_distances(inventories, reference_iso="eng"):
    """
    Calcule la distance de Jaccard entre chaque langue et l'anglais.

    Parameters
    ----------
    inventories : dict
        {iso_code: set_of_phonemes}
    reference_iso : str
        ISO code de la langue de référence (anglais = 'eng')

    Returns
    -------
    dict
        {iso_code: jaccard_distance}
    """
    if reference_iso not in inventories or len(inventories[reference_iso]) == 0:
        logger.error(f"Reference language '{reference_iso}' not found or empty!")
        return {}

    english_phonemes = inventories[reference_iso]
    logger.info(f"English phoneme inventory: {len(english_phonemes)} phonemes")

    distances = {}
    for iso, phonemes in inventories.items():
        if iso == reference_iso:
            continue
        d = jaccard_distance(phonemes, english_phonemes)
        if not np.isnan(d):
            distances[iso] = d
            logger.info(f"  {iso}: Jaccard distance = {d:.4f} "
                        f"(shared: {len(phonemes & english_phonemes)}/{len(phonemes | english_phonemes)})")
        else:
            logger.warning(f"  {iso}: Could not compute distance (empty inventory)")

    return distances


# ============================================================================
# STEP 4 : CALCULER WER MOYEN PAR L1
# ============================================================================

def compute_wer_by_l1(wer_df, wer_column, languages=None):
    """
    Calcule le WER moyen par L1.
    Les natifs (native_language == 'english') sont exclus automatiquement.
    """
    logger.info(f"Computing WER by L1 using column '{wer_column}'")

    if wer_column not in wer_df.columns:
        available = [c for c in wer_df.columns if "wer" in c]
        logger.error(f"Column '{wer_column}' not found! Available: {available}")
        return {}

    l2_df = wer_df[wer_df["native_language"] != "english"].copy()

    if languages is not None:
        languages_lower = [l.lower() for l in languages]
        l2_df = l2_df[l2_df["native_language"].str.lower().isin(languages_lower)]

    wer_by_l1 = (
        l2_df.groupby("native_language")[wer_column]
        .mean()
        .to_dict()
    )

    wer_by_l1 = {k.lower(): v for k, v in wer_by_l1.items()}
    logger.info(f"Computed WER for {len(wer_by_l1)} languages")
    return wer_by_l1


# ============================================================================
# STEP 5 : CORRÉLATION + BOOTSTRAP CI
# ============================================================================

def compute_correlation_with_bootstrap(x_by_lang, y_by_lang, x_label="X", y_label="Y",
                                        n_bootstrap=1000, ci=95):
    """
    Corrélation Spearman + Pearson + CI par bootstrap.
    Aligne automatiquement sur les langues communes.
    """
    common_langs = sorted(set(x_by_lang.keys()) & set(y_by_lang.keys()))

    if len(common_langs) < 3:
        logger.error(f"Not enough common languages ({len(common_langs)})")
        logger.info(f"  In {x_label}: {sorted(x_by_lang.keys())}")
        logger.info(f"  In {y_label}: {sorted(y_by_lang.keys())}")
        return None

    x = np.array([x_by_lang[l] for l in common_langs])
    y = np.array([y_by_lang[l] for l in common_langs])

    logger.info(f"Correlation on {len(common_langs)} languages")

    rho, p_spearman = spearmanr(x, y)
    r,   p_pearson  = pearsonr(x, y)

    # Bootstrap CI pour Spearman
    bootstrap_rhos = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(x), size=len(x), replace=True)
        rho_b, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(rho_b):
            bootstrap_rhos.append(rho_b)

    ci_lower = np.percentile(bootstrap_rhos, (100 - ci) / 2)
    ci_upper = np.percentile(bootstrap_rhos, 100 - (100 - ci) / 2)

    return {
        "spearman_rho": rho,
        "spearman_p":   p_spearman,
        "pearson_r":    r,
        "pearson_p":    p_pearson,
        "ci_lower":     ci_lower,
        "ci_upper":     ci_upper,
        "languages":    common_langs,
        "x":            x,
        "y":            y,
    }


# ============================================================================
# STEP 6 : SCATTER PLOT
# ============================================================================

def plot_correlation(corr_result, x_label="Phonological Distance (Jaccard)",
                     y_label="Word Error Rate (WER)",
                     title="Correlation: Phonological Distance vs WER",
                     output_path=None):
    """
    Scatter plot avec régression linéaire + stats box.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    langs = corr_result["languages"]
    x = corr_result["x"]
    y = corr_result["y"]

    ax.scatter(x, y, s=120, alpha=0.7, edgecolors="k", linewidths=1.5, zorder=3)

    for lang, xi, yi in zip(langs, x, y):
        ax.annotate(
            lang.capitalize(), (xi, yi),
            xytext=(6, 6), textcoords="offset points",
            fontsize=10, alpha=0.85, weight="bold"
        )

    # Régression linéaire
    slope, intercept, r_value, _, _ = linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, "r--", linewidth=2.5, alpha=0.7,
            label=f"Linear fit (R² = {r_value**2:.3f})")

    # Stats box
    rho = corr_result["spearman_rho"]
    p   = corr_result["spearman_p"]
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

    stats_text = (
        f"Spearman ρ = {rho:.3f} {sig}\n"
        f"p-value = {p:.4f}\n"
        f"95% CI: [{corr_result['ci_lower']:.3f}, {corr_result['ci_upper']:.3f}]\n"
        f"n = {len(langs)} languages"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    ax.set_xlabel(x_label, fontsize=13, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved to {output_path}")

    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Correlation: Phonological Distance (PHOIBLE Jaccard) vs WER"
    )
    parser.add_argument("--wer_data",     type=Path, required=True,
                        help="Path to speakers_with_metrics.csv")
    parser.add_argument("--wer_column",   type=str,  required=True,
                        help="WER column name (e.g. 'wer__whisper-large-v3')")
    parser.add_argument("--phoible_csv",  type=Path, default=None,
                        help="Path to local phoible.csv (téléchargé si absent)")
    parser.add_argument("--languages",    nargs="+", default=None,
                        help="Langues à inclure. Si absent, toutes les langues communes.")
    parser.add_argument("--model_name",   type=str,  default="Model",
                        help="Nom du modèle ASR pour le titre du plot")
    parser.add_argument("--output_plot",  type=Path, default=None)
    parser.add_argument("--output_stats", type=Path, default=None)

    args = parser.parse_args()

    # 1. WER par L1
    wer_df   = pd.read_csv(args.wer_data)
    wer_by_l1 = compute_wer_by_l1(wer_df, args.wer_column, languages=args.languages)

    if not wer_by_l1:
        logger.error("No WER data, exiting")
        exit(1)

    # 2. Langues à traiter (intersection avec le mapping ISO)
    langs_to_process = list(wer_by_l1.keys()) + ["english"]
    iso_codes = []
    lang_to_iso = {}
    for lang in langs_to_process:
        iso = LANGUAGE_NAME_TO_ISO.get(lang.lower())
        if iso:
            iso_codes.append(iso)
            lang_to_iso[lang] = iso
        else:
            logger.warning(f"No ISO code for '{lang}', skipping")

    iso_codes = list(set(iso_codes))
    logger.info(f"Languages to process: {len(iso_codes)} ISO codes")

    # 3. Charger PHOIBLE
    phoible_df = load_phoible(args.phoible_csv)

    # 4. Inventaires de phonèmes
    logger.info("Extracting phoneme inventories...")
    inventories = extract_phoneme_inventories(phoible_df, iso_codes)

    # 5. Distances de Jaccard par rapport à l'anglais
    logger.info("Computing Jaccard distances to English...")
    iso_distances = compute_phonological_distances(inventories, reference_iso="eng")

    # Convertir iso → nom de langue
    iso_to_lang = {v: k for k, v in lang_to_iso.items() if k != "english"}
    phon_dist_by_l1 = {}
    for iso, dist in iso_distances.items():
        lang = iso_to_lang.get(iso)
        if lang and lang in wer_by_l1:
            phon_dist_by_l1[lang] = dist

    logger.info(f"Phonological distances computed for {len(phon_dist_by_l1)} languages:")
    for lang, d in sorted(phon_dist_by_l1.items(), key=lambda x: x[1]):
        logger.info(f"  {lang:15s}: {d:.4f}")

    # 6. Corrélation
    corr_result = compute_correlation_with_bootstrap(
        phon_dist_by_l1, wer_by_l1,
        x_label="Phonological Distance (Jaccard)",
        y_label="WER"
    )

    if corr_result is None:
        exit(1)

    # 7. Résultats
    logger.info("\n" + "="*70)
    logger.info("CORRELATION: Phonological Distance vs WER")
    logger.info("="*70)
    logger.info(f"WER column      : {args.wer_column}")
    logger.info(f"Languages       : {corr_result['languages']}")
    logger.info(f"Spearman ρ = {corr_result['spearman_rho']:.4f} (p = {corr_result['spearman_p']:.4f})")
    logger.info(f"Pearson r  = {corr_result['pearson_r']:.4f} (p = {corr_result['pearson_p']:.4f})")
    logger.info(f"95% CI: [{corr_result['ci_lower']:.4f}, {corr_result['ci_upper']:.4f}]")

    p = corr_result["spearman_p"]
    if p < 0.001:   logger.info("*** HIGHLY SIGNIFICANT ***")
    elif p < 0.01:  logger.info("** SIGNIFICANT **")
    elif p < 0.05:  logger.info("* MARGINALLY SIGNIFICANT *")
    else:           logger.info("NOT SIGNIFICANT")
    logger.info("="*70)

    # 8. Plot
    plot_correlation(
        corr_result,
        x_label="Phonological Distance to English (Jaccard)",
        y_label="Word Error Rate (WER)",
        title=f"Phonological Distance vs WER — {args.model_name}",
        output_path=args.output_plot,
    )

    # 9. Sauvegarder stats
    if args.output_stats:
        stats = {
            "model":          args.model_name,
            "wer_column":     args.wer_column,
            "n_languages":    len(corr_result["languages"]),
            "languages":      corr_result["languages"],
            "spearman_rho":   float(corr_result["spearman_rho"]),
            "spearman_p":     float(corr_result["spearman_p"]),
            "pearson_r":      float(corr_result["pearson_r"]),
            "pearson_p":      float(corr_result["pearson_p"]),
            "ci_95_lower":    float(corr_result["ci_lower"]),
            "ci_95_upper":    float(corr_result["ci_upper"]),
            "phon_distances": {k: float(v) for k, v in phon_dist_by_l1.items()},
            "wers":           {k: float(v) for k, v in wer_by_l1.items()},
        }
        Path(args.output_stats).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_stats, "w") as f:
            import json
            json.dump(stats, f, indent=2)
        logger.info(f"Stats saved to {args.output_stats}")

    logger.info("✅ Done!")