"""
Distance-to-Native Analysis for Speech Accent Representations (CORRECTED)
==========================================================================

Compute layer-wise distance between L2 accents and native English speakers.

Pipeline:
1. Word-level pooling: mean pool frames per word → word embeddings
2. Compute native centroid PER WORD (not per utterance!) - ONLY US SPEAKERS
3. For each audio, compute distance for EACH word to its word-specific centroid
4. Average word-level distances → utterance-level score
5. Aggregate across speakers and plot

Author: Noureddine Khaous
"""


import matplotlib.pyplot as plt
import numpy as np

# Fixed language order (optional but recommended)
DEFAULT_LANG_ORDER = [
    "spanish",
    "french",
    "mandarin",
    "arabic",
    "korean",
    "german",
    "vietnamese",
]

# Stable tab10 colors
TAB10 = plt.cm.tab10(np.linspace(0, 1, 10))

LANG2COLOR = {
    "spanish": TAB10[0],
    "french": TAB10[1],
    "mandarin": TAB10[2],
    "arabic": TAB10[3],
    "korean": TAB10[4],
    "dutch": TAB10[5],
    "german": TAB10[6],
    "italian": TAB10[7],
    "turkish": TAB10[8],
    "vietnamese": TAB10[9],
}


import pickle
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from loguru import logger
from tqdm.auto import tqdm
from scipy.spatial.distance import cosine

tqdm.pandas()


# ============================================================================
# STEP 1: POOLING (Frame → Word only)
# ============================================================================

def pool_word_embeddings(df, repr_column="repr", normalization="none"):
    logger.info(f"Pooling frame-level → word-level (normalization: {normalization})")
    
    # Mean pool
    word_df = (
        df.groupby(["speaker", "annotation"])[repr_column]
        .apply(lambda x: np.mean(np.vstack(x), axis=0))
        .reset_index()
        .rename(columns={repr_column: "word_emb"})
    )
    
    # Normalisation après pooling
    if normalization == "l2":
        word_df["word_emb"] = word_df["word_emb"].apply(
            lambda x: x / (np.linalg.norm(x) + 1e-8)
        )
    
    elif normalization == "center":
        # Soustraire la moyenne globale du corpus
        all_embs = np.vstack(word_df["word_emb"].values)
        mu = all_embs.mean(axis=0)
        word_df["word_emb"] = word_df["word_emb"].apply(lambda x: x - mu)
    
    elif normalization == "center_l2":
        # Centrer puis L2-normaliser (recommandation de ton prof)
        all_embs = np.vstack(word_df["word_emb"].values)
        mu = all_embs.mean(axis=0)
        word_df["word_emb"] = word_df["word_emb"].apply(
            lambda x: (x - mu) / (np.linalg.norm(x - mu) + 1e-8)
        )
    elif normalization == "center_std":
        # Centrer puis normaliser par écart-type
        all_embs = np.vstack(word_df["word_emb"].values)
        mu = all_embs.mean(axis=0)
        std = all_embs.std(axis=0)
        word_df["word_emb"] = word_df["word_emb"].apply(
            lambda x: (x - mu) / (std + 1e-8)
        )
    elif normalization == "none":
        pass  # No normalization
    else:
        raise ValueError(f"Unknown normalization type: {normalization}")
    return word_df

# ============================================================================
# STEP 2: COMPUTE WORD-SPECIFIC CENTROIDS (US SPEAKERS ONLY)
# ============================================================================

def compute_word_centroids(word_df, meta_df):
    """
    Compute native centroid PER WORD - ONLY FOR US SPEAKERS.
    
    For each word in the vocabulary (e.g., "please", "call", "stella"):
    - Collect all US native speaker embeddings for that word
    - Compute their centroid
    
    Parameters
    ----------
    word_df : pd.DataFrame
        Word-level embeddings with columns: [speaker, annotation, word_emb]
    meta_df : pd.DataFrame
        Metadata with columns: [speaker, is_native, country]
    
    Returns
    -------
    dict
        {word: centroid_array} for each word in vocabulary
    """
    logger.info("Computing native centroids per word (US SPEAKERS ONLY)")
    
    # Merge metadata
    word_df_with_meta = word_df.merge(meta_df[["speaker", "is_native", "country"]], on="speaker")
    
    # Get unique words
    vocab = word_df_with_meta["annotation"].unique()
    
    centroids = {}
    
    for word in tqdm(vocab, desc="Computing word centroids", leave=False):
        # Get all US native embeddings for this word
        us_native_word_embs = word_df_with_meta[
            (word_df_with_meta["annotation"] == word) & 
            (word_df_with_meta["is_native"]) &
            (word_df_with_meta["country"].str.lower() == "usa")
        ]["word_emb"].values
        
        if len(us_native_word_embs) == 0:
            logger.warning(f"No US native examples for word '{word}', skipping")
            continue
        
        # Stack and compute centroid
        us_native_word_embs = np.vstack(us_native_word_embs)
        centroid = np.mean(us_native_word_embs, axis=0)
        
        # L2-normalize
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        
        centroids[word] = centroid
    
    logger.info(f"Computed centroids for {len(centroids)} words")
    return centroids


# ============================================================================
# STEP 3: COMPUTE WORD-LEVEL DISTANCES, THEN AVERAGE PER UTTERANCE
# ============================================================================

def compute_distances_per_utterance(word_df, meta_df, word_centroids):
    """
    For each audio file:
    1. Compute distance for EACH word to its word-specific centroid
    2. Average these word-level distances → utterance-level score
    
    Parameters
    ----------
    word_df : pd.DataFrame
        Word-level embeddings
    meta_df : pd.DataFrame
        Metadata
    word_centroids : dict
        {word: centroid} dictionary
    
    Returns
    -------
    pd.DataFrame
        Per-utterance distances with columns: [speaker, is_native, native_language, distance]
    """
    logger.info("Computing word-level distances and averaging per utterance")
    
    # Merge metadata
    word_df_with_meta = word_df.merge(meta_df[["speaker", "is_native", "native_language"]], on="speaker")
    
    # Compute distance for each word
    def compute_word_distance(row):
        word = row["annotation"]
        
        if word not in word_centroids:
            return np.nan  # Skip words without centroid
        
        centroid = word_centroids[word]
        return cosine(row["word_emb"], centroid)
    
    word_df_with_meta["word_distance"] = word_df_with_meta.apply(compute_word_distance, axis=1)
    
    # Remove NaN distances (words without centroids)
    word_df_with_meta = word_df_with_meta.dropna(subset=["word_distance"])
    
    # Average word-level distances per utterance (per speaker)
    utt_distances = (
        word_df_with_meta.groupby(["speaker", "is_native", "native_language"])["word_distance"]
        .mean()
        .reset_index()
        .rename(columns={"word_distance": "distance"})
    )
    
    return utt_distances


# ============================================================================
# STEP 4: NATIVE BASELINE WITH LEAVE-ONE-OUT (US SPEAKERS ONLY)
# ============================================================================

def compute_native_baseline_loo(word_df, meta_df):
    """
    Leave-one-out baseline for US native speakers.
    
    For each US native speaker k:
    1. For each word, compute centroid WITHOUT speaker k (but still only US speakers)
    2. Compute word-level distances for speaker k
    3. Average → utterance-level score for k
    
    Parameters
    ----------
    word_df : pd.DataFrame
        Word-level embeddings
    meta_df : pd.DataFrame
        Metadata
    
    Returns
    -------
    list of float
        Utterance-level distances for each US native speaker
    """
    logger.info("Computing US native leave-one-out baseline")
    
    # Merge metadata
    word_df_with_meta = word_df.merge(meta_df[["speaker", "is_native", "country"]], on="speaker")
    
    # Get US native speakers only
    us_native_speakers = word_df_with_meta[
        (word_df_with_meta["is_native"]) & 
        (word_df_with_meta["country"].str.lower() == "usa")
    ]["speaker"].unique()
    
    logger.info(f"Found {len(us_native_speakers)} US native speakers for LOO baseline")
    
    utterance_distances = []
    
    for speaker_k in tqdm(us_native_speakers, desc="LOO baseline", leave=False):
        word_distances = []
        
        # Get all words spoken by speaker k
        speaker_words = word_df_with_meta[word_df_with_meta["speaker"] == speaker_k]
        
        for _, row in speaker_words.iterrows():
            word = row["annotation"]
            word_emb = row["word_emb"]
            
            # Get all US native embeddings for this word EXCEPT speaker k
            other_us_native_embs = word_df_with_meta[
                (word_df_with_meta["annotation"] == word) & 
                (word_df_with_meta["is_native"]) & 
                (word_df_with_meta["country"].str.lower() == "usa") &
                (word_df_with_meta["speaker"] != speaker_k)
            ]["word_emb"].values
            
            if len(other_us_native_embs) == 0:
                continue  # Skip if no other US native examples
            
            # Compute LOO centroid
            loo_centroid = np.mean(np.vstack(other_us_native_embs), axis=0)
            loo_centroid = loo_centroid / (np.linalg.norm(loo_centroid) + 1e-8)
            
            # Distance to LOO centroid
            dist = cosine(word_emb, loo_centroid)
            word_distances.append(dist)
        
        # Average word-level distances for this speaker
        if word_distances:
            utterance_distances.append(np.mean(word_distances))
    
    return utterance_distances


# ============================================================================
# STEP 5: BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_ci(values, n_bootstrap=1000, ci=95):
    """
    Compute bootstrap confidence interval.
    
    Parameters
    ----------
    values : array-like
        Data points
    n_bootstrap : int
        Number of bootstrap samples
    ci : int
        Confidence interval percentage (e.g., 95)
    
    Returns
    -------
    tuple (mean, lower_ci, upper_ci)
    """
    values = np.array(values)
    
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    mean = np.mean(values)
    lower = np.percentile(bootstrap_means, (100 - ci) / 2)
    upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
    
    return mean, lower, upper


# ============================================================================
# STEP 6: PLOT FIGURE (now accepts partial results)
# ============================================================================

# def plot_figure(results, languages_to_plot=None, show_individual_l1=False, 
#                 output_path=None, include_loo=False):
#     """
#     Plot distance-to-native vs layer.
#     """

#     plt.close('all')
#     fig, ax = plt.subplots(figsize=(14, 8))

#     layers = results["layers"]

#     # -----------------------------
#     # Individual L1
#     # -----------------------------
#     if show_individual_l1 and "per_L1" in results:
#         if languages_to_plot is None:
#             languages_to_plot = list(results["per_L1"].keys())

#         colors = plt.cm.tab20(np.linspace(0, 1, len(languages_to_plot)))

#         for lang, color in zip(languages_to_plot, colors):
#             if lang in results["per_L1"]:
#                 mean = np.array(results["per_L1"][lang]["mean_L2"])
#                 ax.plot(layers, mean, label=lang.capitalize(),
#                         color=color, linewidth=2, alpha=0.7)

#                 ci_lower = np.array(results["per_L1"][lang]["ci_L2_lower"])
#                 ci_upper = np.array(results["per_L1"][lang]["ci_L2_upper"])
#                 ax.fill_between(layers, ci_lower, ci_upper,
#                                 color=color, alpha=0.15)

#     # -----------------------------
#     # L2 overall
#     # -----------------------------
#     ax.plot(layers, results["mean_L2"], 'o-',
#             label='L2 speakers',
#             color='blue', linewidth=3, markersize=6, zorder=10)

#     ax.fill_between(
#         layers,
#         results["ci_L2_lower"],
#         results["ci_L2_upper"],
#         alpha=0.2,
#         color='blue',
#         zorder=8
#     )

#     # -----------------------------
#     # Native baseline (ONLY IF REQUESTED)
#     # -----------------------------
#     if include_loo:
#         ax.plot(layers, results["mean_native"], 's-',
#                 label='US native baseline (LOO)',
#                 color='green', linewidth=3, markersize=6, zorder=11)

#         ax.fill_between(
#             layers,
#             results["ci_native_lower"],
#             results["ci_native_upper"],
#             alpha=0.2,
#             color='green',
#             zorder=8
#         )

#     # -----------------------------
#     # Cosmetics
#     # -----------------------------
#     ax.set_xlabel("Layer Index", fontsize=12)
#     ax.set_ylabel("Cosine Distance to US Native Centroid", fontsize=12)
#     ax.set_title("Distance to US Native Centroid Across Layers",
#                  fontsize=14, fontweight='bold')

#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
#     ax.grid(True, linestyle='--', alpha=0.5)

#     plt.tight_layout()

#     if output_path:
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Figure saved to {output_path}")

#     plt.close(fig)







# ============================================================================
# CONSTANTES COULEURS (stables entre modèles)
# ============================================================================

DEFAULT_LANG_ORDER = [
    "spanish", "french", "mandarin", "arabic",
    "korean", "german", "vietnamese",
]

TAB10 = plt.cm.tab10(np.linspace(0, 1, 10))

LANG2COLOR = {
    "spanish":    TAB10[0],
    "french":     TAB10[1],
    "mandarin":   TAB10[2],
    "arabic":     TAB10[3],
    "korean":     TAB10[4],
    "dutch":      TAB10[5],
    "german":     TAB10[6],
    "italian":    TAB10[7],
    "turkish":    TAB10[8],
    "vietnamese": TAB10[9],
}


def _to_float_list(xs):
    return [float(x) if x is not None else np.nan for x in xs]


# ============================================================================
# STEP 6: PLOT FIGURE
# ============================================================================

def plot_figure(
    results,
    languages_to_plot=None,
    show_individual_l1=False,
    output_path=None,
    include_loo=False,
    show_ci_per_l1=True,
    show_ci_overall=True,
    model_name="",
    lang2color=LANG2COLOR,
):
    """
    Plot distance-to-native vs layer avec couleurs fixes par langue.
    """
    plt.close("all")
    fig, ax = plt.subplots(figsize=(14, 8))

    layers = results["layers"]

    # ---- Per-L1 curves (couleurs fixes)
    if show_individual_l1 and "per_L1" in results and results["per_L1"]:
        langs = languages_to_plot or DEFAULT_LANG_ORDER

        for lang in langs:
            key = lang.lower()
            if key not in results["per_L1"]:
                continue

            color = lang2color.get(key, "gray")
            mean  = _to_float_list(results["per_L1"][key]["mean_L2"])

            ax.plot(layers, mean,
                    label=key.capitalize(),
                    color=color, linewidth=2, alpha=0.85, zorder=3)

            if show_ci_per_l1:
                lo = _to_float_list(results["per_L1"][key]["ci_L2_lower"])
                hi = _to_float_list(results["per_L1"][key]["ci_L2_upper"])
                ax.fill_between(layers, lo, hi, color=color, alpha=0.12, zorder=2)

    # ---- Overall L2 (bleu)
    ax.plot(layers, _to_float_list(results["mean_L2"]),
            "o-", label="L2 speakers",
            color="blue", linewidth=3, markersize=6, zorder=10)

    if show_ci_overall:
        ax.fill_between(
            layers,
            _to_float_list(results["ci_L2_lower"]),
            _to_float_list(results["ci_L2_upper"]),
            alpha=0.2, color="blue", zorder=8,
        )

    # ---- Native baseline LOO (vert) — seulement si demandé
    if include_loo and not all(np.isnan(_to_float_list(results["mean_native"]))):
        ax.plot(layers, _to_float_list(results["mean_native"]),
                "s-", label="US native baseline (LOO)",
                color="green", linewidth=3, markersize=6, zorder=11)

        if show_ci_overall:
            ax.fill_between(
                layers,
                _to_float_list(results["ci_native_lower"]),
                _to_float_list(results["ci_native_upper"]),
                alpha=0.2, color="green", zorder=8,
            )

    # ---- Cosmetics
    title = "Distance to US Native Centroid Across Layers"
    if model_name:
        title += f" — {model_name}"

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Cosine Distance to US Native Centroid", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved to {output_path}")

    plt.close(fig)





def compute_intra_cluster_loo(word_df, meta_df, cluster_col="native_language"):
    """
    Intra-cluster cohesion (LOO) within each L1 cluster.
    For each L2 speaker and each word: distance to LOO centroid of SAME L1 cluster for that word.
    Then average per speaker.
    """
    logger.info(f"Computing intra-cluster LOO distances (cluster={cluster_col})")

    df = word_df.merge(
        meta_df[["speaker", "is_native", "native_language"]],
        on="speaker",
        how="left"
    )

    # only L2 speakers for intra metric
    df = df[~df["is_native"]].copy()

    df[cluster_col] = df[cluster_col].astype(str).str.lower().str.strip()
    df["annotation"] = df["annotation"].astype(str).str.strip()

    def _sum_stack(arrs):
        return np.sum(np.vstack(arrs), axis=0)

    # sum/count per (cluster, word)
    grp_all = (
        df.groupby([cluster_col, "annotation"])["word_emb"]
        .agg(sum_emb=_sum_stack, count="count")
        .reset_index()
    )

    # sum/count per (cluster, word, speaker)
    grp_spk = (
        df.groupby([cluster_col, "annotation", "speaker"])["word_emb"]
        .agg(sum_emb_spk=_sum_stack, count_spk="count")
        .reset_index()
    )

    m = grp_spk.merge(grp_all, on=[cluster_col, "annotation"], how="left")

    def loo_centroid(row):
        denom = row["count"] - row["count_spk"]
        if denom <= 0:
            return None
        c = (row["sum_emb"] - row["sum_emb_spk"]) / float(denom)
        c = c / (np.linalg.norm(c) + 1e-8)
        return c

    m["loo_centroid"] = m.apply(loo_centroid, axis=1)

    # join back per speaker-word embedding
    speaker_word = df.groupby(["speaker", cluster_col, "annotation"])["word_emb"].first().reset_index()

    tmp = speaker_word.merge(
        m[[cluster_col, "annotation", "speaker", "loo_centroid"]],
        on=[cluster_col, "annotation", "speaker"],
        how="left"
    )

    def dist_row(r):
        c = r["loo_centroid"]
        if c is None:
            return np.nan
        return cosine(r["word_emb"], c)

    tmp["word_distance_intra"] = tmp.apply(dist_row, axis=1)
    tmp = tmp.dropna(subset=["word_distance_intra"])

    # average per speaker
    out = (
        tmp.groupby(["speaker", cluster_col])["word_distance_intra"]
        .mean()
        .reset_index()
        .rename(columns={"word_distance_intra": "distance_intra", cluster_col: "native_language"})
    )

    return out




# ============================================================================
# STEP 7: PROCESS ALL LAYERS WITH INCREMENTAL SAVING
# ============================================================================

# def process_all_layers(input_dir, meta_df, output_results_path, output_figure_path, model_name, 
#                        layer_range=range(33), normalization="none", compute_per_l1=False,
#                        languages_to_plot=None, show_individual_l1=False, include_loo=False):
#     """
#     Process all layers and compute distances.
#     SAVES AND PLOTS AFTER EACH LAYER.
    
#     Parameters
#     ----------
#     input_dir : Path
#         Root directory containing layer_X/ subdirectories
#     meta_df : pd.DataFrame
#         Metadata with columns: [speaker, is_native, native_language, country]
#     output_results_path : Path
#         Path to save results pickle
#     output_figure_path : Path
#         Path to save figure
#     layer_range : range
#         Which layers to process (default: 0-32 for Whisper)
#     normalization : str
#         Normalization to apply to embeddings ("none", "l2", "center", "center_l2")
#     compute_per_l1 : bool
#         Whether to compute statistics per L1 language
#     languages_to_plot : list, optional
#         List of languages to plot
#     show_individual_l1 : bool
#         Whether to show individual L1s
    
#     Returns
#     -------
#     dict
#         Final results
#     """
#     # results = {
#     #     "layers": [],
#     #     "mean_L2": [],
#     #     "ci_L2_lower": [],
#     #     "ci_L2_upper": [],
#     #     "mean_native": [],
#     #     "ci_native_lower": [],
#     #     "ci_native_upper": []
#     # }
    
#     # if compute_per_l1:
#     #     results["per_L1"] = {}
#     #     # Initialize per-L1 dictionaries
#     #     l1_languages = meta_df[~meta_df["is_native"]]["native_language"].str.lower().unique()
#     #     for lang in l1_languages:
#     #         results["per_L1"][lang] = {
#     #             "mean_L2": [],
#     #             "ci_L2_lower": [],
#     #             "ci_L2_upper": []
#     #         }
    
#     # Charger le pkl existant si disponible (pour reprendre)
#     if output_results_path.exists():
#         logger.info(f"Found existing results at {output_results_path}, loading...")
#         with open(output_results_path, "rb") as f:
#             results = pickle.load(f)
#         logger.info(f"Resuming from layer {max(results['layers']) + 1} (already done: {results['layers']})")
#     else:
#         results = {
#             "layers": [],
#             "mean_inter": [],
#             "ci_inter_lower": [],
#             "ci_inter_upper": [],
#             "mean_intra": [],
#             "ci_intra_lower": [],
#             "ci_intra_upper": [],
#             "mean_native": [],
#             "ci_native_lower": [],
#             "ci_native_upper": []
#         }
    
#     if compute_per_l1:
#         if "per_L1" not in results:
#             results["per_L1"] = {}
#         l1_languages = meta_df[~meta_df["is_native"]]["native_language"].str.lower().unique()
#         for lang in l1_languages:
#             if lang not in results["per_L1"]:
#                 results["per_L1"][lang] = {
#                     "mean_L2": [],
#                     "ci_L2_lower": [],
#                     "ci_L2_upper": []
#                 }
                
    
#     for layer in tqdm(layer_range, desc="Processing layers"):
#         layer_dir = input_dir / f"layer_{layer}"
        
#         if not layer_dir.exists():
#             logger.warning(f"Layer {layer} directory not found, skipping")
#             continue
        
#         logger.info(f"Processing layer_{layer}")
        
#         # Get list of speakers to include from metadata
#         speakers_to_include = set(meta_df["speaker"].unique())
#         logger.info(f"Loading data for {len(speakers_to_include)} speakers from metadata")
     
#         # 1. Load frame-aligned data (ONLY for speakers in metadata)
#         logger.info(f"Loading frame-aligned data for layer {layer} (only speakers in metadata)")
#         dfs = []
#         for pkl_file in layer_dir.glob("*.pkl"):
#             # Extract speaker ID from filename (e.g., "ARABIC29_aligned.pkl" → "ARABIC29")
#             speaker_id = pkl_file.stem.replace("_aligned", "").upper()
            
#             # Only load if speaker is in metadata
#             if speaker_id in speakers_to_include:
#                 dfs.append(pd.read_pickle(pkl_file))
        
#         if not dfs:
#             logger.warning(f"No matching files found for layer {layer}, skipping")
#             continue
        
#         df = pd.concat(dfs, axis=0, ignore_index=True)
#         logger.info(f"Loaded {len(dfs)} files for layer {layer}")
        
#         # Clean annotations
#         df["annotation"] = df["annotation"].str.strip()
#         df = df[df["annotation"] != ""]
        
#         # 2. Pool to word-level
#         word_df = pool_word_embeddings(df, repr_column="repr", normalization=normalization)
        
#         # 3. Compute word-specific US native centroids
#         word_centroids = compute_word_centroids(word_df, meta_df)
        
#         if len(word_centroids) == 0:
#             logger.error(f"No word centroids computed for layer {layer}, skipping")
#             continue
        
#         # 4. Compute distances per utterance (average of word-level distances)
#         dist_df = compute_distances_per_utterance(word_df, meta_df, word_centroids)
        
#         # calculer les distances intra-cluster LOO (optionnel, mais peut être intéressant à analyser)
#         # NEW: intra-cluster LOO within each L1
#         intra_df = compute_intra_cluster_loo(word_df, meta_df, cluster_col="native_language")
#         intra_vals = intra_df["distance_intra"].values
#         mean_intra, ci_intra_lower, ci_intra_upper = bootstrap_ci(intra_vals)
                
#         # 5. Compute US native baseline (LOO, word-level then averaged)
#         if include_loo:
#             native_baseline = compute_native_baseline_loo(word_df, meta_df)
#             mean_native, ci_native_lower, ci_native_upper = bootstrap_ci(native_baseline)
#         else:
#             native_baseline = None
#             mean_native, ci_native_lower, ci_native_upper = np.nan, np.nan, np.nan
                
                
#         # 6. Get L2 distances (all L2 speakers combined)
#         L2_distances = dist_df[~dist_df["is_native"]]["distance"].values
        
#         if len(L2_distances) == 0:
#             logger.warning(f"No L2 distances found for layer {layer}")
#             continue
        
#         # 7. Bootstrap CIs for overall L2
#         mean_L2, ci_L2_lower, ci_L2_upper = bootstrap_ci(L2_distances)
        
#         # 8. Store overall results
#         results["layers"].append(layer)
        
#         # results["mean_L2"].append(mean_L2)
#         # results["ci_L2_lower"].append(ci_L2_lower)
#         # results["ci_L2_upper"].append(ci_L2_upper)
        
#         results["mean_inter"].append(mean_L2)
#         results["ci_inter_lower"].append(ci_L2_lower)
#         results["ci_inter_upper"].append(ci_L2_upper)
        
#         results["mean_intra"].append(mean_intra)
#         results["ci_intra_lower"].append(ci_intra_lower)
#         results["ci_intra_upper"].append(ci_intra_upper)
        
#         results["mean_native"].append(mean_native)
#         results["ci_native_lower"].append(ci_native_lower)
#         results["ci_native_upper"].append(ci_native_upper)
        
#         # 9. Compute per-L1 statistics if requested
#         if compute_per_l1:
#             for lang in results["per_L1"].keys():
#                 # Get distances for this specific L1
#                 lang_distances = dist_df[
#                     (~dist_df["is_native"]) & 
#                     (dist_df["native_language"].str.lower() == lang)
#                 ]["distance"].values
                
#                 if len(lang_distances) > 0:
#                     mean_lang, ci_lang_lower, ci_lang_upper = bootstrap_ci(lang_distances)
#                     results["per_L1"][lang]["mean_L2"].append(mean_lang)
#                     results["per_L1"][lang]["ci_L2_lower"].append(ci_lang_lower)
#                     results["per_L1"][lang]["ci_L2_upper"].append(ci_lang_upper)
#                 else:
#                     # No data for this language at this layer
#                     results["per_L1"][lang]["mean_L2"].append(np.nan)
#                     results["per_L1"][lang]["ci_L2_lower"].append(np.nan)
#                     results["per_L1"][lang]["ci_L2_upper"].append(np.nan)
        
#         # 10. SAVE RESULTS AFTER EACH LAYER
#         logger.info(f"Saving results after layer {layer} to {output_results_path}")
#         output_results_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(output_results_path, "wb") as f:
#             pickle.dump(results, f)
        
#         # 11. PLOT FIGURE AFTER EACH LAYER
#         if output_figure_path:
#             logger.info(f"Plotting and saving figure after layer {layer} to {output_figure_path}")
#             output_figure_path.parent.mkdir(parents=True, exist_ok=True)

        
#         plot_figure(
#             results,
#             languages_to_plot=languages_to_plot,
#             show_individual_l1=show_individual_l1,
#             output_path=output_figure_path,
#             include_loo=include_loo,
#             show_ci_per_l1=False,       # ou False si tu veux pas les CI par langue
#             show_ci_overall=True,
#             model_name=model_name,      # ou passe-le en argument de process_all_layers
#         )
    
#     return results

def process_all_layers(
    input_dir,
    meta_df,
    output_results_path,
    model_name="",
    layer_range=range(33),
    normalization="none",
    compute_per_l1=False,
    include_loo=False,
    include_intra=False,
):
    """
    Process all layers and compute:
      1) inter-cluster distance: L2 -> US native word centroids (as before)
      2) intra-cluster cohesion: LOO within each L1 cluster (per word, then avg per speaker)

    Saves results to pickle after each processed layer.

    Parameters
    ----------
    input_dir : Path
        Root directory containing layer_X/ subdirectories
    meta_df : pd.DataFrame
        Metadata with columns: [speaker, is_native, native_language, country]
    output_results_path : Path
        Path to save results pickle
    model_name : str
        Optional name stored in results for traceability
    layer_range : range
        Which layers to process
    normalization : str
        Embedding normalization ("none", "l2", "center", "center_l2", "center_std")
    compute_per_l1 : bool
        Whether to compute stats per L1 language (inter + intra)
    include_loo : bool
        Whether to compute US native baseline LOO (costly)

    Returns
    -------
    dict
        Final results
    """
    # -------------------------
    # Resume or init results
    # -------------------------
    if output_results_path.exists():
        logger.info(f"Found existing results at {output_results_path}, loading...")
        with open(output_results_path, "rb") as f:
            results = pickle.load(f)

        if "layers" in results and len(results["layers"]) > 0:
            logger.info(
                f"Resuming from layer {max(results['layers']) + 1} "
                f"(already done: {results['layers']})"
            )
        else:
            logger.info("Existing results file found but empty 'layers'. Starting from scratch.")
    else:
        results = {
            "model_name": model_name,
            "normalization": normalization,
            "layers": [],

            # inter-cluster (L2 -> US natives)
            "mean_inter": [],
            "ci_inter_lower": [],
            "ci_inter_upper": [],

            # intra-cluster (LOO within L1 clusters)
            "mean_intra": [],
            "ci_intra_lower": [],
            "ci_intra_upper": [],

            # optional native baseline (US LOO)
            "mean_native": [],
            "ci_native_lower": [],
            "ci_native_upper": [],
        }

    # -------------------------
    # Ensure keys exist (for old pickles / backward compatibility)
    # THIS IS WHERE YOUR SNIPPET GOES ✅
    # -------------------------
    required_keys = [
        "model_name", "normalization", "layers",
        "mean_inter", "ci_inter_lower", "ci_inter_upper",
        "mean_intra", "ci_intra_lower", "ci_intra_upper",
        "mean_native", "ci_native_lower", "ci_native_upper",
    ]
    for k in required_keys:
        if k not in results:
            # strings for metadata, lists for series
            results[k] = "" if k in ["model_name", "normalization"] else []

    # keep metadata up to date if you rerun with different args
    results["model_name"] = model_name
    results["normalization"] = normalization

    # -------------------------
    # Init per-L1 containers if requested
    # -------------------------
    if compute_per_l1:
        if "per_L1_inter" not in results:
            results["per_L1_inter"] = {}
        if "per_L1_intra" not in results:
            results["per_L1_intra"] = {}

        l1_languages = (
            meta_df[~meta_df["is_native"]]["native_language"]
            .astype(str).str.lower().str.strip()
            .unique()
        )

        for lang in l1_languages:
            if lang not in results["per_L1_inter"]:
                results["per_L1_inter"][lang] = {"mean": [], "ci_lower": [], "ci_upper": []}
            if lang not in results["per_L1_intra"]:
                results["per_L1_intra"][lang] = {"mean": [], "ci_lower": [], "ci_upper": []}

    # -------------------------
    # Process layers
    # -------------------------
    for layer in tqdm(layer_range, desc="Processing layers"):
        layer_dir = input_dir / f"layer_{layer}"

        # Skip if missing directory
        if not layer_dir.exists():
            logger.warning(f"Layer {layer} directory not found, skipping")
            continue

        # Skip if already processed (resume)
        if layer in results["layers"]:
            logger.info(f"Layer {layer} already in results, skipping")
            continue

        logger.info(f"Processing layer_{layer}")

        # Speakers to include
        speakers_to_include = set(meta_df["speaker"].unique())
        logger.info(f"Loading data for {len(speakers_to_include)} speakers from metadata")

        # 1) Load frame-aligned data (only for speakers in metadata)
        dfs = []
        for pkl_file in layer_dir.glob("*.pkl"):
            speaker_id = pkl_file.stem.replace("_aligned", "").upper()
            if speaker_id in speakers_to_include:
                dfs.append(pd.read_pickle(pkl_file))

        if not dfs:
            logger.warning(f"No matching files found for layer {layer}, skipping")
            continue

        df = pd.concat(dfs, axis=0, ignore_index=True)
        logger.info(f"Loaded {len(dfs)} files for layer {layer}")

        # Clean annotations
        df["annotation"] = df["annotation"].astype(str).str.strip()
        df = df[df["annotation"] != ""]

        # 2) Pool frame -> word
        word_df = pool_word_embeddings(df, repr_column="repr", normalization=normalization)

        # 3) Compute word-specific US native centroids (inter metric)
        word_centroids = compute_word_centroids(word_df, meta_df)
        if len(word_centroids) == 0:
            logger.error(f"No word centroids computed for layer {layer}, skipping")
            continue

        # 4) Inter-cluster distances (avg per speaker in your current code)
        dist_df = compute_distances_per_utterance(word_df, meta_df, word_centroids)

        # overall L2 inter
        inter_vals = dist_df[~dist_df["is_native"]]["distance"].values
        if len(inter_vals) == 0:
            logger.warning(f"No L2 inter distances found for layer {layer}, skipping")
            continue
        mean_inter, ci_inter_lower, ci_inter_upper = bootstrap_ci(inter_vals)

        # 5) Intra-cluster cohesion (LOO within each L1 cluster)
        # intra_df = compute_intra_cluster_loo(word_df, meta_df, cluster_col="native_language")
        # intra_vals = intra_df["distance_intra"].values
        # if len(intra_vals) == 0:
        #     logger.warning(f"No L2 intra distances found for layer {layer}, skipping")
        #     continue
        # mean_intra, ci_intra_lower, ci_intra_upper = bootstrap_ci(intra_vals)
        
        
        
        if include_intra:
            intra_df = compute_intra_cluster_loo(word_df, meta_df, cluster_col="native_language")
            intra_vals = intra_df["distance_intra"].values

            if len(intra_vals) == 0:
                logger.warning(f"No L2 intra distances found for layer {layer}")
                mean_intra, ci_intra_lower, ci_intra_upper = np.nan, np.nan, np.nan
            else:
                mean_intra, ci_intra_lower, ci_intra_upper = bootstrap_ci(intra_vals)
        else:
            intra_df = None
            mean_intra, ci_intra_lower, ci_intra_upper = np.nan, np.nan, np.nan
    
    
    

        # 6) Optional: US native baseline LOO (your existing function)
        if include_loo:
            native_baseline = compute_native_baseline_loo(word_df, meta_df)
            mean_native, ci_native_lower, ci_native_upper = bootstrap_ci(native_baseline)
        else:
            mean_native, ci_native_lower, ci_native_upper = np.nan, np.nan, np.nan

        # 7) Store results for this layer
        results["layers"].append(layer)

        results["mean_inter"].append(mean_inter)
        results["ci_inter_lower"].append(ci_inter_lower)
        results["ci_inter_upper"].append(ci_inter_upper)

        results["mean_intra"].append(mean_intra)
        results["ci_intra_lower"].append(ci_intra_lower)
        results["ci_intra_upper"].append(ci_intra_upper)

        results["mean_native"].append(mean_native)
        results["ci_native_lower"].append(ci_native_lower)
        results["ci_native_upper"].append(ci_native_upper)

        # 8) Per-L1 stats (optional)
        if compute_per_l1:
            for lang in results["per_L1_inter"].keys():
                # --- inter per L1
                lang_inter = dist_df[
                    (~dist_df["is_native"]) &
                    (dist_df["native_language"].astype(str).str.lower().str.strip() == lang)
                ]["distance"].values
                if len(lang_inter) > 0:
                    m, lo, hi = bootstrap_ci(lang_inter)
                else:
                    m, lo, hi = np.nan, np.nan, np.nan
                results["per_L1_inter"][lang]["mean"].append(m)
                results["per_L1_inter"][lang]["ci_lower"].append(lo)
                results["per_L1_inter"][lang]["ci_upper"].append(hi)

                # --- intra per L1
                # lang_intra = intra_df[
                #     (intra_df["native_language"].astype(str).str.lower().str.strip() == lang)
                # ]["distance_intra"].values
                # if len(lang_intra) > 0:
                #     m, lo, hi = bootstrap_ci(lang_intra)
                # else:
                #     m, lo, hi = np.nan, np.nan, np.nan
                # results["per_L1_intra"][lang]["mean"].append(m)
                # results["per_L1_intra"][lang]["ci_lower"].append(lo)
                # results["per_L1_intra"][lang]["ci_upper"].append(hi)
                
                        # intra per L1 (seulement si include_intra)
                if include_intra and intra_df is not None:
                    lang_intra = intra_df[
                        (intra_df["native_language"].astype(str).str.lower().str.strip() == lang)
                    ]["distance_intra"].values
                    if len(lang_intra) > 0:
                        m, lo, hi = bootstrap_ci(lang_intra)
                    else:
                        m, lo, hi = np.nan, np.nan, np.nan

                    results["per_L1_intra"][lang]["mean"].append(m)
                    results["per_L1_intra"][lang]["ci_lower"].append(lo)
                    results["per_L1_intra"][lang]["ci_upper"].append(hi)
                else:
                    # keep alignment with layers
                    results["per_L1_intra"][lang]["mean"].append(np.nan)
                    results["per_L1_intra"][lang]["ci_lower"].append(np.nan)
                    results["per_L1_intra"][lang]["ci_upper"].append(np.nan)
                    
            

        # 9) Save after each layer
        logger.info(f"Saving results after layer {layer} to {output_results_path}")
        output_results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_results_path, "wb") as f:
            pickle.dump(results, f)

    logger.info("✅ Finished processing all layers")
    return results


# ============================================================================
# MAIN
# ============================================================================

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Compute distance-to-native across layers (WORD-LEVEL CENTROIDS - US SPEAKERS ONLY)"
#     )
#     parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing layer_X/ subdirectories")
#     parser.add_argument("--metadata", type=Path, required=True, help="Metadata CSV with columns: speaker, is_native, native_language, country")
#     parser.add_argument("--output_results", type=Path, required=True, help="Output pickle file for results")
#     parser.add_argument("--output_figure", type=Path, default=None, help="Output path for figure (optional)")
#     parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings")
#     parser.add_argument("--normalization",
#                         type=str,
#                         default="none",
#                         # choices=["none", "l2", "center", "center_l2"],
#                         help="Type de normalisation: none | l2 | center | center_l2", 
#                         required=True
#                         )
#     parser.add_argument("--min_layer", type=int, default=0)
#     parser.add_argument("--max_layer", type=int, default=32)
#     parser.add_argument("--compute_per_l1", action="store_true", help="Compute statistics per L1 language")
#     parser.add_argument("--show_individual_l1", action="store_true", help="Show individual L1 languages in plot")
#     parser.add_argument("--languages_to_plot", nargs="+", default=None, 
#                        help="List of languages to plot (e.g., spanish french mandarin). If not specified, plots all.")
#     parser.add_argument("--include_loo",
#                         action="store_true",
#                         help="Compute US native baseline using leave-one-out (costly). If not set, skips it."
#                         )
#     parser.add_argument("--plot_no_per_l1", action="store_true")
#     parser.add_argument("--plot_no_ci", action="store_true")
#     parser.add_argument("--plot_no_markers", action="store_true")
#     parser.add_argument("--model_name", type=str, default="", help="Model name to include in figure title (optional)")

#     args = parser.parse_args()
    
#     # Load metadata
#     logger.info(f"Loading metadata from {args.metadata}")
#     meta_df = pd.read_csv(args.metadata)
    
#     # Ensure required columns exist
#     assert "speaker" in meta_df.columns, "Missing 'speaker' column in metadata"
#     assert "is_native" in meta_df.columns, "Missing 'is_native' column in metadata"
#     assert "native_language" in meta_df.columns, "Missing 'native_language' column in metadata"
#     assert "country" in meta_df.columns, "Missing 'country' column in metadata"
    
#     # Count US native speakers
#     us_native_count = meta_df[
#         (meta_df["is_native"]) & 
#         (meta_df["country"].str.lower() == "usa")
#     ].shape[0]
    
#     logger.info(f"Total samples: {len(meta_df)}")
#     logger.info(f"US native speakers: {us_native_count}")
#     logger.info(f"L2 speakers: {(~meta_df['is_native']).sum()}")
    
#     if us_native_count == 0:
#         logger.error("No US native speakers found in metadata! Please check 'country' column for 'USA' values.")
#         exit(1)
    
#     # Process all layers WITH INCREMENTAL SAVING AND PLOTTING
#     results = process_all_layers(
#         input_dir=args.input_dir,
#         meta_df=meta_df,
#         output_results_path=args.output_results,
#         output_figure_path=args.output_figure,
#         layer_range=range(args.min_layer, args.max_layer + 1),
#         normalization=args.normalization,
#         compute_per_l1=args.compute_per_l1 or args.show_individual_l1,
#         languages_to_plot=args.languages_to_plot,
#         show_individual_l1=args.show_individual_l1,
#         include_loo=args.include_loo,
#         model_name=args.model_name
#     )
    
#     logger.info("✅ Done!")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute inter- and intra-cluster distances across layers"
    )

    parser.add_argument("--input_dir", type=Path, required=True,
                        help="Directory containing layer_X/ subdirectories")

    parser.add_argument("--metadata", type=Path, required=True,
                        help="Metadata CSV with columns: speaker, is_native, native_language, country")

    parser.add_argument("--output_results", type=Path, required=True,
                        help="Output pickle file for results")

    parser.add_argument("--normalization",
                        type=str,
                        default="none",
                        required=True,
                        help="Normalization: none | l2 | center | center_l2 | center_std")

    parser.add_argument("--min_layer", type=int, default=0)
    parser.add_argument("--max_layer", type=int, default=32)

    parser.add_argument("--compute_per_l1", action="store_true",
                        help="Compute statistics per L1 language")

    parser.add_argument("--include_loo", action="store_true",
                        help="Compute US native baseline using leave-one-out")

    parser.add_argument("--model_name", type=str, default="",
                        help="Model name stored in results metadata")
    
    parser.add_argument(
                        "--include_intra",
                        default=True,
                        action="store_true",
                        help="Compute intra-cluster cohesion (LOO within each L1 cluster). If not set, intra is skipped.")

    args = parser.parse_args()

    # -------------------------
    # Load metadata
    # -------------------------
    logger.info(f"Loading metadata from {args.metadata}")
    meta_df = pd.read_csv(args.metadata)

    # Ensure required columns exist
    assert "speaker" in meta_df.columns
    assert "is_native" in meta_df.columns
    assert "native_language" in meta_df.columns
    assert "country" in meta_df.columns

    # -------------------------
    # IMPORTANT: normalize metadata
    # (sinon mismatch avec speaker_id.upper())
    # -------------------------
    meta_df["speaker"] = meta_df["speaker"].astype(str).str.strip().str.upper()
    meta_df["country"] = meta_df["country"].astype(str).str.strip()
    meta_df["native_language"] = meta_df["native_language"].astype(str).str.strip()

    # -------------------------
    # Basic stats
    # -------------------------
    us_native_count = meta_df[
        (meta_df["is_native"]) &
        (meta_df["country"].str.lower() == "usa")
    ].shape[0]

    logger.info(f"Total samples: {len(meta_df)}")
    logger.info(f"US native speakers: {us_native_count}")
    logger.info(f"L2 speakers: {(~meta_df['is_native']).sum()}")

    if us_native_count == 0:
        logger.error("No US native speakers found in metadata!")
        exit(1)

    # -------------------------
    # Process layers (NO PLOT)
    # -------------------------
    results = process_all_layers(
        input_dir=args.input_dir,
        meta_df=meta_df,
        output_results_path=args.output_results,
        model_name=args.model_name,
        layer_range=range(args.min_layer, args.max_layer + 1),
        normalization=args.normalization,
        compute_per_l1=args.compute_per_l1,
        include_loo=args.include_loo,
        include_intra=args.include_intra,
    )

    logger.info("✅ Done!")