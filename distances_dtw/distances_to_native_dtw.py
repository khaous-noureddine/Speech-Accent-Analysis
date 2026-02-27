"""
DTW Distance-to-Native Analysis for Speech Accent Representations
=================================================================

Mesure couche par couche la distance DTW entre locuteurs L2 et natifs américains
dans l'espace des représentations SSL/ASR (wav2vec2, HuBERT, Whisper...).

Stratégie d'approximation (évite O(N_L2 × N_native)) :
-------------------------------------------------------
  Pour chaque mot W et chaque speaker S (L2 ou natif US) :
    → Tirer k natifs US au hasard (seed fixe)
    → Calculer DTW(seq_S_W, seq_natif_j) pour j=1..k
    → Moyenne = estimation de E_native[DTW(S, native)] pour ce mot
  Puis moyenner sur tous les mots → score par speaker.

  Justification : par la loi des grands nombres, la moyenne empirique
  sur k tirages converge vers l'espérance réelle avec erreur en O(sigma/sqrt(k)).
  Avec k=5, le coût passe de O(N_L2 x N_native) à O(k x N_L2).

  Pour la baseline US : même protocole, mais on compare un natif US
  aux AUTRES natifs US (leave-one-out implicite via exclusion du speaker).

Distance locale DTW : cosine distance = 1 - cosine_similarity(frame_i, frame_j)
DTW normalisée par longueur du chemin optimal (évite le biais de longueur).
Librairie : librosa.sequence.dtw

Usage:
    python dtw_distance_to_native.py \
        --input_dir /path/to/layers \
        --metadata /path/to/metadata.csv \
        --output_pkl results/dtw_results.pkl \
        --output_figure results/dtw_figure.png \
        --k 5 \
        --band_ratio 0.1 \
        --downsample 2 \
        --min_layer 0 \
        --max_layer 24 \
        --seed 42 \
        --normalize

Author: Noureddine Khaous
"""

import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa

from loguru import logger
from tqdm.auto import tqdm

tqdm.pandas()


# ============================================================================
# STEP 1 : COLLECTER LES SÉQUENCES DE FRAMES PAR MOT
# ============================================================================

def collect_word_sequences(df, meta_df, repr_column="repr", normalization="none", downsample=1):
    """
    Pour chaque (speaker, mot), collecter et empiler les frames -> matrice (T x d).

    Parameters
    ----------
    df : pd.DataFrame
        Données frame-level avec colonnes [speaker, annotation, repr]
    meta_df : pd.DataFrame
        Metadata avec colonnes [speaker, is_native, native_language, country]
    repr_column : str
        Colonne contenant les embeddings numpy
    normalize : bool
        L2-normaliser chaque frame avant DTW
    downsample : int
        Facteur de sous-échantillonnage temporel (1 = pas de downsampling)

    Returns
    -------
    pd.DataFrame
        Colonnes : [speaker, annotation, seq, is_native, native_language, country]
        seq : np.ndarray de shape (T, d)
    """
    logger.info("Collecting word sequences (frames -> matrices)")

    # Grouper par (speaker, annotation) -> matrice de frames
    seq_df = (
        df.groupby(["speaker", "annotation"])[repr_column]
        .apply(lambda x: np.vstack(x.values))
        .reset_index()
        .rename(columns={repr_column: "seq"})
    )

    # Downsample temporel si demandé
    if downsample > 1:
        seq_df["seq"] = seq_df["seq"].apply(lambda x: x[::downsample])
        logger.info(f"Downsampled sequences by factor {downsample}")

    # # L2-normaliser chaque frame
    # if normalize:
    #     def l2_norm_rows(mat):
    #         norms = np.linalg.norm(mat, axis=1, keepdims=True)
    #         return mat / (norms + 1e-8)
    #     seq_df["seq"] = seq_df["seq"].apply(l2_norm_rows)
    
    # Normalisation complète : centrage + L2-norm
    # if normalize:
    #     # 1. Calculer la moyenne globale sur TOUT le corpus de cette couche
    #     all_frames = np.vstack(seq_df["seq"].values)
    #     mu = all_frames.mean(axis=0, keepdims=True)  # shape: (1, d)
    #     logger.info(f"Computed global mean, shape: {mu.shape}")
        
    #     # 2. Centrer + L2-normaliser
    #     def center_and_l2_norm(mat):
    #         # Centrer
    #         centered = mat - mu
    #         # L2-norm par frame
    #         norms = np.linalg.norm(centered, axis=1, keepdims=True)
    #         return centered / (norms + 1e-8)
        
    #     seq_df["seq"] = seq_df["seq"].apply(center_and_l2_norm)
    #     logger.info("Applied centering + L2 normalization")
    
    
    
    if normalization == "l2":
        def l2_norm_rows(mat):
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            return mat / (norms + 1e-8)
        seq_df["seq"] = seq_df["seq"].apply(l2_norm_rows)

    elif normalization == "center":
        all_frames = np.vstack(seq_df["seq"].values)
        mu = all_frames.mean(axis=0, keepdims=True)
        seq_df["seq"] = seq_df["seq"].apply(lambda mat: mat - mu)

    elif normalization == "center_l2":
        all_frames = np.vstack(seq_df["seq"].values)
        mu = all_frames.mean(axis=0, keepdims=True)
        def center_and_l2_norm(mat):
            centered = mat - mu
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            return centered / (norms + 1e-8)
        seq_df["seq"] = seq_df["seq"].apply(center_and_l2_norm)
        
    elif normalization == "center_std":
        all_frames = np.vstack(seq_df["seq"].values)
        mu = all_frames.mean(axis=0, keepdims=True)
        std = all_frames.std(axis=0, keepdims=True)
        seq_df["seq"] = seq_df["seq"].apply(
            lambda mat: (mat - mu) / (std + 1e-8)
        )

    # normalization == "none" → rien à faire

    logger.info(f"Normalization applied: {normalization}")





    # Vérifier que toutes les séquences ont au moins 1 frame
    before = len(seq_df)
    seq_df = seq_df[seq_df["seq"].apply(lambda x: x.shape[0] > 0)]
    if len(seq_df) < before:
        logger.warning(f"Dropped {before - len(seq_df)} empty sequences")

    # Merger avec metadata
    seq_df = seq_df.merge(
        meta_df[["speaker", "is_native", "native_language", "country"]],
        on="speaker",
        how="inner"
    )

    logger.info(
        f"Collected {len(seq_df)} word sequences "
        f"for {seq_df['speaker'].nunique()} speakers"
    )
    return seq_df


# ============================================================================
# STEP 2 : DTW NORMALISÉE AVEC LIBROSA
# ============================================================================

# def dtw_cosine_normalized(seq_a, seq_b, band_ratio=0.1):
#     """
#     Calcule la distance DTW normalisée entre deux séquences d'embeddings.

#     Distance locale : cosine distance = 1 - cosine_similarity(frame_i, frame_j)
#     Fenêtre Sakoe-Chiba : band_ratio x max(len_a, len_b)
#     Normalisation : D[-1, -1] / len(warping_path)

#     Parameters
#     ----------
#     seq_a : np.ndarray, shape (T_a, d)
#     seq_b : np.ndarray, shape (T_b, d)
#     band_ratio : float

#     Returns
#     -------
#     float
#     """
#     # Matrice de coût local : cosine distance
#     # Si vecteurs L2-normalisés : seq_a @ seq_b.T = similarités cosine
#     cost_matrix = 1.0 - (seq_a @ seq_b.T)
#     cost_matrix = np.clip(cost_matrix.astype(np.float32), 0.0, 2.0)

#     # Fenêtre Sakoe-Chiba
#     band_rad = max(1, int(band_ratio * max(seq_a.shape[0], seq_b.shape[0])))

#     # librosa retourne (D_accumulée, warping_path)
#     D, wp = librosa.sequence.dtw(
#         C=cost_matrix,
#         subseq=False,
#         band_rad=band_rad,
#         backtrack=True,
#     )

#     path_length = len(wp)
#     if path_length == 0:
#         return np.nan

#     return float(D[-1, -1]) / path_length

def dtw_cosine_normalized(seq_a, seq_b, band_ratio=0.1):
    """
    Calcule la distance DTW normalisée entre deux séquences d'embeddings.
    """
    from scipy.spatial.distance import cdist
    
    # Matrice de coût local : cosine distance (robuste, marche même sans L2-norm)
    cost_matrix = cdist(seq_a, seq_b, metric="cosine")
    
    # Fenêtre Sakoe-Chiba
    band_rad = max(1, int(band_ratio * max(seq_a.shape[0], seq_b.shape[0])))
    
    # DTW avec librosa
    D, wp = librosa.sequence.dtw(
        C=cost_matrix,
        subseq=False,
        band_rad=band_rad,
        backtrack=True,
    )
    
    path_length = len(wp)
    if path_length == 0:
        return np.nan
    
    return float(D[-1, -1]) / path_length


# ============================================================================
# STEP 3 : ESPÉRANCE DTW
# ============================================================================
# Approximer E[DTW(speaker, natifs_US)] sans calculer toutes les paires.
def expected_dtw_to_natives(
    speaker_seq,
    native_seqs,
    native_speakers,
    k,
    rng,
    band_ratio=0.1,
    exclude_speaker=None,
):
    """
    Estime E_native[DTW(speaker_seq, native)] par moyenne sur k natifs tirés aléatoirement.

    Pour la baseline US, exclude_speaker permet un LOO implicite.
    """
    # Filtrer le speaker lui-même si LOO
    if exclude_speaker is not None:
        filtered = [
            (seq, spk)
            for seq, spk in zip(native_seqs, native_speakers)
            if spk != exclude_speaker
        ]
        if not filtered:
            return np.nan
        native_seqs_use = [s for s, _ in filtered]
    else:
        native_seqs_use = native_seqs

    n_available = len(native_seqs_use)
    if n_available == 0:
        return np.nan

    replace = k > n_available
    indices = rng.choice(n_available, size=k, replace=replace)

    distances = []
    for idx in indices:
        d = dtw_cosine_normalized(speaker_seq, native_seqs_use[int(idx)], band_ratio=band_ratio)
        if not np.isnan(d):
            distances.append(d)

    return float(np.mean(distances)) if distances else np.nan


# ============================================================================
# STEP 4 : DISTANCES PAR SPEAKER
# ============================================================================

def compute_speaker_distances(seq_df, k, rng, band_ratio=0.1):
    """
    Pour chaque speaker : estime E_native[DTW] par mot, puis moyenne sur les mots.
    """
    logger.info("Computing DTW distances per speaker (sampled expectation)")

    # Index natifs US par mot
    us_mask = seq_df["is_native"] & (seq_df["country"].str.lower() == "usa")
    us_df = seq_df[us_mask]

    word_to_natives = {}
    for word, group in us_df.groupby("annotation"):
        word_to_natives[word] = {
            "seqs": group["seq"].tolist(),
            "speakers": group["speaker"].tolist(),
        }

    logger.info(f"Vocabulary covered by US natives: {len(word_to_natives)} words")

    results = []

    for speaker in tqdm(seq_df["speaker"].unique(), desc="Speakers"):
        speaker_rows = seq_df[seq_df["speaker"] == speaker]

        is_native = bool(speaker_rows["is_native"].iloc[0])
        native_language = speaker_rows["native_language"].iloc[0]
        country = str(speaker_rows["country"].iloc[0])
        is_us_native = is_native and country.lower() == "usa"

        word_distances = []

        for _, row in speaker_rows.iterrows():
            word = row["annotation"]
            if word not in word_to_natives:
                continue

            native_info = word_to_natives[word]
            dist = expected_dtw_to_natives(
                speaker_seq=row["seq"],
                native_seqs=native_info["seqs"],
                native_speakers=native_info["speakers"],
                k=k,
                rng=rng,
                band_ratio=band_ratio,
                exclude_speaker=speaker if is_us_native else None,
            )
            if not np.isnan(dist):
                word_distances.append(dist)

        if word_distances:
            results.append({
                "speaker": speaker,
                "is_native": is_native,
                "native_language": native_language,
                "country": country,
                "distance": float(np.mean(word_distances)),
                "n_words": len(word_distances),
            })
        else:
            logger.warning(f"No valid word distances for speaker '{speaker}'")

    return pd.DataFrame(results)


# ============================================================================
# STEP 5 : BOOTSTRAP CI
# ============================================================================

def bootstrap_ci(values, n_bootstrap=1000, ci=95):
    values = np.array(values)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    bootstrap_means = [
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ]
    mean = float(np.mean(values))
    lower = float(np.percentile(bootstrap_means, (100 - ci) / 2))
    upper = float(np.percentile(bootstrap_means, 100 - (100 - ci) / 2))
    return mean, lower, upper


# ============================================================================
# STEP 6 : FIGURE
# ============================================================================

def plot_figure(results, output_path=None):
    plt.close("all")
    fig, ax = plt.subplots(figsize=(14, 8))

    layers = results["layers"]
    l1_langs = list(results["per_L1"].keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(l1_langs), 1)))

    for lang, color in zip(l1_langs, colors):
        data = results["per_L1"][lang]
        mean = np.array(data["mean_L2"])
        lower = np.array(data["ci_L2_lower"])
        upper = np.array(data["ci_L2_upper"])
        valid = ~np.isnan(mean)
        if valid.sum() == 0:
            continue
        ax.plot(np.array(layers)[valid], mean[valid],
                label=lang.capitalize(), color=color, linewidth=1.8, alpha=0.8)
        ax.fill_between(np.array(layers)[valid], lower[valid], upper[valid],
                        color=color, alpha=0.12)

    # L2 global
    ax.plot(layers, results["mean_L2"], "o-", label="L2 speakers",
            color="blue", linewidth=3, markersize=5, zorder=10)
    ax.fill_between(layers, results["ci_L2_lower"], results["ci_L2_upper"],
                    color="blue", alpha=0.18, zorder=8)

    # Baseline US
    ax.plot(layers, results["mean_native"], "s-", label="US native baseline",
            color="green", linewidth=3, markersize=5, zorder=11)
    ax.fill_between(layers, results["ci_native_lower"], results["ci_native_upper"],
                    color="green", alpha=0.18, zorder=9)

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("DTW Distance to US Native (normalized)", fontsize=12)
    ax.set_title("DTW Distance to US Native Across Layers", fontsize=14, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved -> {output_path}")
    plt.close(fig)


# ============================================================================
# STEP 7 : BOUCLE PRINCIPALE
# ============================================================================

def process_all_layers(input_dir, meta_df, output_pkl, output_figure,
                       layer_range, k, band_ratio, downsample, seed, normalization="none"):
    # Reprise si pkl existe
    if output_pkl.exists():
        logger.info(f"Resuming from: {output_pkl}")
        with open(output_pkl, "rb") as f:
            results = pickle.load(f)
        done_layers = set(results["layers"])
    else:
        results = {
            "layers": [], "mean_L2": [], "ci_L2_lower": [], "ci_L2_upper": [],
            "mean_native": [], "ci_native_lower": [], "ci_native_upper": [],
            "per_L1": {},
        }
        done_layers = set()

    # Init per_L1
    l1_languages = (
        meta_df[~meta_df["is_native"]]["native_language"]
        .str.lower().dropna().unique()
    )
    for lang in l1_languages:
        if lang not in results["per_L1"]:
            results["per_L1"][lang] = {"mean_L2": [], "ci_L2_lower": [], "ci_L2_upper": []}

    rng = np.random.default_rng(seed)
    speakers_to_include = set(meta_df["speaker"].str.upper().unique())

    for layer in tqdm(layer_range, desc="Layers"):

        if layer in done_layers:
            logger.info(f"Layer {layer} already done, skipping")
            continue

        layer_dir = input_dir / f"layer_{layer}"
        if not layer_dir.exists():
            logger.warning(f"Layer {layer} not found, skipping")
            continue

        logger.info(f"\n{'='*60}\nProcessing layer {layer}\n{'='*60}")

        # Chargement
        dfs = []
        for pkl_file in sorted(layer_dir.glob("*.pkl")):
            speaker_id = pkl_file.stem.replace("_aligned", "").upper()
            if speaker_id in speakers_to_include:
                try:
                    dfs.append(pd.read_pickle(pkl_file))
                except Exception as e:
                    logger.warning(f"Failed to load {pkl_file}: {e}")

        if not dfs:
            logger.warning(f"No files for layer {layer}, skipping")
            continue

        df = pd.concat(dfs, axis=0, ignore_index=True)
        df["annotation"] = df["annotation"].str.strip()
        df = df[df["annotation"] != ""]
        df["speaker"] = df["speaker"].str.upper()

        # Vérification dimensions
        sample_dim = df["repr"].iloc[0].shape[0]
        bad_dim = df["repr"].apply(lambda x: x.shape[0] != sample_dim)
        if bad_dim.any():
            logger.warning(f"Dropping {bad_dim.sum()} frames with wrong embedding dim")
            df = df[~bad_dim]

        # Séquences par mot
        # seq_df = collect_word_sequences(df, meta_df, repr_column="repr",
        #                                  normalize=normalize, downsample=downsample)
        
        seq_df = collect_word_sequences(df, meta_df, repr_column="repr",
                                 normalization=normalization, downsample=downsample)
        
        
        if len(seq_df) == 0:
            continue

        # Distances par speaker
        dist_df = compute_speaker_distances(seq_df=seq_df, k=k, rng=rng, band_ratio=band_ratio)
        if len(dist_df) == 0:
            continue

        # Agrégation L2 global
        l2_dist = dist_df[~dist_df["is_native"]]["distance"].values
        if len(l2_dist) == 0:
            continue
        mean_L2, ci_L2_lo, ci_L2_hi = bootstrap_ci(l2_dist)

        # Agrégation baseline US
        us_mask = dist_df["is_native"] & (dist_df["country"].str.lower() == "usa")
        nat_dist = dist_df[us_mask]["distance"].values
        mean_nat, ci_nat_lo, ci_nat_hi = bootstrap_ci(nat_dist)

        results["layers"].append(layer)
        results["mean_L2"].append(mean_L2)
        results["ci_L2_lower"].append(ci_L2_lo)
        results["ci_L2_upper"].append(ci_L2_hi)
        results["mean_native"].append(mean_nat)
        results["ci_native_lower"].append(ci_nat_lo)
        results["ci_native_upper"].append(ci_nat_hi)

        # Par L1
        for lang in results["per_L1"].keys():
            lang_dist = dist_df[
                (~dist_df["is_native"]) &
                (dist_df["native_language"].str.lower() == lang)
            ]["distance"].values
            m, lo, hi = bootstrap_ci(lang_dist) if len(lang_dist) > 0 else (np.nan, np.nan, np.nan)
            results["per_L1"][lang]["mean_L2"].append(m)
            results["per_L1"][lang]["ci_L2_lower"].append(lo)
            results["per_L1"][lang]["ci_L2_upper"].append(hi)

        # Sauvegarde incrémentale
        output_pkl.parent.mkdir(parents=True, exist_ok=True)
        with open(output_pkl, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Saved -> {output_pkl}")

        if output_figure:
            output_figure.parent.mkdir(parents=True, exist_ok=True)
            plot_figure(results, output_path=output_figure)

        done_layers.add(layer)

    logger.info("Done!")
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DTW Distance-to-Native across SSL/ASR layers (sampled expectation)"
    )
    parser.add_argument("--input_dir",     type=Path, required=True)
    parser.add_argument("--metadata",      type=Path, required=True)
    parser.add_argument("--output_pkl",    type=Path, required=True)
    parser.add_argument("--output_figure", type=Path, default=None)
    parser.add_argument("--k",             type=int,   default=5,
                        help="Nb de natifs US tirés par mot (defaut: 5)")
    parser.add_argument("--band_ratio",    type=float, default=0.1,
                        help="Fenetre Sakoe-Chiba (defaut: 0.1)")
    parser.add_argument("--downsample",    type=int,   default=1,
                        help="Facteur downsampling temporel (defaut: 1)")
    parser.add_argument("--min_layer",     type=int,   default=0)
    parser.add_argument("--max_layer",     type=int,   default=32)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--normalize",     action="store_true",
                        help="L2-normaliser les embeddings avant DTW")
    parser.add_argument("--languages",     nargs="+",  default=None,
                        help="Liste des L1 à traiter (ex: spanish french arabic). "
                        "Si non spécifié, toutes les langues sont traitées.")
    parser.add_argument(
    "--normalization",
    type=str,
    default="none",
    choices=["none", "l2", "center", "center_l2"],
    help="Type de normalisation: none | l2 | center | center_l2"
    )

    args = parser.parse_args()

    meta_df = pd.read_csv(args.metadata)
    
    # Filtrer les L2 par langue si spécifié
    if args.languages is not None:
        langs_lower = [l.lower() for l in args.languages]
        l2_mask = meta_df["is_native"]  # garder tous les natifs US
        l2_lang_mask = (
            ~meta_df["is_native"] &
            meta_df["native_language"].str.lower().isin(langs_lower)
        )
        meta_df = meta_df[l2_mask | l2_lang_mask]
        logger.info(f"Filtered to languages: {langs_lower}")
        logger.info(f"Remaining speakers: {len(meta_df)}")
    
    
    
    for col in ["speaker", "is_native", "native_language", "country"]:
        assert col in meta_df.columns, f"Missing column '{col}'"
    meta_df["speaker"] = meta_df["speaker"].str.upper()

    us_count = meta_df[meta_df["is_native"] & (meta_df["country"].str.lower() == "usa")].shape[0]
    logger.info(f"US natives: {us_count} | L2 speakers: {(~meta_df['is_native']).sum()}")
    logger.info(f"L1 langs: {sorted(meta_df[~meta_df['is_native']]['native_language'].dropna().unique())}")

    if us_count == 0:
        logger.error("No US native speakers found!")
        exit(1)

    process_all_layers(
        input_dir=args.input_dir,
        meta_df=meta_df,
        output_pkl=args.output_pkl,
        output_figure=args.output_figure,
        layer_range=range(args.min_layer, args.max_layer + 1),
        k=args.k,
        band_ratio=args.band_ratio,
        downsample=args.downsample,
        seed=args.seed,
        normalization=args.normalization,
    )