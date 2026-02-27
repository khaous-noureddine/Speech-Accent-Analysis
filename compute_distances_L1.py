"""
Distance-to-Native Analysis for Speech Accent Representations (CORRECTED)
==========================================================================

Compute layer-wise distance between L2 accents and native English speakers.

Pipeline:
1. Word-level pooling: mean pool frames per word → word embeddings
2. Compute native centroid PER WORD (not per utterance!)
3. For each audio, compute distance for EACH word to its word-specific centroid
4. Average word-level distances → utterance-level score
5. Aggregate across speakers and plot

Author: Noureddine Khaous
"""

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

def pool_word_embeddings(df, repr_column="repr", normalize=True):
    """
    Mean pool frame-level embeddings to word-level.
    
    Parameters
    ----------
    df : pd.DataFrame
        Frame-aligned data with columns: [repr, annotation, speaker, ...]
    repr_column : str
        Column name containing embeddings
    normalize : bool
        Whether to L2-normalize after pooling
    
    Returns
    -------
    pd.DataFrame
        Word-level embeddings with columns: [speaker, annotation, word_emb]
    """
    logger.info("Pooling frame-level → word-level")
    
    # Group by (speaker, annotation) and mean pool
    word_df = (
        df.groupby(["speaker", "annotation"])[repr_column]
        .apply(lambda x: np.mean(np.vstack(x), axis=0))  # Mean pooling
        .reset_index()
        .rename(columns={repr_column: "word_emb"})
    )
    
    if normalize:
        # L2-normalize each word embedding
        word_df["word_emb"] = word_df["word_emb"].apply(
            lambda x: x / (np.linalg.norm(x) + 1e-8)
        )
    
    return word_df

# ============================================================================
# STEP 2: COMPUTE WORD-SPECIFIC CENTROIDS
# ============================================================================

def compute_word_centroids(word_df, meta_df):
    """
    Compute native centroid PER WORD.
    
    For each word in the vocabulary (e.g., "please", "call", "stella"):
    - Collect all native speaker embeddings for that word
    - Compute their centroid
    
    Parameters
    ----------
    word_df : pd.DataFrame
        Word-level embeddings with columns: [speaker, annotation, word_emb]
    meta_df : pd.DataFrame
        Metadata with columns: [speaker, is_native, L1]
    
    Returns
    -------
    dict
        {word: centroid_array} for each word in vocabulary
    """
    logger.info("Computing native centroids per word")
    
    # Merge metadata
    word_df = word_df.merge(meta_df, on="speaker")
    
    # Get unique words
    vocab = word_df["annotation"].unique()
    
    centroids = {}
    
    for word in tqdm(vocab, desc="Computing word centroids"):
        # Get all native embeddings for this word
        native_word_embs = word_df[
            (word_df["annotation"] == word) & (word_df["is_native"])
        ]["word_emb"].values
        
        if len(native_word_embs) == 0:
            logger.warning(f"No native examples for word '{word}', skipping")
            continue
        
        # Stack and compute centroid
        native_word_embs = np.vstack(native_word_embs)
        centroid = np.mean(native_word_embs, axis=0)
        
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
    word_df = word_df.merge(meta_df, on="speaker")
    
    # Compute distance for each word
    def compute_word_distance(row):
        word = row["annotation"]
        
        if word not in word_centroids:
            return np.nan  # Skip words without centroid
        
        centroid = word_centroids[word]
        return cosine(row["word_emb"], centroid)
    
    word_df["word_distance"] = word_df.apply(compute_word_distance, axis=1)
    
    # Remove NaN distances (words without centroids)
    word_df = word_df.dropna(subset=["word_distance"])
    
    # Average word-level distances per utterance
    utt_distances = (
        word_df.groupby(["speaker", "is_native", "native_language"])["word_distance"]
        .mean()
        .reset_index()
        .rename(columns={"word_distance": "distance"})
    )
    
    return utt_distances


# ============================================================================
# STEP 4: NATIVE BASELINE WITH LEAVE-ONE-OUT
# ============================================================================

def compute_native_baseline_loo(word_df, meta_df):
    """
    Leave-one-out baseline for native speakers.
    
    For each native speaker k:
    1. For each word, compute centroid WITHOUT speaker k
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
        Utterance-level distances for each native speaker
    """
    logger.info("Computing native leave-one-out baseline")
    
    # Merge metadata
    word_df = word_df.merge(meta_df, on="speaker")
    
    # Get native speakers
    native_speakers = word_df[word_df["is_native"]]["speaker"].unique()
    
    utterance_distances = []
    
    for speaker_k in tqdm(native_speakers, desc="LOO baseline"):
        word_distances = []
        
        # Get all words spoken by speaker k
        speaker_words = word_df[word_df["speaker"] == speaker_k]
        
        for _, row in speaker_words.iterrows():
            word = row["annotation"]
            word_emb = row["word_emb"]
            
            # Get all native embeddings for this word EXCEPT speaker k
            other_native_embs = word_df[
                (word_df["annotation"] == word) & 
                (word_df["is_native"]) & 
                (word_df["speaker"] != speaker_k)
            ]["word_emb"].values
            
            if len(other_native_embs) == 0:
                continue  # Skip if no other native examples
            
            # Compute LOO centroid
            loo_centroid = np.mean(np.vstack(other_native_embs), axis=0)
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
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    mean = np.mean(values)
    lower = np.percentile(bootstrap_means, (100 - ci) / 2)
    upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
    
    return mean, lower, upper


# ============================================================================
# STEP 6: PROCESS ALL LAYERS
# ============================================================================

def process_all_layers(input_dir, meta_df, layer_range=range(33), normalize=True):
    """
    Process all layers and compute distances.
    
    Parameters
    ----------
    input_dir : Path
        Root directory containing layer_X/ subdirectories
    meta_df : pd.DataFrame
        Metadata with columns: [filename, is_native, L1]
    layer_range : range
        Which layers to process (default: 0-32 for Whisper)
    normalize : bool
        Whether to L2-normalize embeddings
    
    Returns
    -------
    dict
        {
            'layers': [...],
            'mean_L2': [...],
            'ci_L2_lower': [...],
            'ci_L2_upper': [...],
            'mean_native': [...],
            'ci_native_lower': [...],
            'ci_native_upper': [...]
        }
    """
    results = {
        "layers": [],
        "mean_L2": [],
        "ci_L2_lower": [],
        "ci_L2_upper": [],
        "mean_native": [],
        "ci_native_lower": [],
        "ci_native_upper": []
    }
    
    for layer in tqdm(layer_range, desc="Processing layers"):
        layer_dir = input_dir / f"layer_{layer}"
        
        if not layer_dir.exists():
            logger.warning(f"Layer {layer} directory not found, skipping")
            continue
        
        logger.info(f"Processing layer_{layer}")
        
        # Get list of speakers to include from metadata
        speakers_to_include = set(meta_df["speaker"].unique())
        logger.info(f"Loading data for {len(speakers_to_include)} speakers from metadata")
        
        # 1. Load frame-aligned data (ONLY for speakers in metadata)
        dfs = []
        for pkl_file in layer_dir.glob("*.pkl"):
            # Extract speaker ID from filename (e.g., "ARABIC29_aligned.pkl" → "ARABIC29")
            speaker_id = pkl_file.stem.replace("_aligned", "").upper()
            
            # Only load if speaker is in metadata
            if speaker_id in speakers_to_include:
                dfs.append(pd.read_pickle(pkl_file))
        
        if not dfs:
            logger.warning(f"No matching files found for layer {layer}, skipping")
            continue
        
        df = pd.concat(dfs, axis=0)
        logger.info(f"Loaded {len(dfs)} files for layer {layer}")
        
        # Clean annotations
        df["annotation"] = df["annotation"].str.strip()
        df = df[df["annotation"] != ""]
        
        # 2. Pool to word-level
        word_df = pool_word_embeddings(df, repr_column="repr", normalize=normalize)
        
        # 3. Compute word-specific native centroids
        word_centroids = compute_word_centroids(word_df, meta_df)
        
        # 4. Compute distances per utterance (average of word-level distances)
        dist_df = compute_distances_per_utterance(word_df, meta_df, word_centroids)
        
        # 5. Compute native baseline (LOO, word-level then averaged)
        native_baseline = compute_native_baseline_loo(word_df, meta_df)
        
        # 6. Get L2 distances
        L2_distances = dist_df[~dist_df["is_native"]]["distance"].values
        
        # 7. Bootstrap CIs
        mean_L2, ci_L2_lower, ci_L2_upper = bootstrap_ci(L2_distances)
        mean_native, ci_native_lower, ci_native_upper = bootstrap_ci(native_baseline)
        
        # 8. Store results
        results["layers"].append(layer)
        results["mean_L2"].append(mean_L2)
        results["ci_L2_lower"].append(ci_L2_lower)
        results["ci_L2_upper"].append(ci_L2_upper)
        results["mean_native"].append(mean_native)
        results["ci_native_lower"].append(ci_native_lower)
        results["ci_native_upper"].append(ci_native_upper)
    
    return results


# ============================================================================
# STEP 7: PLOT FIGURE 1
# ============================================================================

def plot_figure1(results, output_path=None):
    """
    Plot distance-to-native vs layer.
    
    Parameters
    ----------
    results : dict
        Results from process_all_layers()
    output_path : Path, optional
        Where to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layers = results["layers"]
    
    # Plot L2 speakers
    ax.plot(layers, results["mean_L2"], 'o-', label='L2 speakers', color='blue', linewidth=2)
    ax.fill_between(
        layers,
        results["ci_L2_lower"],
        results["ci_L2_upper"],
        alpha=0.3,
        color='blue'
    )
    
    # Plot native baseline
    ax.plot(layers, results["mean_native"], 's-', label='Native baseline (LOO)', color='green', linewidth=2)
    ax.fill_between(
        layers,
        results["ci_native_lower"],
        results["ci_native_upper"],
        alpha=0.3,
        color='green'
    )
    
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Cosine Distance to Native Centroid", fontsize=12)
    ax.set_title("Distance to Native Centroid Across Layers", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {output_path}")
    
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute distance-to-native across layers (WORD-LEVEL CENTROIDS)"
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing layer_X/ subdirectories")
    parser.add_argument("--metadata", type=Path, required=True, help="Metadata CSV with columns: speaker, is_native, native_language")
    parser.add_argument("--output_results", type=Path, required=True, help="Output pickle file for results")
    parser.add_argument("--output_figure", type=Path, default=None, help="Output path for figure (optional)")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings")
    parser.add_argument("--min_layer", type=int, default=0)
    parser.add_argument("--max_layer", type=int, default=32)
    
    args = parser.parse_args()
    
    # Load metadata
    logger.info(f"Loading metadata from {args.metadata}")
    meta_df = pd.read_csv(args.metadata)
    
    # Ensure required columns exist
    assert "speaker" in meta_df.columns, "Missing 'speaker' column in metadata"
    assert "is_native" in meta_df.columns, "Missing 'is_native' column in metadata"
    assert "native_language" in meta_df.columns, "Missing 'native_language' column in metadata"
    
    logger.info(f"Total samples: {len(meta_df)}")
    logger.info(f"Native speakers: {meta_df['is_native'].sum()}")
    logger.info(f"L2 speakers: {(~meta_df['is_native']).sum()}")
    
    # Process all layers
    results = process_all_layers(
        input_dir=args.input_dir,
        meta_df=meta_df,
        layer_range=range(args.min_layer, args.max_layer + 1),
        normalize=args.normalize
    )
    
    # Save results
    logger.info(f"Saving results to {args.output_results}")
    args.output_results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_results, "wb") as f:
        pickle.dump(results, f)
    
    # Plot figure
    if args.output_figure:
        args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    plot_figure1(results, output_path=args.output_figure)
    
    logger.info("✅ Done!")