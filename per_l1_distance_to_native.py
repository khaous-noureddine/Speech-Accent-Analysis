#!/usr/bin/env python3
"""
Distance-to-Native (word-centric) across layers, per selected L1 groups.

We reuse your pipeline:
1) Word-level pooling: mean pool frames per word → word embeddings
2) Compute native centroid PER WORD
3) For each speaker/audio: distance for EACH word to its word-specific centroid
4) Average word distances → utterance-level score
5) Aggregate by L1 group (subset) and compute mean + bootstrap CI per layer
6) Save results to a .pkl

Author: Noureddine Khaous (adapted)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from loguru import logger
from scipy.spatial.distance import cosine

# -----------------------------------------------------------------------------
# Config: L1 groups to keep (must match meta_df["native_language"] after lower())
# -----------------------------------------------------------------------------
SELECTED_L1 = [
    "spanish", "arabic", "mandarin", "french", "korean", "portuguese", "russian",
    "dutch", "turkish", "german", "polish", "italian", "japanese", "macedonian",
    "cantonese", "farsi", "vietnamese", "swedish", "romanian",
]

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def l2_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)

def bootstrap_ci(values, n_bootstrap=1000, ci=95, seed=0):
    """
    Bootstrap CI on the MEAN.
    Returns: (mean, lower, upper). If values empty -> (nan, nan, nan)
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        boot_means.append(sample.mean())
    boot_means = np.asarray(boot_means)

    mean = values.mean()
    alpha = (100 - ci) / 2
    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)
    return mean, lower, upper

# -----------------------------------------------------------------------------
# Step 1: Pool frames -> word embeddings (per speaker, per word)
# -----------------------------------------------------------------------------
def pool_word_embeddings(df: pd.DataFrame, repr_column="repr", normalize=True) -> pd.DataFrame:
    """
    Input df: frame-aligned rows with at least [speaker, annotation, repr]
    Output: word_df with [speaker, annotation, word_emb]
    """
    # clean
    df = df.copy()
    df["annotation"] = df["annotation"].astype(str).str.strip()
    df = df[df["annotation"] != ""]

    # mean pool frames within (speaker, word)
    word_df = (
        df.groupby(["speaker", "annotation"], sort=False)[repr_column]
        .apply(lambda x: np.mean(np.vstack(x.to_list()), axis=0))
        .reset_index()
        .rename(columns={repr_column: "word_emb"})
    )

    if normalize:
        word_df["word_emb"] = word_df["word_emb"].apply(l2_normalize)

    return word_df

# -----------------------------------------------------------------------------
# Step 2: native centroids per word
# -----------------------------------------------------------------------------
def compute_word_centroids(word_df: pd.DataFrame, meta_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Return dict {word: centroid} using ONLY native speakers.
    """
    merged = word_df.merge(meta_df[["speaker", "is_native"]], on="speaker", how="inner")
    native_only = merged[merged["is_native"]].copy()

    centroids = {}
    for word, grp in native_only.groupby("annotation", sort=False):
        embs = np.vstack(grp["word_emb"].to_list())
        c = embs.mean(axis=0)
        centroids[word] = l2_normalize(c)
    return centroids

# -----------------------------------------------------------------------------
# Step 3: distances per speaker (average across words)
# -----------------------------------------------------------------------------
def compute_distances_per_speaker(word_df: pd.DataFrame, meta_df: pd.DataFrame, word_centroids: dict) -> pd.DataFrame:
    """
    Compute per-speaker utterance-level score as mean of word distances to word-specific native centroid.
    Returns df with columns: [speaker, is_native, native_language, distance]
    """
    merged = word_df.merge(meta_df[["speaker", "is_native", "native_language"]], on="speaker", how="inner")

    def row_dist(row):
        w = row["annotation"]
        if w not in word_centroids:
            return np.nan
        return cosine(row["word_emb"], word_centroids[w])

    merged["word_distance"] = merged.apply(row_dist, axis=1)
    merged = merged.dropna(subset=["word_distance"])

    out = (
        merged.groupby(["speaker", "is_native", "native_language"], sort=False)["word_distance"]
        .mean()
        .reset_index()
        .rename(columns={"word_distance": "distance"})
    )
    return out

# -----------------------------------------------------------------------------
# Loader for each layer
# -----------------------------------------------------------------------------
def load_layer_frames(layer_dir: Path, speakers_to_include: set[str]) -> pd.DataFrame:
    """
    Each file is a .pkl holding frame-aligned rows (with 'repr','speaker','annotation', etc).
    We only load those whose speaker_id is in speakers_to_include.
    """
    dfs = []
    for pkl_file in layer_dir.glob("*.pkl"):
        # speaker id rule from your code
        speaker_id = pkl_file.stem.replace("_aligned", "").upper()
        if speaker_id in speakers_to_include:
            dfs.append(pd.read_pickle(pkl_file))

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, axis=0, ignore_index=True)

# -----------------------------------------------------------------------------
# Main processing across layers
# -----------------------------------------------------------------------------
def process_all_layers_per_l1(
    input_dir: Path,
    meta_df: pd.DataFrame,
    layer_range,
    selected_l1: list[str],
    normalize=True,
    n_bootstrap=1000,
    ci=95,
    seed=0,
) -> dict:
    """
    Returns:
    {
      'layers': [...],
      'per_L1': {
         'french': {'mean_L2': [...], 'ci_L2_lower': [...], 'ci_L2_upper': [...]},
         ...
      }
      # (optional) you can also store native baseline if you want later
    }
    """
    # normalize meta fields
    meta_df = meta_df.copy()
    meta_df["speaker"] = meta_df["speaker"].astype(str).str.upper()
    meta_df["native_language"] = meta_df["native_language"].astype(str).str.strip().str.lower()

    # keep only selected L1 for L2, plus natives
    selected_set = set([s.lower() for s in selected_l1])
    meta_keep = meta_df[(meta_df["is_native"]) | (meta_df["native_language"].isin(selected_set))].copy()

    speakers_to_include = set(meta_keep["speaker"].unique())
    logger.info(f"Speakers included: {len(speakers_to_include)} (natives + selected L1 groups)")

    # init results dict
    per_L1 = {
        l1: {"mean_L2": [], "ci_L2_lower": [], "ci_L2_upper": []}
        for l1 in selected_l1
    }
    layers_out = []

    for layer in tqdm(list(layer_range), desc="Processing layers"):
        layer_dir = input_dir / f"layer_{layer}"
        if not layer_dir.exists():
            logger.warning(f"Missing {layer_dir}, skipping.")
            continue

        logger.info(f"Layer {layer}: loading frames...")
        frames_df = load_layer_frames(layer_dir, speakers_to_include)
        if frames_df.empty:
            logger.warning(f"Layer {layer}: no data loaded, skipping.")
            continue

        # Ensure speaker column exists; if not, infer from file? (assuming it's present)
        if "speaker" not in frames_df.columns:
            raise ValueError("Frames dataframe must contain a 'speaker' column.")

        # standardize speaker ids
        frames_df["speaker"] = frames_df["speaker"].astype(str).str.upper()

        # Pool to word level
        word_df = pool_word_embeddings(frames_df, repr_column="repr", normalize=normalize)

        # Compute native centroids per word
        centroids = compute_word_centroids(word_df, meta_keep)

        # Compute per-speaker distances (mean over words)
        dist_df = compute_distances_per_speaker(word_df, meta_keep, centroids)

        # Store layer index
        layers_out.append(layer)

        # For each selected L1 group, compute mean+CI on L2 speakers of that group
        for l1 in selected_l1:
            l1_low = l1.lower()
            vals = dist_df[(~dist_df["is_native"]) & (dist_df["native_language"] == l1_low)]["distance"].values
            m, lo, hi = bootstrap_ci(vals, n_bootstrap=n_bootstrap, ci=ci, seed=seed + layer)
            per_L1[l1]["mean_L2"].append(m)
            per_L1[l1]["ci_L2_lower"].append(lo)
            per_L1[l1]["ci_L2_upper"].append(hi)

    return {"layers": layers_out, "per_L1": per_L1}

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Compute distance-to-native across layers for selected L1 groups (word-centroid method)."
    )
    p.add_argument("--input_dir", type=Path, required=True, help="Directory containing layer_X/ subdirectories")
    p.add_argument("--metadata", type=Path, required=True, help="CSV with columns: speaker,is_native,native_language")
    p.add_argument("--output_pkl", type=Path, required=True, help="Output .pkl path")
    p.add_argument("--min_layer", type=int, default=0)
    p.add_argument("--max_layer", type=int, default=32)
    p.add_argument("--normalize", action="store_true", help="L2-normalize pooled embeddings (recommended)")
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--ci", type=int, default=95)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logger.info(f"Loading metadata: {args.metadata}")
    meta_df = pd.read_csv(args.metadata)

    # Validate metadata columns
    for col in ["speaker", "is_native", "native_language"]:
        if col not in meta_df.columns:
            raise ValueError(f"Metadata missing required column: {col}")

    layer_range = range(args.min_layer, args.max_layer + 1)

    results = process_all_layers_per_l1(
        input_dir=args.input_dir,
        meta_df=meta_df,
        layer_range=layer_range,
        selected_l1=SELECTED_L1,
        normalize=args.normalize,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
        seed=args.seed,
    )

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_pkl, "wb") as f:
        pickle.dump(results, f)

    logger.info(f"✅ Saved per-L1 results to: {args.output_pkl}")
    logger.info("Keys: results['layers'], results['per_L1'][L1]['mean_L2'|'ci_L2_lower'|'ci_L2_upper']")

if __name__ == "__main__":
    main()
