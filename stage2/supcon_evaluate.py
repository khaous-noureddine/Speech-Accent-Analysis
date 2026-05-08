"""
supcon_eval.py

Evaluation of supervised contrastive representations for accent invariance.

Measures whether the model has learned to:
  - bring together embeddings of the same utterance across different accents (alignment)
  - spread embeddings uniformly on the unit hypersphere (uniformity / no collapse)
  - retrieve same-utterance neighbours in embedding space (Recall@K)
  - discard accent information (low accent probe accuracy)
  - preserve linguistic content (high utterance probe accuracy)

Alignment is broken down into three complementary views:
  - alignment_pos   : mean L2² distance between positive pairs (same utt, diff speaker) ↓
  - alignment_neg   : mean L2² distance between negative pairs (diff utt, random sample) ↑
  - alignment_ratio : pos / neg  →  should decrease as training progresses              ↓
  - alignment_cos   : mean cosine similarity between positive pairs                     ↑

Each metric is computed in TWO spaces:
  - proj     : after the projection MLP + L2 norm [B, 256]  — what SupCon optimized
  - backbone : mean-pooled transformer output, L2 norm [B, 1024] — what Stage 3 ASR will use

Usage:
    python supcon_eval.py \\
        --checkpoint checkpoints/checkpoint_epoch030.pt \\
        --arctic_parquet_path data/processed/arctic/corpus.parquet \\
        --l2_arctic_parquet_path data/processed/l2_arctic/corpus.parquet \\
        --output_dir eval/epoch030 \\
        --split eval

    python supcon_eval.py \\
        --hf_model facebook/wav2vec2-large-xlsr-53 \\
        --arctic_parquet_path data/processed/arctic/corpus.parquet \\
        --l2_arctic_parquet_path data/processed/l2_arctic/corpus.parquet \\
        --output_dir eval/baseline

To call from train.py after each epoch:
    from supcon_eval import run_eval
    result = run_eval(model, eval_loader, device, metrics=args.eval_metrics)
    result.log(prefix=f"Epoch {epoch:03d}")
    for tag, val in result.tensorboard_scalars().items():
        writer.add_scalar(tag, val, epoch)
"""

from __future__ import annotations

import sys
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loguru import logger

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import StratifiedShuffleSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not found — probe accuracy metrics will be skipped.")

sys.path.insert(0, str(Path(__file__).parent.parent))
from stage2.supcon_data import SupConSpeechDataset
from stage2.supcon_xlsr import SupConXLSR


VALID_METRICS = {
    "alignment",        # pos L2², neg L2², ratio, cosine sim
    "uniformity",
    "retrieval_at_1",
    "retrieval_at_5",
    "retrieval_at_10",
    "accent_probe_acc",
    "utterance_probe_acc",
}


@dataclass
class EvalResult:
    """
    All evaluation metrics for one checkpoint / epoch.
    Every metric is computed in two spaces:
      _proj     : after projection MLP  (what SupCon directly optimized)
      _backbone : mean-pooled transformer output (what Stage 3 ASR will use)

    Alignment is split into four complementary signals:
      alignment_pos   ↓  mean L2² between positive pairs (same utt, diff speaker)
      alignment_neg   ↑  mean L2² between negative pairs (diff utt, random sample)
      alignment_ratio ↓  pos / neg  — the key training signal; should decrease every epoch
      alignment_cos   ↑  mean cosine similarity between positive pairs  (more readable)
    """

    # Alignment — positive / negative / ratio / cosine
    alignment_pos_proj:       float = float("nan")   # ↓
    alignment_pos_backbone:   float = float("nan")   # ↓

    alignment_neg_proj:       float = float("nan")   # ↑
    alignment_neg_backbone:   float = float("nan")   # ↑

    alignment_ratio_proj:     float = float("nan")   # ↓  pos/neg
    alignment_ratio_backbone: float = float("nan")   # ↓  pos/neg

    alignment_cos_proj:       float = float("nan")   # ↑  cosine sim of positive pairs
    alignment_cos_backbone:   float = float("nan")   # ↑

    # Uniformity
    uniformity_proj:          float = float("nan")   # ↓
    uniformity_backbone:      float = float("nan")   # ↓

    # Retrieval
    retrieval_at_1_proj:      float = float("nan")   # ↑
    retrieval_at_1_backbone:  float = float("nan")   # ↑

    retrieval_at_5_proj:      float = float("nan")   # ↑
    retrieval_at_5_backbone:  float = float("nan")   # ↑

    retrieval_at_10_proj:     float = float("nan")   # ↑
    retrieval_at_10_backbone: float = float("nan")   # ↑

    # Linear probes
    accent_probe_acc_proj:        float = float("nan")   # ↓ should be LOW
    accent_probe_acc_backbone:    float = float("nan")   # ↓

    utterance_probe_acc_proj:     float = float("nan")   # ↑ should be HIGH
    utterance_probe_acc_backbone: float = float("nan")   # ↑

    # Meta
    n_samples:    int = 0
    n_utterances: int = 0
    n_speakers:   int = 0
    n_corpora:    int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def log(self, prefix: str = "") -> None:
        tag = f"[{prefix}] " if prefix else ""
        logger.info(f"{tag}── Eval results ──────────────────────────────────────────")
        logger.info(f"{tag}  {'metric':<32} {'proj':>10} {'backbone':>10}")
        logger.info(f"{tag}  {'─'*54}")

        def fmt(v: float) -> str:
            return f"{v:.4f}" if not np.isnan(v) else "   —  "

        rows = [
            ("alignment_pos   ↓", self.alignment_pos_proj,       self.alignment_pos_backbone),
            ("alignment_neg   ↑", self.alignment_neg_proj,       self.alignment_neg_backbone),
            ("alignment_ratio ↓", self.alignment_ratio_proj,     self.alignment_ratio_backbone),
            ("alignment_cos   ↑", self.alignment_cos_proj,       self.alignment_cos_backbone),
            ("uniformity      ↓", self.uniformity_proj,          self.uniformity_backbone),
            ("retrieval@1     ↑", self.retrieval_at_1_proj,      self.retrieval_at_1_backbone),
            ("retrieval@5     ↑", self.retrieval_at_5_proj,      self.retrieval_at_5_backbone),
            ("retrieval@10    ↑", self.retrieval_at_10_proj,     self.retrieval_at_10_backbone),
            ("accent_probe    ↓", self.accent_probe_acc_proj,    self.accent_probe_acc_backbone),
            ("utterance_probe ↑", self.utterance_probe_acc_proj, self.utterance_probe_acc_backbone),
        ]
        for name, vp, vb in rows:
            logger.info(f"{tag}  {name:<32} {fmt(vp):>10} {fmt(vb):>10}")

        logger.info(
            f"{tag}  n_samples={self.n_samples}  "
            f"n_utterances={self.n_utterances}  "
            f"n_speakers={self.n_speakers}  "
            f"n_corpora={self.n_corpora}"
        )

    def tensorboard_scalars(self) -> dict[str, float]:
        """
        Flat dict suitable for writer.add_scalar() in train.py.

        Typical usage in train.py:
            result = run_eval(model, eval_loader, device, metrics=args.eval_metrics)
            result.log(prefix=f"Epoch {epoch:03d}")
            for tag, val in result.tensorboard_scalars().items():
                if not np.isnan(val):
                    writer.add_scalar(tag, val, epoch)
        """
        return {
            # Alignment
            "Eval/alignment_pos_proj":        self.alignment_pos_proj,
            "Eval/alignment_pos_backbone":    self.alignment_pos_backbone,
            "Eval/alignment_neg_proj":        self.alignment_neg_proj,
            "Eval/alignment_neg_backbone":    self.alignment_neg_backbone,
            "Eval/alignment_ratio_proj":      self.alignment_ratio_proj,
            "Eval/alignment_ratio_backbone":  self.alignment_ratio_backbone,
            "Eval/alignment_cos_proj":        self.alignment_cos_proj,
            "Eval/alignment_cos_backbone":    self.alignment_cos_backbone,
            # Uniformity
            "Eval/uniformity_proj":           self.uniformity_proj,
            "Eval/uniformity_backbone":       self.uniformity_backbone,
            # Retrieval
            "Eval/retrieval_at_1_proj":       self.retrieval_at_1_proj,
            "Eval/retrieval_at_1_backbone":   self.retrieval_at_1_backbone,
            "Eval/retrieval_at_5_proj":       self.retrieval_at_5_proj,
            "Eval/retrieval_at_5_backbone":   self.retrieval_at_5_backbone,
            "Eval/retrieval_at_10_proj":      self.retrieval_at_10_proj,
            "Eval/retrieval_at_10_backbone":  self.retrieval_at_10_backbone,
            # Probes
            "Eval/accent_probe_proj":         self.accent_probe_acc_proj,
            "Eval/accent_probe_backbone":     self.accent_probe_acc_backbone,
            "Eval/utterance_probe_proj":      self.utterance_probe_acc_proj,
            "Eval/utterance_probe_backbone":  self.utterance_probe_acc_backbone,
        }


def extract_embeddings(
    model:  SupConXLSR,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """
    Run a full forward pass over `loader` and collect embeddings from both spaces.

    Returns
    -------
    z_proj        : np.ndarray [N, 256]   — after projection MLP, L2-normed
    z_backbone    : np.ndarray [N, 1024]  — mean-pooled transformer output, L2-normed
    utterance_ids : list[str]  length N
    speaker_ids   : list[str]  length N
    corpus_tags   : list[str]  length N   ("arctic" | "l2_arctic")
    """
    model.eval()

    all_proj:          list[np.ndarray] = []
    all_backbone:      list[np.ndarray] = []
    all_utterance_ids: list[str]        = []
    all_speaker_ids:   list[str]        = []
    all_corpus_tags:   list[str]        = []

    with torch.no_grad():
        for batch in loader:
            audio          = batch["audio"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            out = model(audio, attention_mask=attention_mask)

            # projected space — what SupCon directly optimized
            z_proj = out["embeddings"]                   # [B, 256], already L2-normed

            # backbone space — mean-pooled transformer output, before MLP
            z_back = F.normalize(out["pooled"], dim=-1)  # [B, 1024], L2-normed

            all_proj.append(z_proj.cpu().numpy())
            all_backbone.append(z_back.cpu().numpy())
            all_utterance_ids.extend(batch["utterance_id"])
            all_speaker_ids.extend(batch["speaker_id"])

            tags = batch.get("corpus", ["unknown"] * len(batch["utterance_id"]))
            all_corpus_tags.extend(tags)

    z_proj     = np.concatenate(all_proj,     axis=0)  # [N, 256]
    z_backbone = np.concatenate(all_backbone, axis=0)  # [N, 1024]
    return z_proj, z_backbone, all_utterance_ids, all_speaker_ids, all_corpus_tags


# ---------------------------------------------------------------------------
# Metric 1 — Alignment  (pos / neg / ratio / cosine)
# ---------------------------------------------------------------------------
def compute_alignment(
    embeddings:    np.ndarray,
    utterance_ids: list[str],
    speaker_ids:   list[str],
    n_neg_samples: int = 5_000,
    seed:          int = 0,
) -> dict[str, float]:
    """
    Compute alignment metrics across positive and negative pairs.

    Positive pair : same utterance_id, different speaker_id
                    → same content, different accent — should be CLOSE
    Negative pair : different utterance_id (random sample of size n_neg_samples)
                    → different content             — should be FAR

    Since embeddings are L2-normed (‖z‖ = 1):
        cosine_sim(zᵢ, zⱼ) = zᵢ · zⱼ          ∈ [-1, 1]
        L2²(zᵢ, zⱼ)        = 2 - 2·cos(zᵢ,zⱼ) ∈ [ 0, 4]

    Both are computed from the same dot products — zero extra cost.

    Returns
    -------
    {
        "alignment_pos"   : float  ↓  mean L2²  of positive pairs   (Wang & Isola 2020)
        "alignment_neg"   : float  ↑  mean L2²  of negative pairs
        "alignment_ratio" : float  ↓  pos / neg — the key training signal
        "alignment_cos"   : float  ↑  mean cosine similarity of positive pairs
    }
    """
    utt_arr     = np.array(utterance_ids)
    speaker_arr = np.array(speaker_ids)
    unique_utts = np.unique(utt_arr)
    N           = len(embeddings)
    rng         = np.random.default_rng(seed=seed)

    # ── 1. Positive pairs ──────────────────────────────────────────────────
    # Same utterance_id, different speaker_id
    pos_l2sq: list[float] = []
    pos_cos:  list[float] = []

    for utt in unique_utts:
        indices  = np.where(utt_arr == utt)[0]
        if len(indices) < 2:
            continue

        z        = embeddings[indices]        # [K, D]
        speakers = speaker_arr[indices]

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if speakers[i] == speakers[j]:
                    continue                   # skip same-speaker pairs
                cos  = float(np.dot(z[i], z[j]))   # ∈ [-1, 1]  (‖z‖=1)
                l2sq = 2.0 - 2.0 * cos             # exact, no subtraction needed
                pos_l2sq.append(l2sq)
                pos_cos.append(cos)

    if not pos_l2sq:
        logger.warning("No positive pairs found — alignment undefined.")
        return {
            "alignment_pos":   float("nan"),
            "alignment_neg":   float("nan"),
            "alignment_ratio": float("nan"),
            "alignment_cos":   float("nan"),
        }

    # ── 2. Negative pairs ──────────────────────────────────────────────────
    # Different utterance_id — build the full pool of valid (i, j) pairs then sample.
    # For large N, triu_indices would allocate O(N²) memory; we cap at a safe threshold
    # and fall back to rejection sampling above it.
    MAX_DENSE_N = 3_000   # ~4.5M pairs at float32 ≈ 18 MB — acceptable

    if N <= MAX_DENSE_N:
        i_idx, j_idx = np.triu_indices(N, k=1)
        mask_neg     = utt_arr[i_idx] != utt_arr[j_idx]
        neg_i        = i_idx[mask_neg]
        neg_j        = j_idx[mask_neg]

        n_to_sample  = min(n_neg_samples, len(neg_i))
        chosen       = rng.choice(len(neg_i), size=n_to_sample, replace=False)
        pairs_i      = neg_i[chosen]
        pairs_j      = neg_j[chosen]
    else:
        # Rejection sampling — efficient when negatives >> positives (typical case)
        pairs_i_list: list[int] = []
        pairs_j_list: list[int] = []
        attempts = 0
        max_attempts = n_neg_samples * 8

        while len(pairs_i_list) < n_neg_samples and attempts < max_attempts:
            i, j = rng.choice(N, size=2, replace=False)
            attempts += 1
            if utt_arr[i] != utt_arr[j]:
                pairs_i_list.append(i)
                pairs_j_list.append(j)

        if not pairs_i_list:
            logger.warning("Could not sample any negative pairs.")
            return {
                "alignment_pos":   float(np.mean(pos_l2sq)),
                "alignment_neg":   float("nan"),
                "alignment_ratio": float("nan"),
                "alignment_cos":   float(np.mean(pos_cos)),
            }

        pairs_i = np.array(pairs_i_list)
        pairs_j = np.array(pairs_j_list)

    # Vectorised L2² for all sampled negative pairs
    diff     = embeddings[pairs_i] - embeddings[pairs_j]      # [M, D]
    neg_l2sq = np.einsum("md,md->m", diff, diff)              # [M]  = ‖diff‖²
    mean_neg = float(neg_l2sq.mean())

    # ── 3. Aggregate ───────────────────────────────────────────────────────
    mean_pos = float(np.mean(pos_l2sq))
    mean_cos = float(np.mean(pos_cos))
    ratio    = mean_pos / mean_neg if mean_neg > 0.0 else float("nan")

    logger.info(
        f"    alignment_pos={mean_pos:.4f}  alignment_neg={mean_neg:.4f}  "
        f"ratio={ratio:.4f}  cos_pos={mean_cos:.4f}  "
        f"(n_pos={len(pos_l2sq)}, n_neg={len(neg_l2sq)})"
    )

    return {
        "alignment_pos":   mean_pos,
        "alignment_neg":   mean_neg,
        "alignment_ratio": ratio,
        "alignment_cos":   mean_cos,
    }


# ---------------------------------------------------------------------------
# Metric 2 — Uniformity
# ---------------------------------------------------------------------------
def compute_uniformity(
    embeddings: np.ndarray,
    max_pairs:  int = 10_000,
) -> float:
    """
    Log mean pairwise Gaussian kernel on the unit hypersphere.
    uniformity = log( E[ exp(-2 * ||z_i - z_j||^2) ] )
    More negative = better coverage, less collapse.

    Reference: Wang & Isola, ICML 2020.
    """
    N = len(embeddings)

    if N > max_pairs:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(N, size=max_pairs, replace=False)
        z   = embeddings[idx]
    else:
        z = embeddings

    # ||z_i - z_j||^2 = 2 - 2 * z_i·z_j  (since ||z||=1)
    sim     = z @ z.T                          # [N, N]
    sq_dist = 2.0 - 2.0 * sim                 # [N, N]

    triu_idx  = np.triu_indices(len(z), k=1)
    sq_dist_u = sq_dist[triu_idx]

    return float(np.log(np.exp(-2.0 * sq_dist_u).mean() + 1e-8))


# ---------------------------------------------------------------------------
# Metric 3 — Retrieval @ K
# ---------------------------------------------------------------------------
def compute_retrieval(
    embeddings:    np.ndarray,
    utterance_ids: list[str],
    ks:            list[int] = (1, 5, 10),
) -> dict[str, float]:
    """
    For each sample i, retrieve K nearest neighbours by cosine similarity.
    Hit@K = 1 if at least one neighbour shares the same utterance_id.
    Recall@K = mean Hit@K over all samples (self excluded).
    Higher is better.
    """
    utt_arr = np.array(utterance_ids)
    sim     = embeddings @ embeddings.T    # [N, N]
    np.fill_diagonal(sim, -np.inf)

    results: dict[str, float] = {}
    max_k     = max(ks)
    top_k_all = np.argpartition(sim, -max_k, axis=1)[:, -max_k:]  # [N, max_k]

    for k in ks:
        hits: list[int] = []
        for i in range(len(embeddings)):
            top_k = top_k_all[i][np.argsort(sim[i, top_k_all[i]])[-k:]]
            hit   = int(utt_arr[i] in utt_arr[top_k])
            hits.append(hit)
        results[f"retrieval_at_{k}"] = float(np.mean(hits))

    return results


# ---------------------------------------------------------------------------
# Metric 4 & 5 — Linear probes
# ---------------------------------------------------------------------------
def compute_probe(
    embeddings:   np.ndarray,
    labels:       list[str],
    label_name:   str,
    test_size:    float = 0.3,
    random_state: int   = 42,
) -> float:
    """
    Logistic regression on frozen embeddings → accuracy on held-out split.
    For accent probe   : labels = corpus tag  → should be LOW  (accent-invariant)
    For utterance probe: labels = utterance_id → should be HIGH (content-preserving)
    """
    if not SKLEARN_AVAILABLE:
        return float("nan")

    le        = LabelEncoder()
    y         = le.fit_transform(labels)
    n_classes = len(le.classes_)

    if n_classes < 2:
        logger.warning(f"Probe '{label_name}': only {n_classes} class — skipping.")
        return float("nan")

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(embeddings, y))

    clf = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs",
        multi_class="multinomial", random_state=random_state, n_jobs=-1,
    )
    clf.fit(embeddings[train_idx], y[train_idx])
    acc = float(clf.score(embeddings[test_idx], y[test_idx]))

    chance = 1.0 / n_classes
    logger.info(
        f"  Probe '{label_name}': acc={acc:.4f}  "
        f"chance={chance:.4f}  n_classes={n_classes}"
    )
    return acc


# ---------------------------------------------------------------------------
# Main evaluation function (callable from train.py)
# ---------------------------------------------------------------------------
def run_eval(
    model:         SupConXLSR,
    loader:        DataLoader,
    device:        torch.device,
    retrieval_ks:  list[int]        = (1, 5, 10),
    metrics:       list[str] | None = None,
    n_neg_samples: int              = 5_000,
) -> EvalResult:
    """
    Full evaluation pipeline. Call from train.py after each epoch:

        from supcon_eval import run_eval
        result = run_eval(model, eval_loader, device, metrics=args.eval_metrics)
        result.log(prefix=f"Epoch {epoch:03d}")
        for tag, val in result.tensorboard_scalars().items():
            if not np.isnan(val):
                writer.add_scalar(tag, val, epoch)

    Parameters
    ----------
    model         : SupConXLSR (will be set to eval mode internally)
    loader        : DataLoader over the evaluation split
    device        : torch.device
    retrieval_ks  : K values for Recall@K (ignored if no retrieval metric requested)
    metrics       : subset of VALID_METRICS. None = compute all.
    n_neg_samples : number of random negative pairs for alignment_neg / ratio (default 5000)

    Returns
    -------
    EvalResult with proj + backbone variants of every metric.
    """

    # ── Validate ───────────────────────────────────────────────────────────
    if metrics is None:
        metrics = list(VALID_METRICS)

    unknown = set(metrics) - VALID_METRICS
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Valid: {VALID_METRICS}")

    logger.info(f"Running contrastive eval — metrics: {sorted(metrics)}")

    need_retrieval = any(m.startswith("retrieval_at_") for m in metrics)

    # ── 1. Extract embeddings ──────────────────────────────────────────────
    z_proj, z_backbone, utterance_ids, speaker_ids, corpus_tags = extract_embeddings(
        model, loader, device
    )
    N = len(z_proj)
    logger.info(
        f"  Extracted {N} embeddings — "
        f"{len(set(utterance_ids))} utterances, "
        f"{len(set(speaker_ids))} speakers, "
        f"{len(set(corpus_tags))} corpora"
    )

    # ── 2. Alignment (pos / neg / ratio / cosine) ──────────────────────────
    align_proj:     dict[str, float] = {}
    align_backbone: dict[str, float] = {}

    if "alignment" in metrics:
        logger.info(f"  Computing alignment (n_neg_samples={n_neg_samples}) …")
        align_proj     = compute_alignment(
            z_proj,     utterance_ids, speaker_ids, n_neg_samples=n_neg_samples
        )
        align_backbone = compute_alignment(
            z_backbone, utterance_ids, speaker_ids, n_neg_samples=n_neg_samples
        )

    # ── 3. Uniformity ──────────────────────────────────────────────────────
    uniformity_proj     = float("nan")
    uniformity_backbone = float("nan")
    if "uniformity" in metrics:
        logger.info("  Computing uniformity …")
        uniformity_proj     = compute_uniformity(z_proj)
        uniformity_backbone = compute_uniformity(z_backbone)
        logger.info(f"    proj={uniformity_proj:.4f}  backbone={uniformity_backbone:.4f}")

    # ── 4. Retrieval @ K ───────────────────────────────────────────────────
    retrieval_proj:     dict[str, float] = {}
    retrieval_backbone: dict[str, float] = {}
    if need_retrieval:
        requested_ks = sorted({
            int(m.split("_")[-1])
            for m in metrics if m.startswith("retrieval_at_")
        })
        logger.info(f"  Computing retrieval @ {requested_ks} …")
        retrieval_proj     = compute_retrieval(z_proj,     utterance_ids, ks=requested_ks)
        retrieval_backbone = compute_retrieval(z_backbone, utterance_ids, ks=requested_ks)
        for k in retrieval_proj:
            logger.info(
                f"    {k}  proj={retrieval_proj[k]:.4f}  "
                f"backbone={retrieval_backbone[k]:.4f}"
            )

    # ── 5. Linear probes ───────────────────────────────────────────────────
    accent_probe_proj     = float("nan")
    accent_probe_backbone = float("nan")
    utt_probe_proj        = float("nan")
    utt_probe_backbone    = float("nan")

    if "accent_probe_acc" in metrics:
        logger.info("  Computing accent probe …")
        accent_probe_proj     = compute_probe(z_proj,     corpus_tags,   label_name="accent/proj")
        accent_probe_backbone = compute_probe(z_backbone, corpus_tags,   label_name="accent/backbone")

    if "utterance_probe_acc" in metrics:
        logger.info("  Computing utterance probe …")
        utt_probe_proj        = compute_probe(z_proj,     utterance_ids, label_name="utterance/proj")
        utt_probe_backbone    = compute_probe(z_backbone, utterance_ids, label_name="utterance/backbone")

    # ── 6. Pack & return ───────────────────────────────────────────────────
    nan = float("nan")
    result = EvalResult(
        # Alignment
        alignment_pos_proj        = align_proj.get("alignment_pos",   nan),
        alignment_pos_backbone    = align_backbone.get("alignment_pos",   nan),
        alignment_neg_proj        = align_proj.get("alignment_neg",   nan),
        alignment_neg_backbone    = align_backbone.get("alignment_neg",   nan),
        alignment_ratio_proj      = align_proj.get("alignment_ratio", nan),
        alignment_ratio_backbone  = align_backbone.get("alignment_ratio", nan),
        alignment_cos_proj        = align_proj.get("alignment_cos",   nan),
        alignment_cos_backbone    = align_backbone.get("alignment_cos",   nan),
        # Uniformity
        uniformity_proj           = uniformity_proj,
        uniformity_backbone       = uniformity_backbone,
        # Retrieval
        retrieval_at_1_proj       = retrieval_proj.get("retrieval_at_1",  nan),
        retrieval_at_1_backbone   = retrieval_backbone.get("retrieval_at_1",  nan),
        retrieval_at_5_proj       = retrieval_proj.get("retrieval_at_5",  nan),
        retrieval_at_5_backbone   = retrieval_backbone.get("retrieval_at_5",  nan),
        retrieval_at_10_proj      = retrieval_proj.get("retrieval_at_10", nan),
        retrieval_at_10_backbone  = retrieval_backbone.get("retrieval_at_10", nan),
        # Probes
        accent_probe_acc_proj         = accent_probe_proj,
        accent_probe_acc_backbone     = accent_probe_backbone,
        utterance_probe_acc_proj      = utt_probe_proj,
        utterance_probe_acc_backbone  = utt_probe_backbone,
        # Meta
        n_samples    = N,
        n_utterances = len(set(utterance_ids)),
        n_speakers   = len(set(speaker_ids)),
        n_corpora    = len(set(corpus_tags)),
    )

    result.log()
    return result


# ---------------------------------------------------------------------------
# Helpers for the CLI
# ---------------------------------------------------------------------------
def build_eval_loader(args) -> tuple[DataLoader, SupConSpeechDataset]:
    """Build a simple sequential DataLoader over the eval split."""

    def collate_eval(batch: list[dict]) -> dict:
        max_len = max(b["audio"].shape[0] for b in batch)
        audio   = torch.zeros(len(batch), max_len)
        mask    = torch.zeros(len(batch), max_len)
        for i, b in enumerate(batch):
            L = b["audio"].shape[0]
            audio[i, :L] = b["audio"]
            mask[i,  :L] = 1.0
        return {
            "audio":          audio,
            "attention_mask": mask,
            "utterance_id":   [b["utterance_id"] for b in batch],
            "speaker_id":     [b["speaker_id"]   for b in batch],
            "corpus":         [b.get("corpus", "unknown") for b in batch],
            "labels":         torch.tensor([b["label"] for b in batch], dtype=torch.long),
        }

    eval_dataset = SupConSpeechDataset(
        parquet_paths={
            "arctic":    args.arctic_parquet_path,
            "l2_arctic": args.l2_arctic_parquet_path,
        },
        split="eval",
        sample_rate=args.sample_rate,
        max_audio_len_s=args.max_audio_len_s,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_eval,
        pin_memory=True,
    )
    return eval_loader, eval_dataset


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> SupConXLSR:
    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"]

    proj_hidden_dim = state["projection.net.0.weight"].shape[0]
    proj_out_dim    = state["projection.net.2.weight"].shape[0]
    vocab_size      = state["ctc_head.linear.weight"].shape[0]

    model = SupConXLSR(
        proj_hidden_dim=proj_hidden_dim,
        proj_out_dim=proj_out_dim,
        vocab_size=vocab_size,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    logger.info(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}) from {checkpoint_path}")
    return model


def load_model_from_hf(
    model_name:      str,
    device:          torch.device,
    proj_hidden_dim: int = 512,
    proj_out_dim:    int = 256,
    vocab_size:      int = 32,
) -> SupConXLSR:
    model = SupConXLSR(
        model_name=model_name,
        proj_hidden_dim=proj_hidden_dim,
        proj_out_dim=proj_out_dim,
        vocab_size=vocab_size,
    ).to(device)
    model.eval()
    logger.info(f"Loaded HuggingFace model: {model_name}")
    return model


def save_results(result: EvalResult, output_dir: Path, checkpoint_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{checkpoint_name}_metrics.json"
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"Metrics saved → {path}")