"""Metrics for content alignment, cross-accent retrieval, and linear probes."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-norm embeddings cannot be normalized.")
    return embeddings / norms


def _pair_indices(
    words: np.ndarray, accents: np.ndarray, speakers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    left, right = np.triu_indices(len(words), k=1)
    distinct_speaker = speakers[left] != speakers[right]
    cross_accent = accents[left] != accents[right]
    positive = distinct_speaker & cross_accent & (words[left] == words[right])
    negative = distinct_speaker & (words[left] != words[right])
    return np.column_stack((left[positive], right[positive])), np.column_stack(
        (left[negative], right[negative])
    )


def alignment_metrics(
    embeddings: np.ndarray,
    words: Iterable[str],
    accents: Iterable[str],
    speakers: Iterable[str],
    *,
    seed: int,
) -> dict[str, float | int]:
    """Compute balanced cosine distances for cross-accent positive pairs."""

    embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float64))
    words = np.asarray(list(words), dtype=str)
    accents = np.asarray(list(accents), dtype=str)
    speakers = np.asarray(list(speakers), dtype=str)
    positive, negative = _pair_indices(words, accents, speakers)
    if not len(positive) or not len(negative):
        raise ValueError("Analysis requires both positive and negative pairs.")
    rng = np.random.default_rng(seed)
    if len(negative) > len(positive):
        negative = negative[rng.choice(len(negative), len(positive), replace=False)]

    def distances(pairs: np.ndarray) -> np.ndarray:
        return 1.0 - np.sum(
            embeddings[pairs[:, 0]] * embeddings[pairs[:, 1]], axis=1
        )

    positive_distance = float(distances(positive).mean())
    negative_distance = float(distances(negative).mean())
    return {
        "positive_cosine_distance": positive_distance,
        "negative_cosine_distance": negative_distance,
        "alignment_ratio": positive_distance / negative_distance,
        "positive_pairs": int(len(positive)),
        "negative_pairs": int(len(negative)),
    }


def cross_accent_retrieval(
    embeddings: np.ndarray,
    words: Iterable[str],
    accents: Iterable[str],
    speakers: Iterable[str],
    *,
    ks: tuple[int, ...] = (1, 5),
) -> dict[str, float | int]:
    """Retrieve the same word exclusively from other accents and speakers."""

    embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float64))
    words = np.asarray(list(words), dtype=str)
    accents = np.asarray(list(accents), dtype=str)
    speakers = np.asarray(list(speakers), dtype=str)
    similarities = embeddings @ embeddings.T
    recalls: dict[int, list[float]] = defaultdict(list)
    average_precisions: list[float] = []
    for query in range(len(words)):
        candidates = np.flatnonzero(
            (accents != accents[query]) & (speakers != speakers[query])
        )
        relevant = words[candidates] == words[query]
        if not relevant.any():
            continue
        order = np.argsort(-similarities[query, candidates], kind="stable")
        ranked_relevant = relevant[order]
        for k in ks:
            recalls[k].append(float(ranked_relevant[:k].any()))
        precision = np.cumsum(ranked_relevant) / np.arange(1, len(order) + 1)
        average_precisions.append(float(precision[ranked_relevant].mean()))
    if not average_precisions:
        raise ValueError("No query has a same-word example from another accent.")
    return {
        **{f"recall_at_{k}": float(np.mean(recalls[k])) for k in ks},
        "map": float(np.mean(average_precisions)),
        "queries": len(average_precisions),
    }


def linear_probe(
    embeddings: np.ndarray,
    labels: Iterable[str],
    speakers: Iterable[str],
    *,
    seed: int,
    test_size: float = 0.3,
) -> dict[str, float | int]:
    """Fit a speaker-disjoint multinomial linear probe."""

    labels = np.asarray(list(labels), dtype=str)
    speakers = np.asarray(list(speakers), dtype=str)
    splitter = GroupShuffleSplit(n_splits=50, test_size=test_size, random_state=seed)
    all_classes = set(labels)
    selected_split = None
    for train, test in splitter.split(embeddings, labels, groups=speakers):
        train_classes, test_classes = set(labels[train]), set(labels[test])
        if train_classes == all_classes and test_classes == all_classes:
            selected_split = (train, test)
            break
    if selected_split is None:
        raise ValueError(
            "Could not build a speaker-disjoint probe split containing every class "
            "in both train and test. Increase examples_per_word_accent."
        )
    train, test = selected_split
    classifier = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=seed
    )
    classifier.fit(embeddings[train], labels[train])
    predictions = classifier.predict(embeddings[test])
    return {
        "accuracy": float(accuracy_score(labels[test], predictions)),
        "macro_f1": float(f1_score(labels[test], predictions, average="macro")),
        "chance_accuracy": float(1.0 / len(np.unique(labels[test]))),
        "train_examples": int(len(train)),
        "test_examples": int(len(test)),
        "classes": int(len(np.unique(labels))),
    }


def grouped_bootstrap_alignment(
    embeddings: np.ndarray,
    words: Iterable[str],
    accents: Iterable[str],
    speakers: Iterable[str],
    *,
    seed: int,
    replicates: int,
) -> dict[str, dict[str, float]]:
    """Word-cluster bootstrap confidence intervals for alignment metrics."""

    words = np.asarray(list(words), dtype=str)
    accents = np.asarray(list(accents), dtype=str)
    speakers = np.asarray(list(speakers), dtype=str)
    unique_words = np.unique(words)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for replicate in range(replicates):
        selected = rng.choice(unique_words, len(unique_words), replace=True)
        indices = np.concatenate([np.flatnonzero(words == word) for word in selected])
        try:
            result = alignment_metrics(
                embeddings[indices], words[indices], accents[indices], speakers[indices],
                seed=seed + replicate + 1,
            )
        except ValueError:
            continue
        for key in ("positive_cosine_distance", "negative_cosine_distance", "alignment_ratio"):
            samples[key].append(float(result[key]))
    if not samples:
        raise ValueError("No valid bootstrap replicate could be computed.")
    return {
        key: {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for key, values in samples.items()
    }
