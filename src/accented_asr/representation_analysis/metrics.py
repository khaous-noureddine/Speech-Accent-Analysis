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


def mean_center(
    embeddings: np.ndarray, reference_embeddings: np.ndarray
) -> np.ndarray:
    """Subtract a reference-set mean, then restore unit-length vectors."""
    center = np.asarray(reference_embeddings, dtype=np.float64).mean(axis=0)
    return l2_normalize(np.asarray(embeddings, dtype=np.float64) - center)


def directional_cross_accent_alignment(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_contents: Iterable[str],
    gallery_contents: Iterable[str],
    *,
    seed: int,
) -> dict[str, float | int]:
    """Compare held-out-accent queries only against seen-accent examples."""
    query_embeddings = l2_normalize(np.asarray(query_embeddings, dtype=np.float64))
    gallery_embeddings = l2_normalize(np.asarray(gallery_embeddings, dtype=np.float64))
    query_contents = np.asarray(list(query_contents), dtype=str)
    gallery_contents = np.asarray(list(gallery_contents), dtype=str)
    same = query_contents[:, None] == gallery_contents[None, :]
    positive = np.argwhere(same)
    negative = np.argwhere(~same)
    if not len(positive) or not len(negative):
        raise ValueError("Directional analysis requires positive and negative pairs.")
    rng = np.random.default_rng(seed)
    negative = negative[rng.choice(len(negative), len(positive), replace=False)]

    def distance(pairs: np.ndarray) -> float:
        similarities = np.sum(
            query_embeddings[pairs[:, 0]] * gallery_embeddings[pairs[:, 1]], axis=1
        )
        return float((1.0 - similarities).mean())

    positive_distance = distance(positive)
    negative_distance = distance(negative)
    return {
        "positive_cosine_distance": positive_distance,
        "negative_cosine_distance": negative_distance,
        "alignment_ratio": positive_distance / negative_distance,
        "positive_pairs": int(len(positive)),
        "negative_pairs": int(len(negative)),
    }


def directional_cross_accent_retrieval(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_contents: Iterable[str],
    gallery_contents: Iterable[str],
    *,
    ks: tuple[int, ...] = (1, 5),
) -> dict[str, float | int]:
    """Retrieve matching content in a seen-accent gallery for held-out queries."""
    query_embeddings = l2_normalize(np.asarray(query_embeddings, dtype=np.float64))
    gallery_embeddings = l2_normalize(np.asarray(gallery_embeddings, dtype=np.float64))
    query_contents = np.asarray(list(query_contents), dtype=str)
    gallery_contents = np.asarray(list(gallery_contents), dtype=str)
    similarities = query_embeddings @ gallery_embeddings.T
    recalls: dict[int, list[float]] = defaultdict(list)
    average_precisions = []
    for query, content in enumerate(query_contents):
        relevant = gallery_contents == content
        if not relevant.any():
            continue
        order = np.argsort(-similarities[query], kind="stable")
        ranked = relevant[order]
        for k in ks:
            recalls[k].append(float(ranked[:k].any()))
        precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
        average_precisions.append(float(precision[ranked].mean()))
    if not average_precisions:
        raise ValueError("No held-out query has matching gallery content.")
    return {
        **{f"recall_at_{k}": float(np.mean(recalls[k])) for k in ks},
        "map": float(np.mean(average_precisions)),
        "queries": len(average_precisions),
    }


def fixed_split_linear_probe(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_labels: Iterable[str],
    test_labels: Iterable[str],
    *,
    seed: int,
) -> dict[str, float | int]:
    """Fit an accent probe on predefined speaker-disjoint partitions."""
    train_labels = np.asarray(list(train_labels), dtype=str)
    test_labels = np.asarray(list(test_labels), dtype=str)
    if set(train_labels) != set(test_labels):
        raise ValueError("Probe train and test partitions must contain the same classes.")
    classifier = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=seed
    )
    classifier.fit(train_embeddings, train_labels)
    predictions = classifier.predict(test_embeddings)
    return {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "macro_f1": float(f1_score(test_labels, predictions, average="macro")),
        "chance_accuracy": float(1.0 / len(np.unique(test_labels))),
        "train_examples": int(len(train_labels)),
        "test_examples": int(len(test_labels)),
        "classes": int(len(np.unique(train_labels))),
    }


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


def cross_speaker_alignment(
    embeddings: np.ndarray,
    contents: Iterable[str],
    speakers: Iterable[str],
    *,
    seed: int,
) -> dict[str, float | int]:
    """Compare same-content and different-content pairs across speakers."""
    embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float64))
    contents = np.asarray(list(contents), dtype=str)
    speakers = np.asarray(list(speakers), dtype=str)
    left, right = np.triu_indices(len(contents), k=1)
    cross_speaker = speakers[left] != speakers[right]
    positive = np.column_stack(
        (left[cross_speaker & (contents[left] == contents[right])],
         right[cross_speaker & (contents[left] == contents[right])])
    )
    negative = np.column_stack(
        (left[cross_speaker & (contents[left] != contents[right])],
         right[cross_speaker & (contents[left] != contents[right])])
    )
    if not len(positive) or not len(negative):
        raise ValueError("Analysis requires positive and negative cross-speaker pairs.")
    rng = np.random.default_rng(seed)
    negative = negative[rng.choice(len(negative), len(positive), replace=False)]

    def mean_distance(pairs: np.ndarray) -> float:
        similarity = np.sum(embeddings[pairs[:, 0]] * embeddings[pairs[:, 1]], axis=1)
        return float((1.0 - similarity).mean())

    positive_distance = mean_distance(positive)
    negative_distance = mean_distance(negative)
    return {
        "positive_cosine_distance": positive_distance,
        "negative_cosine_distance": negative_distance,
        "alignment_ratio": positive_distance / negative_distance,
        "positive_pairs": int(len(positive)),
        "negative_pairs": int(len(negative)),
    }


def cross_speaker_retrieval(
    embeddings: np.ndarray,
    contents: Iterable[str],
    speakers: Iterable[str],
    *,
    ks: tuple[int, ...] = (1, 5),
) -> dict[str, float | int]:
    """Retrieve the same content using candidates from other speakers only."""
    embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float64))
    contents = np.asarray(list(contents), dtype=str)
    speakers = np.asarray(list(speakers), dtype=str)
    similarities = embeddings @ embeddings.T
    recalls: dict[int, list[float]] = defaultdict(list)
    average_precisions: list[float] = []
    for query in range(len(contents)):
        candidates = np.flatnonzero(speakers != speakers[query])
        relevant = contents[candidates] == contents[query]
        if not relevant.any():
            continue
        order = np.argsort(-similarities[query, candidates], kind="stable")
        ranked = relevant[order]
        for k in ks:
            recalls[k].append(float(ranked[:k].any()))
        precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
        average_precisions.append(float(precision[ranked].mean()))
    if not average_precisions:
        raise ValueError("No query has a same-content cross-speaker candidate.")
    return {
        **{f"recall_at_{k}": float(np.mean(recalls[k])) for k in ks},
        "map": float(np.mean(average_precisions)),
        "queries": len(average_precisions),
    }


def grouped_bootstrap_cross_speaker(
    embeddings: np.ndarray,
    contents: Iterable[str],
    speakers: Iterable[str],
    *,
    seed: int,
    replicates: int,
) -> dict[str, dict[str, float]]:
    """Content-cluster bootstrap intervals for cross-speaker alignment."""
    contents = np.asarray(list(contents), dtype=str)
    speakers = np.asarray(list(speakers), dtype=str)
    unique_contents = np.unique(contents)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for replicate in range(replicates):
        selected = rng.choice(unique_contents, len(unique_contents), replace=True)
        indices = np.concatenate([np.flatnonzero(contents == item) for item in selected])
        try:
            result = cross_speaker_alignment(
                embeddings[indices], contents[indices], speakers[indices],
                seed=seed + replicate + 1,
            )
        except ValueError:
            continue
        for key in (
            "positive_cosine_distance", "negative_cosine_distance", "alignment_ratio"
        ):
            samples[key].append(float(result[key]))
    if not samples:
        raise ValueError("No valid content-bootstrap replicate could be computed.")
    return {
        key: {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for key, values in samples.items()
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
