from __future__ import annotations

import numpy as np
import pandas as pd

from accented_asr.representation_analysis.data import balanced_word_sample
from accented_asr.representation_analysis.metrics import (
    alignment_metrics,
    cross_accent_retrieval,
    grouped_bootstrap_alignment,
    linear_probe,
)


def synthetic_geometry():
    words, accents, speakers, vectors = [], [], [], []
    for word_index, word in enumerate(("cat", "dog", "tree")):
        for accent_index, accent in enumerate(("a", "b", "c")):
            for repeat in range(3):
                vector = np.zeros(6, dtype=np.float64)
                vector[word_index] = 1.0
                vector[3 + accent_index] = 0.05
                words.append(word)
                accents.append(accent)
                speakers.append(f"{accent}-{repeat}-{word}")
                vectors.append(vector)
    return np.asarray(vectors), words, accents, speakers


def test_alignment_and_cross_accent_retrieval_capture_word_geometry():
    embeddings, words, accents, speakers = synthetic_geometry()
    alignment = alignment_metrics(
        embeddings, words, accents, speakers, seed=13
    )
    retrieval = cross_accent_retrieval(embeddings, words, accents, speakers)
    assert alignment["positive_cosine_distance"] < alignment["negative_cosine_distance"]
    assert alignment["alignment_ratio"] < 0.1
    assert retrieval["recall_at_1"] == 1.0
    assert retrieval["map"] == 1.0


def test_bootstrap_and_probes_return_finite_metrics():
    embeddings, words, accents, speakers = synthetic_geometry()
    intervals = grouped_bootstrap_alignment(
        embeddings, words, accents, speakers, seed=13, replicates=20
    )
    probe = linear_probe(embeddings, words, speakers, seed=13)
    assert intervals["alignment_ratio"]["lower_95"] >= 0
    assert intervals["alignment_ratio"]["upper_95"] < 0.1
    assert 0 <= probe["macro_f1"] <= 1


def test_balanced_sample_uses_distinct_speakers(tmp_path):
    rows = []
    for word in ("cat", "dog"):
        for accent in ("a", "b", "c"):
            for speaker_index in range(3):
                rows.append(
                    {
                        "normalized_word": word,
                        "accent": accent,
                        "speaker_id": f"{accent}-{speaker_index}",
                        "audio_path": f"{word}-{accent}-{speaker_index}.wav",
                        "start_s": 0.0,
                        "end_s": 0.5,
                        "split": "dev",
                    }
                )
    parquet = tmp_path / "words.parquet"
    pd.DataFrame(rows).to_parquet(parquet, index=False)
    sample = balanced_word_sample(
        parquet,
        split="dev",
        min_accents_per_word=3,
        examples_per_word_accent=2,
        max_words=10,
        seed=13,
    )
    assert len(sample) == 12
    assert sample.groupby(["normalized_word", "accent"])["speaker_id"].nunique().eq(2).all()
