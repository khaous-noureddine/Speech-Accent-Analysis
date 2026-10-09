from __future__ import annotations

import numpy as np
import pandas as pd

from accented_asr.representation_analysis.data import balanced_word_sample
from accented_asr.representation_analysis.metrics import (
    accent_neighborhood,
    accent_separation,
    alignment_metrics,
    cross_accent_retrieval,
    directional_cross_accent_alignment,
    directional_cross_accent_retrieval,
    fixed_split_linear_probe,
    grouped_bootstrap_alignment,
    linear_probe,
    mean_center,
    cross_speaker_alignment,
    cross_speaker_retrieval,
    grouped_bootstrap_cross_speaker,
)
from accented_asr.representation_analysis.run_saa import (
    add_probe_split,
    select_saa_sample,
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


def test_cross_speaker_metrics_retrieve_same_content():
    embeddings, words, _accents, speakers = synthetic_geometry()
    alignment = cross_speaker_alignment(embeddings, words, speakers, seed=13)
    retrieval = cross_speaker_retrieval(embeddings, words, speakers)
    assert alignment["positive_cosine_distance"] < alignment["negative_cosine_distance"]
    assert retrieval["recall_at_1"] == 1.0
    assert retrieval["map"] == 1.0
    intervals = grouped_bootstrap_cross_speaker(
        embeddings, words, speakers, seed=13, replicates=10
    )
    assert intervals["alignment_ratio"]["lower_95"] >= 0


def test_directional_cross_accent_metrics_use_query_to_gallery_only():
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    gallery = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    query_contents = ["a", "b"]
    gallery_contents = ["a", "b", "a", "b"]
    alignment = directional_cross_accent_alignment(
        query, gallery, query_contents, gallery_contents, seed=13
    )
    retrieval = directional_cross_accent_retrieval(
        query, gallery, query_contents, gallery_contents
    )
    assert alignment["positive_cosine_distance"] == 0.0
    assert retrieval["recall_at_1"] == 1.0
    assert retrieval["map"] == 1.0


def test_centering_and_fixed_probe_are_finite():
    train = np.asarray([
        [2.0, 0.0], [2.1, 0.0], [0.0, 2.0], [0.0, 2.1]
    ])
    test = np.asarray([[1.9, 0.0], [0.0, 1.9]])
    centered = mean_center(np.vstack((train, test)), train)
    assert np.isfinite(centered).all()
    probe = fixed_split_linear_probe(
        train, test, ["a", "a", "b", "b"], ["a", "b"], seed=13
    )
    assert probe["accuracy"] == 1.0


def test_saa_accent_metrics_detect_l1_structure():
    embeddings = np.asarray([
        [1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]
    ])
    labels = ["a", "a", "b", "b"]
    separation = accent_separation(embeddings, labels, seed=13)
    neighborhood = accent_neighborhood(embeddings, labels)
    assert (
        separation["within_l1_cosine_distance"]
        < separation["between_l1_cosine_distance"]
    )
    assert separation["l1_separation_gap"] > 0
    assert neighborhood["same_l1_at_1"] == 1.0


def test_saa_selection_is_balanced_and_probe_split_is_stratified():
    rows = []
    for language, count in (("a", 8), ("b", 7), ("c", 2)):
        for index in range(count):
            rows.append({
                "audio_path": f"{language}-{index}.wav",
                "speaker_id": f"{language}-{index}",
                "native_language": language,
                "prompt_id": "stella_passage",
            })
    sample = select_saa_sample(
        pd.DataFrame(rows), min_speakers_per_l1=5, max_speakers_per_l1=6,
        max_l1=2, seed=13,
    )
    assert sample.groupby("native_language").size().to_dict() == {"a": 6, "b": 6}
    split = add_probe_split(sample, test_size=1 / 3, seed=13)
    assert set(split.loc[split.probe_split == "train", "native_language"]) == {
        "a", "b"
    }
    assert set(split.loc[split.probe_split == "test", "native_language"]) == {
        "a", "b"
    }


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
