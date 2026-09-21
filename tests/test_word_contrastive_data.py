from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from accented_asr.data.prepare_word_contrastive import (  # noqa: E402
    assign_speaker_splits,
    eligible_words,
    normalize_word,
    parse_textgrid_words,
    validate_occurrences,
)
from accented_asr.data.prepare_mswc_word_contrastive import (  # noqa: E402
    resolve_audio_paths,
    source_filename,
)
from accented_asr.data.prepare_mswc_heldout_subset import build_subset  # noqa: E402


def occurrence_inventory() -> pd.DataFrame:
    rows = []
    for accent in ("A", "B", "C"):
        for speaker_index in range(3):
            for word in ("hello", "world"):
                rows.append(
                    {
                        "dataset": "synthetic",
                        "language": "en",
                        "accent": accent,
                        "speaker_id": f"{accent}{speaker_index}",
                        "utterance_id": f"{accent}{speaker_index}-{word}",
                        "word": word,
                        "normalized_word": word,
                        "audio_path": f"{accent}/{speaker_index}.wav",
                        "start_s": 0.1,
                        "end_s": 0.5,
                        "duration_s": 0.4,
                        "alignment_source": "test",
                        "occurrence_id": f"{accent}-{speaker_index}-{word}",
                    }
                )
    return pd.DataFrame(rows)


class NormalizationTests(unittest.TestCase):
    def test_unicode_normalization_preserves_letters(self) -> None:
        self.assertEqual(normalize_word("  ÉCOLE! "), "école")
        self.assertEqual(normalize_word("l’homme"), "l'homme")

    def test_mswc_filename_maps_back_to_common_voice_clip(self) -> None:
        self.assertEqual(
            source_filename("hello/common_voice_en_123__2.opus"),
            "common_voice_en_123.mp3",
        )

    def test_mswc_audio_link_resolves_without_global_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / "en" / "clips" / "hello" / "sample.opus"
            clip.parent.mkdir(parents=True)
            clip.touch()
            self.assertEqual(
                resolve_audio_paths(root, pd.Series(["hello/sample.opus"])),
                [clip],
            )


class TextGridTests(unittest.TestCase):
    def test_extracts_only_word_tier(self) -> None:
        text = '''File type = "ooTextFile"
item [1]:
    class = "IntervalTier"
    name = "words"
    intervals [1]:
        xmin = 0.0
        xmax = 0.2
        text = ""
    intervals [2]:
        xmin = 0.2
        xmax = 0.6
        text = "hello"
item [2]:
    class = "IntervalTier"
    name = "phones"
    intervals [1]:
        xmin = 0.2
        xmax = 0.3
        text = "HH"
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.TextGrid"
            path.write_text(text)
            self.assertEqual(parse_textgrid_words(path), [(0.2, 0.6, "hello")])


class CoverageAndSplitTests(unittest.TestCase):
    def test_coverage_is_based_on_unique_speakers_per_accent(self) -> None:
        selected, vocabulary = eligible_words(
            occurrence_inventory(),
            min_accents=3,
            min_speakers_per_accent=3,
            min_duration_s=0.1,
            max_duration_s=1.0,
        )
        self.assertEqual(set(selected["normalized_word"]), {"hello", "world"})
        self.assertEqual(set(vocabulary["n_accents"]), {3})

    def test_leave_one_accent_and_speakers_are_disjoint(self) -> None:
        assigned = assign_speaker_splits(
            occurrence_inventory(), heldout_accent="C", seed=13,
            dev_speakers_per_accent=1,
        )
        self.assertEqual(set(assigned.loc[assigned.split == "test", "accent"]), {"C"})
        speaker_sets = {
            split: set(assigned.loc[assigned.split == split, "speaker_id"])
            for split in ("train", "dev", "test")
        }
        self.assertFalse(speaker_sets["train"] & speaker_sets["dev"])
        self.assertFalse(speaker_sets["train"] & speaker_sets["test"])

    def test_duplicate_occurrence_is_rejected(self) -> None:
        frame = occurrence_inventory()
        frame.loc[1, "occurrence_id"] = frame.loc[0, "occurrence_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate occurrence"):
            validate_occurrences(frame)


class MswcHeldoutSubsetTests(unittest.TestCase):
    def test_builds_exact_size_with_unseen_test_accent(self) -> None:
        rows = []
        for accent in "abcdefg":
            for word_index in range(12):
                word = f"word{word_index}"
                for source_split, speakers in (("train", 4), ("dev", 2)):
                    for speaker_index in range(speakers):
                        speaker = f"{accent}-{source_split}-{speaker_index}"
                        for repetition in range(2):
                            occurrence = f"{accent}-{word}-{source_split}-{speaker_index}-{repetition}"
                            rows.append({
                                "dataset": "synthetic_mswc",
                                "language": "en",
                                "accent": accent,
                                "speaker_id": speaker,
                                "utterance_id": occurrence,
                                "source_transcript": word,
                                "word": word,
                                "normalized_word": word,
                                "audio_path": f"audio/{occurrence}.opus",
                                "start_s": 0.0,
                                "end_s": 1.0,
                                "duration_s": 1.0,
                                "alignment_source": "test",
                                "split": source_split,
                            })
        args = SimpleNamespace(
            target_hours=0.1,
            heldout_accent="g",
            seen_accents=6,
            min_train_speakers_per_accent=3,
            min_dev_accents=5,
            dev_fraction=0.1,
            test_fraction=0.1,
            seed=13,
        )
        subset, report = build_subset(pd.DataFrame(rows), args)
        self.assertEqual(len(subset), 360)
        self.assertAlmostEqual(subset["duration_s"].sum() / 3600, 0.1)
        self.assertEqual(set(subset.loc[subset.split == "test", "accent"]), {"g"})
        self.assertNotIn("g", set(subset.loc[subset.split != "test", "accent"]))
        self.assertEqual(report["speaker_overlap"], {
            "train_dev": 0, "train_test": 0, "dev_test": 0,
        })
        self.assertEqual(report["status"], "passed")


if __name__ == "__main__":
    unittest.main()
