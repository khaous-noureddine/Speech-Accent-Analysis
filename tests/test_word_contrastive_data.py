from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

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
from accented_asr.data.prepare_mswc_word_contrastive import source_filename  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
