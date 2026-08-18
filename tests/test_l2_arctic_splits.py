from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from accented_asr.data.l2_arctic_splits import (  # noqa: E402
    PROTOCOL_LOO,
    PROTOCOL_MAIN,
    SplitValidationError,
    assign_records,
    build_all_manifests,
    build_manifest,
    build_validation_report,
    validate_manifest,
)


L1S = ["Arabic", "Chinese", "Hindi", "Korean", "Spanish", "Vietnamese"]


def synthetic_inventory(n_prompts: int = 20) -> list[dict]:
    return [
        {
            "native_language": l1,
            "accent": l1,
            "speaker_id": f"{l1[:2].upper()}_{speaker_index}",
            "prompt_id": f"arctic_a{prompt_index:04d}",
            "utterance_id": f"arctic_a{prompt_index:04d}",
            "audio_path": f"/{l1}/{speaker_index}/{prompt_index}.wav",
            "duration_s": 1.0,
        }
        for l1 in L1S
        for speaker_index in range(4)
        for prompt_index in range(n_prompts)
    ]


class MainSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = synthetic_inventory()
        self.manifest = build_manifest(
            self.records, protocol=PROTOCOL_MAIN, split_seed=42
        )
        self.assigned = assign_records(self.records, self.manifest)

    def test_global_prompts_are_disjoint(self) -> None:
        prompts = {
            split: {
                row["prompt_id"] for row in self.assigned if row["split"] == split
            }
            for split in ("train", "dev", "test")
        }
        self.assertFalse(prompts["train"] & prompts["dev"])
        self.assertFalse(prompts["train"] & prompts["test"])
        self.assertFalse(prompts["dev"] & prompts["test"])

    def test_speakers_are_disjoint(self) -> None:
        speakers = {
            split: {
                row["speaker_id"] for row in self.assigned if row["split"] == split
            }
            for split in ("train", "dev", "test")
        }
        self.assertEqual([len(speakers[s]) for s in speakers], [12, 6, 6])
        self.assertFalse(speakers["train"] & speakers["dev"])
        self.assertFalse(speakers["train"] & speakers["test"])
        self.assertFalse(speakers["dev"] & speakers["test"])

    def test_validation_report_records_zero_leakage(self) -> None:
        report = build_validation_report(self.assigned, self.manifest)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(set(report["prompt_overlap"].values()), {0})
        self.assertEqual(set(report["speaker_overlap"].values()), {0})

    def test_same_seed_is_reproducible(self) -> None:
        repeated = build_manifest(
            list(reversed(self.records)), protocol=PROTOCOL_MAIN, split_seed=42
        )
        self.assertEqual(self.manifest, repeated)

    def test_different_seed_changes_assignments(self) -> None:
        other = build_manifest(self.records, protocol=PROTOCOL_MAIN, split_seed=77)
        self.assertNotEqual(self.manifest["sha256"], other["sha256"])
        self.assertNotEqual(self.manifest["prompt_splits"], other["prompt_splits"])

    def test_transcript_change_changes_source_fingerprint(self) -> None:
        changed = copy.deepcopy(self.records)
        changed[0]["transcript"] = "corrected transcript"
        other = build_manifest(changed, protocol=PROTOCOL_MAIN, split_seed=42)
        self.assertNotEqual(
            self.manifest["source_fingerprint"], other["source_fingerprint"]
        )


class LeaveOneL1OutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = synthetic_inventory()

    def test_builds_one_fold_per_l1(self) -> None:
        manifests = build_all_manifests(self.records, split_seed=42)
        self.assertEqual(len(manifests), 7)
        self.assertEqual(
            {m["heldout_l1"] for m in manifests if m["protocol"] == PROTOCOL_LOO},
            set(L1S),
        )

    def test_heldout_l1_only_appears_in_test(self) -> None:
        manifest = build_manifest(
            self.records,
            protocol=PROTOCOL_LOO,
            split_seed=42,
            heldout_l1="Arabic",
        )
        assigned = assign_records(self.records, manifest)
        train_dev_l1s = {
            row["native_language"]
            for row in assigned
            if row["split"] in {"train", "dev"}
        }
        test_l1s = {
            row["native_language"] for row in assigned if row["split"] == "test"
        }
        self.assertNotIn("Arabic", train_dev_l1s)
        self.assertEqual(test_l1s, {"Arabic"})

    def test_test_uses_all_four_heldout_speakers(self) -> None:
        manifest = build_manifest(
            self.records,
            protocol=PROTOCOL_LOO,
            split_seed=42,
            heldout_l1="Vietnamese",
        )
        assigned = assign_records(self.records, manifest)
        test_speakers = {
            row["speaker_id"] for row in assigned if row["split"] == "test"
        }
        self.assertEqual(len(test_speakers), 4)


class ValidationTests(unittest.TestCase):
    def test_rejects_fourth_split_prompt_leakage(self) -> None:
        records = synthetic_inventory()
        manifest = build_manifest(records, protocol=PROTOCOL_MAIN, split_seed=42)
        broken = copy.deepcopy(manifest)
        broken["prompt_splits"]["dev"].append(broken["prompt_splits"]["train"][0])
        # Recompute the hash so validation reaches the leakage check.
        from accented_asr.data.l2_arctic_splits import manifest_sha256

        broken["sha256"] = manifest_sha256(broken)
        with self.assertRaisesRegex(SplitValidationError, "prompt leakage"):
            validate_manifest(broken)

    def test_rejects_inventory_with_wrong_speaker_count(self) -> None:
        records = [
            row for row in synthetic_inventory() if row["speaker_id"] != "AR_3"
        ]
        with self.assertRaisesRegex(SplitValidationError, "exactly four speakers"):
            build_manifest(records, protocol=PROTOCOL_MAIN, split_seed=42)

    def test_rejects_missing_l1(self) -> None:
        records = [
            row
            for row in synthetic_inventory()
            if row["native_language"] != "Vietnamese"
        ]
        with self.assertRaisesRegex(SplitValidationError, "Expected L1 groups"):
            build_manifest(records, protocol=PROTOCOL_MAIN, split_seed=42)

    def test_rejects_duplicate_speaker_prompt(self) -> None:
        records = synthetic_inventory()
        records.append(dict(records[0]))
        with self.assertRaisesRegex(SplitValidationError, "Duplicate speaker/prompt"):
            build_manifest(records, protocol=PROTOCOL_MAIN, split_seed=42)


if __name__ == "__main__":
    unittest.main()
