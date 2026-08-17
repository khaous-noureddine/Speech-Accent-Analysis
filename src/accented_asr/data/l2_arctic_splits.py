"""Leakage-safe, deterministic split protocols for L2-ARCTIC.

The split logic intentionally works on plain dictionaries so it can be tested
without Pandas or audio dependencies.  The command-line adapter is responsible
for reading and writing Parquet files.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


SPLITS = ("train", "dev", "test")
PROTOCOL_MAIN = "main"
PROTOCOL_LOO = "leave_one_l1_out"
EXPECTED_L1S = frozenset(
    {"Arabic", "Chinese", "Hindi", "Korean", "Spanish", "Vietnamese"}
)


class SplitValidationError(ValueError):
    """Raised when an inventory or generated split violates the protocol."""


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.8
    dev: float = 0.1
    test: float = 0.1

    def validate(self) -> None:
        values = (self.train, self.dev, self.test)
        if any(value <= 0 for value in values):
            raise SplitValidationError("All split ratios must be positive.")
        if abs(sum(values) - 1.0) > 1e-9:
            raise SplitValidationError("Split ratios must sum to 1.0.")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Return a stable hash while ignoring an existing ``sha256`` field."""

    payload = dict(manifest)
    payload.pop("sha256", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _stable_shuffle(items: Iterable[str], seed: int, namespace: str) -> list[str]:
    values = sorted(set(items))
    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    rng.shuffle(values)
    return values


def _allocate_counts(size: int, ratios: SplitRatios) -> tuple[int, int, int]:
    """Allocate all items with a largest-remainder rule."""

    ratios.validate()
    if size < 3:
        raise SplitValidationError("At least three common prompts are required.")

    raw = [size * ratios.train, size * ratios.dev, size * ratios.test]
    counts = [int(value) for value in raw]
    remainder = size - sum(counts)
    order = sorted(range(3), key=lambda i: (raw[i] - counts[i], -i), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1

    if any(count == 0 for count in counts):
        raise SplitValidationError("Each prompt split must contain at least one prompt.")
    return counts[0], counts[1], counts[2]


def _partition_prompts(
    prompts: Iterable[str], seed: int, ratios: SplitRatios
) -> dict[str, list[str]]:
    shuffled = _stable_shuffle(prompts, seed, "global-prompts")
    n_train, n_dev, _ = _allocate_counts(len(shuffled), ratios)
    return {
        "train": sorted(shuffled[:n_train]),
        "dev": sorted(shuffled[n_train : n_train + n_dev]),
        "test": sorted(shuffled[n_train + n_dev :]),
    }


def inventory_index(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[str]], list[str]]:
    """Validate an inventory and return speakers by L1 and global prompts.

    A prompt is eligible only when every speaker has an example for it. This
    conservative policy makes prompt composition identical across L1 groups.
    """

    required = {"speaker_id", "native_language", "prompt_id"}
    if not records:
        raise SplitValidationError("The L2-ARCTIC inventory is empty.")

    speakers_by_l1: dict[str, set[str]] = defaultdict(set)
    prompts_by_speaker: dict[str, set[str]] = defaultdict(set)
    speaker_to_l1: dict[str, str] = {}
    seen_speaker_prompts: set[tuple[str, str]] = set()

    for row_index, record in enumerate(records):
        missing = required - set(record)
        if missing:
            raise SplitValidationError(
                f"Inventory row {row_index} is missing columns: {sorted(missing)}"
            )
        speaker = str(record["speaker_id"])
        l1 = str(record["native_language"])
        prompt = str(record["prompt_id"])
        if not speaker or not l1 or not prompt or "None" in {speaker, l1, prompt}:
            raise SplitValidationError(
                f"Inventory row {row_index} contains an empty identifier."
            )
        pair = (speaker, prompt)
        if pair in seen_speaker_prompts:
            raise SplitValidationError(
                f"Duplicate speaker/prompt pair in inventory: {pair}"
            )
        seen_speaker_prompts.add(pair)
        previous_l1 = speaker_to_l1.setdefault(speaker, l1)
        if previous_l1 != l1:
            raise SplitValidationError(
                f"Speaker {speaker!r} belongs to both {previous_l1!r} and {l1!r}."
            )
        speakers_by_l1[l1].add(speaker)
        prompts_by_speaker[speaker].add(prompt)

    found_l1s = set(speakers_by_l1)
    if found_l1s != EXPECTED_L1S:
        raise SplitValidationError(
            f"Expected L1 groups {sorted(EXPECTED_L1S)}, found {sorted(found_l1s)}."
        )

    invalid = {
        l1: sorted(speakers)
        for l1, speakers in speakers_by_l1.items()
        if len(speakers) != 4
    }
    if invalid:
        raise SplitValidationError(
            "Each L1 must contain exactly four speakers; invalid groups: "
            f"{invalid}"
        )

    all_speakers = sorted(speaker_to_l1)
    common_prompts = sorted(
        set.intersection(*(prompts_by_speaker[speaker] for speaker in all_speakers))
    )
    if len(common_prompts) < 3:
        raise SplitValidationError(
            "Fewer than three prompts are shared by every L2-ARCTIC speaker."
        )

    normalized_speakers = {
        l1: sorted(speakers) for l1, speakers in sorted(speakers_by_l1.items())
    }
    return normalized_speakers, common_prompts


def _source_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    identifiers = sorted(
        (
            str(row["native_language"]),
            str(row["speaker_id"]),
            str(row["prompt_id"]),
        )
        for row in records
    )
    return hashlib.sha256(_canonical_json(identifiers).encode("utf-8")).hexdigest()


def _main_speaker_roles(
    speakers_by_l1: Mapping[str, Sequence[str]], seed: int
) -> dict[str, dict[str, list[str]]]:
    roles: dict[str, dict[str, list[str]]] = {}
    for l1, speakers in sorted(speakers_by_l1.items()):
        shuffled = _stable_shuffle(speakers, seed, f"main-speakers:{l1}")
        roles[l1] = {
            "train": sorted(shuffled[:2]),
            "dev": [shuffled[2]],
            "test": [shuffled[3]],
        }
    return roles


def _loo_speaker_roles(
    speakers_by_l1: Mapping[str, Sequence[str]], seed: int, heldout_l1: str
) -> dict[str, dict[str, list[str]]]:
    if heldout_l1 not in speakers_by_l1:
        raise SplitValidationError(f"Unknown held-out L1: {heldout_l1!r}")

    roles: dict[str, dict[str, list[str]]] = {}
    for l1, speakers in sorted(speakers_by_l1.items()):
        shuffled = _stable_shuffle(speakers, seed, f"loo-speakers:{l1}")
        if l1 == heldout_l1:
            roles[l1] = {"train": [], "dev": [], "test": sorted(shuffled)}
        else:
            roles[l1] = {
                "train": sorted(shuffled[:3]),
                "dev": [shuffled[3]],
                "test": [],
            }
    return roles


def build_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    protocol: str,
    split_seed: int,
    ratios: SplitRatios = SplitRatios(),
    heldout_l1: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic manifest for one L2-ARCTIC protocol/fold."""

    speakers_by_l1, common_prompts = inventory_index(records)
    prompt_splits = _partition_prompts(common_prompts, split_seed, ratios)

    if protocol == PROTOCOL_MAIN:
        if heldout_l1 is not None:
            raise SplitValidationError("The main protocol does not accept heldout_l1.")
        speaker_roles = _main_speaker_roles(speakers_by_l1, split_seed)
    elif protocol == PROTOCOL_LOO:
        if heldout_l1 is None:
            raise SplitValidationError("leave_one_l1_out requires heldout_l1.")
        speaker_roles = _loo_speaker_roles(speakers_by_l1, split_seed, heldout_l1)
    else:
        raise SplitValidationError(f"Unknown split protocol: {protocol!r}")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "l2_arctic",
        "protocol": protocol,
        "split_seed": split_seed,
        "heldout_l1": heldout_l1,
        "ratios": {
            "train": ratios.train,
            "dev": ratios.dev,
            "test": ratios.test,
        },
        "source_fingerprint": _source_fingerprint(records),
        "eligible_prompt_policy": "intersection_across_all_speakers",
        "eligible_prompt_count": len(common_prompts),
        "prompt_splits": prompt_splits,
        "speaker_roles": speaker_roles,
    }
    manifest["sha256"] = manifest_sha256(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate global prompt, speaker, and held-out-L1 separation."""

    if manifest.get("sha256") != manifest_sha256(manifest):
        raise SplitValidationError("Manifest SHA-256 does not match its contents.")

    prompt_splits = manifest["prompt_splits"]
    prompt_sets = {split: set(prompt_splits[split]) for split in SPLITS}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = prompt_sets[left] & prompt_sets[right]
            if overlap:
                raise SplitValidationError(
                    f"Global prompt leakage between {left} and {right}: "
                    f"{sorted(overlap)[:5]}"
                )

    roles = manifest["speaker_roles"]
    speakers_by_split = {
        split: {
            speaker
            for l1_roles in roles.values()
            for speaker in l1_roles[split]
        }
        for split in SPLITS
    }
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = speakers_by_split[left] & speakers_by_split[right]
            if overlap:
                raise SplitValidationError(
                    f"Speaker leakage between {left} and {right}: {sorted(overlap)}"
                )

    if manifest["protocol"] == PROTOCOL_LOO:
        heldout_l1 = manifest["heldout_l1"]
        for l1, l1_roles in roles.items():
            if l1 == heldout_l1:
                if l1_roles["train"] or l1_roles["dev"] or not l1_roles["test"]:
                    raise SplitValidationError("Held-out L1 roles are invalid.")
            elif l1_roles["test"]:
                raise SplitValidationError(
                    f"Seen L1 {l1!r} unexpectedly contains test speakers."
                )


def assign_records(
    records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Apply a manifest, dropping deliberately unused speaker/prompt crossings."""

    validate_manifest(manifest)
    prompt_to_split = {
        prompt: split
        for split, prompts in manifest["prompt_splits"].items()
        for prompt in prompts
    }
    speaker_to_split = {
        speaker: split
        for l1_roles in manifest["speaker_roles"].values()
        for split, speakers in l1_roles.items()
        for speaker in speakers
    }

    assigned: list[dict[str, Any]] = []
    for record in records:
        prompt_split = prompt_to_split.get(str(record["prompt_id"]))
        speaker_split = speaker_to_split.get(str(record["speaker_id"]))
        if prompt_split is None or prompt_split != speaker_split:
            continue
        row = dict(record)
        row.update(
            {
                "split": prompt_split,
                "split_protocol": manifest["protocol"],
                "split_seed": manifest["split_seed"],
                "heldout_l1": manifest["heldout_l1"],
                "split_manifest_sha256": manifest["sha256"],
            }
        )
        assigned.append(row)

    validate_assigned_records(assigned, manifest)
    return assigned


def validate_assigned_records(
    records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> None:
    if not records:
        raise SplitValidationError("The manifest produced no assigned records.")

    for split in SPLITS:
        if not any(row["split"] == split for row in records):
            raise SplitValidationError(f"Generated split {split!r} is empty.")

    for field in ("prompt_id", "speaker_id"):
        values = {
            split: {str(row[field]) for row in records if row["split"] == split}
            for split in SPLITS
        }
        for index, left in enumerate(SPLITS):
            for right in SPLITS[index + 1 :]:
                overlap = values[left] & values[right]
                if overlap:
                    raise SplitValidationError(
                        f"{field} leakage between {left} and {right}: "
                        f"{sorted(overlap)[:5]}"
                    )

    heldout_l1 = manifest.get("heldout_l1")
    if heldout_l1 is not None:
        train_dev_l1s = {
            str(row["native_language"])
            for row in records
            if row["split"] in {"train", "dev"}
        }
        test_l1s = {
            str(row["native_language"])
            for row in records
            if row["split"] == "test"
        }
        if heldout_l1 in train_dev_l1s or test_l1s != {heldout_l1}:
            raise SplitValidationError(
                "The leave-one-L1-out assignment leaks or misroutes the held-out L1."
            )


def build_validation_report(
    records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Return an auditable summary after enforcing all split invariants."""

    validate_assigned_records(records, manifest)
    values = {
        field: {
            split: {
                str(row[field]) for row in records if row["split"] == split
            }
            for split in SPLITS
        }
        for field in ("prompt_id", "speaker_id", "native_language")
    }
    pair_names = (("train", "dev"), ("train", "test"), ("dev", "test"))
    return {
        "manifest_sha256": manifest["sha256"],
        "status": "passed",
        "counts": {
            split: {
                "examples": sum(row["split"] == split for row in records),
                "prompts": len(values["prompt_id"][split]),
                "speakers": len(values["speaker_id"][split]),
                "l1s": len(values["native_language"][split]),
            }
            for split in SPLITS
        },
        "prompt_overlap": {
            f"{left}_{right}": len(
                values["prompt_id"][left] & values["prompt_id"][right]
            )
            for left, right in pair_names
        },
        "speaker_overlap": {
            f"{left}_{right}": len(
                values["speaker_id"][left] & values["speaker_id"][right]
            )
            for left, right in pair_names
        },
        "l1s_by_split": {
            split: sorted(values["native_language"][split]) for split in SPLITS
        },
    }


def build_all_manifests(
    records: Sequence[Mapping[str, Any]],
    *,
    split_seed: int,
    ratios: SplitRatios = SplitRatios(),
) -> list[dict[str, Any]]:
    """Return the main manifest followed by all leave-one-L1-out folds."""

    speakers_by_l1, _ = inventory_index(records)
    manifests = [
        build_manifest(
            records,
            protocol=PROTOCOL_MAIN,
            split_seed=split_seed,
            ratios=ratios,
        )
    ]
    manifests.extend(
        build_manifest(
            records,
            protocol=PROTOCOL_LOO,
            split_seed=split_seed,
            ratios=ratios,
            heldout_l1=l1,
        )
        for l1 in sorted(speakers_by_l1)
    )
    return manifests
