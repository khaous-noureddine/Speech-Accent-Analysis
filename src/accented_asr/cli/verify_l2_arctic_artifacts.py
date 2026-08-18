"""Verify canonical L2-ARCTIC inventory, manifests, Parquets, and audio paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import soundfile as sf

from accented_asr.data.l2_arctic_inventory import inventory_sha256
from accented_asr.data.l2_arctic_splits import (
    PROTOCOL_LOO,
    PROTOCOL_MAIN,
    assign_records,
    build_validation_report,
    validate_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-report", type=Path, required=True)
    parser.add_argument("--split-data-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--check-audio",
        action="store_true",
        help="Open every inventory audio header in addition to checking its path.",
    )
    return parser.parse_args()


def _artifact_directories(root: Path) -> list[Path]:
    main = root / "main"
    loo_root = root / "leave_one_l1_out"
    folds = sorted(path for path in loo_root.iterdir() if path.is_dir())
    directories = [main, *folds]
    if not main.is_dir() or len(folds) != 6:
        raise AssertionError(
            f"Expected main plus 6 leave-one-L1-out manifests, found {directories}"
        )
    return directories


def _data_directory(manifest_dir: Path, manifest_root: Path, data_root: Path) -> Path:
    return data_root / manifest_dir.relative_to(manifest_root)


def _records_by_key(records: list[dict]) -> dict[tuple[str, str], dict]:
    return {
        (str(record["speaker_id"]), str(record["prompt_id"])): record
        for record in records
    }


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    inventory = pd.read_parquet(args.inventory).to_dict("records")
    inventory_report = json.loads(args.inventory_report.read_text(encoding="utf-8"))

    computed_inventory_hash = inventory_sha256(inventory)
    assert computed_inventory_hash == inventory_report["inventory_sha256"], (
        "Inventory fingerprint mismatch: "
        f"{computed_inventory_hash} != {inventory_report['inventory_sha256']}"
    )
    assert len(inventory) == inventory_report["n_inventory_examples"]
    assert len({row["speaker_id"] for row in inventory}) == 24
    assert (
        len({row["prompt_id"] for row in inventory})
        == inventory_report["n_eligible_prompts"]
    )

    missing_audio = []
    unreadable_audio = []
    for record in inventory:
        audio_path = repository_root / str(record["audio_path"])
        if not audio_path.is_file():
            missing_audio.append(str(audio_path))
        elif args.check_audio:
            try:
                info = sf.info(str(audio_path))
                if info.frames <= 0:
                    unreadable_audio.append(str(audio_path))
            except Exception:
                unreadable_audio.append(str(audio_path))
    assert not missing_audio, f"Missing audio files: {missing_audio[:5]}"
    assert not unreadable_audio, f"Unreadable audio files: {unreadable_audio[:5]}"

    seen_heldout_l1s = set()
    source_fingerprints = set()
    summaries = []
    for metadata_dir in _artifact_directories(args.manifest_dir):
        manifest = json.loads((metadata_dir / "manifest.json").read_text(encoding="utf-8"))
        stored_hash = (metadata_dir / "manifest.content.sha256").read_text(
            encoding="utf-8"
        ).strip()
        stored_report = json.loads(
            (metadata_dir / "validation_report.json").read_text(encoding="utf-8")
        )
        validate_manifest(manifest)
        assert stored_hash == manifest["sha256"]
        source_fingerprints.add(manifest["source_fingerprint"])

        data_dir = _data_directory(metadata_dir, args.manifest_dir, args.split_data_dir)
        actual_records = pd.read_parquet(data_dir / "corpus.parquet").to_dict("records")
        expected_records = assign_records(inventory, manifest)
        actual_by_key = _records_by_key(actual_records)
        expected_by_key = _records_by_key(expected_records)
        assert actual_by_key.keys() == expected_by_key.keys()
        for key, expected in expected_by_key.items():
            actual = actual_by_key[key]
            for field in (
                "split",
                "split_protocol",
                "split_seed",
                "heldout_l1",
                "split_manifest_sha256",
                "audio_path",
                "transcript",
            ):
                assert actual[field] == expected[field], f"Mismatch for {key}: {field}"

        computed_report = build_validation_report(actual_records, manifest)
        assert computed_report == stored_report
        assert set(stored_report["prompt_overlap"].values()) == {0}
        assert set(stored_report["speaker_overlap"].values()) == {0}

        if manifest["protocol"] == PROTOCOL_MAIN:
            assert manifest["heldout_l1"] is None
        elif manifest["protocol"] == PROTOCOL_LOO:
            seen_heldout_l1s.add(manifest["heldout_l1"])
        else:
            raise AssertionError(f"Unexpected protocol: {manifest['protocol']}")
        summaries.append(
            (
                manifest["protocol"],
                manifest["heldout_l1"] or "-",
                len(actual_records),
                manifest["sha256"][:12],
            )
        )

    assert len(source_fingerprints) == 1, "Manifests do not share one source inventory."
    assert len(seen_heldout_l1s) == 6

    print("protocol              heldout_l1  examples  manifest")
    for protocol, heldout_l1, examples, manifest_hash in summaries:
        print(f"{protocol:21s} {heldout_l1:11s} {examples:8d}  {manifest_hash}")
    print(f"Inventory: {len(inventory):,} examples, SHA-256 {computed_inventory_hash}")
    print(f"Audio check: {'headers verified' if args.check_audio else 'paths verified'}")
    print("All L2-ARCTIC artifacts passed validation.")


if __name__ == "__main__":
    main()
