#!/usr/bin/env python3
"""Generate matched shuffled-content SupCon controls for sequential L2 adaptation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(
    "experiments/sequential-l2-adaptation/librispeech-100h/"
    "wav2vec2-large-lv60"
)
TARGET_ROOT = Path(
    "experiments/controls/sequential-l2-shuffled-content/librispeech-100h/"
    "wav2vec2-large-lv60"
)
ACCENTS = ("arabic", "chinese", "hindi", "korean", "spanish", "vietnamese")
SEED = 13


def write(path: Path, document: dict) -> None:
    destination = ROOT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(document, sort_keys=False, width=1000), encoding="utf-8"
    )


def main() -> None:
    for accent in ACCENTS:
        source_dir = SOURCE_ROOT / accent / f"seed={SEED}"
        target_dir = TARGET_ROOT / accent / f"seed={SEED}"
        training = yaml.safe_load((ROOT / source_dir / "config.yaml").read_text())
        training = deepcopy(training)
        training["experiment"]["name"] = f"sequential-l2-shuffled-content-{accent}"
        training["sequential_l2_adaptation"]["training"]["output_dir"] = str(
            TARGET_ROOT / accent / "outputs"
        )
        write(target_dir / "config.yaml", training)

        evaluation = yaml.safe_load(
            (ROOT / source_dir / "evaluation-ctc_supcon.yaml").read_text()
        )
        evaluation = deepcopy(evaluation)
        payload = evaluation["evaluation"]
        payload["name"] = f"sequential-l2-shuffled-content-{accent}-seed-{SEED}"
        payload["objective"] = "ctc_shuffled_supcon"
        payload["checkpoint"] = str(
            TARGET_ROOT
            / accent
            / "outputs"
            / f"seed={SEED}"
            / "ctc_shuffled_supcon"
            / "checkpoint_best.pt"
        )
        payload["sequential_config"] = str(target_dir / "config.yaml")
        payload["output_dir"] = str(
            TARGET_ROOT / accent / "evaluation" / "ctc_shuffled_supcon"
        )
        write(target_dir / "evaluation-ctc_shuffled_supcon.yaml", evaluation)


if __name__ == "__main__":
    main()
