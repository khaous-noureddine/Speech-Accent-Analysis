#!/usr/bin/env python3
"""Generate a controlled SupCon-weight sweep for sequential L2 adaptation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / (
    "experiments/sequential-l2-adaptation/librispeech-100h/"
    "wav2vec2-large-lv60/arabic/seed=13/config.yaml"
)
BASE_EVALUATION = ROOT / (
    "experiments/sequential-l2-adaptation/librispeech-100h/"
    "wav2vec2-large-lv60/arabic/seed=13/evaluation-ctc_supcon.yaml"
)
EXPERIMENT_ROOT = Path(
    "experiments/ablations/sequential-l2-supcon-weight/librispeech-100h/"
    "wav2vec2-large-lv60/arabic"
)
WEIGHTS = (
    ("lambda-0p01", 0.01),
    ("lambda-0p03", 0.03),
    ("lambda-0p10", 0.10),
    ("lambda-0p20", 0.20),
)


def write(path: Path, document: dict) -> None:
    destination = ROOT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(document, sort_keys=False, width=1000), encoding="utf-8")


def main() -> None:
    training_base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    evaluation_base = yaml.safe_load(BASE_EVALUATION.read_text(encoding="utf-8"))

    for slug, weight in WEIGHTS:
        experiment_dir = EXPERIMENT_ROOT / slug
        seed_dir = experiment_dir / "seed=13"

        training = deepcopy(training_base)
        training["experiment"]["name"] = f"sequential-l2-arabic-{slug}"
        training["sequential_l2_adaptation"]["training"]["supcon_weight"] = weight
        training["sequential_l2_adaptation"]["training"]["output_dir"] = str(
            experiment_dir / "outputs"
        )
        write(seed_dir / "config.yaml", training)

        evaluation = deepcopy(evaluation_base)
        payload = evaluation["evaluation"]
        payload["name"] = f"sequential-l2-ctc-supcon-arabic-{slug}-seed-13"
        payload["checkpoint"] = str(
            experiment_dir / "outputs/seed=13/ctc_supcon/checkpoint_best.pt"
        )
        payload["sequential_config"] = str(seed_dir / "config.yaml")
        payload["output_dir"] = str(experiment_dir / "evaluation/ctc_supcon")
        write(seed_dir / "evaluation-ctc_supcon.yaml", evaluation)


if __name__ == "__main__":
    main()
