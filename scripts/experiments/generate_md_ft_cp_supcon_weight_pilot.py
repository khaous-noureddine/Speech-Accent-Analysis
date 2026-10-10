#!/usr/bin/env python3
"""Generate the lambda=0.03 MD-FT + CP-SupCon pilot configs."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path("experiments/joint/librispeech-100h/wav2vec2-large-lv60")
PILOT_ROOT = Path(
    "experiments/ablations/md-ft-cp-supcon-weight/librispeech-100h/"
    "wav2vec2-large-lv60/utterance"
)
ACCENTS = ("arabic", "chinese")
SUPCON_WEIGHT = 0.03


def read_yaml(path: Path) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def write_yaml(path: Path, document: dict) -> None:
    destination = ROOT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(document, sort_keys=False, width=1000), encoding="utf-8"
    )


def main() -> None:
    for accent in ACCENTS:
        source_dir = (
            MODEL_ROOT / "md-ft-cp-supcon" / "utterance" / accent / "full-transformer"
        )
        target_dir = PILOT_ROOT / accent / "lambda-0p03" / "full-transformer"
        objective = (
            f"wav2vec2-large-lv60_md-ft-cp-supcon-utterance_{accent}_lambda-0p03"
        )

        training = copy.deepcopy(read_yaml(source_dir / "config.yaml"))
        training["experiment"]["name"] = objective
        settings = training["joint_training"]["training"]
        settings["supcon_weight"] = SUPCON_WEIGHT
        settings["output_dir"] = str(target_dir / "outputs")
        write_yaml(target_dir / "config.yaml", training)

        evaluation = copy.deepcopy(read_yaml(source_dir / "evaluation.yaml"))
        settings = evaluation["evaluation"]
        settings["name"] = f"{objective}_evaluation"
        settings["objective"] = objective
        settings["checkpoint"] = str(
            target_dir / "outputs/seed=13/checkpoint_best.pt"
        )
        settings["joint_config"] = str(target_dir / "config.yaml")
        settings["output_dir"] = str(target_dir / "outputs")
        write_yaml(target_dir / "evaluation.yaml", evaluation)


if __name__ == "__main__":
    main()
