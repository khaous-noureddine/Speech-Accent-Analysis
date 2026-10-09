#!/usr/bin/env python3
"""Generate six label-shuffled content-parallel SupCon controls."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
BASE = Path("experiments/joint/librispeech-100h/wav2vec2-large-lv60")
ACCENTS = ("arabic", "chinese", "hindi", "korean", "spanish", "vietnamese")


def load(path: Path) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def save(path: Path, document: dict) -> None:
    destination = ROOT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(document, sort_keys=False, width=1000), encoding="utf-8"
    )


def main() -> None:
    for accent in ACCENTS:
        source = BASE / "utterance-supcon" / accent / "full-transformer"
        target = BASE / "controls/shuffled-label-supcon" / accent / "full-transformer"
        objective = f"wav2vec2-large-lv60_shuffled-label-supcon_{accent}"

        training = copy.deepcopy(load(source / "config.yaml"))
        training["experiment"]["name"] = objective
        training["experiment"]["task"] = "joint_ctc_auxiliary"
        settings = training["joint_training"]["training"]
        settings["auxiliary_objective"] = "shuffled_parallel_supcon"
        settings["auxiliary_weight"] = 0.0
        settings["output_dir"] = str(target / "outputs")
        save(target / "config.yaml", training)

        evaluation = copy.deepcopy(load(source / "evaluation.yaml"))
        values = evaluation["evaluation"]
        values["name"] = f"{objective}_evaluation"
        values["objective"] = objective
        values["checkpoint"] = str(target / "outputs/seed=13/checkpoint_best.pt")
        values["joint_config"] = str(target / "config.yaml")
        values["output_dir"] = str(target / "outputs")
        save(target / "evaluation.yaml", evaluation)


if __name__ == "__main__":
    main()
