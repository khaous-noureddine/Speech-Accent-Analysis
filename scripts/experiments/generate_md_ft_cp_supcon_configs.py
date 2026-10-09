#!/usr/bin/env python3
"""Generate MD-FT + content-parallel SupCon training and evaluation configs."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path("experiments/joint/librispeech-100h/wav2vec2-large-lv60")
ACCENTS = ("arabic", "chinese", "hindi", "korean", "spanish", "vietnamese")


def read_yaml(path: Path) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def write_yaml(path: Path, document: dict) -> None:
    destination = ROOT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(document, sort_keys=False, width=1000), encoding="utf-8"
    )


def combined_training(source: dict, *, name: str, output_dir: Path) -> dict:
    document = copy.deepcopy(source)
    document["experiment"]["name"] = name
    document["experiment"]["task"] = "joint_ctc_auxiliary"
    training = document["joint_training"]["training"]
    training["supcon_weight"] = 0.1
    training["auxiliary_objective"] = "multidomain_ctc_supcon"
    training["auxiliary_weight"] = 0.1
    training["output_dir"] = str(output_dir / "outputs")
    return document


def combined_evaluation(
    source: dict, *, name: str, objective: str, experiment_dir: Path
) -> dict:
    document = copy.deepcopy(source)
    evaluation = document["evaluation"]
    evaluation["name"] = name
    evaluation["objective"] = objective
    evaluation["checkpoint"] = str(
        experiment_dir / "outputs/seed=13/checkpoint_best.pt"
    )
    evaluation["joint_config"] = str(experiment_dir / "config.yaml")
    evaluation["output_dir"] = str(experiment_dir / "outputs")
    return document


def main() -> None:
    for accent in ACCENTS:
        source_dir = MODEL_ROOT / "utterance-supcon" / accent / "full-transformer"
        target_dir = (
            MODEL_ROOT / "md-ft-cp-supcon" / "utterance" / accent / "full-transformer"
        )
        objective = f"wav2vec2-large-lv60_md-ft-cp-supcon-utterance_{accent}"
        write_yaml(
            target_dir / "config.yaml",
            combined_training(
                read_yaml(source_dir / "config.yaml"),
                name=objective,
                output_dir=target_dir,
            ),
        )
        write_yaml(
            target_dir / "evaluation.yaml",
            combined_evaluation(
                read_yaml(source_dir / "evaluation.yaml"),
                name=f"{objective}_evaluation",
                objective=objective,
                experiment_dir=target_dir,
            ),
        )

    dataset = "mswc-common-voice-50h"
    source_dir = MODEL_ROOT / "word-supcon" / dataset / "full-transformer"
    target_dir = MODEL_ROOT / "md-ft-cp-supcon" / "word" / dataset / "full-transformer"
    objective = "wav2vec2-large-lv60_md-ft-cp-supcon-word_mswc-cv-50h"
    write_yaml(
        target_dir / "config.yaml",
        combined_training(
            read_yaml(source_dir / "config.yaml"),
            name=objective,
            output_dir=target_dir,
        ),
    )
    write_yaml(
        target_dir / "evaluation.yaml",
        combined_evaluation(
            read_yaml(source_dir / "evaluation.yaml"),
            name=f"{objective}_evaluation",
            objective=objective,
            experiment_dir=target_dir,
        ),
    )


if __name__ == "__main__":
    main()
