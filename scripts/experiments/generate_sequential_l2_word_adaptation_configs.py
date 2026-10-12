#!/usr/bin/env python3
"""Generate isolated sequential L2-ARCTIC word-SupCon experiments."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(
    "experiments/sequential-l2-word-adaptation/librispeech-100h/"
    "wav2vec2-large-lv60"
)
ACCENTS = ("arabic", "chinese", "hindi", "korean", "spanish", "vietnamese")
SEED = 13
SOURCE = (
    "experiments/baselines/internal/librispeech-100h/wav2vec2-large-lv60/"
    "ctc-only/full-transformer/outputs/seed=13/checkpoint_best.pt"
)


def write(path: Path, document: dict) -> None:
    destination = ROOT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(document, sort_keys=False, width=1000), encoding="utf-8"
    )


def training_config(accent: str) -> dict:
    experiment_dir = EXPERIMENT_ROOT / accent
    return {
        "experiment": {
            "name": f"sequential-l2-word-{accent}",
            "fold": accent,
            "heldout_accent": accent,
            "seeds": [SEED],
        },
        "sequential_l2_adaptation": {
            "data": {
                # Full utterances supply the L2-ARCTIC CTC loss.
                "parquet_path": (
                    "data/processed/l2_arctic_leave_one_accent_out/"
                    f"{accent}/corpus.parquet"
                ),
                # Aligned word segments supply the lexical SupCon loss.
                "contrastive_unit": "word",
                "word_parquet_path": (
                    f"data/processed/l2_arctic_words/{accent}/corpus.parquet"
                ),
                "sample_rate": 16000,
                "max_audio_len_s": 10.0,
                "max_word_len_s": 1.0,
                "validate_audio": True,
                "num_workers": 4,
            },
            "sampler": {
                "prompts_per_batch": 8,
                "speakers_per_prompt": 5,
                "words_per_batch": 8,
                "accents_per_word": 5,
                "batches_per_epoch": 100,
            },
            "model": {
                "model_name": "facebook/wav2vec2-large-lv60",
                "tokenizer_path": "configs/tokenizers/librispeech_char",
                "source_checkpoint": SOURCE,
                "projection_hidden_size": 512,
                "projection_size": 256,
                "temperature": 0.1,
                "freeze_feature_encoder": True,
                "gradient_checkpointing": True,
            },
            "training": {
                "warmup_epochs": 1,
                "adaptation_epochs": 40,
                "early_stopping_patience": 5,
                "supcon_weight": 0.05,
                "supcon_ramp_ratio": 0.1,
                "backbone_lr": 1e-5,
                "head_lr": 1e-5,
                "projection_lr": 1e-5,
                "weight_decay": 0.01,
                "lr_warmup_ratio": 0.1,
                "gradient_clip": 1.0,
                "mixed_precision": True,
                "output_dir": str(experiment_dir / "outputs"),
                "device": "cuda",
                "log_every_steps": 50,
                "tensorboard": True,
                "tensorboard_subdir": "tensorboard",
            },
            "development": {
                "selection_metric": "l2_dev_wer",
                "batch_size": 4,
            },
        },
    }


def evaluation_config(accent: str) -> dict:
    experiment_dir = EXPERIMENT_ROOT / accent
    config_path = experiment_dir / f"seed={SEED}" / "config.yaml"
    checkpoint = experiment_dir / "outputs" / f"seed={SEED}" / "ctc_word_supcon/checkpoint_best.pt"
    return {
        "evaluation": {
            "name": f"sequential-l2-word-ctc-word-supcon-{accent}-seed-{SEED}",
            "model": "wav2vec2-large-lv60",
            "objective": "ctc_word_supcon",
            "fold": accent,
            "seed": SEED,
            "checkpoint": str(checkpoint),
            "stage3_config": None,
            "joint_config": None,
            "sequential_config": str(config_path),
            "source": "sequential_l2",
            "decoder": "greedy",
            "output_dir": str(experiment_dir / "evaluation" / "ctc_word_supcon"),
            "batch_size": 2,
            "num_workers": 2,
            "device": "auto",
            "datasets": {
                "l2_arctic": {
                    "split": "test",
                    "parquet": f"data/processed/l2_arctic_leave_one_accent_out/{accent}/corpus.parquet",
                    "raw_dir": "data/raw/l2_arctic/speakers",
                },
                "librispeech_test_clean": {
                    "split": "test",
                    "parquet": "data/processed/librispeech_test_clean/corpus.parquet",
                    "raw_dir": "data/raw/librispeech/test/LibriSpeech/test-clean",
                },
                "librispeech_test_other": {
                    "split": "test",
                    "parquet": "data/processed/librispeech_test_other/corpus.parquet",
                    "raw_dir": "data/raw/librispeech/test/LibriSpeech/test-other",
                },
                "aesrc": {
                    "split": "test",
                    "parquet": "data/processed/aesrc/corpus.parquet",
                    "raw_dir": "data/raw/aesrc/data",
                },
                "speech_accent_archive": {
                    "split": "test",
                    "parquet": "data/processed/speech_accent_archive/corpus.parquet",
                    "raw_dir": "data/raw/speech_accent_archive",
                },
                "edacc": {
                    "split": "test",
                    "parquet": "data/processed/edacc/corpus.parquet",
                    "raw_dir": "data/raw/edacc",
                },
            },
        }
    }


def main() -> None:
    for accent in ACCENTS:
        base = EXPERIMENT_ROOT / accent / f"seed={SEED}"
        write(base / "config.yaml", training_config(accent))
        write(base / "evaluation-ctc_word_supcon.yaml", evaluation_config(accent))


if __name__ == "__main__":
    main()
