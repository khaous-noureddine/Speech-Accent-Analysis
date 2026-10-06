"""Generate the fold-specific configs for prior-method baseline experiments."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
ACCENTS = ("arabic", "chinese", "hindi", "korean", "spanish", "vietnamese")
METHODS = {
    "accent-dat": {"objective": "accent_dat", "weight": 0.1},
    "accent-mtl": {"objective": "accent_mtl", "weight": 0.1},
    "multidomain-ctc": {"objective": "multidomain_ctc", "weight": 0.1},
    "augmented-view-supcon": {"objective": "augmented_view_supcon", "weight": 0.1},
}
DATASETS = {
    "l2_arctic": {
        "split": "test",
        "parquet": None,
        "raw_dir": "data/raw/l2_arctic/speakers",
    },
    "librispeech_test_clean": {
        "split": "test",
        "parquet": "data/processed/librispeech_test_clean/corpus.parquet",
        "raw_dir": "data/raw/librispeech/test/LibriSpeech/test-clean",
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
}


def training_document(method: str, accent: str) -> dict:
    spec = METHODS[method]
    name = f"wav2vec2-large-lv60_{method}_{accent}_full-transformer"
    output = (
        "experiments/baselines/prior-methods/librispeech-100h/"
        f"wav2vec2-large-lv60/{method}/{accent}/full-transformer/outputs"
    )
    return {
        "experiment": {
            "name": name,
            "task": "joint_ctc_auxiliary",
            "fold": accent,
            "heldout_accent": accent,
            "seeds": [13],
        },
        "joint_training": {
            "data": {
                "librispeech_train_parquet": "data/processed/librispeech_train/corpus.parquet",
                "librispeech_dev_parquet": "data/processed/librispeech_eval/corpus.parquet",
                "librispeech_train_raw_dir": "data/raw/librispeech/train/LibriSpeech/train-clean-100",
                "librispeech_dev_raw_dir": "data/raw/librispeech/eval/LibriSpeech/dev-clean",
                "l2_parquet": f"data/processed/l2_arctic_leave_one_accent_out/{accent}/corpus.parquet",
                "l2_raw_dir": "data/raw/l2_arctic/speakers",
                "sample_rate": 16000,
                "max_librispeech_duration_s": 20.0,
                "max_l2_duration_s": 10.0,
                "validate_audio": True,
                "num_workers": 4,
            },
            "sampler": {
                "librispeech_batch_size": 16,
                "prompts_per_batch": 8,
                "speakers_per_prompt": 5,
                "l2_batches_per_cycle": 100,
                "l2_dev_batches": 12,
            },
            "model": {
                "model_name": "facebook/wav2vec2-large-lv60",
                "tokenizer_path": "configs/tokenizers/librispeech_char",
                "projection_hidden_size": 512,
                "projection_size": 256,
                "classifier_hidden_size": 512,
                "temperature": 0.07 if method == "augmented-view-supcon" else 0.1,
                "frozen_transformer_layers": 0,
                "freeze_feature_encoder": True,
                "gradient_checkpointing": True,
            },
            "training": {
                "head_warmup_epochs": 1,
                "max_steps": 50000,
                "supcon_weight": 0.0,
                "auxiliary_objective": spec["objective"],
                "auxiliary_weight": spec["weight"],
                "grl_scale": 1.0,
                "augmentation_noise_std": 0.005,
                "augmentation_time_mask_ratio": 0.05,
                "backbone_lr": 1e-5,
                "head_lr": 1e-4,
                "projection_lr": 1e-5,
                "weight_decay": 0.01,
                "scheduler_warmup_ratio": 0.1,
                "scheduler_hold_ratio": 0.4,
                "final_lr_scale": 0.05,
                "gradient_clip": 1.0,
                "mixed_precision": True,
                "tensorboard": True,
                "tensorboard_subdir": "tensorboard",
                "device": "cuda",
                "log_every_steps": 100,
                "output_dir": output,
            },
            "evaluation": {
                "selection_metric": "librispeech_dev_wer",
                "eval_every_steps": 5000,
            },
        },
    }


def evaluation_document(method: str, accent: str, config_path: Path) -> dict:
    training = training_document(method, accent)
    experiment = training["experiment"]
    output = training["joint_training"]["training"]["output_dir"]
    datasets = {name: dict(values) for name, values in DATASETS.items()}
    datasets["l2_arctic"]["parquet"] = (
        f"data/processed/l2_arctic_leave_one_accent_out/{accent}/corpus.parquet"
    )
    return {
        "evaluation": {
            "name": f"{experiment['name']}_evaluation",
            "model": "wav2vec2-large-lv60",
            "objective": experiment["name"],
            "fold": accent,
            "seed": 13,
            "source": "joint",
            "checkpoint": f"{output}/seed=13/checkpoint_best.pt",
            "stage3_config": None,
            "joint_config": str(config_path.relative_to(ROOT)),
            "decoder": "greedy",
            "output_dir": output,
            "batch_size": 2,
            "num_workers": 2,
            "device": "auto",
            "datasets": datasets,
        }
    }


def main() -> None:
    base = ROOT / "experiments/baselines/prior-methods/librispeech-100h/wav2vec2-large-lv60"
    for method in METHODS:
        for accent in ACCENTS:
            directory = base / method / accent / "full-transformer"
            directory.mkdir(parents=True, exist_ok=True)
            config_path = directory / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(training_document(method, accent), sort_keys=False),
                encoding="utf-8",
            )
            (directory / "evaluation.yaml").write_text(
                yaml.safe_dump(
                    evaluation_document(method, accent, config_path), sort_keys=False
                ),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
