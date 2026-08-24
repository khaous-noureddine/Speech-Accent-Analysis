"""Snakemake orchestration for the resubmission Stage 2 experiments.

Safe first commands:

    pixi run snakemake -s Snakefile -n stage2_smoke_all
    CUDA_VISIBLE_DEVICES=0 pixi run snakemake -s Snakefile \
        stage2_smoke_all --cores 4 --resources gpu=1

The full campaign is selected explicitly with the ``stage2_all`` target.
Slurm settings belong to the Snakemake Slurm profile/executor, not to the
scientific experiment configurations.
"""

from pathlib import Path

import yaml


EXPERIMENTS = [
    "wav2vec2-base_supcon-ctc",
    "wav2vec2-base_supcon-only",
    "wav2vec2-base_ctc-only",
]

EXPERIMENT_CONDITIONS = {
    "wav2vec2-base_supcon-ctc": ("A", "supcon_ctc"),
    "wav2vec2-base_supcon-only": ("E", "supcon_only"),
    "wav2vec2-base_ctc-only": ("F", "ctc_only"),
}

ACCENTS = [
    "arabic",
    "chinese",
    "hindi",
    "korean",
    "spanish",
    "vietnamese",
]

SEEDS = [13, 42, 77]
SMOKE_EXPERIMENTS = EXPERIMENTS
SMOKE_ACCENTS = ["arabic"]
SMOKE_SEEDS = [13]

# Per-job resources are centralized here, never duplicated in experiment YAML.
STAGE2_THREADS = 4
STAGE2_GPUS = 1
STAGE2_MEMORY_MB = 32000
STAGE2_RUNTIME_MINUTES = 1440
SMOKE_RUNTIME_MINUTES = 60

CONFIG_PATTERN = "experiments/stage2/{experiment}/{accent}/config.yaml"
OUTPUT_PATTERN = (
    "experiments/stage2/{experiment}/{accent}/outputs/"
    "seed={seed}/{artifact}"
)
SMOKE_OUTPUT_PATTERN = (
    "experiments/stage2/{experiment}/{accent}/outputs/"
    "seed={seed}/smoke/{artifact}"
)


def validate_experiment_configs():
    """Fail before scheduling if a config disagrees with its directory."""
    for experiment in EXPERIMENTS:
        expected_condition, expected_mode = EXPERIMENT_CONDITIONS[experiment]
        for accent in ACCENTS:
            path = Path(
                CONFIG_PATTERN.format(experiment=experiment, accent=accent)
            )
            if not path.is_file():
                raise ValueError(f"Missing Stage 2 config: {path}")
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            metadata = document["experiment"]
            expected = {
                "name": experiment,
                "stage": 2,
                "condition": expected_condition,
                "loss_mode": expected_mode,
                "fold": accent,
                "heldout_accent": accent,
                "seeds": SEEDS,
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise ValueError(
                        f"{path}: experiment.{key}={metadata.get(key)!r}; "
                        f"expected {value!r}."
                    )


validate_experiment_configs()


wildcard_constraints:
    experiment="|".join(EXPERIMENTS),
    accent="|".join(ACCENTS),
    seed="|".join(map(str, SEEDS))


localrules: stage2_smoke_all, stage2_all


rule stage2_smoke_all:
    """Default: bounded A/E/F checks on Arabic with seed 13."""
    input:
        expand(
            SMOKE_OUTPUT_PATTERN,
            experiment=SMOKE_EXPERIMENTS,
            accent=SMOKE_ACCENTS,
            seed=SMOKE_SEEDS,
            artifact=["checkpoint_best.pt"],
        )


rule stage2_all:
    """Complete campaign: 3 conditions x 6 accents x 3 seeds = 54 runs."""
    input:
        expand(
            OUTPUT_PATTERN,
            experiment=EXPERIMENTS,
            accent=ACCENTS,
            seed=SEEDS,
            artifact=["checkpoint_best.pt"],
        )


rule stage2_adaptation:
    """Run one complete condition/accent/seed Stage 2 adaptation."""
    input:
        config=CONFIG_PATTERN,
        parquet=(
            "data/processed/l2_arctic_leave_one_accent_out/"
            "{accent}/corpus.parquet"
        ),
        manifest=(
            "data/processed/l2_arctic_leave_one_accent_out/"
            "{accent}/manifest.json"
        ),
        runner="scripts/local/run_adaptation.sh",
        train="src/accented_asr/adaptation/train.py",
        data="src/accented_asr/adaptation/data.py",
        model="src/accented_asr/adaptation/model.py",
        vocab="configs/tokenizers/librispeech_char/vocab.json",
    output:
        best=OUTPUT_PATTERN.replace("{artifact}", "checkpoint_best.pt"),
        final=OUTPUT_PATTERN.replace("{artifact}", "checkpoint_final.pt"),
        metrics=OUTPUT_PATTERN.replace("{artifact}", "metrics.jsonl"),
        resolved=OUTPUT_PATTERN.replace("{artifact}", "config.resolved.json"),
    log:
        OUTPUT_PATTERN.replace("{artifact}", "training.log"),
    benchmark:
        OUTPUT_PATTERN.replace("{artifact}", "benchmark.tsv"),
    threads: STAGE2_THREADS
    resources:
        gpu=STAGE2_GPUS,
        mem_mb=STAGE2_MEMORY_MB,
        runtime=STAGE2_RUNTIME_MINUTES,
    shell:
        r"""
        mkdir -p "$(dirname {log})"
        {input.runner} {input.config} {wildcards.seed} > {log} 2>&1
        """


rule stage2_adaptation_smoke:
    """Run the bounded form of one Stage 2 adaptation."""
    input:
        config=CONFIG_PATTERN,
        parquet=(
            "data/processed/l2_arctic_leave_one_accent_out/"
            "{accent}/corpus.parquet"
        ),
        manifest=(
            "data/processed/l2_arctic_leave_one_accent_out/"
            "{accent}/manifest.json"
        ),
        runner="scripts/local/run_adaptation.sh",
        train="src/accented_asr/adaptation/train.py",
        data="src/accented_asr/adaptation/data.py",
        model="src/accented_asr/adaptation/model.py",
        vocab="configs/tokenizers/librispeech_char/vocab.json",
    output:
        best=SMOKE_OUTPUT_PATTERN.replace("{artifact}", "checkpoint_best.pt"),
        final=SMOKE_OUTPUT_PATTERN.replace("{artifact}", "checkpoint_final.pt"),
        metrics=SMOKE_OUTPUT_PATTERN.replace("{artifact}", "metrics.jsonl"),
        resolved=SMOKE_OUTPUT_PATTERN.replace("{artifact}", "config.resolved.json"),
    log:
        SMOKE_OUTPUT_PATTERN.replace("{artifact}", "training.log"),
    benchmark:
        SMOKE_OUTPUT_PATTERN.replace("{artifact}", "benchmark.tsv"),
    threads: STAGE2_THREADS
    resources:
        gpu=STAGE2_GPUS,
        mem_mb=STAGE2_MEMORY_MB,
        runtime=SMOKE_RUNTIME_MINUTES,
    shell:
        r"""
        mkdir -p "$(dirname {log})"
        {input.runner} {input.config} {wildcards.seed} --smoke > {log} 2>&1
        """
