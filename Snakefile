"""Run one config-defined Stage 2 adaptation.

The Snakefile never enumerates experiments, accents, or seeds. A shell or
Slurm launcher selects one experiment configuration and one run seed:

    pixi run snakemake -s Snakefile stage2_adaptation \
        --configfile experiments/stage2/wav2vec2-large-lv60/supcon-only/arabic/config.yaml \
        --config run_seed=13 --cores 1

Add ``run_smoke=true`` to the CLI config for a bounded smoke run.
"""

from pathlib import Path


if len(workflow.configfiles) != 1:
    raise ValueError("Pass exactly one Stage 2 YAML file with --configfile.")

if "experiment" not in config or "stage2_adaptation" not in config:
    raise ValueError(
        "The config must contain experiment and stage2_adaptation sections."
    )

RUN_SEED = config.get("run_seed")
if RUN_SEED is None:
    raise ValueError("Pass the selected seed with --config run_seed=<seed>.")
RUN_SEED = int(RUN_SEED)

EXPERIMENT = config["experiment"]
ADAPTATION = config["stage2_adaptation"]
DATA = ADAPTATION["data"]
TRAINING = ADAPTATION["training"]
DECLARED_SEEDS = [int(seed) for seed in EXPERIMENT["seeds"]]

if RUN_SEED not in DECLARED_SEEDS:
    raise ValueError(
        f"run_seed={RUN_SEED} is not declared in experiment.seeds={DECLARED_SEEDS}."
    )

RUN_SMOKE = str(config.get("run_smoke", "false")).lower() in {"1", "true", "yes"}
CONFIG_PATH = str(workflow.configfiles[0])
PARQUET_PATH = DATA["parquet_path"]
FOLD_DIR = str(Path(PARQUET_PATH).parent)
RUN_DIR = f"{TRAINING['output_dir']}/seed={RUN_SEED}"
SMOKE_ARGUMENT = ""
if RUN_SMOKE:
    RUN_DIR = f"{RUN_DIR}/smoke"
    SMOKE_ARGUMENT = "--smoke"


rule stage2_adaptation:
    """Train one Stage 2 condition/accent/seed selected by the launcher."""
    input:
        config=CONFIG_PATH,
        parquet=PARQUET_PATH,
        manifest=f"{FOLD_DIR}/manifest.json",
        runner="scripts/local/run_adaptation.sh",
        train="src/accented_asr/adaptation/train.py",
        data="src/accented_asr/adaptation/data.py",
        model="src/accented_asr/adaptation/model.py",
        vocab="configs/tokenizers/librispeech_char/vocab.json",
    output:
        best=f"{RUN_DIR}/checkpoint_best.pt",
        final=f"{RUN_DIR}/checkpoint_final.pt",
        metrics=f"{RUN_DIR}/metrics.jsonl",
        resolved=f"{RUN_DIR}/config.resolved.json",
    log:
        f"{RUN_DIR}/training.log",
    benchmark:
        f"{RUN_DIR}/benchmark.tsv",
    params:
        seed=RUN_SEED,
        smoke_argument=SMOKE_ARGUMENT,
    shell:
        r"""
        mkdir -p "$(dirname {log})"
        {input.runner} {input.config} {params.seed} {params.smoke_argument} > {log} 2>&1
        """
