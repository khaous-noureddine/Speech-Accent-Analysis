"""Run one config-defined Stage 2 adaptation.

The Snakefile never enumerates experiments, accents, or seeds. A shell or
Slurm launcher selects one experiment configuration and one run seed:

    pixi run snakemake -s Snakefile stage2_adaptation \
        --configfile experiments/stage2/wav2vec2-large-lv60/supcon-only/arabic/config.yaml \
        --config run_seed=13 --cores 1

Add ``run_smoke=true`` to the CLI config for a bounded smoke run.
"""

from pathlib import Path


L2_ARCTIC_ACCENTS = ("arabic", "chinese", "hindi", "korean", "spanish", "vietnamese")
L2_ARCTIC_RAW_DIR = config.get("l2_arctic_raw_dir", "data/raw/l2_arctic/speakers")


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
L2_ARCTIC_PROCESSED_DIR = str(Path(FOLD_DIR).parent)
RUN_DIR = f"{TRAINING['output_dir']}/seed={RUN_SEED}"
SMOKE_ARGUMENT = ""
if RUN_SMOKE:
    RUN_DIR = f"{RUN_DIR}/smoke"
    SMOKE_ARGUMENT = "--smoke"


rule prepare_l2_arctic_splits:
    """Build all leave-one-accent-out folds directly from raw L2-ARCTIC."""
    input:
        raw=L2_ARCTIC_RAW_DIR,
        prepare="src/accented_asr/data/prepare_l2_arctic.py",
        splits="src/accented_asr/data/l2_arctic_splits.py",
    output:
        inventory=f"{L2_ARCTIC_PROCESSED_DIR}/inventory.parquet",
        inventory_report=f"{L2_ARCTIC_PROCESSED_DIR}/inventory_report.json",
        summary=f"{L2_ARCTIC_PROCESSED_DIR}/fold_summary.csv",
        parquets=expand(
            f"{L2_ARCTIC_PROCESSED_DIR}/{{accent}}/corpus.parquet",
            accent=L2_ARCTIC_ACCENTS,
        ),
        manifests=expand(
            f"{L2_ARCTIC_PROCESSED_DIR}/{{accent}}/manifest.json",
            accent=L2_ARCTIC_ACCENTS,
        ),
        reports=expand(
            f"{L2_ARCTIC_PROCESSED_DIR}/{{accent}}/validation_report.json",
            accent=L2_ARCTIC_ACCENTS,
        ),
        stats=expand(
            f"{L2_ARCTIC_PROCESSED_DIR}/{{accent}}/split_stats.csv",
            accent=L2_ARCTIC_ACCENTS,
        ),
        hashes=expand(
            f"{L2_ARCTIC_PROCESSED_DIR}/{{accent}}/manifest.content.sha256",
            accent=L2_ARCTIC_ACCENTS,
        ),
    params:
        output_dir=L2_ARCTIC_PROCESSED_DIR,
    shell:
        r"""
        PYTHONPATH=src python -m accented_asr.data.prepare_l2_arctic \
            --corpus-dir {input.raw:q} \
            --output-dir {params.output_dir:q} \
            --repository-root . \
            --split-seed 20260817
        """


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
