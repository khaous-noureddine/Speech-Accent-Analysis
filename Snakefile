"""Run one config-defined Stage 2 or Stage 3 experiment."""

from pathlib import Path


if len(workflow.configfiles) != 1:
    raise ValueError("Pass exactly one experiment YAML file with --configfile.")
HAS_STAGE2 = "stage2_adaptation" in config
HAS_STAGE3 = "stage3_finetuning" in config
HAS_EVALUATION = "evaluation" in config
if sum((HAS_STAGE2, HAS_STAGE3, HAS_EVALUATION)) != 1:
    raise ValueError(
        "Configure exactly one of stage2_adaptation, stage3_finetuning, or evaluation."
    )
if not HAS_EVALUATION and "experiment" not in config:
    raise ValueError("Training configs must contain an experiment section.")

RUN_SEED = None
if not HAS_EVALUATION:
    RUN_SEED = config.get("run_seed")
    if RUN_SEED is None:
        raise ValueError("Pass the selected seed with --config run_seed=<seed>.")
    RUN_SEED = int(RUN_SEED)
    EXPERIMENT = config["experiment"]
    DECLARED_SEEDS = [int(seed) for seed in EXPERIMENT["seeds"]]
    if RUN_SEED not in DECLARED_SEEDS:
        raise ValueError(
            f"run_seed={RUN_SEED} is not declared in experiment.seeds={DECLARED_SEEDS}."
        )

RUN_SMOKE = str(config.get("run_smoke", "false")).lower() in {"1", "true", "yes"}
SMOKE_ARGUMENT = "--smoke" if RUN_SMOKE else ""
CONFIG_PATH = str(workflow.configfiles[0])


if HAS_STAGE2:
    L2_ARCTIC_ACCENTS = (
        "arabic", "chinese", "hindi", "korean", "spanish", "vietnamese"
    )
    L2_ARCTIC_RAW_DIR = config.get(
        "l2_arctic_raw_dir", "data/raw/l2_arctic/speakers"
    )
    ADAPTATION = config["stage2_adaptation"]
    DATA = ADAPTATION["data"]
    TRAINING = ADAPTATION["training"]
    PARQUET_PATH = DATA["parquet_path"]
    FOLD_DIR = str(Path(PARQUET_PATH).parent)
    L2_ARCTIC_PROCESSED_DIR = str(Path(FOLD_DIR).parent)
    RUN_DIR = f"{TRAINING['output_dir']}/seed={RUN_SEED}"
    if RUN_SMOKE:
        RUN_DIR = f"{RUN_DIR}/smoke"

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
            PYTHONPATH=src python -m accented_asr.adaptation.train \
                --config {input.config:q} \
                --repository-root . \
                --seed {params.seed} \
                {params.smoke_argument} > {log} 2>&1
            """


if HAS_STAGE3:
    FINETUNING = config["stage3_finetuning"]
    DATA = FINETUNING["data"]
    MODEL = FINETUNING["model"]
    TRAINING = FINETUNING["training"]
    TRAIN_PARQUET = DATA["train_parquet"]
    DEV_PARQUET = DATA["dev_parquet"]
    TRAIN_PROCESSED_DIR = str(Path(TRAIN_PARQUET).parent)
    DEV_PROCESSED_DIR = str(Path(DEV_PARQUET).parent)
    INITIALIZATION = MODEL.get("initialization", "stage2")
    STAGE2_CHECKPOINT = None
    if INITIALIZATION == "stage2":
        STAGE2_CHECKPOINT = f"{MODEL['stage2_output_dir']}/seed={RUN_SEED}/checkpoint_best.pt"
    elif INITIALIZATION != "base":
        raise ValueError(f"Unknown Stage 3 initialization: {INITIALIZATION}")
    STAGE2_CHECKPOINT_INPUT = [STAGE2_CHECKPOINT] if STAGE2_CHECKPOINT else []
    RUN_DIR = f"{TRAINING['output_dir']}/seed={RUN_SEED}"
    if RUN_SMOKE:
        RUN_DIR = f"{RUN_DIR}/smoke"

    rule prepare_librispeech_train:
        """Build the train-clean-100 parquet and WAV directory from raw FLAC."""
        input:
            importer="corpus/import_librispeech.py",
            utils="utils.py",
        output:
            parquet=TRAIN_PARQUET,
        log:
            f"{TRAIN_PROCESSED_DIR}/preparation.log",
        params:
            raw=DATA["train_raw_dir"],
            audio_dir=f"{TRAIN_PROCESSED_DIR}/wavs",
        shell:
            r"""
            test -d {params.raw:q} || {{
                echo "Missing raw LibriSpeech train subset: {params.raw}" >&2
                exit 1
            }}
            mkdir -p "$(dirname {log:q})"
            python {input.importer:q} \
                --corpus_dir {params.raw:q} \
                --output_parquet {output.parquet:q} \
                --audio_dir {params.audio_dir:q} \
                --split train > {log:q} 2>&1
            """


    rule prepare_librispeech_dev:
        """Build the dev-clean parquet and WAV directory from raw FLAC."""
        input:
            importer="corpus/import_librispeech.py",
            utils="utils.py",
        output:
            parquet=DEV_PARQUET,
        log:
            f"{DEV_PROCESSED_DIR}/preparation.log",
        params:
            raw=DATA["dev_raw_dir"],
            audio_dir=f"{DEV_PROCESSED_DIR}/wavs",
        shell:
            r"""
            test -d {params.raw:q} || {{
                echo "Missing raw LibriSpeech development subset: {params.raw}" >&2
                exit 1
            }}
            mkdir -p "$(dirname {log:q})"
            python {input.importer:q} \
                --corpus_dir {params.raw:q} \
                --output_parquet {output.parquet:q} \
                --audio_dir {params.audio_dir:q} \
                --split eval > {log:q} 2>&1
            """

    rule stage3_asr_finetuning:
        """Fine-tune one selected Stage 2 encoder on LibriSpeech CTC ASR."""
        input:
            config=CONFIG_PATH,
            stage2_checkpoint=STAGE2_CHECKPOINT_INPUT,
            train_parquet=TRAIN_PARQUET,
            dev_parquet=DEV_PARQUET,
            train="src/accented_asr/asr/train.py",
            data="src/accented_asr/asr/data.py",
            model="src/accented_asr/asr/model.py",
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
            PYTHONPATH=src python -m accented_asr.asr.train \
                --config {input.config:q} \
                --repository-root . \
                --seed {params.seed} \
                {params.smoke_argument} > {log} 2>&1
            """


if HAS_EVALUATION:
    EVALUATION = config["evaluation"]
    EVAL_DATASETS_CONFIG = EVALUATION["datasets"]
    EVAL_DATASETS = tuple(EVAL_DATASETS_CONFIG)
    if not EVAL_DATASETS:
        raise ValueError("Evaluation config must declare at least one dataset.")
    EXTERNAL_EVAL_DATASETS = tuple(
        dataset for dataset in EVAL_DATASETS if dataset != "l2_arctic"
    )
    EVAL_OUTPUT_PATTERN = (
        f"{EVALUATION['output_dir']}/seed={EVALUATION['seed']}/"
        f"{EVALUATION['decoder']}/{{dataset}}"
    )
    EVAL_CAMPAIGN_DIR = (
        f"{EVALUATION['output_dir']}/seed={EVALUATION['seed']}/"
        f"{EVALUATION['decoder']}"
    )
    EVAL_SUMMARY = f"{EVAL_CAMPAIGN_DIR}/metrics_summary.json"
    if RUN_SMOKE:
        EVAL_OUTPUT_PATTERN = f"{EVAL_OUTPUT_PATTERN}/smoke"
        EVAL_SUMMARY = f"{EVAL_CAMPAIGN_DIR}/metrics_summary.smoke.json"
    EVAL_SMOKE_ARGUMENT = "--smoke" if RUN_SMOKE else ""

    def evaluation_parquet(wildcards):
        return EVAL_DATASETS_CONFIG[wildcards.dataset]["parquet"]

    def evaluation_raw_dir(wildcards):
        return EVAL_DATASETS_CONFIG[wildcards.dataset]["raw_dir"]

    rule evaluate_all:
        """Evaluate one checkpoint on every dataset declared by its config."""
        input:
            summary=EVAL_SUMMARY,

    rule aggregate_evaluation:
        """Collect every dataset WER for one checkpoint in a single JSON file."""
        input:
            metrics=expand(
                f"{EVAL_OUTPUT_PATTERN}/metrics.json", dataset=EVAL_DATASETS
            ),
            aggregator="src/accented_asr/evaluation/metrics.py",
        output:
            summary=EVAL_SUMMARY,
        log:
            f"{EVAL_CAMPAIGN_DIR}/metrics_summary.log",
        shell:
            r"""
            PYTHONPATH=src python -m accented_asr.evaluation.metrics \
                --metrics {input.metrics:q} \
                --output {output.summary:q} > {log:q} 2>&1
            """

    if EXTERNAL_EVAL_DATASETS:
        rule prepare_evaluation_dataset:
            """Prepare one canonical external evaluation corpus when absent."""
            input:
                preparer="src/accented_asr/data/prepare_evaluation_data.py",
            output:
                parquet="data/processed/{dataset}/corpus.parquet",
            log:
                "data/processed/{dataset}/preparation.log",
            params:
                raw_dir=evaluation_raw_dir,
            wildcard_constraints:
                dataset="|".join(EXTERNAL_EVAL_DATASETS),
            shell:
                r"""
                test -d {params.raw_dir:q} || {{
                    echo "Missing raw {wildcards.dataset} data: {params.raw_dir}" >&2
                    exit 1
                }}
                mkdir -p "$(dirname {log:q})"
                PYTHONPATH=src python -m accented_asr.data.prepare_evaluation_data \
                    --dataset {wildcards.dataset:q} \
                    --raw-dir {params.raw_dir:q} \
                    --output-parquet {output.parquet:q} \
                    --repository-root . \
                    --seed 20260817 > {log:q} 2>&1
                """

    rule evaluate_greedy:
        """Decode and score one Stage 3 checkpoint on one fixed test split."""
        input:
            config=CONFIG_PATH,
            checkpoint=EVALUATION["checkpoint"],
            stage3_config=EVALUATION["stage3_config"],
            parquet=evaluation_parquet,
            runner="src/accented_asr/evaluation/run.py",
            metrics="src/accented_asr/evaluation/metrics.py",
            model="src/accented_asr/asr/model.py",
            vocab="configs/tokenizers/librispeech_char/vocab.json",
        output:
            predictions=f"{EVAL_OUTPUT_PATTERN}/predictions.parquet",
            metrics=f"{EVAL_OUTPUT_PATTERN}/metrics.json",
            resolved=f"{EVAL_OUTPUT_PATTERN}/config.resolved.json",
        log:
            f"{EVAL_OUTPUT_PATTERN}/evaluation.log",
        benchmark:
            f"{EVAL_OUTPUT_PATTERN}/benchmark.tsv",
        params:
            smoke_argument=EVAL_SMOKE_ARGUMENT,
        shell:
            r"""
            mkdir -p "$(dirname {log:q})"
            PYTHONPATH=src python -m accented_asr.evaluation.run \
                --config {input.config:q} \
                --dataset {wildcards.dataset:q} \
                --repository-root . \
                {params.smoke_argument} > {log:q} 2>&1
            """
