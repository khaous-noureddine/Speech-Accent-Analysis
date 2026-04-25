"""
Launch Dta Prep Only:
snakemake --snakefile Snakefile --cores 1 \
  data/processed/arctic/corpus.parquet \
  data/processed/l2_arctic/corpus.parquet


-----------
Launch 

-----------
Launch all:
snakemake --snakefile Snakefile --cores 1
"""


# snakemake --snakefile Snakefile \
#   --cores 1 \
#   --resources gpu=1 \
#   --config device=cuda n_gpus=1


# # Lancer seulement speech_accents
# snakemake --snakefile Snakefile --cores 1 \
#     data/processed/speech_accents/corpus.parquet

# # Lancer seulement arctic
# snakemake --snakefile Snakefile --cores 1 \
#     data/processed/arctic/corpus.parquet

    
configfile: "config.yaml"


# --------------------------------------#
# All at once                           #
# --------------------------------------#
rule all:
    input:
        config["data_preparation"]["arctic"]["parquet_path"],
        config["data_preparation"]["l2_arctic"]["parquet_path"],
        directory(config["supervised_contrastive_training"]["training"]["checkpoint_dir"])



# --------------------------------------#
# Data Preparation                      #
# --------------------------------------#
rule prepare_speech_accent_corpus:
    input:
        data_dir       = config["data_preparation"]["speech_accents"]["raw_data_dir"]

    output:
        parquet        = config["data_preparation"]["speech_accents"]["parquet_path"]

    params:
        output_dir     = config["data_preparation"]["speech_accents"]["processed_data_dir"],
        script         = workflow.basedir + "/import_speech_accent.py"

    shell:
        """
        pixi run python {params.script} \
            --corpus_dir {input.data_dir} \
            --output_parquet {output.parquet} \
            --audio_dir {params.output_dir}/wavs
        """


rule prepare_arctic_corpus:
    input:
        data_dir       = config["data_preparation"]["arctic"]["raw_data_dir"]

    output:
        parquet        = config["data_preparation"]["arctic"]["parquet_path"]

    params:
        output_dir     = config["data_preparation"]["arctic"]["processed_data_dir"],
        script         = workflow.basedir + "/import_arctic.py"

    shell:
        """
        pixi run python {params.script} \
            --corpus_dir {input.data_dir} \
            --output_parquet {output.parquet} \
            --audio_dir {params.output_dir}/wavs
        """


rule prepare_l2_arctic_corpus:
    input:
        data_dir       = config["data_preparation"]["l2_arctic"]["raw_data_dir"]

    output:
        parquet        = config["data_preparation"]["l2_arctic"]["parquet_path"]

    params:
        output_dir     = config["data_preparation"]["l2_arctic"]["processed_data_dir"],
        script         = workflow.basedir + "/import_l2_arctic.py"

    shell:
        """
        pixi run python {params.script} \
            --corpus_dir {input.data_dir} \
            --output_parquet {output.parquet} \
            --audio_dir {params.output_dir}/wavs
        """


# --------------------------------------#
# Supervised Constrastive Learning      #
# --------------------------------------#

rule supervised_contrastive_training:
    input:
        script                = "supcon_train.py",
        arctic_parquet        = config["supervised_contrastive_training"]["data"]["arctic_parquet_path"],
        l2_arctic_parquet     = config["supervised_contrastive_training"]["data"]["l2_arctic_parquet_path"]

    output:
        checkpoint_dir        = directory(config["supervised_contrastive_training"]["training"]["checkpoint_dir"])

    params:
        sample_rate           = config["supervised_contrastive_training"]["data"]["sample_rate"],
        max_audio_len_s       = config["supervised_contrastive_training"]["data"]["max_audio_len_s"],
        split                 = config["supervised_contrastive_training"]["data"]["split"],

        number_of_workers     = config["supervised_contrastive_training"]["sampler"]["number_of_workers"],
        k_utterances          = config["supervised_contrastive_training"]["sampler"]["k_utterances"],
        s_speakers            = config["supervised_contrastive_training"]["sampler"]["s_speakers"],
        n_batches             = config["supervised_contrastive_training"]["sampler"]["n_batches"],
        seed                  = config["supervised_contrastive_training"]["sampler"]["seed"],

        model_name            = config["supervised_contrastive_training"]["model"]["model_name"],
        proj_hidden_dim       = config["supervised_contrastive_training"]["model"]["proj_hidden_dim"],
        proj_out_dim          = config["supervised_contrastive_training"]["model"]["proj_out_dim"],
        vocab_size            = config["supervised_contrastive_training"]["model"]["vocab_size"],
        min_frozen_layer      = config["supervised_contrastive_training"]["model"]["min_frozen_layer"],
        max_frozen_layer      = config["supervised_contrastive_training"]["model"]["max_frozen_layer"],
        ctc_lambda            = config["supervised_contrastive_training"]["model"]["ctc_lambda"],
        temperature           = config["supervised_contrastive_training"]["model"]["temperature"],

        epochs                = config["supervised_contrastive_training"]["training"]["epochs"],
        learning_rate         = config["supervised_contrastive_training"]["training"]["learning_rate"],
        device                = config["supervised_contrastive_training"]["training"]["device"],
        log_every_n_steps     = config["supervised_contrastive_training"]["training"]["log_every_n_steps"]

    shell:
        """
        python {input.script} \
            --arctic_parquet_path {input.arctic_parquet} \
            --l2_arctic_parquet_path {input.l2_arctic_parquet} \
            --sample_rate {params.sample_rate} \
            --max_audio_len_s {params.max_audio_len_s} \
            --split {params.split} \
            --num_workers {params.number_of_workers} \
            --k_utterances {params.k_utterances} \
            --s_speakers {params.s_speakers} \
            --n_batches {params.n_batches} \
            --seed {params.seed} \
            --model_name {params.model_name} \
            --proj_hidden_dim {params.proj_hidden_dim} \
            --proj_out_dim {params.proj_out_dim} \
            --vocab_size {params.vocab_size} \
            --min_frozen_layer {params.min_frozen_layer} \
            --max_frozen_layer {params.max_frozen_layer} \
            --ctc_lambda {params.ctc_lambda} \
            --temperature {params.temperature} \
            --epochs {params.epochs} \
            --lr {params.learning_rate} \
            --device {params.device} \
            --save_dir {output.checkpoint_dir} \
            --log_every_n_steps {params.log_every_n_steps}
        """