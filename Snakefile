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


# rules 
rule all:
    input:
        config["arctic"]["parquet_path"],
        config["l2_arctic"]["parquet_path"],




rule prepare_speech_accent_corpus:
    input:
        data_dir = config["speech_accents"]["raw_data_dir"]
    output:
        parquet = config["speech_accents"]["parquet_path"]
    params:
        output_dir = config["speech_accents"]["processed_data_dir"],
        script = workflow.basedir + "/import_speech_accent.py"   # chemin absolu + nom corrigé
    shell:
        """
        python {params.script} \\
        --corpus_dir  {input.data_dir} \\
        --output_parquet {output.parquet} \\
        --audio_dir   {params.output_dir}/wavs
        """


rule prepare_arctic_corpus:
    input:
        data_dir = config["arctic"]["raw_data_dir"]
    output:
        parquet = config["arctic"]["parquet_path"]
    params:
        output_dir = config["arctic"]["processed_data_dir"],
        script = workflow.basedir + "/import_arctic.py"
    shell:
        """
        python {params.script} \
        --corpus_dir  {input.data_dir} \
        --output_parquet {output.parquet} \
        --audio_dir   {params.output_dir}/wavs
        """


rule prepare_l2_arctic_corpus:
    input:
        data_dir = config["l2_arctic"]["raw_data_dir"]
    output:
        parquet = config["l2_arctic"]["parquet_path"]
    params:
        output_dir = config["l2_arctic"]["processed_data_dir"],
        script = "import_l2_arctic.py"
    shell:
        """
        python {params.script} \
        --corpus_dir  {input.data_dir} \
        --output_parquet {output.parquet} \
        --audio_dir   {params.output_dir}/wavs
        """




#         script = workflow.basedir + "/import_l2_arctic.py"