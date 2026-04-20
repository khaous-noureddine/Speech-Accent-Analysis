# run with :

# snakemake --snakefile Snakefile \
#   --cores 1 \
#   --resources gpu=1 \
#   --config device=cuda n_gpus=1

configfile: "config.yaml"






# rules 



rule prepare_speech_accent_corpus:
    input:
    output:
    params:
    shell:
        """
        pixi run python corpus/speech_accents/import_speech_accents.py \
            --corpus_dir  speech_accents \
        """



















# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
TIMIT_EMB_ROOT = "wav2vec2_learning_curve/evaluation/timit/embeddings"
RESULTS_ROOT   = "wav2vec2_learning_curve/evaluation/timit/results"

from pathlib import Path
import re


def is_checkpoint_model(model_name):
    return config["models"][model_name].get("from_checkpoint", False)

def checkpoint_models():
    return [m for m in config["models"] if is_checkpoint_model(m)]

def regular_models():
    return [m for m in config["models"] if not is_checkpoint_model(m)]


def list_checkpoints(model_name):
    ckpt_dir = Path(config["models"][model_name]["checkpoints_dir"])
    out = []

    for f in ckpt_dir.glob("checkpoint_*.pt"):
        m = re.match(r"^checkpoint_\d+_(\d+)\.pt$", f.name)
        if m:
            steps = m.group(1)
            out.append((steps, str(f)))

    return sorted(out, key=lambda x: int(x[0]))


def checkpoint_path_for(model_name, steps):
    ckpts = dict(list_checkpoints(model_name))
    if steps not in ckpts:
        raise ValueError(f"Aucun checkpoint trouvé pour {model_name} à step={steps}")
    return ckpts[steps]


def get_all_targets():
    targets = []

    for model in regular_models():
        for layer in config["models"][model]["layers"]:
            targets.append(
                f"{RESULTS_ROOT}/probe/{model}/layer_{layer}/metrics.json"
            )

    for model in checkpoint_models():
        for steps, _ in list_checkpoints(model):
            for layer in config["models"][model]["layers"]:
                targets.append(
                    f"{RESULTS_ROOT}/probe/{model}_{steps}/layer_{layer}/metrics.json"
                )

    return targets



# ---------------------------------------------------------------------------
# Target rules
# ---------------------------------------------------------------------------
rule all:
    input:
        get_all_targets()




# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


rule preprocess_timit_corpus:
    input:
        corpusdir = config["timit"]["timit_root"],
        script    = "corpus/timit/import_timit.py"
    output:
        parquet = config["timit"]["parquet_path"]
    params:
        output_dir = config["timit"]["output_dir"]
    shell:
        """
        pixi run python {input.script} \
            --corpus_dir {input.corpusdir} \
            --output_dir {params.output_dir} \
            --sample_rate 16000
        """



# ------- REGULAR MODELS --------

# extract embeddings
rule extract_timit_embeddings_regular:
    input:
        parquet = config["timit"]["parquet_path"],
        script  = "representations_analysis/extract_representations.py"
    output:
        done = f"{TIMIT_EMB_ROOT}/{{model}}/.done"
    params:
        output_dir        = lambda w: f"{TIMIT_EMB_ROOT}/{w.model}",
        audio_col         = config["timit"]["audio_col"],
        device            = config.get("device", "cpu"),
        acoustic_features = lambda w: config["models"][w.model].get("acoustic_features", ""),
    wildcard_constraints:
        model="|".join(re.escape(m) for m in regular_models())
    shell:
        """
        pixi run python {input.script} \
            --parquet    {input.parquet} \
            --audio_col  {params.audio_col} \
            --output_dir {params.output_dir} \
            --model      {wildcards.model} \
            --device     {params.device} \
            $([ -n "{params.acoustic_features}" ] && echo "--acoustic_features {params.acoustic_features}")
        touch {output.done}
        """
# probe
rule probe_regular:
    input:
        parquet    = config["timit"]["parquet_path"],
        script     = "phoneme_distance/phoneme_probe.py",
        timit_done = f"{TIMIT_EMB_ROOT}/{{model}}/.done",
    output:
        json       = f"{RESULTS_ROOT}/probe/{{model}}/layer_{{layer}}/metrics.json",
        confusion  = f"{RESULTS_ROOT}/probe/{{model}}/layer_{{layer}}/confusion.csv",
        classifier = f"{RESULTS_ROOT}/probe/{{model}}/layer_{{layer}}/classifier.joblib",
    params:
        embeddings    = lambda w: f"{TIMIT_EMB_ROOT}/{w.model}",
        split_train   = config["timit"]["split"],
        boundary_trim = config["probe"]["boundary_trim"],
        n_bootstrap   = config["probe"]["n_bootstrap"],
        ci_alpha      = config["probe"]["ci_alpha"],
        C             = config["probe"]["C"],
        max_frames    = config["probe"]["max_frames"],
        max_iter      = config["probe"]["max_iter"],
    wildcard_constraints:
        model="|".join(re.escape(m) for m in regular_models()),
        layer=r"\d+"
    shell:
        """
        pixi run python {input.script} \
            --timit_parquet     {input.parquet} \
            --embeddings        {params.embeddings} \
            --layer             {wildcards.layer} \
            --output_json       {output.json} \
            --output_confusion  {output.confusion} \
            --output_model      {output.classifier} \
            --split_train       {params.split_train} \
            --split_test        test \
            --boundary_trim     {params.boundary_trim} \
            --n_bootstrap       {params.n_bootstrap} \
            --ci_alpha          {params.ci_alpha} \
            --C                 {params.C} \
            --max_frames        {params.max_frames} \
            --max_iter          {params.max_iter}
        """



# -------- CHECKPOINT MODELS --------

# extract embeddings
rule extract_timit_embeddings_checkpoint:
    input:
        parquet = config["timit"]["parquet_path"],
        script  = "representations_analysis/extract_representations.py"
    output:
        embdir = temp(directory(f"{TIMIT_EMB_ROOT}/{{model}}_{{steps}}"))
    params:
        output_dir      = lambda w: f"{TIMIT_EMB_ROOT}/{w.model}_{w.steps}",
        audio_col       = config["timit"]["audio_col"],
        device          = config.get("device", "cpu"),
        checkpoint_path = lambda w: checkpoint_path_for(w.model, w.steps),
    wildcard_constraints:
        model="|".join(re.escape(m) for m in checkpoint_models()),
        steps=r"\d+"
    shell:
        """
        pixi run python {input.script} \
            --parquet         {input.parquet} \
            --audio_col       {params.audio_col} \
            --output_dir      {params.output_dir} \
            --model           {wildcards.model} \
            --device          {params.device} \
            --checkpoint_path {params.checkpoint_path}
        """

# probe 
rule probe_checkpoint:
    input:
        parquet    = config["timit"]["parquet_path"],
        script     = "phoneme_distance/phoneme_probe.py",
        embdir     = f"{TIMIT_EMB_ROOT}/{{model}}_{{steps}}"
    output:
        json       = f"{RESULTS_ROOT}/probe/{{model}}_{{steps}}/layer_{{layer}}/metrics.json",
        confusion  = f"{RESULTS_ROOT}/probe/{{model}}_{{steps}}/layer_{{layer}}/confusion.csv",
        classifier = f"{RESULTS_ROOT}/probe/{{model}}_{{steps}}/layer_{{layer}}/classifier.joblib",
    params:
        embeddings    = lambda w: f"{TIMIT_EMB_ROOT}/{w.model}_{w.steps}",
        split_train   = config["timit"]["split"],
        boundary_trim = config["probe"]["boundary_trim"],
        n_bootstrap   = config["probe"]["n_bootstrap"],
        ci_alpha      = config["probe"]["ci_alpha"],
        C             = config["probe"]["C"],
        max_frames    = config["probe"]["max_frames"],
        max_iter      = config["probe"]["max_iter"],
    wildcard_constraints:
        model="|".join(re.escape(m) for m in checkpoint_models()),
        steps=r"\d+",
        layer=r"\d+"
    shell:
        """
        pixi run python {input.script} \
            --timit_parquet     {input.parquet} \
            --embeddings        {params.embeddings} \
            --layer             {wildcards.layer} \
            --output_json       {output.json} \
            --output_confusion  {output.confusion} \
            --output_model      {output.classifier} \
            --split_train       {params.split_train} \
            --split_test        test \
            --boundary_trim     {params.boundary_trim} \
            --n_bootstrap       {params.n_bootstrap} \
            --ci_alpha          {params.ci_alpha} \
            --C                 {params.C} \
            --max_frames        {params.max_frames} \
            --max_iter          {params.max_iter}
        """