#!/usr/bin/env bash
#SBATCH --partition=GPU-A100
#SBATCH --account=efl
#SBATCH --job-name=pending-eval
#SBATCH --array=0-14%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm/pending-%x-%A_%a.out

set -euo pipefail

accents=(arabic chinese hindi korean spanish vietnamese)
variants=(freeze-18 full-transformer)
task_id="${SLURM_ARRAY_TASK_ID:?Submit this launcher as a Slurm array}"

case "${task_id}" in
  0)
    # The only missing result from the main seed-13 evaluation grid.
    label="supcon-only/korean/beam_4gram"
    decoder="beam_4gram"
    config_path="experiments/eval/wav2vec2-large-lv60/supcon-only/korean/config.yaml"
    ;;
  1|2)
    # Complete the four missing external datasets for both Arabic joint runs.
    variant="${variants[task_id - 1]}"
    label="joint/${variant}/arabic/greedy"
    decoder="greedy"
    config_path="experiments/joint-training/wav2vec2-large-lv60/utterance-supcon/arabic/${variant}/evaluation.yaml"
    ;;
  *)
    # Evaluate every utterance-level joint model with the fixed 4-gram LM.
    joint_index=$((task_id - 3))
    variant="${variants[joint_index / 6]}"
    accent="${accents[joint_index % 6]}"
    label="joint/${variant}/${accent}/beam_4gram"
    decoder="beam_4gram"
    config_path="experiments/joint-training/wav2vec2-large-lv60/utterance-supcon/${accent}/${variant}/evaluation.yaml"
    ;;
esac

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"

test -f "${config_path}" || {
  echo "Missing evaluation config: ${config_path}" >&2
  exit 1
}

lm_dir="language_models/wav2vec2-base-100h-with-lm"
if [[ "${decoder}" == "beam_4gram" ]]; then
  test -s "${lm_dir}/language_model/4-gram.bin" || {
    echo "Missing 4-gram LM. Run scripts/download_language_model.sh first." >&2
    exit 1
  }
fi

echo "Evaluation task ${task_id}: ${label}"
exec pixi run snakemake -s Snakefile evaluate_all \
  --configfile "${config_path}" \
  --config \
    run_smoke=false \
    evaluation_decoder="${decoder}" \
    evaluation_lm_dir="${lm_dir}" \
    evaluation_beam_width=100 \
  --rerun-incomplete --nolock --cores 1

# Launch all remaining seed-13 evaluations, with at most two A100 GPUs:
# sbatch scripts/slurm/run_pending_evaluations.sbatch
