#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=word-joint-eval
#SBATCH --array=0-3%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm/word-joint-eval-%A_%a.out

set -euo pipefail

task_id="${SLURM_ARRAY_TASK_ID:?Submit this launcher as a Slurm array}"
accented_hours="${1:-50h}"
if [[ "${accented_hours}" == "500h" ]]; then
  experiment_dir="mswc-common-voice-500h"
  case "${task_id}" in
    0) decoder="greedy" ;;
    1) decoder="beam_4gram" ;;
    *) echo "For 500h, submit --array=0-1; got task ${task_id}." >&2; exit 2 ;;
  esac
else
  case "${task_id}" in
    0) experiment_dir="mswc-common-voice-50h"; decoder="greedy" ;;
    1) experiment_dir="mswc-common-voice-50h"; decoder="beam_4gram" ;;
    2) experiment_dir="mswc-common-voice-50h-librispeech-960"; decoder="greedy" ;;
    3) experiment_dir="mswc-common-voice-50h-librispeech-960"; decoder="beam_4gram" ;;
    *) echo "Unexpected array task: ${task_id}" >&2; exit 2 ;;
  esac
fi

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"

config_path="experiments/joint-training/wav2vec2-large-lv60/word-supcon/${experiment_dir}/full-transformer/evaluation.yaml"
lm_dir="language_models/wav2vec2-base-100h-with-lm"
if [[ "${decoder}" == "beam_4gram" ]]; then
  test -s "${lm_dir}/language_model/4-gram.bin" || {
    echo "Missing 4-gram LM. Run scripts/download_language_model.sh first." >&2
    exit 1
  }
fi

echo "Evaluating ${experiment_dir}/full-transformer with ${decoder}"
exec pixi run snakemake -s Snakefile evaluate_all \
  --configfile "${config_path}" \
  --config \
    run_smoke=false \
    evaluation_decoder="${decoder}" \
    evaluation_lm_dir="${lm_dir}" \
    evaluation_beam_width=100 \
  --rerun-incomplete --nolock --cores 1

# Evaluate 100h and 960h, Greedy and LM, with at most two H200 GPUs:
# sbatch scripts/slurm/run_word_joint_evaluation.sbatch
# Evaluate the 500 h accented-data experiment:
# sbatch --array=0-1%2 scripts/slurm/run_word_joint_evaluation.sbatch 500h
