#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=gamma-eval
#SBATCH --array=0-9%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=outputs/slurm/gamma-eval-%A_%a.out

set -euo pipefail

task_id="${SLURM_ARRAY_TASK_ID:?Submit this launcher as a Slurm array}"
gamma_values=(0.01 0.05 0.1 0.5 1.0)
gamma_slugs=(0p01 0p05 0p1 0p5 1p0)

gamma_index=$((task_id / 2))
decoder_index=$((task_id % 2))
gamma="${gamma_values[gamma_index]:?Invalid Slurm array index}"
slug="${gamma_slugs[gamma_index]}"
case "${decoder_index}" in
  0) decoder="greedy" ;;
  1) decoder="beam_4gram" ;;
esac

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
mkdir -p outputs/slurm

experiment_dir="experiments/ablations/supcon-weight/librispeech-100h/wav2vec2-large-lv60/utterance-supcon/arabic/full-transformer/gamma-${slug}"
config_path="${experiment_dir}/evaluation.yaml"
test -f "${config_path}" || { echo "Missing evaluation config: ${config_path}" >&2; exit 1; }

lm_dir="language_models/wav2vec2-base-100h-with-lm"
if [[ "${decoder}" == "beam_4gram" ]]; then
  test -s "${lm_dir}/language_model/4-gram.bin" || {
    echo "Missing 4-gram LM. Run scripts/download_language_model.sh first." >&2
    exit 1
  }
fi

echo "Evaluating Arabic gamma=${gamma} with ${decoder}"
exec pixi run snakemake -s Snakefile evaluate_all \
  --configfile "${config_path}" \
  --config \
    run_smoke=false \
    evaluation_decoder="${decoder}" \
    evaluation_lm_dir="${lm_dir}" \
    evaluation_beam_width=100 \
  --rerun-incomplete --nolock --cores 1

# All gamma values, greedy and 4-gram, at most two GPUs simultaneously:
# sbatch --array=0-9%2 scripts/slurm/run_gamma_ablation_evaluation.sbatch
