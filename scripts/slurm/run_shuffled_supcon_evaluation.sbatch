#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=shuffle-eval
#SBATCH --array=0-11%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=outputs/slurm/shuffle-eval-%A_%a.out

set -euo pipefail

task_id="${SLURM_ARRAY_TASK_ID:?Submit this launcher as an array from 0 to 11}"
accents=(arabic chinese hindi korean spanish vietnamese)
accent="${accents[$((task_id / 2))]:?Invalid array index}"
[[ $((task_id % 2)) -eq 0 ]] && decoder="greedy" || decoder="beam_4gram"

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
mkdir -p outputs/slurm
experiment="experiments/joint/librispeech-100h/wav2vec2-large-lv60/controls/shuffled-label-supcon/${accent}/full-transformer"
config="${experiment}/evaluation.yaml"
checkpoint="${experiment}/outputs/seed=13/checkpoint_best.pt"
test -s "${checkpoint}" || { echo "Missing checkpoint: ${checkpoint}" >&2; exit 1; }

lm_dir="language_models/wav2vec2-base-100h-with-lm"
if [[ "${decoder}" == "beam_4gram" ]]; then
  test -s "${lm_dir}/language_model/4-gram.bin" || {
    echo "Missing 4-gram language model." >&2
    exit 1
  }
fi

echo "Evaluating shuffled-label CP-SupCon ${accent} with ${decoder}"
exec pixi run snakemake -s Snakefile evaluate_all \
  --configfile "${config}" \
  --config run_smoke=false evaluation_decoder="${decoder}" \
    evaluation_lm_dir="${lm_dir}" evaluation_beam_width=100 \
  --rerun-incomplete --nolock --cores 1
