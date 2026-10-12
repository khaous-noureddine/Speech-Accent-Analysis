#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=seq-l2-aug-view
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm/seq-l2-aug-view-%A_%a.out

set -euo pipefail

pairs=(
  arabic:13 arabic:42 arabic:77
  korean:13 korean:42 korean:77
)
pair="${pairs[${SLURM_ARRAY_TASK_ID:?Submit as a six-task array}]:?Invalid array index}"
accent="${pair%%:*}"
seed="${pair##*:}"

root="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "${root}"

warmup="experiments/sequential-l2-adaptation/librispeech-100h/wav2vec2-large-lv60/${accent}/outputs/seed=${seed}/warmup/checkpoint.pt"
test -s "${warmup}" || {
  echo "Missing completed warm-up: ${warmup}" >&2
  exit 1
}

exec bash scripts/slurm/run_sequential_l2_adaptation.sbatch \
  "${accent}" ctc_augmented_supcon "${seed}"
