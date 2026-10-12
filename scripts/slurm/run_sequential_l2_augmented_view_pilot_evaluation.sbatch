#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=seq-l2-aug-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm/seq-l2-aug-eval-%A_%a.out

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

exec bash scripts/slurm/run_sequential_l2_evaluation.sbatch \
  "${accent}" ctc_augmented_supcon "${seed}"
