#!/usr/bin/env bash
# Submit the matched shuffled-content control, six folds and seed 13.

set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
mkdir -p outputs/slurm

PYTHONPATH=src pixi run python \
  scripts/experiments/generate_sequential_l2_shuffled_content_configs.py

warmup_job=$(sbatch --parsable --array=0-5%2 \
  scripts/slurm/run_sequential_l2_shuffled_content_warmup.sbatch)
adapt_job=$(sbatch --parsable --dependency="afterok:${warmup_job}" --array=0-5%2 \
  scripts/slurm/run_sequential_l2_shuffled_content_adaptation.sbatch)
eval_job=$(sbatch --parsable --dependency="afterok:${adapt_job}" --array=0-11%2 \
  scripts/slurm/run_sequential_l2_shuffled_content_evaluation.sbatch)

printf 'Warm-up: %s\nAdaptation: %s\nEvaluation: %s\n' \
  "${warmup_job}" "${adapt_job}" "${eval_job}"
printf 'Six folds, seed 13, max two H200 GPUs at a time.\n'
