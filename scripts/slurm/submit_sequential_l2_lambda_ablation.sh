#!/usr/bin/env bash
# Submit the one-seed sequential utterance-level SupCon weight sweep.

set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
mkdir -p outputs/slurm

PYTHONPATH=src pixi run python \
  scripts/experiments/generate_sequential_l2_lambda_ablation_configs.py

warmup_job=$(sbatch --parsable scripts/slurm/run_sequential_l2_lambda_warmup.sbatch)
adapt_job=$(sbatch --parsable --dependency="afterok:${warmup_job}" --array=0-4%2 \
  scripts/slurm/run_sequential_l2_lambda_adaptation.sbatch)
eval_job=$(sbatch --parsable --dependency="afterok:${adapt_job}" --array=0-9%2 \
  scripts/slurm/run_sequential_l2_lambda_evaluation.sbatch)

printf 'Shared warm-up: %s\nAdaptation sweep: %s\nEvaluation sweep: %s\n' \
  "${warmup_job}" "${adapt_job}" "${eval_job}"
printf 'Weights: lambda = 0.01, 0.03, 0.05, 0.10, 0.20 (Arabic, seed 13)\n'
printf 'At most two H200 GPUs are used at any time.\n'
