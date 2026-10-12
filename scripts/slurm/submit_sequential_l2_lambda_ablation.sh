#!/usr/bin/env bash
# Submit the one-seed sequential utterance-level SupCon weight sweep.

set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
mkdir -p outputs/slurm

PYTHONPATH=src pixi run python \
  scripts/experiments/generate_sequential_l2_lambda_ablation_configs.py

reference_warmup="experiments/sequential-l2-adaptation/librispeech-100h/wav2vec2-large-lv60/arabic/outputs/seed=13/warmup/checkpoint.pt"
test -s "${reference_warmup}" || {
  echo "Missing reference warm-up: ${reference_warmup}" >&2
  echo "Run or recover the existing Arabic seed-13 sequential 0.05 experiment first." >&2
  exit 1
}

adapt_job=$(sbatch --parsable --array=0-3%2 \
  scripts/slurm/run_sequential_l2_lambda_adaptation.sbatch)
eval_job=$(sbatch --parsable --dependency="afterok:${adapt_job}" --array=0-7%2 \
  scripts/slurm/run_sequential_l2_lambda_evaluation.sbatch)

printf 'Reference warm-up: %s\nAdaptation sweep: %s\nEvaluation sweep: %s\n' \
  "${reference_warmup}" "${adapt_job}" "${eval_job}"
printf 'New weights: lambda = 0.01, 0.03, 0.10, 0.20 (Arabic, seed 13); lambda = 0.05 is reused.\n'
printf 'At most two H200 GPUs are used at any time.\n'
