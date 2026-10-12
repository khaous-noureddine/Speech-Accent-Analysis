#!/usr/bin/env bash
# Submit the six-fold, one-seed sequential L2 word-SupCon experiment.

set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
mkdir -p outputs/slurm

seed="${1:-13}"
[[ "${seed}" == "13" ]] || {
  echo "This pilot is intentionally configured only for seed 13." >&2
  exit 2
}

warmup_job=$(sbatch --parsable --array=0-5%2 \
  scripts/slurm/run_sequential_l2_word_warmup.sbatch "${seed}")
adapt_job=$(sbatch --parsable --dependency="afterok:${warmup_job}" --array=0-5%2 \
  scripts/slurm/run_sequential_l2_word_adaptation.sbatch "${seed}")
eval_job=$(sbatch --parsable --dependency="afterok:${adapt_job}" --array=0-5%2 \
  scripts/slurm/run_sequential_l2_word_evaluation.sbatch "${seed}")

printf 'Warm-up:    %s\nAdaptation: %s\nEvaluation: %s\n' \
  "${warmup_job}" "${adapt_job}" "${eval_job}"
printf '\nAt most two GPUs are used at any time.\n'
