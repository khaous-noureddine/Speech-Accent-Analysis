#!/usr/bin/env bash
# Submit Arabic and Korean augmented-view controls for one matched seed.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p outputs/slurm
seed="${1:-13}"
case "${seed}" in 13|42|77) ;; *) echo "Usage: $0 [13|42|77]" >&2; exit 2 ;; esac

train_job=$(sbatch --parsable --array=0-1%2 \
  scripts/slurm/run_sequential_l2_augmented_view_pilot.sbatch "${seed}")
eval_job=$(sbatch --parsable --dependency="afterok:${train_job}" --array=0-1%2 \
  scripts/slurm/run_sequential_l2_augmented_view_pilot_evaluation.sbatch "${seed}")

printf 'Training: %s\nEvaluation: %s\n' "${train_job}" "${eval_job}"
printf 'Pairs: Arabic/Korean x seed %s; max two GPUs.\n' "${seed}"
