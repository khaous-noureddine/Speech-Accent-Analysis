#!/usr/bin/env bash
# Submit Arabic and Korean augmented-view controls for seeds 13, 42, and 77.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p outputs/slurm

train_job=$(sbatch --parsable --array=0-5%2 \
  scripts/slurm/run_sequential_l2_augmented_view_pilot.sbatch)
eval_job=$(sbatch --parsable --dependency="afterok:${train_job}" --array=0-5%2 \
  scripts/slurm/run_sequential_l2_augmented_view_pilot_evaluation.sbatch)

printf 'Training: %s\nEvaluation: %s\n' "${train_job}" "${eval_job}"
printf 'Pairs: Arabic/Korean x seeds 13, 42, 77; max two GPUs.\n'
