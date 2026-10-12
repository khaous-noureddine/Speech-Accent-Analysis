#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
accent_selector="${1:-arabic}"
seed_selector="${2:-all}"
mkdir -p outputs/slurm

case "${accent_selector}" in
  all) accents=(arabic chinese hindi korean spanish vietnamese) ;;
  arabic|chinese|hindi|korean|spanish|vietnamese)
    accents=("${accent_selector}")
    ;;
  *) echo "Accent must be a fold name or all: ${accent_selector}" >&2; exit 2 ;;
esac
case "${seed_selector}" in
  all) seeds=(13 42 77) ;;
  13|42|77) seeds=("${seed_selector}") ;;
  *) echo "Seed must be 13, 42, 77, or all." >&2; exit 2 ;;
esac

for accent in "${accents[@]}"; do
  for seed in "${seeds[@]}"; do
    warmup=$(sbatch --parsable \
      scripts/slurm/run_sequential_l2_warmup.sbatch "${accent}" "${seed}")
    parallel=$(sbatch --parsable --dependency="afterok:${warmup}" \
      scripts/slurm/run_sequential_l2_adaptation.sbatch \
      "${accent}" ctc_supcon "${seed}")
    augmented=$(sbatch --parsable --dependency="afterok:${warmup}" \
      scripts/slurm/run_sequential_l2_adaptation.sbatch \
      "${accent}" ctc_augmented_supcon "${seed}")
    parallel_eval=$(sbatch --parsable --dependency="afterok:${parallel}" \
      scripts/slurm/run_sequential_l2_evaluation.sbatch \
      "${accent}" ctc_supcon "${seed}")
    augmented_eval=$(sbatch --parsable --dependency="afterok:${augmented}" \
      scripts/slurm/run_sequential_l2_evaluation.sbatch \
      "${accent}" ctc_augmented_supcon "${seed}")
    printf 'accent=%s seed=%s warmup=%s parallel=%s augmented=%s parallel_eval=%s augmented_eval=%s\n' \
      "${accent}" "${seed}" "${warmup}" "${parallel}" "${augmented}" \
      "${parallel_eval}" "${augmented_eval}"
  done
done
