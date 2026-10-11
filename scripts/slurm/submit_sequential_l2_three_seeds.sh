#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
accent="${1:-arabic}"
mkdir -p outputs/slurm

for seed in 13 42 77; do
  warmup=$(sbatch --parsable scripts/slurm/run_sequential_l2_warmup.sbatch "${accent}" "${seed}")
  ctc=$(sbatch --parsable --dependency="afterok:${warmup}" scripts/slurm/run_sequential_l2_adaptation.sbatch "${accent}" ctc "${seed}")
  supcon=$(sbatch --parsable --dependency="afterok:${warmup}" scripts/slurm/run_sequential_l2_adaptation.sbatch "${accent}" ctc_supcon "${seed}")
  ctc_eval=$(sbatch --parsable --dependency="afterok:${ctc}" scripts/slurm/run_sequential_l2_evaluation.sbatch "${accent}" ctc "${seed}")
  supcon_eval=$(sbatch --parsable --dependency="afterok:${supcon}" scripts/slurm/run_sequential_l2_evaluation.sbatch "${accent}" ctc_supcon "${seed}")
  printf 'seed=%s warmup=%s ctc=%s supcon=%s ctc_eval=%s supcon_eval=%s\n' \
    "${seed}" "${warmup}" "${ctc}" "${supcon}" "${ctc_eval}" "${supcon_eval}"
done
