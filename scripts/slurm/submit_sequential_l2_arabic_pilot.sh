#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p outputs/slurm
seed="${1:-13}"

warmup=$(sbatch --parsable scripts/slurm/run_sequential_l2_warmup.sbatch arabic "${seed}")
ctc=$(sbatch --parsable --dependency="afterok:${warmup}" scripts/slurm/run_sequential_l2_adaptation.sbatch arabic ctc "${seed}")
supcon=$(sbatch --parsable --dependency="afterok:${warmup}" scripts/slurm/run_sequential_l2_adaptation.sbatch arabic ctc_supcon "${seed}")
ctc_eval=$(sbatch --parsable --dependency="afterok:${ctc}" scripts/slurm/run_sequential_l2_evaluation.sbatch arabic ctc "${seed}")
supcon_eval=$(sbatch --parsable --dependency="afterok:${supcon}" scripts/slurm/run_sequential_l2_evaluation.sbatch arabic ctc_supcon "${seed}")

printf 'warmup=%s\nctc=%s\nsupcon=%s\nctc_eval=%s\nsupcon_eval=%s\n' \
  "${warmup}" "${ctc}" "${supcon}" "${ctc_eval}" "${supcon_eval}"
