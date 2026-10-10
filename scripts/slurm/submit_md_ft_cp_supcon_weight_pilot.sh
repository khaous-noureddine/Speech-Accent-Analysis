#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repository_root}"
mkdir -p outputs/slurm

arabic_train=$(sbatch --parsable \
  scripts/slurm/run_md_ft_cp_supcon_weight_pilot.sbatch arabic 13)
arabic_eval=$(sbatch --parsable \
  --dependency="afterok:${arabic_train}" \
  --array=0-1%2 \
  scripts/slurm/run_md_ft_cp_supcon_weight_pilot_evaluation.sbatch arabic)
chinese_train=$(sbatch --parsable \
  --dependency="afterok:${arabic_eval}" \
  scripts/slurm/run_md_ft_cp_supcon_weight_pilot.sbatch chinese 13)
chinese_eval=$(sbatch --parsable \
  --dependency="afterok:${chinese_train}" \
  --array=0-1%2 \
  scripts/slurm/run_md_ft_cp_supcon_weight_pilot_evaluation.sbatch chinese)

printf 'Arabic train: %s\n' "${arabic_train}"
printf 'Arabic eval:  %s\n' "${arabic_eval}"
printf 'Chinese train: %s\n' "${chinese_train}"
printf 'Chinese eval:  %s\n' "${chinese_eval}"
