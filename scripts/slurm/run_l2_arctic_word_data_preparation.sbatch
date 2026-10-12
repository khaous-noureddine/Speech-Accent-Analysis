#!/usr/bin/env bash
#SBATCH --partition=SMP-256c
#SBATCH --account=efl
#SBATCH --job-name=l2-word-data
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/l2-word-data-%j.out

set -euo pipefail

root="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "${root}"
mkdir -p outputs/slurm

exec pixi run snakemake \
  --snakefile Snakefile \
  prepare_word_contrastive_dataset \
  --configfile configs/data/word_contrastive_l2_arctic.yaml \
  --cores "${SLURM_CPUS_PER_TASK:-8}" \
  --rerun-incomplete \
  --nolock \
  --printshellcmds
