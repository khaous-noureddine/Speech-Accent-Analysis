#!/usr/bin/env bash
#SBATCH --partition=SMP-256c
#SBATCH --account=efl
#SBATCH --job-name=word-data-scale
#SBATCH --array=0-2%1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=outputs/slurm/word-data-scale-%A_%a.out

set -euo pipefail

task_id="${SLURM_ARRAY_TASK_ID:?Submit this launcher as an array}"
sizes=(25h 100h 250h)
hours="${sizes[task_id]:?Expected array task 0-2}"

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
mkdir -p outputs/slurm

config="configs/data/word_contrastive_mswc_en_${hours}.yaml"
test -f "${config}" || { echo "Missing dataset config: ${config}" >&2; exit 1; }

exec pixi run snakemake -s Snakefile prepare_mswc_word_heldout_subset \
  --configfile "${config}" --rerun-incomplete --nolock --cores 1

# sbatch --array=0-2%1 scripts/slurm/run_word_data_subset_preparation.sbatch
