#!/usr/bin/env bash
#SBATCH --partition=GPU-A100
#SBATCH --account=efl
#SBATCH --job-name=stage2-supcon
#SBATCH --array=0-5%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=stage2-%x-%A_%a.out

set -euo pipefail

accents=(arabic chinese hindi korean spanish vietnamese)
accent="${accents[SLURM_ARRAY_TASK_ID]}"
seed="${1:-13}"
smoke=false
[[ "${2:-}" == "--smoke" ]] && smoke=true

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"

config_path="experiments/stage2/wav2vec2-large-lv60/supcon-only/${accent}/config.yaml"

exec pixi run snakemake -s Snakefile stage2_adaptation \
  --configfile "${config_path}" \
  --config run_seed="${seed}" run_smoke="${smoke}" \
  --cores 1
