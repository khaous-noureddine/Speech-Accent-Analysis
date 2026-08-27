#!/usr/bin/env bash
#SBATCH --partition=GPU-A100
#SBATCH --account=efl
#SBATCH --job-name=eval-greedy
#SBATCH --array=0-5%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=eval-%x-%A_%a.out

set -euo pipefail

accents=(arabic chinese hindi korean spanish vietnamese)
objective="${1:-supcon-only}"
seed="${2:-13}"
smoke=false
[[ "${3:-}" == "--smoke" ]] && smoke=true

if [[ "${objective}" != "supcon-only" ]]; then
  echo "Only supcon-only evaluation configs are currently available." >&2
  exit 2
fi
if [[ "${seed}" != "13" ]]; then
  echo "Only seed 13 evaluation configs are currently available." >&2
  exit 2
fi

accent="${accents[SLURM_ARRAY_TASK_ID]}"
config_path="experiments/eval/wav2vec2-large-lv60/${objective}/${accent}/config.yaml"

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"

exec pixi run snakemake -s Snakefile evaluate_all \
  --configfile "${config_path}" \
  --config run_smoke="${smoke}" \
  --nolock \
  --rerun-incomplete \
  --cores 1

# All six folds, sequentially on at most one GPU:
# sbatch scripts/slurm/run_evaluation.sbatch supcon-only 13
#
# Spanish only (array index 4):
# sbatch --array=4 scripts/slurm/run_evaluation.sbatch supcon-only 13
#
# Smoke all six folds:
# sbatch scripts/slurm/run_evaluation.sbatch supcon-only 13 --smoke
