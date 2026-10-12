#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=seq-l2-word-warm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/seq-l2-word-warm-%A_%a.out

set -euo pipefail

accents=(arabic chinese hindi korean spanish vietnamese)
task_id="${SLURM_ARRAY_TASK_ID:?Submit as an accent array.}"
accent="${accents[$task_id]:?Unknown array task: ${task_id}}"
seed="${1:-13}"

root="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "${root}"
mkdir -p outputs/slurm
config="experiments/sequential-l2-word-adaptation/librispeech-100h/wav2vec2-large-lv60/${accent}/seed=${seed}/config.yaml"
test -f "${config}" || { echo "Missing config: ${config}" >&2; exit 1; }

exec env PYTHONPATH=src pixi run python -m accented_asr.sequential_l2.train \
  --config "${config}" --seed "${seed}" --stage warmup --repository-root .
