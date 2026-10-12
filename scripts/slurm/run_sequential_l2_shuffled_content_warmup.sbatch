#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=seq-l2-shuf-warm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/seq-l2-shuf-warm-%A_%a.out

set -euo pipefail

accents=(arabic chinese hindi korean spanish vietnamese)
accent="${accents[${SLURM_ARRAY_TASK_ID:?Submit as a six-fold array}]:?Invalid array index}"
root="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "${root}"

config="experiments/controls/sequential-l2-shuffled-content/librispeech-100h/wav2vec2-large-lv60/${accent}/seed=13/config.yaml"
exec env PYTHONPATH=src pixi run python -m accented_asr.sequential_l2.train \
  --config "${config}" --seed 13 --stage warmup --repository-root .
