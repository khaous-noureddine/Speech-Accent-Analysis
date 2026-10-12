#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=seq-l2-lam-warm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/seq-l2-lam-warm-%j.out

set -euo pipefail

root="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "${root}"
mkdir -p outputs/slurm

config="experiments/ablations/sequential-l2-supcon-weight/librispeech-100h/wav2vec2-large-lv60/arabic/lambda-0p05/seed=13/config.yaml"
test -f "${config}" || { echo "Missing config: ${config}" >&2; exit 1; }

exec env PYTHONPATH=src pixi run python -m accented_asr.sequential_l2.train \
  --config "${config}" --seed 13 --stage warmup --repository-root .
