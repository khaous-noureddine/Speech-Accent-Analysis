#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=seq-l2-lambda
#SBATCH --array=0-3%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm/seq-l2-lambda-%A_%a.out

set -euo pipefail

slugs=(lambda-0p01 lambda-0p03 lambda-0p10 lambda-0p20)
slug="${slugs[${SLURM_ARRAY_TASK_ID:?Submit as a four-setting array}]:?Invalid array index}"

root="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "${root}"
mkdir -p outputs/slurm

base="experiments/ablations/sequential-l2-supcon-weight/librispeech-100h/wav2vec2-large-lv60/arabic"
reference_warmup="experiments/sequential-l2-adaptation/librispeech-100h/wav2vec2-large-lv60/arabic/outputs/seed=13/warmup/checkpoint.pt"
warmup="${base}/${slug}/outputs/seed=13/warmup/checkpoint.pt"
config="${base}/${slug}/seed=13/config.yaml"

test -f "${config}" || { echo "Missing config: ${config}" >&2; exit 1; }
test -s "${reference_warmup}" || { echo "Missing shared warm-up: ${reference_warmup}" >&2; exit 1; }
mkdir -p "$(dirname "${warmup}")"
[[ "${warmup}" == "${reference_warmup}" ]] || ln -f "${reference_warmup}" "${warmup}"

exec env PYTHONPATH=src pixi run python -m accented_asr.sequential_l2.train \
  --config "${config}" --seed 13 --stage adapt --objective ctc_supcon --repository-root .
