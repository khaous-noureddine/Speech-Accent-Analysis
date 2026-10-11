#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=seq-l2-adapt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm/seq-l2-adapt-%j.out

set -euo pipefail

accent="${1:?Usage: sbatch $0 ACCENT OBJECTIVE [SEED]}"
objective="${2:?Objective must be ctc or ctc_supcon}"
seed="${3:-13}"
case "${objective}" in ctc|ctc_supcon) ;; *) exit 2 ;; esac

root="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "${root}"
mkdir -p outputs/slurm
config="experiments/sequential-l2-adaptation/librispeech-100h/wav2vec2-large-lv60/${accent}/seed=${seed}/config.yaml"

exec env PYTHONPATH=src pixi run python -m accented_asr.sequential_l2.train \
  --config "${config}" \
  --seed "${seed}" \
  --stage adapt \
  --objective "${objective}" \
  --repository-root .
