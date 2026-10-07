#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=utt-emb-analysis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=outputs/slurm/utt-emb-analysis-%j.out

set -euo pipefail

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
config="experiments/analysis/utterance-supcon-embeddings/librispeech-100h-l2-arctic/config.yaml"

exec env PYTHONPATH=src pixi run python \
  -m accented_asr.representation_analysis.run_utterance \
  --config "${config}" --repository-root .
