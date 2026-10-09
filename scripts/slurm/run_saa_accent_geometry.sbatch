#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=saa-geom
#SBATCH --array=0-5%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/saa-geometry-%A_%a.out

set -euo pipefail

task_id="${SLURM_ARRAY_TASK_ID:?Submit this launcher as an array from 0 to 5}"
folds=(arabic chinese hindi korean spanish vietnamese)
fold="${folds[task_id]:?Invalid array index}"
repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
mkdir -p outputs/slurm

exec env PYTHONPATH=src pixi run \
  --manifest-path environments/representation-cuda124/pixi.toml \
  python -m accented_asr.representation_analysis.run_saa \
  --config experiments/analysis/saa-accent-geometry/librispeech-100h/config.yaml \
  --repository-root . \
  --fold "${fold}"
