#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=shuffle-supcon
#SBATCH --array=0-5%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/slurm/shuffle-supcon-%A_%a.out

set -euo pipefail

task_id="${SLURM_ARRAY_TASK_ID:?Submit this launcher as an array from 0 to 5}"
accents=(arabic chinese hindi korean spanish vietnamese)
accent="${accents[task_id]:?Invalid array index}"
seed="${1:-13}"
[[ "${seed}" == "13" ]] || { echo "Only seed 13 is configured." >&2; exit 2; }

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
mkdir -p outputs/slurm
config="experiments/joint/librispeech-100h/wav2vec2-large-lv60/controls/shuffled-label-supcon/${accent}/full-transformer/config.yaml"

echo "Training shuffled-label CP-SupCon (${accent}, seed=${seed})"
exec pixi run snakemake -s Snakefile joint_ctc_supcon_training \
  --configfile "${config}" \
  --config run_seed="${seed}" run_smoke=false \
  --rerun-incomplete --nolock --cores 1
