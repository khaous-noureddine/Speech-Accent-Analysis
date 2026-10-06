#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=prior-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/slurm/prior-train-%A_%a.out

set -euo pipefail

method="${1:?Usage: sbatch [--array=0-5] $0 METHOD [ACCENT] [SEED]}"
seed="${3:-13}"
accents=(arabic chinese hindi korean spanish vietnamese)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  accent="${accents[SLURM_ARRAY_TASK_ID]:?Invalid Slurm array index}"
else
  accent="${2:-arabic}"
fi

case "${method}" in
  accent-dat|accent-mtl|multidomain-ctc|augmented-view-supcon) ;;
  *) echo "Unknown method: ${method}" >&2; exit 2 ;;
esac
[[ " ${accents[*]} " == *" ${accent} "* ]] || { echo "Unknown accent: ${accent}" >&2; exit 2; }
[[ "${seed}" == "13" ]] || { echo "Only seed 13 is configured." >&2; exit 2; }

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
mkdir -p outputs/slurm
config="experiments/baselines/prior-methods/librispeech-100h/wav2vec2-large-lv60/${method}/${accent}/full-transformer/config.yaml"
test -f "${config}" || { echo "Missing config: ${config}" >&2; exit 1; }

exec pixi run snakemake -s Snakefile joint_ctc_supcon_training \
  --configfile "${config}" \
  --config run_seed="${seed}" run_smoke=false \
  --rerun-incomplete --nolock --cores 1

