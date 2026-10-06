#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=prior-eval
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=outputs/slurm/prior-eval-%A_%a.out

set -euo pipefail

task_id="${SLURM_ARRAY_TASK_ID:?This launcher uses an array for greedy and beam decoding}"
method="${1:?Usage: sbatch $0 METHOD ACCENT}"
accent="${2:?Usage: sbatch $0 METHOD ACCENT}"
case "${method}" in
  accent-dat|accent-mtl|multidomain-ctc|augmented-view-supcon) ;;
  *) echo "Unknown method: ${method}" >&2; exit 2 ;;
esac
case "${accent}" in
  arabic|chinese|hindi|korean|spanish|vietnamese) ;;
  *) echo "Unknown accent: ${accent}" >&2; exit 2 ;;
esac
case "${task_id}" in
  0) decoder="greedy" ;;
  1) decoder="beam_4gram" ;;
  *) echo "Expected array task 0 or 1." >&2; exit 2 ;;
esac

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
mkdir -p outputs/slurm
config="experiments/baselines/prior-methods/librispeech-100h/wav2vec2-large-lv60/${method}/${accent}/full-transformer/evaluation.yaml"
lm_dir="language_models/wav2vec2-base-100h-with-lm"
if [[ "${decoder}" == "beam_4gram" ]]; then
  test -s "${lm_dir}/language_model/4-gram.bin" || {
    echo "Missing 4-gram LM. Run scripts/download_language_model.sh first." >&2
    exit 1
  }
fi

exec pixi run snakemake -s Snakefile evaluate_all \
  --configfile "${config}" \
  --config run_smoke=false evaluation_decoder="${decoder}" \
    evaluation_lm_dir="${lm_dir}" evaluation_beam_width=100 \
  --rerun-incomplete --nolock --cores 1
