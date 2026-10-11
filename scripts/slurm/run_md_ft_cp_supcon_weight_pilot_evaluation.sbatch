#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=mdcp-eval
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=outputs/slurm/mdcp-eval-%A_%a.out

set -euo pipefail

task_id="${SLURM_ARRAY_TASK_ID:?This launcher uses tasks 0 and 1}"
accent="${1:?Usage: sbatch --array=0-1 $0 <arabic|chinese>}"
case "${accent}" in
  arabic|chinese) ;;
  *) echo "Expected accent Arabic or Chinese; got ${accent}." >&2; exit 2 ;;
esac
case "${task_id}" in
  0) decoder="greedy" ;;
  1) decoder="beam_4gram" ;;
  *) echo "Expected array task 0 or 1." >&2; exit 2 ;;
esac

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
mkdir -p outputs/slurm

experiment_dir="experiments/ablations/md-ft-cp-supcon-weight/librispeech-100h/wav2vec2-large-lv60/utterance/${accent}/lambda-0p03/full-transformer"
config="${experiment_dir}/evaluation.yaml"
checkpoint="${experiment_dir}/outputs/seed=13/checkpoint_best.pt"
lm_dir="language_models/wav2vec2-base-100h-with-lm"
test -f "${config}" || { echo "Missing config: ${config}" >&2; exit 1; }
test -s "${checkpoint}" || { echo "Missing checkpoint: ${checkpoint}" >&2; exit 1; }
if [[ "${decoder}" == "beam_4gram" ]]; then
  test -s "${lm_dir}/language_model/4-gram.bin" || {
    echo "Missing 4-gram language model." >&2
    exit 1
  }
fi

echo "Evaluating ${accent} lambda=0.03 with ${decoder}"
exec pixi run snakemake -s Snakefile evaluate_all \
  --configfile "${config}" \
  --config run_smoke=false evaluation_decoder="${decoder}" \
    evaluation_lm_dir="${lm_dir}" evaluation_beam_width=100 \
  --rerun-incomplete \
  --consider-ancient prepare_evaluation_dataset=preparer \
  --nolock --cores 1
