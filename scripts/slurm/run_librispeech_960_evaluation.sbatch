#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=ls960-eval
#SBATCH --array=0-2%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm/ls960-%x-%A_%a.out

set -euo pipefail

task_id="${SLURM_ARRAY_TASK_ID:?Submit this launcher as a Slurm array}"
accent="${1:-arabic}"
case "${accent}" in
  arabic|chinese|hindi|korean|spanish|vietnamese) ;;
  *) echo "Unknown accent: ${accent}" >&2; exit 2 ;;
esac
case "${task_id}" in
  0) objective="joint-supcon"; decoder="greedy" ;;
  1) objective="joint-supcon"; decoder="beam_4gram" ;;
  2) objective="ctc-only"; decoder="beam_4gram" ;;
  *) echo "Unexpected array task: ${task_id}" >&2; exit 2 ;;
esac

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
if [[ "${objective}" == "ctc-only" ]]; then
  [[ "${accent}" == "arabic" ]] || {
    echo "CTC-only is fold-independent and stored under arabic." >&2
    exit 2
  }
  config_path="experiments/joint/librispeech-960h/wav2vec2-large-lv60/utterance-supcon/arabic/ctc-only/full-transformer/evaluation.yaml"
else
  config_path="experiments/joint/librispeech-960h/wav2vec2-large-lv60/utterance-supcon/${accent}/full-transformer/evaluation.yaml"
fi
test -f "${config_path}" || {
  echo "Missing evaluation config: ${config_path}" >&2
  exit 1
}
lm_dir="language_models/wav2vec2-base-100h-with-lm"
if [[ "${decoder}" == "beam_4gram" ]]; then
  test -s "${lm_dir}/language_model/4-gram.bin" || {
    echo "Missing 4-gram LM. Run scripts/download_language_model.sh first." >&2
    exit 1
  }
fi

exec pixi run snakemake -s Snakefile evaluate_all \
  --configfile "${config_path}" \
  --config \
    run_smoke=false \
    evaluation_decoder="${decoder}" \
    evaluation_lm_dir="${lm_dir}" \
    evaluation_beam_width=100 \
  --rerun-incomplete --nolock --cores 1

# Launch all missing LibriSpeech-960 evaluations, with at most two H200 GPUs:
# sbatch scripts/slurm/run_librispeech_960_evaluation.sbatch
# Evaluate one joint fold with greedy and 4-gram decoding:
# sbatch --array=0-1%2 scripts/slurm/run_librispeech_960_evaluation.sbatch chinese
