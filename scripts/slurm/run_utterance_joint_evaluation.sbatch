#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=utt-joint-eval
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm/utt-joint-eval-%A_%a.out

set -euo pipefail

task_id="${SLURM_ARRAY_TASK_ID:?Submit this launcher as a Slurm array}"
training_hours="${1:-100h}"
accent="${2:-arabic}"
variant="${3:-full-transformer}"

case "${training_hours}" in
  100h|960h) ;;
  *) echo "Expected LibriSpeech hours: 100h or 960h; got ${training_hours}." >&2; exit 2 ;;
esac
case "${accent}" in
  arabic|chinese|hindi|korean|spanish|vietnamese) ;;
  *) echo "Unknown accent: ${accent}" >&2; exit 2 ;;
esac
case "${variant}" in
  freeze-18|full-transformer) ;;
  *) echo "Expected freeze-18 or full-transformer; got ${variant}." >&2; exit 2 ;;
esac
if [[ "${training_hours}" == "960h" && "${variant}" != "full-transformer" ]]; then
  echo "LibriSpeech 960h currently has only the full-transformer variant." >&2
  exit 2
fi
case "${task_id}" in
  0) decoder="greedy" ;;
  1) decoder="beam_4gram" ;;
  *) echo "Unexpected array task: ${task_id}; expected 0-1." >&2; exit 2 ;;
esac

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
config_path="experiments/joint/librispeech-${training_hours}/wav2vec2-large-lv60/utterance-supcon/${accent}/${variant}/evaluation.yaml"
test -f "${config_path}" || { echo "Missing evaluation config: ${config_path}" >&2; exit 1; }

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

# Greedy and 4-gram evaluation for one accent:
# sbatch scripts/slurm/run_utterance_joint_evaluation.sbatch 960h chinese
