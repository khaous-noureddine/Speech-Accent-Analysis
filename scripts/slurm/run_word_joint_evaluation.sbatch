#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=word-joint-eval
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm/word-joint-eval-%A_%a.out

set -euo pipefail

task_id="${SLURM_ARRAY_TASK_ID:?Submit this launcher as a Slurm array}"
training_hours="${1:-100h}"
accented_hours="${2:-50h}"
case "${accented_hours}:${training_hours}" in
  50h:100h) librispeech_dir="librispeech-100h"; dataset_dir="mswc-common-voice-50h" ;;
  50h:960h) librispeech_dir="librispeech-960h"; dataset_dir="mswc-common-voice-50h" ;;
  500h:100h) librispeech_dir="librispeech-100h"; dataset_dir="mswc-common-voice-500h" ;;
  500h:960h) librispeech_dir="librispeech-960h"; dataset_dir="mswc-common-voice-500h" ;;
  *) echo "Expected LibriSpeech hours (100h|960h) then accented-word hours (50h|500h)." >&2; exit 2 ;;
esac
case "${task_id}" in
  0) decoder="greedy" ;;
  1) decoder="beam_4gram" ;;
  *) echo "Unexpected array task: ${task_id}; expected 0-1." >&2; exit 2 ;;
esac

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"

config_path="experiments/joint/${librispeech_dir}/wav2vec2-large-lv60/word-supcon/${dataset_dir}/full-transformer/evaluation.yaml"
lm_dir="language_models/wav2vec2-base-100h-with-lm"
if [[ "${decoder}" == "beam_4gram" ]]; then
  test -s "${lm_dir}/language_model/4-gram.bin" || {
    echo "Missing 4-gram LM. Run scripts/download_language_model.sh first." >&2
    exit 1
  }
fi

echo "Evaluating ${librispeech_dir}/${dataset_dir}/full-transformer with ${decoder}"
exec pixi run snakemake -s Snakefile evaluate_all \
  --configfile "${config_path}" \
  --config \
    run_smoke=false \
    evaluation_decoder="${decoder}" \
    evaluation_lm_dir="${lm_dir}" \
    evaluation_beam_width=100 \
  --rerun-incomplete --nolock --cores 1

# Evaluate one training/data combination with greedy and 4-gram decoding:
# sbatch scripts/slurm/run_word_joint_evaluation.sbatch 100h 50h
# sbatch scripts/slurm/run_word_joint_evaluation.sbatch 960h 50h
# sbatch scripts/slurm/run_word_joint_evaluation.sbatch 100h 500h
# sbatch scripts/slurm/run_word_joint_evaluation.sbatch 960h 500h
