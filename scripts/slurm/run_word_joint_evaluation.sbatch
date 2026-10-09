#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=word-joint-eval
#SBATCH --array=0-3%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=outputs/slurm/word-joint-eval-%A_%a.out

set -euo pipefail

task_id="${SLURM_ARRAY_TASK_ID:?Submit this launcher as a Slurm array}"
training_selector="${1:-all}"
accented_hours="${2:-50h}"

case "${training_selector}" in
  100h|960h)
    training_hours="${training_selector}"
    case "${task_id}" in
      0) decoder="greedy" ;;
      1) decoder="beam_4gram" ;;
      *) echo "For one LibriSpeech size, submit --array=0-1; got task ${task_id}." >&2; exit 2 ;;
    esac
    ;;
  all)
    case "${task_id}" in
      0) training_hours="100h"; decoder="greedy" ;;
      1) training_hours="100h"; decoder="beam_4gram" ;;
      2) training_hours="960h"; decoder="greedy" ;;
      3) training_hours="960h"; decoder="beam_4gram" ;;
      *) echo "For both LibriSpeech sizes, submit --array=0-3; got task ${task_id}." >&2; exit 2 ;;
    esac
    ;;
  *) echo "Expected LibriSpeech hours: 100h, 960h, or all; got ${training_selector}." >&2; exit 2 ;;
esac
case "${accented_hours}" in
  25h|50h|100h|250h|500h) ;;
  *) echo "Expected accented-word hours: 25h, 50h, 100h, 250h, or 500h; got ${accented_hours}." >&2; exit 2 ;;
esac
if [[ "${training_hours}" == "960h" && ! "${accented_hours}" =~ ^(50h|500h)$ ]]; then
  echo "Only 50h and 500h word datasets are configured with LibriSpeech-960h." >&2
  exit 2
fi

librispeech_dir="librispeech-${training_hours}"
dataset_dir="mswc-common-voice-${accented_hours}"

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

# Evaluate 100h and 960h with 500h accented data, Greedy and LM:
# sbatch --array=0-3%2 scripts/slurm/run_word_joint_evaluation.sbatch all 500h
# Evaluate one configuration only:
# sbatch --array=0-1%2 scripts/slurm/run_word_joint_evaluation.sbatch 100h 500h
