#!/usr/bin/env bash
set -euo pipefail

training_hours="${1:?Usage: $0 <100h|960h> <50h|500h> [greedy|beam_4gram|all] [gpu_id]}"
accented_hours="${2:?Usage: $0 <100h|960h> <50h|500h> [greedy|beam_4gram|all] [gpu_id]}"
decoder_selector="${3:-all}"
gpu_id="${4:-0}"

case "${training_hours}" in
  100h|960h) ;;
  *) echo "Expected LibriSpeech hours: 100h or 960h; got ${training_hours}." >&2; exit 2 ;;
esac
case "${accented_hours}" in
  50h|500h) ;;
  *) echo "Expected accented-word hours: 50h or 500h; got ${accented_hours}." >&2; exit 2 ;;
esac
case "${decoder_selector}" in
  greedy|beam_4gram) decoders=("${decoder_selector}") ;;
  all) decoders=(greedy beam_4gram) ;;
  *) echo "Expected decoder: greedy, beam_4gram, or all; got ${decoder_selector}." >&2; exit 2 ;;
esac
[[ "${gpu_id}" =~ ^[0-9]+$ ]] || { echo "Expected a numeric GPU ID; got ${gpu_id}." >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${gpu_id}"

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repository_root}"

librispeech_dir="librispeech-${training_hours}"
dataset_dir="mswc-common-voice-${accented_hours}"
experiment_dir="experiments/joint/${librispeech_dir}/wav2vec2-large-lv60/word-supcon/${dataset_dir}/full-transformer"
config_path="${experiment_dir}/evaluation.yaml"
checkpoint_path="${experiment_dir}/outputs/seed=13/checkpoint_best.pt"
lm_dir="language_models/wav2vec2-base-100h-with-lm"

test -f "${config_path}" || { echo "Missing evaluation config: ${config_path}" >&2; exit 1; }
test -s "${checkpoint_path}" || { echo "Missing checkpoint: ${checkpoint_path}" >&2; exit 1; }

for decoder in "${decoders[@]}"; do
  if [[ "${decoder}" == "beam_4gram" ]]; then
    test -s "${lm_dir}/language_model/4-gram.bin" || {
      echo "Missing 4-gram LM. Run scripts/download_language_model.sh first." >&2
      exit 1
    }
  fi

  echo "Evaluating ${librispeech_dir}/${dataset_dir}/full-transformer with ${decoder} on GPU ${gpu_id}"
  pixi run snakemake -s Snakefile evaluate_all \
    --configfile "${config_path}" \
    --config \
      run_smoke=false \
      evaluation_decoder="${decoder}" \
      evaluation_lm_dir="${lm_dir}" \
      evaluation_beam_width=100 \
    --rerun-incomplete --nolock --cores 1
done

# Examples:
# scripts/local/run_word_joint_evaluation.sh 100h 500h all 0
# scripts/local/run_word_joint_evaluation.sh 960h 50h greedy 1
