#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_root}"

move_outputs() {
  local source="$1"
  local destination="$2"
  [[ -d "${source}" ]] || return 0
  if [[ -e "${destination}" ]]; then
    echo "Refusing to overwrite existing destination: ${destination}" >&2
    echo "Merge or remove it manually, then rerun this script." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${destination}")"
  mv "${source}" "${destination}"
  echo "Moved ${source} -> ${destination}"
}

copy_outputs_without_overwrite() {
  local source="$1"
  local destination="$2"
  [[ -d "${source}" ]] || return 0
  mkdir -p "${destination}"
  cp -a --no-clobber "${source}/." "${destination}/"
  echo "Merged without overwrite ${source} -> ${destination}"
  echo "Legacy source retained for manual verification: ${source}"
}

model="wav2vec2-large-lv60"
accents=(arabic chinese hindi korean spanish vietnamese)
objectives=(ctc-only supcon-ctc supcon-only)

for objective in "${objectives[@]}"; do
  for accent in "${accents[@]}"; do
    root="experiments/separated/librispeech-100h/${model}/utterance-supcon/${objective}/${accent}"
    move_outputs \
      "experiments/stage2/${model}/${objective}/${accent}/outputs" \
      "${root}/stage2/freeze-18/outputs"
    move_outputs \
      "experiments/stage3/${model}/${objective}/${accent}/outputs" \
      "${root}/stage3/full-transformer/outputs"
    copy_outputs_without_overwrite \
      "experiments/eval/${model}/${objective}/${accent}/outputs" \
      "${root}/stage3/full-transformer/outputs"
  done
done

move_outputs \
  "experiments/stage3/${model}/no-stage2/outputs" \
  "experiments/separated/librispeech-100h/${model}/utterance-supcon/no-stage2/global/stage3/full-transformer/outputs"
copy_outputs_without_overwrite \
  "experiments/eval/${model}/no-stage2/outputs" \
  "experiments/separated/librispeech-100h/${model}/utterance-supcon/no-stage2/global/stage3/full-transformer/outputs"

for accent in "${accents[@]}"; do
  for variant in freeze-18 full-transformer; do
    move_outputs \
      "experiments/joint-training/${model}/utterance-supcon/${accent}/${variant}/outputs" \
      "experiments/joint/librispeech-100h/${model}/utterance-supcon/${accent}/${variant}/outputs"
  done
  move_outputs \
    "experiments/scaling/librispeech-960/${model}/${accent}/joint-supcon/outputs" \
    "experiments/joint/librispeech-960h/${model}/utterance-supcon/${accent}/full-transformer/outputs"
done

move_outputs \
  "experiments/scaling/librispeech-960/${model}/arabic/ctc-only/outputs" \
  "experiments/joint/librispeech-960h/${model}/utterance-supcon/arabic/ctc-only/full-transformer/outputs"

move_outputs \
  "experiments/joint-training/${model}/word-supcon/mswc-common-voice-50h/freeze-18/outputs" \
  "experiments/joint/librispeech-100h/${model}/word-supcon/mswc-common-voice-50h/freeze-18/outputs"
move_outputs \
  "experiments/joint-training/${model}/word-supcon/mswc-common-voice-50h/full-transformer/outputs" \
  "experiments/joint/librispeech-100h/${model}/word-supcon/mswc-common-voice-50h/full-transformer/outputs"
move_outputs \
  "experiments/joint-training/${model}/word-supcon/mswc-common-voice-500h/full-transformer/outputs" \
  "experiments/joint/librispeech-100h/${model}/word-supcon/mswc-common-voice-500h/full-transformer/outputs"
move_outputs \
  "experiments/joint-training/${model}/word-supcon/mswc-common-voice-50h-librispeech-960/freeze-18/outputs" \
  "experiments/joint/librispeech-960h/${model}/word-supcon/mswc-common-voice-50h/freeze-18/outputs"
move_outputs \
  "experiments/joint-training/${model}/word-supcon/mswc-common-voice-50h-librispeech-960/full-transformer/outputs" \
  "experiments/joint/librispeech-960h/${model}/word-supcon/mswc-common-voice-50h/full-transformer/outputs"

move_outputs \
  "experiments/external-baselines/facebook-wav2vec2-large-960h-lv60/evaluation/outputs" \
  "experiments/external-baselines/librispeech-960h/facebook-wav2vec2-large-960h-lv60/outputs"
move_outputs \
  "experiments/external-baselines/patrickvonplaten-wav2vec2-large-lv60h-100h/evaluation/outputs" \
  "experiments/external-baselines/librispeech-100h/patrickvonplaten-wav2vec2-large-lv60h-100h/outputs"

echo "Experiment-output migration complete. Empty legacy directories may now be removed."
