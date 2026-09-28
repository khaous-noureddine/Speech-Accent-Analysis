#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_root}"

mode="move"
case "${1:-}" in
  "") ;;
  --live-copy) mode="live-copy" ;;
  --finalize-copy) mode="finalize-copy" ;;
  --verify) mode="verify" ;;
  *)
    echo "Usage: $0 [--live-copy|--finalize-copy|--verify]" >&2
    exit 2
    ;;
esac

verified_checkpoints=0
missing_checkpoints=0
mismatched_checkpoints=0

verify_checkpoints() {
  local source="$1"
  local destination="$2"
  local checkpoint relative target
  [[ -d "${source}" ]] || return 0

  while IFS= read -r -d '' checkpoint; do
    relative="${checkpoint#"${source}/"}"
    target="${destination}/${relative}"
    if [[ ! -f "${target}" ]]; then
      echo "MISSING: ${target}" >&2
      missing_checkpoints=$((missing_checkpoints + 1))
    elif ! cmp -s "${checkpoint}" "${target}"; then
      echo "MISMATCH: ${checkpoint} != ${target}" >&2
      mismatched_checkpoints=$((mismatched_checkpoints + 1))
    else
      verified_checkpoints=$((verified_checkpoints + 1))
    fi
  done < <(find "${source}" -type f \( -name 'checkpoint_best.pt' -o -name 'checkpoint_final.pt' \) -print0)
}

move_outputs() {
  local source="$1"
  local destination="$2"
  [[ -d "${source}" ]] || return 0

  if [[ "${mode}" == "verify" ]]; then
    verify_checkpoints "${source}" "${destination}"
    return 0
  fi

  if [[ "${mode}" == "live-copy" ]]; then
    mkdir -p "${destination}"
    # Keep the live source in place. Existing regular files are hard-linked so
    # subsequent in-place writes remain visible at both paths; the final pass
    # refreshes files that were atomically replaced and copies newly created
    # artifacts.
    cp -al --no-clobber "${source}/." "${destination}/"
    touch "${destination}/.LIVE_MIGRATION_INCOMPLETE"
    echo "Live-copied ${source} -> ${destination} (source retained)"
    return 0
  fi

  if [[ "${mode}" == "finalize-copy" ]]; then
    mkdir -p "${destination}"
    cp -a "${source}/." "${destination}/"
    rm -f "${destination}/.LIVE_MIGRATION_INCOMPLETE"
    echo "Finalized copy ${source} -> ${destination} (source retained)"
    return 0
  fi

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
  if [[ "${mode}" == "verify" ]]; then
    verify_checkpoints "${source}" "${destination}"
    return 0
  fi
  mkdir -p "${destination}"
  if [[ "${mode}" == "live-copy" ]]; then
    cp -al --no-clobber "${source}/." "${destination}/"
    touch "${destination}/.LIVE_MIGRATION_INCOMPLETE"
  elif [[ "${mode}" == "finalize-copy" ]]; then
    cp -a "${source}/." "${destination}/"
    rm -f "${destination}/.LIVE_MIGRATION_INCOMPLETE"
  else
    cp -a --no-clobber "${source}/." "${destination}/"
  fi
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

if [[ "${mode}" == "verify" ]]; then
  echo "Verified checkpoints: ${verified_checkpoints}"
  echo "Missing checkpoints: ${missing_checkpoints}"
  echo "Mismatched checkpoints: ${mismatched_checkpoints}"
  if (( missing_checkpoints > 0 || mismatched_checkpoints > 0 )); then
    exit 1
  fi
  echo "Checkpoint migration verification passed."
elif [[ "${mode}" == "live-copy" ]]; then
  echo "Live copy complete. Do not delete legacy outputs or consume copied checkpoints yet."
  echo "After all jobs finish, run: scripts/migrate_experiment_layout.sh --finalize-copy"
elif [[ "${mode}" == "finalize-copy" ]]; then
  echo "Final copy complete. Legacy outputs were retained for verification."
else
  echo "Experiment-output migration complete. Empty legacy directories may now be removed."
fi
