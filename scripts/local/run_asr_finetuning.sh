#!/usr/bin/env bash
set -euo pipefail

objective="${1:?Usage: $0 <objective> [accent] <seed> [--smoke]}"
if [[ "${objective}" == "no-stage2" ]]; then
  accent=""
  seed="${2:?Usage: $0 no-stage2 <seed> [--smoke]}"
  smoke_argument="${3:-}"
else
  accent="${2:?Usage: $0 <objective> <accent> <seed> [--smoke]}"
  seed="${3:?Usage: $0 <objective> <accent> <seed> [--smoke]}"
  smoke_argument="${4:-}"
fi
smoke=false
[[ "${smoke_argument}" == "--smoke" ]] && smoke=true

case "${objective}" in
  supcon-only|supcon-ctc|ctc-only|no-stage2) ;;
  *)
    echo "Unknown objective: ${objective}" >&2
    echo "Expected supcon-only, supcon-ctc, ctc-only, or no-stage2." >&2
    exit 2
    ;;
esac

if [[ "${objective}" != "no-stage2" ]]; then
  case "${accent}" in
    arabic|chinese|hindi|korean|spanish|vietnamese) ;;
    *)
      echo "Unknown accent: ${accent}" >&2
      echo "Expected arabic, chinese, hindi, korean, spanish, or vietnamese." >&2
      exit 2
      ;;
  esac
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repository_root}"

if [[ "${objective}" == "no-stage2" ]]; then
  config_path="experiments/separated/librispeech-100h/wav2vec2-large-lv60/utterance-supcon/no-stage2/global/stage3/full-transformer/config.yaml"
else
  config_path="experiments/separated/librispeech-100h/wav2vec2-large-lv60/utterance-supcon/${objective}/${accent}/stage3/full-transformer/config.yaml"
fi

exec pixi run snakemake -s Snakefile stage3_asr_finetuning \
  --configfile "${config_path}" \
  --config run_seed="${seed}" run_smoke="${smoke}" \
  --rerun-incomplete \
  --cores 1

# Examples:
# scripts/local/run_asr_finetuning.sh supcon-only spanish 13
# scripts/local/run_asr_finetuning.sh supcon-only spanish 13 --smoke
# scripts/local/run_asr_finetuning.sh no-stage2 13
