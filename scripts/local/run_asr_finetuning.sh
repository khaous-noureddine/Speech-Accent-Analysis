#!/usr/bin/env bash
set -euo pipefail

objective="${1:?Usage: $0 <supcon-only|supcon-ctc|ctc-only> <accent> <seed> [--smoke]}"
accent="${2:?Usage: $0 <objective> <accent> <seed> [--smoke]}"
seed="${3:?Usage: $0 <objective> <accent> <seed> [--smoke]}"
smoke=false
[[ "${4:-}" == "--smoke" ]] && smoke=true

case "${objective}" in
  supcon-only|supcon-ctc|ctc-only) ;;
  *)
    echo "Unknown objective: ${objective}" >&2
    echo "Expected supcon-only, supcon-ctc, or ctc-only." >&2
    exit 2
    ;;
esac

case "${accent}" in
  arabic|chinese|hindi|korean|spanish|vietnamese) ;;
  *)
    echo "Unknown accent: ${accent}" >&2
    echo "Expected arabic, chinese, hindi, korean, spanish, or vietnamese." >&2
    exit 2
    ;;
esac

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repository_root}"

config_path="experiments/stage3/wav2vec2-large-lv60/${objective}/${accent}/config.yaml"

exec pixi run snakemake -s Snakefile stage3_asr_finetuning \
  --configfile "${config_path}" \
  --config run_seed="${seed}" run_smoke="${smoke}" \
  --rerun-incomplete \
  --cores 1

# Examples:
# scripts/local/run_asr_finetuning.sh supcon-only spanish 13
# scripts/local/run_asr_finetuning.sh supcon-only spanish 13 --smoke
