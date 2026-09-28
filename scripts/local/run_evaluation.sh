#!/usr/bin/env bash
set -euo pipefail

config_path="${1:?Usage: $0 <evaluation-config.yaml> [--smoke]}"
smoke=false
[[ "${2:-}" == "--smoke" ]] && smoke=true

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repository_root}"

exec pixi run snakemake -s Snakefile evaluate_all \
  --configfile "${config_path}" \
  --config run_smoke="${smoke}" \
  --nolock \
  --rerun-incomplete \
  --cores 1

# Example:
# scripts/local/run_evaluation.sh \
#   experiments/separated/librispeech-100h/wav2vec2-large-lv60/utterance-supcon/supcon-only/arabic/stage3/full-transformer/config.yaml --smoke
