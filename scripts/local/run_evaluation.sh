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
  --rerun-incomplete \
  --cores 1

# Example:
# scripts/local/run_evaluation.sh \
#   experiments/eval/wav2vec2-large-lv60/supcon-only/arabic/config.yaml --smoke
