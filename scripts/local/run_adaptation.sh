#!/usr/bin/env bash
set -euo pipefail

config_path="$1"
seed="$2"
smoke=false
[[ "${3:-}" == "--smoke" ]] && smoke=true

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repository_root}"

exec pixi run snakemake -s Snakefile stage2_adaptation \
  --configfile "${config_path}" \
  --config run_seed="${seed}" run_smoke="${smoke}" \
  --cores 1
