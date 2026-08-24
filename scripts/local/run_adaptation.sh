#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 CONDITION FOLD SEED [--smoke]" >&2
  exit 2
fi

condition="$1"
fold="$2"
seed="$3"
shift 3

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

if command -v python >/dev/null 2>&1; then
  python_command=(python)
elif command -v pixi >/dev/null 2>&1; then
  python_command=(pixi run python)
else
  echo "No Python environment found. Activate one or install Pixi." >&2
  exit 1
fi

"${python_command[@]}" -m accented_asr.adaptation.train \
  --config "${repository_root}/configs/adaptation/wav2vec2_base.yaml" \
  --repository-root "${repository_root}" \
  --condition "${condition}" \
  --fold "${fold}" \
  --seed "${seed}" \
  "$@"
