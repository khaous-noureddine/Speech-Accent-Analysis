#!/usr/bin/env bash

set -euo pipefail

fold="${1:?Usage: $0 FOLD GPU [MODEL ...]}"
gpu="${2:?Usage: $0 FOLD GPU [MODEL ...]}"
shift 2

case "${fold}" in
  arabic|chinese|hindi|korean|spanish|vietnamese) ;;
  *) echo "Unknown fold: ${fold}" >&2; exit 2 ;;
esac

repository_root="$(git rev-parse --show-toplevel)"
cd "${repository_root}"
mkdir -p outputs/representation-analysis

model_arguments=()
if (( $# > 0 )); then
  model_arguments=(--models "$@")
fi

exec env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src \
  pixi run --manifest-path environments/representation-cuda124/pixi.toml \
  python -m accented_asr.representation_analysis.run_saa \
  --config experiments/analysis/saa-accent-geometry/librispeech-100h/config.yaml \
  --repository-root . \
  --fold "${fold}" \
  "${model_arguments[@]}"
