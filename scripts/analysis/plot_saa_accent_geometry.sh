#!/usr/bin/env bash

set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
cd "${repository_root}"

input_root="experiments/analysis/saa-accent-geometry/librispeech-100h/outputs/seed=13"

exec env PYTHONPATH=src pixi run \
  --manifest-path environments/representation-cuda124/pixi.toml \
  python -m accented_asr.representation_analysis.plot_saa \
  --input-root "${input_root}" \
  "$@"
