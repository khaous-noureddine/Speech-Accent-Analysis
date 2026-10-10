#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
CONFIG="experiments/analysis/utterance-supcon-layer-trajectories/librispeech-100h-l2-arctic/config-all-models.yaml"
OUT="experiments/analysis/utterance-supcon-layer-trajectories/librispeech-100h-l2-arctic/outputs-all-models/seed=13"
ENVIRONMENT="environments/representation-cuda124/pixi.toml"

cd "$ROOT"

PYTHONPATH=src pixi run --manifest-path "$ENVIRONMENT" \
  python -m accented_asr.representation_analysis.run_utterance \
  --config "$CONFIG" \
  --repository-root .

PYTHONPATH=src pixi run --manifest-path "$ENVIRONMENT" \
  python -m accented_asr.representation_analysis.recenter_utterance \
  --input-root "$OUT" \
  --output-dir "$OUT/centered-analysis" \
  --seed 13

PYTHONPATH=src pixi run --manifest-path "$ENVIRONMENT" \
  python -m accented_asr.representation_analysis.plot_layer_trajectories \
  --metrics-by-fold "$OUT/centered-analysis/metrics_by_fold_raw_and_centered.csv" \
  --output-dir "$OUT/figures"

printf 'All-model layer trajectories written to %s/figures\n' "$OUT"
