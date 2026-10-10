#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
CONFIG="experiments/analysis/saa-accent-geometry/librispeech-100h/config.yaml"
OUT="experiments/analysis/saa-layer-trajectories/librispeech-100h/outputs/seed=13"
ENVIRONMENT="environments/representation-cuda124/pixi.toml"
LAYERS=($(seq 1 24))

cd "$ROOT"

for fold in arabic chinese hindi korean spanish vietnamese; do
  models=(asr_only multidomain_ctc utterance_supcon)
  if [[ "$fold" == "arabic" || "$fold" == "chinese" ]]; then
    models+=(md_ft_cp_supcon)
  fi
  PYTHONPATH=src pixi run --manifest-path "$ENVIRONMENT" \
    python -m accented_asr.representation_analysis.run_saa \
    --config "$CONFIG" \
    --repository-root . \
    --fold "$fold" \
    --models "${models[@]}" \
    --layers "${LAYERS[@]}" \
    --output-dir "$OUT"
done

PYTHONPATH=src pixi run --manifest-path "$ENVIRONMENT" \
  python scripts/analysis/aggregate_saa_accent_geometry.py "$OUT"

PYTHONPATH=src pixi run --manifest-path "$ENVIRONMENT" \
  python -m accented_asr.representation_analysis.plot_saa_layer_trajectories \
  --all-folds "$OUT/saa_accent_geometry_all_folds.csv" \
  --output-dir "$OUT/figures"

printf 'SAA layer trajectories written to %s/figures\n' "$OUT"
