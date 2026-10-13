#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=seq-l2-shuf-geom
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm/seq-l2-shuf-geom-%j.out

set -euo pipefail

root="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "${root}"

config="experiments/controls/sequential-l2-shuffled-content/librispeech-100h/wav2vec2-large-lv60/representation-geometry-seed13.yaml"
output="experiments/controls/sequential-l2-shuffled-content/librispeech-100h/wav2vec2-large-lv60/representation-geometry/seed=13"

for accent in arabic chinese hindi korean spanish vietnamese; do
  checkpoint="experiments/controls/sequential-l2-shuffled-content/librispeech-100h/wav2vec2-large-lv60/${accent}/outputs/seed=13/ctc_shuffled_supcon/checkpoint_best.pt"
  test -s "${checkpoint}" || {
    echo "Missing shuffled-control checkpoint: ${checkpoint}" >&2
    exit 1
  }
done

env PYTHONPATH=src pixi run \
  --manifest-path environments/representation-cuda124/pixi.toml \
  python -m accented_asr.representation_analysis.run_utterance \
  --config "${config}" \
  --repository-root . \
  --output-dir "${output}"

env PYTHONPATH=src pixi run \
  --manifest-path environments/representation-cuda124/pixi.toml \
  python -m accented_asr.representation_analysis.recenter_utterance \
  --input-root "${output}" \
  --output-dir "${output}/centered-analysis" \
  --seed 13

echo "Raw metrics: ${output}/metrics_macro.csv"
echo "Mean-centered metrics: ${output}/centered-analysis/metrics_macro_raw_and_centered.csv"
