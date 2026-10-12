#!/usr/bin/env bash
set -euo pipefail

seed="${1:-13}"
gpu_zero="${2:-0}"
gpu_one="${3:-1}"
evaluate_after="${4:-true}"
weights=(lambda-0p01 lambda-0p03 lambda-0p05 lambda-0p10)
reference_weight="lambda-0p05"

[[ "${seed}" == "13" ]] || {
  echo "This first lambda sweep is configured for seed 13 only." >&2
  exit 2
}
[[ "${gpu_zero}" != "${gpu_one}" ]] || {
  echo "Use two distinct GPU IDs." >&2
  exit 2
}

root="$(git rev-parse --show-toplevel)"
cd "${root}"
mkdir -p outputs/sequential-l2-word-lambda-ablation

PYTHONPATH=src pixi run python \
  scripts/experiments/generate_sequential_l2_word_lambda_ablation_configs.py

base="experiments/ablations/sequential-l2-word-supcon-weight/librispeech-100h/wav2vec2-large-lv60/arabic"
reference_config="${base}/${reference_weight}/seed=${seed}/config.yaml"
reference_warmup="${base}/${reference_weight}/outputs/seed=${seed}/warmup/checkpoint.pt"

echo "[1/3] Shared word-CTC warm-up on GPU ${gpu_zero}"
CUDA_VISIBLE_DEVICES="${gpu_zero}" PYTHONPATH=src pixi run \
  --manifest-path environments/representation-cuda124/pixi.toml \
  python -m accented_asr.sequential_l2.train \
  --config "${reference_config}" --seed "${seed}" --stage warmup --repository-root . \
  > >(tee "outputs/sequential-l2-word-lambda-ablation/warmup.log") 2>&1

for slug in "${weights[@]}"; do
  warmup="${base}/${slug}/outputs/seed=${seed}/warmup/checkpoint.pt"
  mkdir -p "$(dirname "${warmup}")"
  if [[ "${warmup}" != "${reference_warmup}" ]]; then
    ln -f "${reference_warmup}" "${warmup}"
  fi
done

run_pair() {
  local left="$1" right="$2"
  local left_config="${base}/${left}/seed=${seed}/config.yaml"
  local right_config="${base}/${right}/seed=${seed}/config.yaml"

  echo "[2/3] Training ${left} on GPU ${gpu_zero}; ${right} on GPU ${gpu_one}"
  CUDA_VISIBLE_DEVICES="${gpu_zero}" PYTHONPATH=src pixi run \
    --manifest-path environments/representation-cuda124/pixi.toml \
    python -m accented_asr.sequential_l2.train \
    --config "${left_config}" --seed "${seed}" --stage adapt \
    --objective ctc_word_supcon --repository-root . \
    > >(tee "outputs/sequential-l2-word-lambda-ablation/${left}.log") 2>&1 &
  local left_pid=$!

  CUDA_VISIBLE_DEVICES="${gpu_one}" PYTHONPATH=src pixi run \
    --manifest-path environments/representation-cuda124/pixi.toml \
    python -m accented_asr.sequential_l2.train \
    --config "${right_config}" --seed "${seed}" --stage adapt \
    --objective ctc_word_supcon --repository-root . \
    > >(tee "outputs/sequential-l2-word-lambda-ablation/${right}.log") 2>&1 &
  local right_pid=$!

  wait "${left_pid}"
  wait "${right_pid}"
}

run_pair "${weights[0]}" "${weights[1]}"
run_pair "${weights[2]}" "${weights[3]}"

if [[ "${evaluate_after}" == "true" ]]; then
  echo "[3/3] Evaluating the four lambda settings on the same two GPUs"
  for index in 0 2; do
    left="${weights[$index]}"
    right="${weights[$((index + 1))]}"
    bash scripts/local/run_sequential_l2_word_lambda_evaluation.sh "${left}" "${gpu_zero}" \
      > >(tee "outputs/sequential-l2-word-lambda-ablation/${left}-eval.log") 2>&1 &
    left_pid=$!
    bash scripts/local/run_sequential_l2_word_lambda_evaluation.sh "${right}" "${gpu_one}" \
      > >(tee "outputs/sequential-l2-word-lambda-ablation/${right}-eval.log") 2>&1 &
    right_pid=$!
    wait "${left_pid}"
    wait "${right_pid}"
  done
fi
