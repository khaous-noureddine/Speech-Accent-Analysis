#!/usr/bin/env bash
set -euo pipefail

accent="${1:-arabic}"
seed="${2:-13}"
ctc_gpu="${3:-0}"
supcon_gpu="${4:-1}"
evaluate_after="${5:-false}"

root="$(git rev-parse --show-toplevel)"
cd "${root}"
mkdir -p outputs/sequential-l2-adaptation
config="experiments/sequential-l2-adaptation/librispeech-100h/wav2vec2-large-lv60/${accent}/seed=${seed}/config.yaml"
prefix="outputs/sequential-l2-adaptation/${accent}-seed-${seed}"

test -f "${config}" || { echo "Missing config: ${config}" >&2; exit 1; }

echo "[1/3] Shared head-only warm-up on GPU ${ctc_gpu}"
CUDA_VISIBLE_DEVICES="${ctc_gpu}" PYTHONPATH=src pixi run \
  --manifest-path environments/representation-cuda124/pixi.toml \
  python -m accented_asr.sequential_l2.train \
  --config "${config}" --seed "${seed}" --stage warmup --repository-root . \
  > >(tee "${prefix}-warmup.log") 2>&1

echo "[2/3] Paired adaptations on GPUs ${ctc_gpu} and ${supcon_gpu}"
CUDA_VISIBLE_DEVICES="${ctc_gpu}" PYTHONPATH=src pixi run \
  --manifest-path environments/representation-cuda124/pixi.toml \
  python -m accented_asr.sequential_l2.train \
  --config "${config}" --seed "${seed}" --stage adapt --objective ctc \
  --repository-root . > >(tee "${prefix}-ctc.log") 2>&1 &
ctc_pid=$!

CUDA_VISIBLE_DEVICES="${supcon_gpu}" PYTHONPATH=src pixi run \
  --manifest-path environments/representation-cuda124/pixi.toml \
  python -m accented_asr.sequential_l2.train \
  --config "${config}" --seed "${seed}" --stage adapt --objective ctc_supcon \
  --repository-root . > >(tee "${prefix}-ctc-supcon.log") 2>&1 &
supcon_pid=$!

failed=0
wait "${ctc_pid}" || failed=1
wait "${supcon_pid}" || failed=1
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one branch failed. Check ${prefix}-*.log" >&2
  exit 1
fi

echo "[3/3] Complete: ${accent}, seed ${seed}"
echo "CTC log: ${prefix}-ctc.log"
echo "SupCon log: ${prefix}-ctc-supcon.log"

if [[ "${evaluate_after}" == "true" ]]; then
  echo "[eval] Evaluating both objectives on GPUs ${ctc_gpu} and ${supcon_gpu}"
  bash scripts/local/run_sequential_l2_evaluation.sh \
    "${accent}" ctc "${seed}" "${ctc_gpu}" \
    > >(tee "${prefix}-ctc-eval.log") 2>&1 &
  ctc_eval_pid=$!
  bash scripts/local/run_sequential_l2_evaluation.sh \
    "${accent}" ctc_supcon "${seed}" "${supcon_gpu}" \
    > >(tee "${prefix}-ctc-supcon-eval.log") 2>&1 &
  supcon_eval_pid=$!

  eval_failed=0
  wait "${ctc_eval_pid}" || eval_failed=1
  wait "${supcon_eval_pid}" || eval_failed=1
  if [[ "${eval_failed}" -ne 0 ]]; then
    echo "At least one evaluation failed. Check ${prefix}-*-eval.log" >&2
    exit 1
  fi
  echo "[eval] Complete: ${accent}, seed ${seed}"
fi
