#!/usr/bin/env bash
set -euo pipefail

method="${1:?Usage: $0 <method> <accent> [greedy|beam_4gram|all] [gpu_id]}"
accent="${2:?Usage: $0 <method> <accent> [greedy|beam_4gram|all] [gpu_id]}"
decoder_selector="${3:-all}"
gpu_id="${4:-0}"

case "${method}" in
  accent-dat|accent-mtl|multidomain-ctc|augmented-view-supcon) ;;
  *) echo "Unknown method: ${method}" >&2; exit 2 ;;
esac
case "${accent}" in
  arabic|chinese|hindi|korean|spanish|vietnamese) ;;
  *) echo "Unknown accent: ${accent}" >&2; exit 2 ;;
esac
case "${decoder_selector}" in
  greedy|beam_4gram) decoders=("${decoder_selector}") ;;
  all) decoders=(greedy beam_4gram) ;;
  *) echo "Expected decoder: greedy, beam_4gram, or all; got ${decoder_selector}." >&2; exit 2 ;;
esac
[[ "${gpu_id}" =~ ^[0-9]+$ ]] || { echo "Expected a numeric GPU ID; got ${gpu_id}." >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${gpu_id}"

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repository_root}"

pixi_command=(pixi run)
if [[ -n "${EVALUATION_PIXI_MANIFEST:-}" ]]; then
  pixi_command+=(--manifest-path "${EVALUATION_PIXI_MANIFEST}")
fi

"${pixi_command[@]}" python -c '
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        f"CUDA is unavailable (PyTorch {torch.__version__}, runtime {torch.version.cuda})."
    )
print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
'

experiment_dir="experiments/baselines/prior-methods/librispeech-100h/wav2vec2-large-lv60/${method}/${accent}/full-transformer"
config_path="${experiment_dir}/evaluation.yaml"
checkpoint_path="${experiment_dir}/outputs/seed=13/checkpoint_best.pt"
lm_dir="language_models/wav2vec2-base-100h-with-lm"

test -f "${config_path}" || { echo "Missing evaluation config: ${config_path}" >&2; exit 1; }
test -s "${checkpoint_path}" || { echo "Missing checkpoint: ${checkpoint_path}" >&2; exit 1; }

for decoder in "${decoders[@]}"; do
  if [[ "${decoder}" == "beam_4gram" ]]; then
    test -s "${lm_dir}/language_model/4-gram.bin" || {
      echo "Missing 4-gram LM. Run scripts/download_language_model.sh first." >&2
      exit 1
    }
  fi

  echo "Evaluating ${method}/${accent} with ${decoder} on GPU ${gpu_id}"
  "${pixi_command[@]}" snakemake -s Snakefile evaluate_all \
    --configfile "${config_path}" \
    --config \
      run_smoke=false \
      evaluation_decoder="${decoder}" \
      evaluation_lm_dir="${lm_dir}" \
      evaluation_beam_width=100 \
    --rerun-incomplete \
    --consider-ancient prepare_evaluation_dataset=preparer \
    --nolock --cores 1
done

# Examples:
# scripts/local/run_prior_method_evaluation.sh accent-dat arabic all 0
# scripts/local/run_prior_method_evaluation.sh accent-mtl chinese greedy 1
