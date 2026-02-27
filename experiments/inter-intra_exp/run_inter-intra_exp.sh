#!/usr/bin/env bash
set -euo pipefail
set -x

# ============================================================
# CONFIG
# ============================================================
PYTHON_BIN=/home/nkhaous/myLLF/speech_accent/representations_distances/.venv/bin/python
SCRIPT=/home/nkhaous/myLLF/speech_accent/representations_distances/meanpooling_distances/compute_distance_to_native_meanpooling.py
METADATA=/home/nkhaous/myLLF/speech_accent/representations_distances/data/metadata_balanced.csv
REPR_DIR=/home/nkhaous/myLLF/speech_accent/representations_distances/representations
PARENT_OUT=/home/nkhaous/myLLF/speech_accent/representations_distances/experiences/inter-intra_exp

# Normalizations to run
# NORMALIZATIONS=(none l2 center center_l2 center_std)
NORMALIZATIONS=(center_std)

# Toggle flags
COMPUTE_PER_L1=1        # 1 -> add --compute_per_l1, 0 -> don't
INCLUDE_NATIVE_LOO=0    # 1 -> add --include_loo, 0 -> don't (costly)

# ============================================================
# HELPERS
# ============================================================
get_layer_range() {
  case "$1" in
    wav2vec) echo "0 24" ;;
    whisper) echo "0 32" ;;
    *)       echo "Unknown model: $1" >&2; exit 1 ;;
  esac
}

run_model() {
  local model="$1"
  read -r min_layer max_layer <<< "$(get_layer_range "${model}")"

  local input_dir="${REPR_DIR}/${model}_aligned"

  for normalization in "${NORMALIZATIONS[@]}"; do
    local out_dir="${PARENT_OUT}/${model}/norm_${normalization}"
    mkdir -p "${out_dir}"

    # Build args
    args=(
      --input_dir      "${input_dir}"
      --metadata       "${METADATA}"
      --output_results "${out_dir}/results_${model}_${normalization}.pkl"
      --normalization  "${normalization}"
      --min_layer      "${min_layer}"
      --max_layer      "${max_layer}"
      --model_name     "${model}"
      --compute_per_l1
      --include_intra
    #   --include_loo
    )

    if [[ "${COMPUTE_PER_L1}" -eq 1 ]]; then
      args+=(--compute_per_l1)
    fi

    if [[ "${INCLUDE_NATIVE_LOO}" -eq 1 ]]; then
      args+=(--include_loo)
    fi

    "${PYTHON_BIN}" "${SCRIPT}" "${args[@]}"
  done
}

# ============================================================
# MAIN
# ============================================================
mkdir -p "${PARENT_OUT}"

for model in wav2vec whisper; do
# for model in wav2vec; do
  run_model "${model}"
done



# python patch_merge_results.py \
#   --old_pkl  /path/to/old_loo.pkl \
#   --intra_pkl /path/to/new_intra.pkl \
#   --out_pkl  /path/to/merged.pkl