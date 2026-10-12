#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=seq-l2-word-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/slurm/seq-l2-word-eval-%A_%a.out

set -euo pipefail

accents=(arabic chinese hindi korean spanish vietnamese)
task_id="${SLURM_ARRAY_TASK_ID:?Submit as an accent array.}"
accent="${accents[$task_id]:?Unknown array task: ${task_id}}"
seed="${1:-13}"

root="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "${root}"
mkdir -p outputs/slurm
experiment="experiments/sequential-l2-word-adaptation/librispeech-100h/wav2vec2-large-lv60/${accent}/seed=${seed}"
config="${experiment}/evaluation-ctc_word_supcon.yaml"
manifest="environments/evaluation-cuda124/pixi.toml"
env_lib="$(dirname "${manifest}")/.pixi/envs/default/lib"
lm_dir="language_models/wav2vec2-base-100h-with-lm"
datasets=(l2_arctic librispeech_test_clean librispeech_test_other aesrc speech_accent_archive edacc)
declare -A parquets=(
  [l2_arctic]="data/processed/l2_arctic_leave_one_accent_out/${accent}/corpus.parquet"
  [librispeech_test_clean]="data/processed/librispeech_test_clean/corpus.parquet"
  [librispeech_test_other]="data/processed/librispeech_test_other/corpus.parquet"
  [aesrc]="data/processed/aesrc/corpus.parquet"
  [speech_accent_archive]="data/processed/speech_accent_archive/corpus.parquet"
  [edacc]="data/processed/edacc/corpus.parquet"
)
output_root="${experiment}/evaluation/ctc_word_supcon/seed=${seed}"

test -f "${config}" || { echo "Missing config: ${config}" >&2; exit 1; }
test -s "${lm_dir}/language_model/4-gram.bin" || {
  echo "Missing 4-gram LM: ${lm_dir}/language_model/4-gram.bin" >&2
  exit 1
}
if [[ -d "${env_lib}" ]]; then
  export LD_LIBRARY_PATH="${env_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

for decoder in greedy beam_4gram; do
  extra=()
  if [[ "${decoder}" == "beam_4gram" ]]; then
    extra=(--lm-dir "${lm_dir}" --beam-width 100)
  fi
  for dataset in "${datasets[@]}"; do
    if [[ ! -s "${parquets[$dataset]}" ]]; then
      echo "SKIP dataset=${dataset}: missing ${parquets[$dataset]}"
      continue
    fi
    if [[ -s "${output_root}/${decoder}/${dataset}/metrics.json" ]]; then
      echo "SKIP dataset=${dataset} decoder=${decoder}: metrics already exist"
      continue
    fi
    echo "Evaluating accent=${accent} seed=${seed} decoder=${decoder} dataset=${dataset}"
    PYTHONPATH=src pixi run --manifest-path "${manifest}" \
      python -m accented_asr.evaluation.run \
      --config "${config}" --dataset "${dataset}" --decoder "${decoder}" \
      --repository-root . "${extra[@]}"
  done
done
