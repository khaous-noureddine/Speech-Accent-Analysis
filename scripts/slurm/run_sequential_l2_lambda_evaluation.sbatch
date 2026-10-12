#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=seq-l2-lam-eval
#SBATCH --array=0-9%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm/seq-l2-lam-eval-%A_%a.out

set -euo pipefail

slugs=(lambda-0p01 lambda-0p03 lambda-0p05 lambda-0p10 lambda-0p20)
task_id="${SLURM_ARRAY_TASK_ID:?Submit as a ten-task array}"
slug="${slugs[$((task_id / 2))]:?Invalid array index}"
decoders=(greedy beam_4gram)
decoder="${decoders[$((task_id % 2))]}"

root="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "${root}"
mkdir -p outputs/slurm

base="experiments/ablations/sequential-l2-supcon-weight/librispeech-100h/wav2vec2-large-lv60/arabic"
config="${base}/${slug}/seed=13/evaluation-ctc_supcon.yaml"
test -f "${config}" || { echo "Missing config: ${config}" >&2; exit 1; }

declare -A parquets=(
  [l2_arctic]="data/processed/l2_arctic_leave_one_accent_out/arabic/corpus.parquet"
  [librispeech_test_clean]="data/processed/librispeech_test_clean/corpus.parquet"
  [librispeech_test_other]="data/processed/librispeech_test_other/corpus.parquet"
  [aesrc]="data/processed/aesrc/corpus.parquet"
  [speech_accent_archive]="data/processed/speech_accent_archive/corpus.parquet"
  [edacc]="data/processed/edacc/corpus.parquet"
)

extra=()
if [[ "${decoder}" == "beam_4gram" ]]; then
  extra=(--lm-dir language_models/wav2vec2-base-100h-with-lm --beam-width 100)
fi

for dataset in l2_arctic librispeech_test_clean librispeech_test_other aesrc speech_accent_archive edacc; do
  if [[ ! -s "${parquets[${dataset}]}" ]]; then
    echo "SKIP dataset=${dataset}: missing ${parquets[${dataset}]}"
    continue
  fi
  PYTHONPATH=src pixi run --manifest-path environments/evaluation-cuda124/pixi.toml \
    python -m accented_asr.evaluation.run \
    --config "${config}" --dataset "${dataset}" --decoder "${decoder}" \
    --repository-root . "${extra[@]}"
done
