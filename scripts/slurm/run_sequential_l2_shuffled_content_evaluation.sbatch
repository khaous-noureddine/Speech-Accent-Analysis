#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=seq-l2-shuf-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm/seq-l2-shuf-eval-%A_%a.out

set -euo pipefail

accents=(arabic chinese hindi korean spanish vietnamese)
task_id="${SLURM_ARRAY_TASK_ID:?Submit as a 12-task array}"
accent="${accents[$((task_id / 2))]:?Invalid array index}"
decoders=(greedy beam_4gram)
decoder="${decoders[$((task_id % 2))]}"

root="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "${root}"

experiment="experiments/controls/sequential-l2-shuffled-content/librispeech-100h/wav2vec2-large-lv60/${accent}/seed=13"
config="${experiment}/evaluation-ctc_shuffled_supcon.yaml"
manifest="environments/evaluation-cuda124/pixi.toml"
lm_dir="language_models/wav2vec2-base-100h-with-lm"
declare -A parquets=(
  [l2_arctic]="data/processed/l2_arctic_leave_one_accent_out/${accent}/corpus.parquet"
  [librispeech_test_clean]="data/processed/librispeech_test_clean/corpus.parquet"
  [librispeech_test_other]="data/processed/librispeech_test_other/corpus.parquet"
  [aesrc]="data/processed/aesrc/corpus.parquet"
  [speech_accent_archive]="data/processed/speech_accent_archive/corpus.parquet"
  [edacc]="data/processed/edacc/corpus.parquet"
)

extra=()
if [[ "${decoder}" == "beam_4gram" ]]; then
  extra=(--lm-dir "${lm_dir}" --beam-width 100)
fi
for dataset in l2_arctic librispeech_test_clean librispeech_test_other aesrc speech_accent_archive edacc; do
  if [[ ! -s "${parquets[${dataset}]}" ]]; then
    echo "SKIP dataset=${dataset}: missing ${parquets[${dataset}]}"
    continue
  fi
  PYTHONPATH=src pixi run --manifest-path "${manifest}" \
    python -m accented_asr.evaluation.run \
    --config "${config}" --dataset "${dataset}" --decoder "${decoder}" \
    --repository-root . "${extra[@]}"
done
