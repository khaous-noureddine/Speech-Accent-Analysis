#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=test-other
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=outputs/slurm/test-other-%A_%a.out

set -euo pipefail

family="${1:?Usage: sbatch --array=... $0 FAMILY}"
task_id="${SLURM_ARRAY_TASK_ID:?Submit this launcher as an array}"
accents=(arabic chinese hindi korean spanish vietnamese)
decoder_index=$((task_id % 2))
[[ "${decoder_index}" -eq 0 ]] && decoder="greedy" || decoder="beam_4gram"
accent="${accents[$((task_id / 2))]:-}"
root="experiments/joint/librispeech-100h/wav2vec2-large-lv60"

case "${family}" in
  asr-only)
    (( task_id < 2 )) || { echo "asr-only expects --array=0-1" >&2; exit 2; }
    experiment="experiments/baselines/internal/librispeech-100h/wav2vec2-large-lv60/ctc-only/full-transformer"
    ;;
  md-ft)
    (( task_id < 12 )) || { echo "md-ft expects --array=0-11" >&2; exit 2; }
    experiment="experiments/baselines/prior-methods/librispeech-100h/wav2vec2-large-lv60/multidomain-ctc/${accent}/full-transformer"
    ;;
  cp-supcon-utterance)
    (( task_id < 12 )) || { echo "cp-supcon-utterance expects --array=0-11" >&2; exit 2; }
    experiment="${root}/utterance-supcon/${accent}/full-transformer"
    ;;
  cp-supcon-word)
    (( task_id < 2 )) || { echo "cp-supcon-word expects --array=0-1" >&2; exit 2; }
    experiment="${root}/word-supcon/mswc-common-voice-50h/full-transformer"
    ;;
  combined-utterance)
    (( task_id < 12 )) || { echo "combined-utterance expects --array=0-11" >&2; exit 2; }
    experiment="${root}/md-ft-cp-supcon/utterance/${accent}/full-transformer"
    ;;
  combined-word)
    (( task_id < 2 )) || { echo "combined-word expects --array=0-1" >&2; exit 2; }
    experiment="${root}/md-ft-cp-supcon/word/mswc-common-voice-50h/full-transformer"
    ;;
  *) echo "Unknown family: ${family}" >&2; exit 2 ;;
esac

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
mkdir -p outputs/slurm
config="${experiment}/evaluation.yaml"
checkpoint="${experiment}/outputs/seed=13/checkpoint_best.pt"
target="${experiment}/outputs/seed=13/${decoder}/librispeech_test_other/metrics.json"
test -s "${checkpoint}" || { echo "Missing checkpoint: ${checkpoint}" >&2; exit 1; }
test -d data/raw/librispeech/test/LibriSpeech/test-other || {
  echo "Download test-other with scripts/download_librispeech_test_other.sh first." >&2
  exit 1
}

lm_dir="language_models/wav2vec2-base-100h-with-lm"
if [[ "${decoder}" == "beam_4gram" ]]; then
  test -s "${lm_dir}/language_model/4-gram.bin" || {
    echo "Missing 4-gram language model." >&2
    exit 1
  }
fi

echo "Evaluating ${family} ${accent:-global} on test-other with ${decoder}"
exec pixi run snakemake -s Snakefile "${target}" \
  --configfile "${config}" \
  --config run_smoke=false evaluation_decoder="${decoder}" \
    evaluation_lm_dir="${lm_dir}" evaluation_beam_width=100 \
  --rerun-incomplete --nolock --cores 1
