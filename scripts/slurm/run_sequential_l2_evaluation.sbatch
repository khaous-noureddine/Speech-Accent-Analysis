#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=seq-l2-eval
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm/seq-l2-eval-%A_%a.out

set -euo pipefail

accent="${1:?Usage: sbatch $0 ACCENT OBJECTIVE [SEED]}"
objective="${2:?Objective must be ctc, ctc_supcon, or ctc_augmented_supcon}"
seed="${3:-13}"
case "${objective}" in
  ctc|ctc_supcon|ctc_augmented_supcon) ;;
  *) echo "Unknown objective: ${objective}" >&2; exit 2 ;;
esac
decoders=(greedy beam_4gram)
decoder="${decoders[$SLURM_ARRAY_TASK_ID]}"

root="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "${root}"
mkdir -p outputs/slurm
experiment="experiments/sequential-l2-adaptation/librispeech-100h/wav2vec2-large-lv60/${accent}/seed=${seed}"
config="${experiment}/evaluation-${objective}.yaml"
extra=()
if [[ "${decoder}" == "beam_4gram" ]]; then
  extra=(--lm-dir language_models/wav2vec2-base-100h-with-lm --beam-width 100)
fi

for dataset in l2_arctic librispeech_test_clean librispeech_test_other aesrc speech_accent_archive edacc; do
  PYTHONPATH=src pixi run \
    --manifest-path environments/evaluation-cuda124/pixi.toml \
    python -m accented_asr.evaluation.run \
    --config "${config}" \
    --dataset "${dataset}" \
    --decoder "${decoder}" \
    --repository-root . \
    "${extra[@]}"
done
