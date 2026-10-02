#!/usr/bin/env bash
#SBATCH --partition=GPU-A100
#SBATCH --account=efl
#SBATCH --job-name=utt-sep-stage3
#SBATCH --array=0-5%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/slurm/stage3-%x-%A_%a.out

set -euo pipefail

accents=(arabic chinese hindi korean spanish vietnamese)
objective="${1:-supcon-only}"
seed="${2:-13}"
smoke=false
[[ "${3:-}" == "--smoke" ]] && smoke=true

case "${objective}" in
  supcon-only|supcon-ctc|ctc-only|no-stage2) ;;
  *)
    echo "Unknown objective: ${objective}" >&2
    echo "Expected supcon-only, supcon-ctc, ctc-only, or no-stage2." >&2
    exit 2
    ;;
esac

if [[ "${objective}" == "no-stage2" ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID}" -ne 0 ]]; then
    echo "Skipping array index ${SLURM_ARRAY_TASK_ID}: no-stage2 has no accent folds."
    exit 0
  fi
  config_path="experiments/baselines/internal/librispeech-100h/wav2vec2-large-lv60/ctc-only/full-transformer/config.yaml"
else
  accent="${accents[SLURM_ARRAY_TASK_ID]}"
  config_path="experiments/separated/librispeech-100h/wav2vec2-large-lv60/utterance-supcon/${objective}/${accent}/stage3/full-transformer/config.yaml"
fi

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"

exec pixi run snakemake -s Snakefile stage3_asr_finetuning \
  --configfile "${config_path}" \
  --config run_seed="${seed}" run_smoke="${smoke}" \
  --rerun-incomplete \
  --cores 1

# Usage:
# One full Spanish run (array index 4):
# sbatch --array=4 scripts/slurm/run_utterance_separated_stage3.sbatch supcon-only 13
#
# Six accents, serially on at most one GPU:
# sbatch scripts/slurm/run_utterance_separated_stage3.sbatch supcon-only 13
# sbatch scripts/slurm/run_utterance_separated_stage3.sbatch supcon-ctc 13
# sbatch scripts/slurm/run_utterance_separated_stage3.sbatch ctc-only 13
#
# One no-Stage-2 baseline (no accent fold):
# sbatch --array=0 scripts/slurm/run_utterance_separated_stage3.sbatch no-stage2 13
