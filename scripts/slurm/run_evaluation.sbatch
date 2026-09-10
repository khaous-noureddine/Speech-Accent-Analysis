#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=eval-greedy
#SBATCH --array=0-5%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=eval-%x-%A_%a.out

set -euo pipefail

accents=(arabic chinese hindi korean spanish vietnamese)
objective="${1:-supcon-only}"
seed="${2:-13}"
smoke=false
[[ "${3:-}" == "--smoke" ]] && smoke=true

case "${objective}" in
  supcon-only|supcon-ctc|ctc-only|no-stage2|facebook-960h-lv60|published-lv60-100h) ;;
  *)
    echo "Expected supcon-only, supcon-ctc, ctc-only, no-stage2, facebook-960h-lv60, or published-lv60-100h." >&2
    exit 2
    ;;
esac
case "${objective}" in
  facebook-960h-lv60)
    [[ "${seed}" == "official" ]] || {
      echo "The official Hugging Face baseline uses seed label 'official'." >&2
      exit 2
    }
    ;;
  published-lv60-100h)
    [[ "${seed}" == "published" ]] || {
      echo "The published 100-hour baseline uses seed label 'published'." >&2
      exit 2
    }
    ;;
  *)
    [[ "${seed}" == "13" ]] || {
      echo "Only seed 13 evaluation configs are currently available." >&2
      exit 2
    }
    ;;
esac

if [[ "${objective}" == "facebook-960h-lv60" ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "The external checkpoint has one fold-independent evaluation."
    exit 0
  fi
  config_path="experiments/external-baselines/facebook-wav2vec2-large-960h-lv60/evaluation/config.yaml"
elif [[ "${objective}" == "published-lv60-100h" ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "The external checkpoint has one fold-independent evaluation."
    exit 0
  fi
  config_path="experiments/external-baselines/patrickvonplaten-wav2vec2-large-lv60h-100h/evaluation/config.yaml"
elif [[ "${objective}" == "no-stage2" ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "no-stage2 has one fold-independent checkpoint; array task 0 owns the campaign."
    exit 0
  fi
  config_path="experiments/eval/wav2vec2-large-lv60/no-stage2/config.yaml"
else
  accent="${accents[SLURM_ARRAY_TASK_ID]}"
  config_path="experiments/eval/wav2vec2-large-lv60/${objective}/${accent}/config.yaml"
fi

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"

exec pixi run snakemake -s Snakefile evaluate_all \
  --configfile "${config_path}" \
  --config run_smoke="${smoke}" \
  --nolock \
  --rerun-incomplete \
  --cores 1

# All six folds, sequentially on at most one GPU:
# sbatch scripts/slurm/run_evaluation.sbatch supcon-only 13
#
# Spanish only (array index 4):
# sbatch --array=4 scripts/slurm/run_evaluation.sbatch supcon-only 13
#
# Smoke all six folds:
# sbatch scripts/slurm/run_evaluation.sbatch supcon-only 13 --smoke
#
# Fold-independent no-stage2 checkpoint on all evaluation datasets:
# sbatch --array=0 scripts/slurm/run_evaluation.sbatch no-stage2 13
#
# First two supcon-ctc folds, sequentially:
# sbatch --array=0-1%1 scripts/slurm/run_evaluation.sbatch supcon-ctc 13
#
# Arabic CTC-only fold:
# sbatch --array=0 scripts/slurm/run_evaluation.sbatch ctc-only 13
#
# Official Meta checkpoint (evaluation only):
# sbatch --array=0 scripts/slurm/run_evaluation.sbatch facebook-960h-lv60 official
#
# Published LV-60 + LibriSpeech 100-hour Transformers checkpoint:
# sbatch --array=0 scripts/slurm/run_evaluation.sbatch published-lv60-100h published
