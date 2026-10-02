#!/usr/bin/env bash
#SBATCH --partition=GPU-A100
#SBATCH --account=efl
#SBATCH --job-name=utt-eval-4gram
#SBATCH --array=0-5%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=outputs/slurm/eval-%x-%A_%a.out

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
  config_path="experiments/baselines/external/librispeech-960h/facebook-wav2vec2-large-960h-lv60/evaluation.yaml"
elif [[ "${objective}" == "published-lv60-100h" ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "The external checkpoint has one fold-independent evaluation."
    exit 0
  fi
  config_path="experiments/baselines/external/librispeech-100h/patrickvonplaten-wav2vec2-large-lv60h-100h/evaluation.yaml"
elif [[ "${objective}" == "no-stage2" ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "no-stage2 has one fold-independent checkpoint; array task 0 owns it."
    exit 0
  fi
  config_path="experiments/baselines/internal/librispeech-100h/wav2vec2-large-lv60/ctc-only/full-transformer/evaluation.yaml"
else
  accent="${accents[SLURM_ARRAY_TASK_ID]}"
  config_path="experiments/separated/librispeech-100h/wav2vec2-large-lv60/utterance-supcon/${objective}/${accent}/stage3/full-transformer/evaluation.yaml"
fi

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"

lm_dir="language_models/wav2vec2-base-100h-with-lm"
test -s "${lm_dir}/language_model/4-gram.bin" || {
  echo "Missing 4-gram LM. Run scripts/download_language_model.sh first." >&2
  exit 1
}

exec pixi run snakemake -s Snakefile evaluate_all \
  --configfile "${config_path}" \
  --config \
    run_smoke="${smoke}" \
    evaluation_decoder="beam_4gram" \
    evaluation_lm_dir="${lm_dir}" \
    evaluation_beam_width=100 \
  --nolock \
  --rerun-incomplete \
  --cores 1

# All six folds sequentially:
# sbatch scripts/slurm/run_utterance_evaluation_4gram.sbatch supcon-only 13
#
# One fold only (Spanish = 4):
# sbatch --array=4 scripts/slurm/run_utterance_evaluation_4gram.sbatch supcon-only 13
#
# Smoke test all folds sequentially:
# sbatch scripts/slurm/run_utterance_evaluation_4gram.sbatch supcon-only 13 --smoke
#
# Fold-independent no-stage2 baseline:
# sbatch --array=0 scripts/slurm/run_utterance_evaluation_4gram.sbatch no-stage2 13
#
# Official Meta checkpoint (evaluation only):
# sbatch --array=0 scripts/slurm/run_utterance_evaluation_4gram.sbatch facebook-960h-lv60 official
#
# Published LV-60 + LibriSpeech 100-hour Transformers checkpoint:
# sbatch --array=0 scripts/slurm/run_utterance_evaluation_4gram.sbatch published-lv60-100h published
