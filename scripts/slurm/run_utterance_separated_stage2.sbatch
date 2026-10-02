#!/usr/bin/env bash
#SBATCH --partition=GPU-A100
#SBATCH --account=efl
#SBATCH --job-name=utt-sep-stage2
#SBATCH --array=0-5%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/slurm/stage2-%x-%A_%a.out

set -euo pipefail

accents=(arabic chinese hindi korean spanish vietnamese)
accent="${accents[SLURM_ARRAY_TASK_ID]}"
objective="${1:-supcon-only}"
seed="${2:-13}"
smoke=false
[[ "${3:-}" == "--smoke" ]] && smoke=true

case "${objective}" in
  supcon-only|supcon-ctc|ctc-only) ;;
  *)
    echo "Unknown objective: ${objective}" >&2
    echo "Expected supcon-only, supcon-ctc, or ctc-only." >&2
    exit 2
    ;;
esac

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"

config_path="experiments/separated/librispeech-100h/wav2vec2-large-lv60/utterance-supcon/${objective}/${accent}/stage2/freeze-18/config.yaml"

exec pixi run snakemake -s Snakefile stage2_adaptation \
  --configfile "${config_path}" \
  --config run_seed="${seed}" run_smoke="${smoke}" \
  --cores 1

# Usage (one serial six-accent campaign per submission):
# sbatch scripts/slurm/run_utterance_separated_stage2.sbatch supcon-only 13
# sbatch scripts/slurm/run_utterance_separated_stage2.sbatch supcon-ctc 13
# sbatch scripts/slurm/run_utterance_separated_stage2.sbatch ctc-only 13
