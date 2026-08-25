#!/usr/bin/env bash
#SBATCH --partition=GPU-A100
#SBATCH --account=efl
#SBATCH --job-name=stage3-asr-ft
#SBATCH --array=0-5%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=stage3-%x-%A_%a.out

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

config_path="experiments/stage3/wav2vec2-large-lv60/${objective}/${accent}/config.yaml"

exec pixi run snakemake -s Snakefile stage3_asr_finetuning \
  --configfile "${config_path}" \
  --config run_seed="${seed}" run_smoke="${smoke}" \
  --cores 1

# Usage:
# One full Spanish run (array index 4):
# sbatch --array=4 scripts/slurm/run_asr_finetuning.sbatch supcon-only 13
#
# Six accents, serially on at most one GPU:
# sbatch scripts/slurm/run_asr_finetuning.sbatch supcon-only 13
# sbatch scripts/slurm/run_asr_finetuning.sbatch supcon-ctc 13
# sbatch scripts/slurm/run_asr_finetuning.sbatch ctc-only 13
