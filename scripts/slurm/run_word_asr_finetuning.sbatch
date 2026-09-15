#!/usr/bin/env bash
#SBATCH --partition=GPU-A100
#SBATCH --account=efl
#SBATCH --job-name=word-asr-ft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=word-asr-ft-%j.out

set -euo pipefail

corpus="${1:-l2-arctic}"
accent="${2:-arabic}"
seed="${3:-13}"
smoke=false
[[ "${4:-}" == "--smoke" ]] && smoke=true

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
config_path="experiments/word-contrastive/wav2vec2-large-lv60/${corpus}/${accent}/asr-finetuning/config.yaml"
test -f "${config_path}" || { echo "Missing config: ${config_path}" >&2; exit 2; }

exec pixi run snakemake -s Snakefile stage3_asr_finetuning \
  --configfile "${config_path}" \
  --config run_seed="${seed}" run_smoke="${smoke}" \
  --cores 1 --rerun-incomplete

# Usage after word adaptation finishes:
# sbatch scripts/slurm/run_word_asr_finetuning.sbatch l2-arctic arabic 13 --smoke
# sbatch scripts/slurm/run_word_asr_finetuning.sbatch l2-arctic arabic 13
