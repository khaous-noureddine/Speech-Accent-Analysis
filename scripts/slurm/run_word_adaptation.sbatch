#!/usr/bin/env bash
#SBATCH --partition=GPU-A100
#SBATCH --account=efl
#SBATCH --job-name=word-supcon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=word-supcon-%j.out

set -euo pipefail

corpus="${1:-l2-arctic}"
accent="${2:-arabic}"
seed="${3:-13}"
smoke=false
[[ "${4:-}" == "--smoke" ]] && smoke=true

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
config_path="experiments/word-contrastive/wav2vec2-large-lv60/${corpus}/${accent}/config.yaml"
test -f "${config_path}" || { echo "Missing config: ${config_path}" >&2; exit 2; }

exec pixi run snakemake -s Snakefile stage2_adaptation \
  --configfile "${config_path}" \
  --config run_seed="${seed}" run_smoke="${smoke}" \
  --cores 1 --rerun-incomplete

# Usage:
# sbatch scripts/slurm/run_word_adaptation.sbatch l2-arctic arabic 13 --smoke
# sbatch scripts/slurm/run_word_adaptation.sbatch l2-arctic arabic 13
