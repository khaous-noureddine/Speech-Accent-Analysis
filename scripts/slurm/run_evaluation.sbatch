#!/usr/bin/env bash
#SBATCH --partition=GPU-A100
#SBATCH --account=efl
#SBATCH --job-name=eval-greedy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=eval-%x-%j.out

set -euo pipefail

config_path="${1:?Usage: sbatch $0 <evaluation-config.yaml> [--smoke]}"
smoke=false
[[ "${2:-}" == "--smoke" ]] && smoke=true

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"

exec pixi run snakemake -s Snakefile evaluate_all \
  --configfile "${config_path}" \
  --config run_smoke="${smoke}" \
  --rerun-incomplete \
  --cores 1

# Example:
# sbatch scripts/slurm/run_evaluation.sbatch \
#   experiments/eval/wav2vec2-large-lv60/supcon-only/arabic/config.yaml --smoke
