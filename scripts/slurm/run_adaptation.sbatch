#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=stage2-adapt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=stage2-%x-%j.out

set -euo pipefail

config_path="$1"
seed="$2"
smoke=false
[[ "${3:-}" == "--smoke" ]] && smoke=true

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repository_root}"

exec pixi run snakemake -s Snakefile stage2_adaptation \
  --configfile "${config_path}" \
  --config run_seed="${seed}" run_smoke="${smoke}" \
  --cores 1
