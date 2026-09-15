#!/usr/bin/env bash
#SBATCH --partition=GPU-A100
#SBATCH --account=efl
#SBATCH --job-name=ls960-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=ls960-%x-%j.out

set -euo pipefail

objective="${1:-joint-supcon}"
smoke=false
[[ "${2:-}" == "--smoke" ]] && smoke=true
case "${objective}" in
  ctc-only|joint-supcon) ;;
  *) echo "Expected ctc-only or joint-supcon, got: ${objective}" >&2; exit 2 ;;
esac

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
config_path="experiments/scaling/librispeech-960/wav2vec2-large-lv60/arabic/${objective}/evaluation.yaml"

exec pixi run snakemake -s Snakefile evaluate_all \
  --configfile "${config_path}" \
  --config run_smoke="${smoke}" \
  --rerun-incomplete --nolock --cores 1
