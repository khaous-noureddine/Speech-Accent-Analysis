#!/usr/bin/env bash
#SBATCH --partition=GPU-A100
#SBATCH --account=efl
#SBATCH --job-name=joint-eval-arabic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=joint-%x-%j.out

set -euo pipefail

variant="${1:-freeze-18}"
smoke=false
[[ "${2:-}" == "--smoke" ]] && smoke=true

case "${variant}" in
  freeze-18|full-transformer) ;;
  *) echo "Expected freeze-18 or full-transformer, got: ${variant}" >&2; exit 2 ;;
esac

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
config_path="experiments/joint-training/wav2vec2-large-lv60/utterance-supcon/arabic/${variant}/evaluation.yaml"

exec pixi run snakemake -s Snakefile evaluate_all \
  --configfile "${config_path}" \
  --config run_smoke="${smoke}" \
  --rerun-incomplete --nolock --cores 1

# sbatch scripts/slurm/run_joint_evaluation.sbatch freeze-18
# sbatch scripts/slurm/run_joint_evaluation.sbatch full-transformer
