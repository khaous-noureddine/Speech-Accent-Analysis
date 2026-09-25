#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=word-joint-50h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/slurm/word-joint-%j.out

set -euo pipefail

variant="${1:-full-transformer}"
seed="${2:-13}"
smoke=false
[[ "${3:-}" == "--smoke" ]] && smoke=true

case "${variant}" in
  freeze-18|full-transformer) ;;
  *) echo "Expected freeze-18 or full-transformer, got: ${variant}" >&2; exit 2 ;;
esac
[[ "${seed}" == "13" ]] || { echo "Only seed 13 is configured." >&2; exit 2; }

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
config_path="experiments/joint-training/wav2vec2-large-lv60/word-supcon/mswc-common-voice-50h/${variant}/config.yaml"

exec pixi run snakemake -s Snakefile joint_ctc_supcon_training \
  --configfile "${config_path}" \
  --config run_seed="${seed}" run_smoke="${smoke}" \
  --rerun-incomplete --nolock --cores 1

# Smoke tests:
# sbatch scripts/slurm/run_word_joint_training.sbatch full-transformer 13 --smoke
# sbatch scripts/slurm/run_word_joint_training.sbatch freeze-18 13 --smoke
# Full experiments:
# sbatch scripts/slurm/run_word_joint_training.sbatch full-transformer 13
# sbatch scripts/slurm/run_word_joint_training.sbatch freeze-18 13
