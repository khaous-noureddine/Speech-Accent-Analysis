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
training_hours="${3:-100h}"
accented_hours="${4:-50h}"
smoke=false
for argument in "$@"; do
  [[ "${argument}" == "--smoke" ]] && smoke=true
done

case "${variant}" in
  freeze-18|full-transformer) ;;
  *) echo "Expected freeze-18 or full-transformer, got: ${variant}" >&2; exit 2 ;;
esac
[[ "${seed}" == "13" ]] || { echo "Only seed 13 is configured." >&2; exit 2; }
case "${accented_hours}:${training_hours}" in
  50h:100h) librispeech_dir="librispeech-100h"; dataset_dir="mswc-common-voice-50h" ;;
  50h:960h) librispeech_dir="librispeech-960h"; dataset_dir="mswc-common-voice-50h" ;;
  500h:100h) librispeech_dir="librispeech-100h"; dataset_dir="mswc-common-voice-500h" ;;
  *)
    echo "Supported combinations: accented/LibriSpeech = 50h/100h, 50h/960h, or 500h/100h." >&2
    exit 2
    ;;
esac

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
config_path="experiments/joint/${librispeech_dir}/wav2vec2-large-lv60/word-supcon/${dataset_dir}/${variant}/config.yaml"

exec pixi run snakemake -s Snakefile joint_ctc_supcon_training \
  --configfile "${config_path}" \
  --config run_seed="${seed}" run_smoke="${smoke}" \
  --rerun-incomplete --nolock --cores 1

# Smoke tests:
# sbatch scripts/slurm/run_word_joint_training.sbatch full-transformer 13 100h --smoke
# sbatch scripts/slurm/run_word_joint_training.sbatch freeze-18 13 100h --smoke
# Full experiments:
# sbatch scripts/slurm/run_word_joint_training.sbatch full-transformer 13 100h
# sbatch scripts/slurm/run_word_joint_training.sbatch freeze-18 13 100h
# LibriSpeech-960 experiments:
# sbatch scripts/slurm/run_word_joint_training.sbatch full-transformer 13 960h
# sbatch scripts/slurm/run_word_joint_training.sbatch freeze-18 13 960h
# 500 h accented words + LibriSpeech 100 h:
# sbatch scripts/slurm/run_word_joint_training.sbatch full-transformer 13 100h 500h
