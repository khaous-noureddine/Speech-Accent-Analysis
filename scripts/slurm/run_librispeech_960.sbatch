#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=ls960-pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/slurm/ls960-%x-%j.out

set -euo pipefail

objective="${1:-joint-supcon}"
seed="${2:-13}"
smoke=false
[[ "${3:-}" == "--smoke" ]] && smoke=true
case "${objective}" in
  ctc-only|joint-supcon) ;;
  *) echo "Expected ctc-only or joint-supcon, got: ${objective}" >&2; exit 2 ;;
esac
[[ "${seed}" == "13" ]] || { echo "Pilot config declares only seed 13." >&2; exit 2; }

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
config_path="experiments/scaling/librispeech-960/wav2vec2-large-lv60/arabic/${objective}/config.yaml"

exec pixi run snakemake -s Snakefile joint_ctc_supcon_training \
  --configfile "${config_path}" \
  --config run_seed="${seed}" run_smoke="${smoke}" \
  --rerun-incomplete --nolock --cores 1

# sbatch scripts/slurm/run_librispeech_960.sbatch ctc-only 13 --smoke
# sbatch scripts/slurm/run_librispeech_960.sbatch joint-supcon 13 --smoke
# sbatch scripts/slurm/run_librispeech_960.sbatch ctc-only 13
# sbatch scripts/slurm/run_librispeech_960.sbatch joint-supcon 13
