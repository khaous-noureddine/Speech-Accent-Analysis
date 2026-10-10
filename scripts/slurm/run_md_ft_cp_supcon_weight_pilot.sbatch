#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=mdcp-l003
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/slurm/mdcp-l003-%j.out

set -euo pipefail

accent="${1:?Usage: sbatch $0 <arabic|chinese> [seed]}"
seed="${2:-13}"
case "${accent}" in
  arabic|chinese) ;;
  *) echo "Expected accent Arabic or Chinese; got ${accent}." >&2; exit 2 ;;
esac
[[ "${seed}" == "13" ]] || { echo "Only seed 13 is configured." >&2; exit 2; }

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
mkdir -p outputs/slurm

experiment_dir="experiments/ablations/md-ft-cp-supcon-weight/librispeech-100h/wav2vec2-large-lv60/utterance/${accent}/lambda-0p03/full-transformer"
config="${experiment_dir}/config.yaml"
test -f "${config}" || { echo "Missing config: ${config}" >&2; exit 1; }

echo "Training MD-FT + CP-SupCon (${accent}, gamma=0.1, lambda=0.03, seed=${seed})"
exec pixi run snakemake -s Snakefile joint_ctc_supcon_training \
  --configfile "${config}" \
  --config run_seed="${seed}" run_smoke=false \
  --rerun-incomplete --nolock --cores 1
