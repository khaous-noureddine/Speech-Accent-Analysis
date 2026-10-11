#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=mdcp-lam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/slurm/mdcp-lambda-%j.out

set -euo pipefail

accent="${1:?Usage: sbatch $0 <arabic|chinese> [seed] [lambda-0p03|lambda-0p01]}"
seed="${2:-13}"
lambda_tag="${3:-lambda-0p03}"
case "${accent}" in
  arabic|chinese) ;;
  *) echo "Expected accent Arabic or Chinese; got ${accent}." >&2; exit 2 ;;
esac
[[ "${seed}" == "13" ]] || { echo "Only seed 13 is configured." >&2; exit 2; }
case "${lambda_tag}" in
  lambda-0p03) lambda_value="0.03" ;;
  lambda-0p01) lambda_value="0.01" ;;
  *) echo "Unknown lambda configuration: ${lambda_tag}." >&2; exit 2 ;;
esac

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
mkdir -p outputs/slurm

experiment_dir="experiments/ablations/md-ft-cp-supcon-weight/librispeech-100h/wav2vec2-large-lv60/utterance/${accent}/${lambda_tag}/full-transformer"
config="${experiment_dir}/config.yaml"
test -f "${config}" || { echo "Missing config: ${config}" >&2; exit 1; }

echo "Training MD-FT + CP-SupCon (${accent}, gamma=0.1, lambda=${lambda_value}, seed=${seed})"
exec pixi run snakemake -s Snakefile joint_ctc_supcon_training \
  --configfile "${config}" \
  --config run_seed="${seed}" run_smoke=false \
  --rerun-incomplete --nolock --cores 1
