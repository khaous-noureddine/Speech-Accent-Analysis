#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=gamma-ablation
#SBATCH --array=0-5%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/slurm/gamma-ablation-%A_%a.out

set -euo pipefail

gamma_values=(0 0.01 0.05 0.1 0.5 1.0)
gamma_slugs=(0p0 0p01 0p05 0p1 0p5 1p0)
smoke=false
for argument in "$@"; do
  [[ "${argument}" == "--smoke" ]] && smoke=true
done

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  index="${SLURM_ARRAY_TASK_ID}"
else
  requested_gamma="${1:-0.1}"
  index=-1
  for candidate in "${!gamma_values[@]}"; do
    if [[ "${gamma_values[candidate]}" == "${requested_gamma}" ]]; then
      index="${candidate}"
      break
    fi
  done
  (( index >= 0 )) || {
    echo "Unsupported gamma: ${requested_gamma}. Expected: ${gamma_values[*]}" >&2
    exit 2
  }
fi

gamma="${gamma_values[index]:?Invalid Slurm array index}"
slug="${gamma_slugs[index]}"
seed="${2:-13}"
[[ "${seed}" == "13" ]] || { echo "Only seed 13 is configured." >&2; exit 2; }

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
mkdir -p outputs/slurm

experiment_dir="experiments/ablations/supcon-weight/librispeech-100h/wav2vec2-large-lv60/utterance-supcon/arabic/full-transformer/gamma-${slug}"
config_path="${experiment_dir}/config.yaml"
test -f "${config_path}" || { echo "Missing config: ${config_path}" >&2; exit 1; }

echo "Training Arabic gamma=${gamma} (seed=${seed})"
exec pixi run snakemake -s Snakefile joint_ctc_supcon_training \
  --configfile "${config_path}" \
  --config run_seed="${seed}" run_smoke="${smoke}" \
  --rerun-incomplete --nolock --cores 1

# Full grid, at most two H200 jobs simultaneously:
# sbatch --array=0-5%2 scripts/slurm/run_gamma_ablation_training.sbatch
# One value only:
# sbatch scripts/slurm/run_gamma_ablation_training.sbatch 0.05 13
# Smoke gamma=0.05:
# sbatch --array=2 scripts/slurm/run_gamma_ablation_training.sbatch --smoke
