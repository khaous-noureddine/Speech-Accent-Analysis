#!/usr/bin/env bash
#SBATCH --partition=MISC-56c
#SBATCH --account=efl
#SBATCH --job-name=supcon-error-analysis
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=outputs/slurm/supcon-error-analysis-%j.out

set -euo pipefail

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"

config_path="${1:-experiments/analysis/supcon-error-analysis/librispeech-960h/config.yaml}"
test -f "${config_path}" || { echo "Missing analysis config: ${config_path}" >&2; exit 1; }

exec env PYTHONPATH=src pixi run python -m accented_asr.error_analysis.run \
  --config "${config_path}" \
  --repository-root .

# sbatch scripts/slurm/run_supcon_error_analysis.sbatch
