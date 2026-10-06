#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=word-emb-analysis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=outputs/slurm/word-emb-analysis-%j.out

set -euo pipefail

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"

config_path="${1:-experiments/analysis/word-supcon-embeddings/librispeech-960h-mswc-common-voice-500h/config.yaml}"
test -f "${config_path}" || { echo "Missing analysis config: ${config_path}" >&2; exit 1; }

exec env PYTHONPATH=src pixi run python -m accented_asr.representation_analysis.run \
  --config "${config_path}" \
  --repository-root .

# sbatch scripts/slurm/run_word_embedding_analysis.sbatch
