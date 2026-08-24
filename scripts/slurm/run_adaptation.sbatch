#!/usr/bin/env bash
#SBATCH --job-name=stage2-adapt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage2-%x-%j.out

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: sbatch $0 CONDITION FOLD SEED [--smoke]" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "${repository_root}/logs"

# Activate the cluster environment before sbatch, or set ENV_ACTIVATE to a
# shell command such as: source /path/to/venv/bin/activate
if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  eval "${ENV_ACTIVATE}"
fi

exec "${repository_root}/scripts/local/run_adaptation.sh" "$@"
