#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_root}"

if [[ "${1:-}" != "--delete" ]]; then
  echo "Usage: $0 --delete" >&2
  echo "This removes verified legacy experiment directories after migration." >&2
  exit 2
fi

if command -v squeue >/dev/null 2>&1; then
  active_jobs="$(squeue -h -u "${USER}" -o '%i %j %T' 2>/dev/null || true)"
  if [[ -n "${active_jobs}" ]]; then
    echo "Refusing cleanup while Slurm jobs are present:" >&2
    echo "${active_jobs}" >&2
    exit 1
  fi
fi

scripts/migrate_experiment_layout.sh --finalize-copy
scripts/migrate_experiment_layout.sh --verify

if find experiments -name '.LIVE_MIGRATION_INCOMPLETE' -print -quit | grep -q .; then
  echo "Refusing cleanup: live-migration markers remain." >&2
  find experiments -name '.LIVE_MIGRATION_INCOMPLETE' -print >&2
  exit 1
fi

legacy_paths=(
  experiments/eval
  experiments/stage2
  experiments/stage3
  experiments/joint-training
  experiments/scaling
  experiments/word-contrastive
  experiments/external-baselines
)

echo "Removing verified legacy paths:"
for path in "${legacy_paths[@]}"; do
  [[ -e "${path}" ]] || continue
  du -sh "${path}" 2>/dev/null || true
  rm -rf -- "${path}"
done

echo "Legacy experiment-layout cleanup complete."
echo "Active top-level directories:"
find experiments -mindepth 1 -maxdepth 1 -type d -printf '  %f\n' | sort
