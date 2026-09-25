#!/usr/bin/env bash
#SBATCH --partition=GPU-A100
#SBATCH --account=efl
#SBATCH --job-name=joint-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=joint-%x-%j.out

set -euo pipefail

variant_selector="${1:-all}"
accent_selector="${2:-all}"
accents=(arabic chinese hindi korean spanish vietnamese)
variants=(freeze-18 full-transformer)
smoke=false
skip_missing=false
for argument in "$@"; do
  [[ "${argument}" == "--smoke" ]] && smoke=true
  [[ "${argument}" == "--skip-missing" ]] && skip_missing=true
done

case "${variant_selector}" in
  all) selected_variants=("${variants[@]}") ;;
  freeze-18|full-transformer) selected_variants=("${variant_selector}") ;;
  *) echo "Expected all, freeze-18 or full-transformer, got: ${variant_selector}" >&2; exit 2 ;;
esac

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  selected_accents=("${accents[SLURM_ARRAY_TASK_ID]:?Invalid Slurm array index}")
elif [[ "${accent_selector}" == "all" ]]; then
  selected_accents=("${accents[@]}")
elif [[ " ${accents[*]} " == *" ${accent_selector} "* ]]; then
  selected_accents=("${accent_selector}")
else
  echo "Expected all or one of: ${accents[*]}; got: ${accent_selector}" >&2
  exit 2
fi

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"

declare -a jobs=()
for variant in "${selected_variants[@]}"; do
  for accent in "${selected_accents[@]}"; do
    config_path="experiments/joint-training/wav2vec2-large-lv60/utterance-supcon/${accent}/${variant}/evaluation.yaml"
    checkpoint="experiments/joint-training/wav2vec2-large-lv60/utterance-supcon/${accent}/${variant}/outputs/seed=13/checkpoint_best.pt"
    if [[ ! -f "${config_path}" ]]; then
      echo "Missing evaluation config: ${config_path}" >&2
      exit 1
    fi
    if [[ ! -f "${checkpoint}" ]]; then
      if [[ "${skip_missing}" == true ]]; then
        echo "Skipping ${variant}/${accent}: missing ${checkpoint}" >&2
        continue
      fi
      echo "Missing checkpoint for ${variant}/${accent}: ${checkpoint}" >&2
      echo "Use --skip-missing to evaluate only completed models." >&2
      exit 1
    fi
    jobs+=("${variant}|${accent}|${config_path}")
  done
done

if (( ${#jobs[@]} == 0 )); then
  echo "No completed model matched the requested selection." >&2
  exit 1
fi

echo "Sequential evaluation campaign: ${#jobs[@]} model(s), one GPU"
printf '  %s\n' "${jobs[@]}"

completed=0
for job in "${jobs[@]}"; do
  IFS='|' read -r variant accent config_path <<< "${job}"
  echo "[$((completed + 1))/${#jobs[@]}] Evaluating ${variant}/${accent}"
  pixi run snakemake -s Snakefile evaluate_all \
    --configfile "${config_path}" \
    --config run_smoke="${smoke}" \
    --rerun-incomplete --nolock --cores 1
  completed=$((completed + 1))
done

echo "Sequential evaluation campaign complete: ${completed}/${#jobs[@]} model(s)."

# One model:
# sbatch scripts/slurm/run_joint_evaluation.sbatch freeze-18 arabic
# One variant, all accents sequentially on one GPU:
# sbatch scripts/slurm/run_joint_evaluation.sbatch freeze-18 all
# Both variants and all accents sequentially on one GPU:
# sbatch scripts/slurm/run_joint_evaluation.sbatch all all
# Evaluate only checkpoints that currently exist:
# sbatch scripts/slurm/run_joint_evaluation.sbatch all all --skip-missing
# Legacy array mode (one accent per task, limited to one concurrent task):
# sbatch --array=0-5%1 scripts/slurm/run_joint_evaluation.sbatch freeze-18
