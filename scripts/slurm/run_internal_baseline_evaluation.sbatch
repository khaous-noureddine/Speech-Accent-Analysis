#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=internal-baseline-eval
#SBATCH --array=0-3%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=outputs/slurm/internal-baseline-eval-%A_%a.out

set -euo pipefail

task_id="${SLURM_ARRAY_TASK_ID:?Submit this launcher as a Slurm array}"
case "${task_id}" in
  0) training_hours="100h"; decoder="greedy" ;;
  1) training_hours="100h"; decoder="beam_4gram" ;;
  2) training_hours="960h"; decoder="greedy" ;;
  3) training_hours="960h"; decoder="beam_4gram" ;;
  *) echo "Unexpected array task ${task_id}; expected 0-3." >&2; exit 2 ;;
esac

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"

experiment_dir="experiments/baselines/internal/librispeech-${training_hours}/wav2vec2-large-lv60/ctc-only/full-transformer"
config_path="${experiment_dir}/evaluation.yaml"
checkpoint_path="${experiment_dir}/outputs/seed=13/checkpoint_best.pt"
lm_dir="language_models/wav2vec2-base-100h-with-lm"

test -f "${config_path}" || { echo "Missing evaluation config: ${config_path}" >&2; exit 1; }
test -s "${checkpoint_path}" || { echo "Missing checkpoint: ${checkpoint_path}" >&2; exit 1; }
if [[ "${decoder}" == "beam_4gram" ]]; then
  test -s "${lm_dir}/language_model/4-gram.bin" || {
    echo "Missing 4-gram LM. Run scripts/download_language_model.sh first." >&2
    exit 1
  }
fi

echo "Re-evaluating internal ${training_hours} CTC baseline with ${decoder}"
exec pixi run snakemake -s Snakefile evaluate_all \
  --configfile "${config_path}" \
  --config \
    run_smoke=false \
    evaluation_decoder="${decoder}" \
    evaluation_lm_dir="${lm_dir}" \
    evaluation_beam_width=100 \
  --forcerun evaluate_checkpoint aggregate_evaluation \
  --rerun-incomplete --nolock --cores 1

# Recompute 100h and 960h internal baselines with greedy and 4-gram decoding:
# sbatch scripts/slurm/run_internal_baseline_evaluation.sbatch
