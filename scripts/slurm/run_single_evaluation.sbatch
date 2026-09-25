#!/usr/bin/env bash
#SBATCH --partition=GPU-A100
#SBATCH --account=efl
#SBATCH --job-name=single-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm/single-eval-%j.out

set -euo pipefail

config_path="${1:?Usage: sbatch run_single_evaluation.sbatch CONFIG [greedy|beam_4gram]}"
decoder="${2:-greedy}"

case "${decoder}" in
  greedy|beam_4gram) ;;
  *) echo "Expected decoder greedy or beam_4gram, got: ${decoder}" >&2; exit 2 ;;
esac

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"

test -f "${config_path}" || {
  echo "Missing evaluation config: ${config_path}" >&2
  exit 1
}

lm_dir="language_models/wav2vec2-base-100h-with-lm"
if [[ "${decoder}" == "beam_4gram" ]]; then
  test -s "${lm_dir}/language_model/4-gram.bin" || {
    echo "Missing 4-gram LM. Run scripts/download_language_model.sh first." >&2
    exit 1
  }
fi

echo "Evaluation config: ${config_path}"
echo "Decoder: ${decoder}"
exec pixi run snakemake -s Snakefile evaluate_all \
  --configfile "${config_path}" \
  --config \
    run_smoke=false \
    evaluation_decoder="${decoder}" \
    evaluation_lm_dir="${lm_dir}" \
    evaluation_beam_width=100 \
  --rerun-incomplete --nolock --cores 1

# Examples:
# sbatch scripts/slurm/run_single_evaluation.sbatch \
#   experiments/eval/wav2vec2-large-lv60/supcon-only/korean/config.yaml \
#   beam_4gram
# sbatch --partition=GPU-H200 scripts/slurm/run_single_evaluation.sbatch \
#   experiments/joint-training/wav2vec2-large-lv60/utterance-supcon/arabic/full-transformer/evaluation.yaml \
#   greedy
