#!/usr/bin/env bash
# Submit the six-fold, one-seed sequential L2 word-SupCon experiment.

set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
mkdir -p outputs/slurm

seed="${1:-13}"
[[ "${seed}" == "13" ]] || {
  echo "This pilot is intentionally configured only for seed 13." >&2
  exit 2
}

accents=(arabic chinese hindi korean spanish vietnamese)
word_data_ready=true
for accent in "${accents[@]}"; do
  for filename in corpus.parquet validation_report.json; do
    if [[ ! -s "data/processed/l2_arctic_words/${accent}/${filename}" ]]; then
      word_data_ready=false
    fi
  done
done

warmup_dependency=()
data_job="already prepared"
if [[ "${word_data_ready}" != true ]]; then
  data_job=$(sbatch --parsable \
    scripts/slurm/run_l2_arctic_word_data_preparation.sbatch)
  warmup_dependency=(--dependency="afterok:${data_job}")
fi

warmup_job=$(sbatch --parsable "${warmup_dependency[@]}" --array=0-5%2 \
  scripts/slurm/run_sequential_l2_word_warmup.sbatch "${seed}")
adapt_job=$(sbatch --parsable --dependency="afterok:${warmup_job}" --array=0-11%2 \
  scripts/slurm/run_sequential_l2_word_adaptation.sbatch "${seed}")
eval_job=$(sbatch --parsable --dependency="afterok:${adapt_job}" --array=0-11%2 \
  scripts/slurm/run_sequential_l2_word_evaluation.sbatch "${seed}")

printf 'Word data:  %s\nWarm-up:    %s\nAdaptation: %s\nEvaluation: %s\n' \
  "${data_job}" "${warmup_job}" "${adapt_job}" "${eval_job}"
printf '\nAt most two GPUs are used at any time.\n'
