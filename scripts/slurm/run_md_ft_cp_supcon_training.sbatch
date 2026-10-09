#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=mdft-cpsupcon
#SBATCH --array=0-6%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/slurm/mdft-cpsupcon-%A_%a.out

set -euo pipefail

task_id="${SLURM_ARRAY_TASK_ID:?Submit this launcher as an array from 0 to 6}"
seed="${1:-13}"
accents=(arabic chinese hindi korean spanish vietnamese)
[[ "${seed}" == "13" ]] || { echo "Only seed 13 is configured." >&2; exit 2; }

root="experiments/joint/librispeech-100h/wav2vec2-large-lv60/md-ft-cp-supcon"
if (( task_id < 6 )); then
  unit="utterance"
  fold="${accents[task_id]}"
else
  unit="word"
  fold="mswc-common-voice-50h"
fi

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
mkdir -p outputs/slurm
config="${root}/${unit}/${fold}/full-transformer/config.yaml"
test -f "${config}" || { echo "Missing config: ${config}" >&2; exit 1; }

echo "Training MD-FT + CP-SupCon (${unit}, ${fold}, seed=${seed})"
exec pixi run snakemake -s Snakefile joint_ctc_supcon_training \
  --configfile "${config}" \
  --config run_seed="${seed}" run_smoke=false \
  --rerun-incomplete --nolock --cores 1

# Six L2-ARCTIC folds followed by the 50 h word-level configuration, sequentially:
# sbatch --array=0-6%1 scripts/slurm/run_md_ft_cp_supcon_training.sbatch
