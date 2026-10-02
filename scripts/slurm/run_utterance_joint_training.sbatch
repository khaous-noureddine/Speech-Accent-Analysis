#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=utt-joint-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/slurm/utt-joint-train-%j.out

set -euo pipefail

training_hours="${1:-100h}"
variant="${2:-full-transformer}"
seed="${3:-13}"
accents=(arabic chinese hindi korean spanish vietnamese)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  accent="${accents[SLURM_ARRAY_TASK_ID]:?Invalid Slurm array index}"
else
  accent="${4:-arabic}"
fi
smoke=false
for argument in "$@"; do
  [[ "${argument}" == "--smoke" ]] && smoke=true
done

case "${training_hours}" in
  100h|960h) ;;
  *) echo "Expected LibriSpeech hours: 100h or 960h; got ${training_hours}." >&2; exit 2 ;;
esac
case "${variant}" in
  freeze-18|full-transformer) ;;
  *) echo "Expected freeze-18 or full-transformer; got ${variant}." >&2; exit 2 ;;
esac
if [[ "${training_hours}" == "960h" && "${variant}" != "full-transformer" ]]; then
  echo "LibriSpeech 960h currently has only the full-transformer variant." >&2
  exit 2
fi
[[ "${seed}" == "13" ]] || { echo "Only seed 13 is configured." >&2; exit 2; }
[[ " ${accents[*]} " == *" ${accent} "* ]] || {
  echo "Unknown accent: ${accent}" >&2
  exit 2
}

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
config_path="experiments/joint/librispeech-${training_hours}/wav2vec2-large-lv60/utterance-supcon/${accent}/${variant}/config.yaml"
test -f "${config_path}" || { echo "Missing config: ${config_path}" >&2; exit 1; }

exec pixi run snakemake -s Snakefile joint_ctc_supcon_training \
  --configfile "${config_path}" \
  --config run_seed="${seed}" run_smoke="${smoke}" \
  --rerun-incomplete --nolock --cores 1

# One accent:
# sbatch scripts/slurm/run_utterance_joint_training.sbatch 100h full-transformer 13 arabic
# All accents, one at a time:
# sbatch --array=0-5%1 scripts/slurm/run_utterance_joint_training.sbatch 960h full-transformer 13
