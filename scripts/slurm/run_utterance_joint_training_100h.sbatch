#!/usr/bin/env bash
#SBATCH --partition=GPU-A100
#SBATCH --account=efl
#SBATCH --job-name=utt-joint-100h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/slurm/joint-%x-%j.out

set -euo pipefail

variant="${1:-freeze-18}"
seed="${2:-13}"
accents=(arabic chinese hindi korean spanish vietnamese)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  accent="${accents[SLURM_ARRAY_TASK_ID]:?Invalid Slurm array index}"
else
  accent="${3:-arabic}"
fi
smoke=false
for argument in "$@"; do
  [[ "${argument}" == "--smoke" ]] && smoke=true
done

case "${variant}" in
  freeze-18|full-transformer) ;;
  *)
    echo "Expected freeze-18 or full-transformer, got: ${variant}" >&2
    exit 2
    ;;
esac

[[ "${seed}" == "13" ]] || {
  echo "Only seed 13 is declared for the initial Arabic experiment." >&2
  exit 2
}
[[ " ${accents[*]} " == *" ${accent} "* ]] || {
  echo "Unknown accent: ${accent}" >&2
  exit 2
}

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"

config_path="experiments/joint/librispeech-100h/wav2vec2-large-lv60/utterance-supcon/${accent}/${variant}/config.yaml"

exec pixi run snakemake -s Snakefile joint_ctc_supcon_training \
  --configfile "${config_path}" \
  --config run_seed="${seed}" run_smoke="${smoke}" \
  --rerun-incomplete \
  --nolock \
  --cores 1

# Two independent smoke jobs (up to two GPUs concurrently):
# sbatch scripts/slurm/run_utterance_joint_training_100h.sbatch freeze-18 13 --smoke
# sbatch scripts/slurm/run_utterance_joint_training_100h.sbatch full-transformer 13 --smoke
#
# Two independent full jobs (up to two GPUs concurrently):
# sbatch scripts/slurm/run_utterance_joint_training_100h.sbatch freeze-18 13
# sbatch scripts/slurm/run_utterance_joint_training_100h.sbatch full-transformer 13
#
# Five remaining accents, sequentially within each variant (two GPUs maximum):
# sbatch --array=1-5%1 scripts/slurm/run_utterance_joint_training_100h.sbatch freeze-18 13
# sbatch --array=1-5%1 scripts/slurm/run_utterance_joint_training_100h.sbatch full-transformer 13
