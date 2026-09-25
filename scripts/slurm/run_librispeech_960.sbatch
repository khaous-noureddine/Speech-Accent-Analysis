#!/usr/bin/env bash
#SBATCH --partition=GPU-H200
#SBATCH --account=efl
#SBATCH --job-name=ls960-pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/slurm/ls960-%x-%j.out

set -euo pipefail

objective="${1:-joint-supcon}"
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
case "${objective}" in
  ctc-only|joint-supcon) ;;
  *) echo "Expected ctc-only or joint-supcon, got: ${objective}" >&2; exit 2 ;;
esac
[[ "${seed}" == "13" ]] || { echo "Pilot config declares only seed 13." >&2; exit 2; }
[[ " ${accents[*]} " == *" ${accent} "* ]] || {
  echo "Unknown accent: ${accent}" >&2
  exit 2
}
if [[ "${objective}" == "ctc-only" && "${accent}" != "arabic" ]]; then
  echo "CTC-only is fold-independent and is stored once under arabic." >&2
  exit 2
fi

repository_root="${SLURM_SUBMIT_DIR:?Run sbatch from the repository root}"
cd "${repository_root}"
config_path="experiments/scaling/librispeech-960/wav2vec2-large-lv60/${accent}/${objective}/config.yaml"

exec pixi run snakemake -s Snakefile joint_ctc_supcon_training \
  --configfile "${config_path}" \
  --config run_seed="${seed}" run_smoke="${smoke}" \
  --rerun-incomplete --nolock --cores 1

# sbatch scripts/slurm/run_librispeech_960.sbatch ctc-only 13 --smoke
# sbatch scripts/slurm/run_librispeech_960.sbatch joint-supcon 13 --smoke
# sbatch scripts/slurm/run_librispeech_960.sbatch ctc-only 13
# sbatch scripts/slurm/run_librispeech_960.sbatch joint-supcon 13
# Five remaining joint folds, sequentially with at most one H200 GPU:
# sbatch --array=1-5%1 scripts/slurm/run_librispeech_960.sbatch joint-supcon 13
