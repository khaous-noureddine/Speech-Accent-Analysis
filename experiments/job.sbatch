#!/bin/bash
#SBATCH -p GPU-H200
#SBATCH --account=efl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=supcon-dtw

source /home/dist/noureddine.khaous/miniconda3/etc/profile.d/conda.sh
conda activate supcon


# snakemake --configfile experiments/baseline-models/configs/config.yaml --cores 1 --latency-wait 30 eval_transcribe > experiments/baseline-models/logs/eval_transcribe.log 2>&1 &
# snakemake --configfile experiments/baseline-models/configs/config.yaml --cores 1 --latency-wait 30 eval_compute_wer > experiments/baseline-models/logs/eval_compute_wer.log 2>&1 &

# snakemake --configfile experiments/cross-validation/fold_00/configs/b.yaml \
#   --cores 1 --latency-wait 30 asr_finetuning > experiments/cross-validation/fold_00/B/logs/stage3_finetune.log 2>&1 &

# snakemake --configfile experiments/cross-validation/stage3/configA.yaml --cores 1 --latency-wait 30 asr_finetuning > experiments/cross-validation/fold_00/logs/stage3_finetune.log 2>&1 &
# snakemake --configfile experiments/cross-validation/fold_00/configs/b.yaml \
#   --cores 1 --latency-wait 30 asr_finetuning > experiments/cross-validation/fold_00/B/logs/stage3_finetune.log 2>&1 &


# snakemake --configfile experiments/cross-validation/fold_00/configs/c.yaml \
#   --cores 1 --latency-wait 30 asr_finetuning > experiments/cross-validation/fold_00/C/logs/stage3_finetune.log 2>&1 &




snakemake   --configfile experiments/stage2-dtw/fold_00/configs/stage2-dtw.yaml   --cores 1   --latency-wait 30   stage2_supcon_dtw_l2cv   > experiments/stage2-dtw/fold_00/logs/snakemake_stage2_dtw.log 2>&1 &
  
  
  
  # sbatch experiments/job.sbatch