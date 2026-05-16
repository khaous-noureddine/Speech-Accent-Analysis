#!/bin/bash
#SBATCH -p GPU-H200
#SBATCH --account=efl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=evaluate
#SBATCH --output=logs/evaluation/%x-%j.out
#SBATCH --error=logs/evaluation/%x-%j.err

source /home/dist/noureddine.khaous/miniconda3/etc/profile.d/conda.sh
conda activate supcon




snakemake --configfile configs/evaluation/eval_config.yaml --cores 1 eval_transcribe > logs/evaluation/log_transcribe.log 2>&1 &  


# sbatch scripts/evaluate.sbatch    