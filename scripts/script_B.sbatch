#!/bin/bash
#SBATCH -p GPU-H200
#SBATCH --account=efl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=B_supcon


source /home/dist/noureddine.khaous/miniconda3/etc/profile.d/conda.sh
conda activate supcon
snakemake --configfile configs/conditions/config_B.yaml --cores 1 asr_finetuning > logs/asr_finetuning_aesrc.log 2>&1 & 

# sbatch scripts/script_B.sbatch