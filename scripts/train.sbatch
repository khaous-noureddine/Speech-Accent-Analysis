#!/bin/bash
#SBATCH -p GPU-H200
#SBATCH --account=efl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=supcon_train


source /home/dist/noureddine.khaous/miniconda3/etc/profile.d/conda.sh
conda activate supcon


snakemake --cores 1 -p checkpoints/v2/ > logs/training_v2.log 2>&1



# sbatch train.sbatch