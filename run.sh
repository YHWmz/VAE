#!/bin/bash

#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
STORE_PATH=./images/result/
mkdir $STORE_PATH
python train.py --store_path $STORE_PATH

