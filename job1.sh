#!/bin/bash

#SBATCH --job-name=test01
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --time=02:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.e

module purge
module load anaconda3
source activate pytorch_env

export CUDA_LAUNCH_BLOCKING=1

python ./run.py

python ./run_eval.py
