#!/bin/bash
#
#SBATCH --job-name=model_training_run
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --partition=owners

ml python/3.9.0

TIMESTAMP=$(date +"%Y%m%d%H%M%S")
python3 main.py train --model feedforward --checkpoint-dir "$GROUP_HOME/cvpr/checkpoint-$TIMESTAMP" --hidden-size 512 --learning-rate 0.01 --opt-type adagrad --dataset-home "$GROUP_HOME/data" --epochs 500
