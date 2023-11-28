#!/bin/bash
#
#SBATCH --job-name=model_training_run
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --partition=owners
#SBATCH --mail-type=ALL

ml python/3.9.0

python3 /home/users/jkaneda/InferBiomechanics/src/main.py analyze --model-type analytical --dataset-home "$GROUP_HOME/data" --checkpoint-dir "$GROUP_HOME/cvpr/checkpoint-20231030112236" --hidden-size 1024