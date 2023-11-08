#!/bin/bash
#
#SBATCH --job-name=model_training_run
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --partition=owners
#SBATCH --mail-type=ALL

ml python/3.9.0

TIMESTAMP=$(date +"%Y%m%d%H%M%S")
#python3 ../main.py train --model feedforward --checkpoint-dir "$GROUP_HOME/cvpr/checkpoint-$TIMESTAMP" --hidden-size 256 --learning-rate 0.01 --opt-type adagrad --dataset-home "$GROUP_HOME/data" --epochs 500
#python3 ../main.py train --model feedforward --checkpoint-dir "$GROUP_HOME/cvpr/checkpoint-20231027022502" --hidden-size 64 --learning-rate 0.01 --opt-type adagrad --dataset-home "$GROUP_HOME/data" --epochs 500
#python3 ../main.py train --model feedforward --checkpoint-dir "$GROUP_HOME/cvpr/checkpoint-$TIMESTAMP" --hidden-size 512 --learning-rate 0.001 --opt-type adagrad --dataset-home "$GROUP_HOME/data" --epochs 50
#python3 /home/users/jkaneda/InferBiomechanics/src/main.py train --model feedforward --checkpoint-dir "$GROUP_HOME/cvpr/checkpoint-$TIMESTAMP" --hidden-size 1024 --learning-rate 0.001 --opt-type adagrad --dataset-home "$GROUP_HOME/data" --epochs 15
python3 /home/users/jkaneda/InferBiomechanics/src/main.py train --model-type feedforward --checkpoint-dir "$GROUP_HOME/cvpr/checkpoint-$TIMESTAMP" --learning-rate 0.1 --dataset-home "$GROUP_HOME/data" --opt-type adam --trial-filter gait walk