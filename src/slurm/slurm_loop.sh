#!/bin/bash
#
#SBATCH --job-name=model_training_run
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --partition=owners

ml python/3.9.0

# Define an array of hidden sizes and learning rates
hidden_sizes=(64 128 256 512 1024)
learning_rates=(0.01 0.001 0.0001)

# Loop through combinations of hidden sizes and learning rates
for hidden_size in "${hidden_sizes[@]}"; do
  for learning_rate in "${learning_rates[@]}"; do
    TIMESTAMP=$(date +"%Y%m%d%H%M%S")
    python3 /home/users/jkaneda/InferBiomechanics/src/main.py train --model feedforward --checkpoint-dir "$GROUP_HOME/cvpr/checkpoint-$TIMESTAMP" --hidden-size "$hidden_size" --learning-rate "$learning_rate" --opt-type adagrad --dataset-home "$GROUP_HOME/data" --epochs 15
  done
done
