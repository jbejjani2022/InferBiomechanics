#!/bin/bash
#
#SBATCH --job-name=make_plots_run
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --partition=owners

ml python/3.9.0

python3 /home/users/jkaneda/InferBiomechanics/src/main.py make-plots --data-path "/home/groups/delp/data" --out-path "/home/groups/delp/jkaneda/figures" --output-histograms