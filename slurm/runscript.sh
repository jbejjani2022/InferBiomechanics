#!/bin/bash
#SBATCH --time=0-12:00
#SBATCH --mem=50G
#SBATCH -c 1
#SBATCH -p gpu_test
#SBATCH --gres=gpu:4
#SBATCH -o output/jobid_%j.out
#SBATCH -e error/jobid_%j.err
#SBATCH --mail-user=camilobrownpinilla@college.harvard.edu
#SBATCH --mail-type=ALL

module load python/3.10.13-fasrc01
source activate paper_reimplementation_env
cd ../src
python main.py train --model mdm --checkpoint-dir "../checkpoints/dffGsrV0" --opt-type adam --dataset-home "/n/holyscratch01/pslade_lab/AddBiomechanicsDataset/addb_dataset" --data-loading-workers 1 --dropout --dropout-prob 0.1 --epochs 300