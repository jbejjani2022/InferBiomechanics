#!/bin/bash
#SBATCH --time=2-00:00
#SBATCH --mem=200G
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:4
#SBATCH -o output/jobid_%j.out
#SBATCH -e error/jobid_%j.err
#SBATCH --mail-user=camilobrownpinilla@college.harvard.edu
#SBATCH --mail-type=ALL

module load python/3.10.13-fasrc01
source activate paper_reimplementation_env
cd ../src

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr}-ib

torchrun --standalone --nproc_per_node=4 main.py train --model mdm --checkpoint-dir "../checkpoints/dffGsrV0/" --opt-type adam --dataset-home "/n/holyscratch01/pslade_lab/AddBiomechanicsDataset/addb_dataset" --data-loading-workers 0 --dropout --dropout-prob 0.1 --epochs 10 --batch-size 256 --learning-rate 8e-4 

