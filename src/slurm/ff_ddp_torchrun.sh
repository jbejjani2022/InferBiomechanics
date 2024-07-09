#!/bin/bash
#SBATCH --time=2-00:00
#SBATCH --mem=200GB
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:4
#SBATCH -o output/jobid_%j.out
#SBATCH -e error/jobid_%j.err
#SBATCH --mail-user=jbejjani@college.harvard.edu
#SBATCH --mail-type=ALL

module load python/3.10.13-fasrc01
source activate inferbiomechanics
cd ..

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr-ib
export WANDB_RUN_GROUP=$SLURM_JOB_ID

torchrun --standalone --nproc_per_node=4 main.py train --model feedforward --checkpoint-dir "../checkpoints/feedforward/ddp-bs512-hd1024-512-256-lr1e-3" --opt-type adam --dataset-home "/n/holyscratch01/pslade_lab/AddBiomechanicsDataset/addb_dataset" --data-loading-workers 1 --batchnorm --dropout --dropout-prob 0.5 --activation tanh --hidden-dims 1024 512 256 --epochs 50 --batch-size 512 --learning-rate 1e-3
