#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 2-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_requeue              # Partition to submit to
#SBATCH --gres=gpu:4
#SBATCH --mem=200GB           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=jbejjani@college.harvard.edu
#SBATCH --mail-type=ALL

# load modules
module load python/3.10.13-fasrc01
source activate paper_reimplementation_env
cd ../src

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export NCCL_DEBUG=INFO
torchrun --standalone --nproc_per_node=4 main.py train --model feedforward --checkpoint-dir "../checkpoints/ddp-batch-size-128" --hidden-dims 512 512 --batchnorm --dropout --dropout-prob 0.5 --activation tanh --learning-rate 0.01 --opt-type adam --dataset-home "/n/holyscratch01/pslade_lab/AddBiomechanicsDataset/addb_dataset" --epochs 300 --batch-size 128 --data-loading-workers 8