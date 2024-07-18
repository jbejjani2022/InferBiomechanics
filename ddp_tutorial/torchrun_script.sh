#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-02:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_test              # Partition to submit to
#SBATCH --gres=gpu:3
#SBATCH --mem=50GB           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jbejjani@college.harvard.edu

# load modules
module load python/3.10.13-fasrc01
source activate inferbiomechanics

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
torchrun --nnodes=1 --nproc_per_node=3 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 elastic_ddp.py