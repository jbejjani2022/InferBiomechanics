#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-00:02          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu              # Partition to submit to
#SBATCH --gres=gpu:2
#SBATCH --mem=1GB           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jbejjani@college.harvard.edu

# load modules
module load python/3.10.13-fasrc01
source activate inferbiomechanics

python test.py