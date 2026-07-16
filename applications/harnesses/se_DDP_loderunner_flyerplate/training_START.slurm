#!/bin/bash
#SBATCH --job-name=flyer_smoke_s001_e0001
#SBATCH --account=xd
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=study001_epoch0001.out
#SBATCH --error=study001_epoch0001.err
#SBATCH -vvv

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$(shuf -i 1024-65535 -n 1)
export MASTER_ADDR
export MASTER_PORT

source ../../../.venv/bin/activate

export OMP_NUM_THREADS=8

srun python -u train_LodeRunner_ddp_flyerplate.py @study001_START.input
