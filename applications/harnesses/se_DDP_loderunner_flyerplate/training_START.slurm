#!/bin/bash

#SBATCH --account=y26_artimis-fmod_g
#SBATCH --partition=standard
#SBATCH --qos=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=120G

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$(shuf -i 1024-65535 -n 1)
export MASTER_ADDR
export MASTER_PORT

sinfo -o "%P %.24G %N"
srun -vv --cpu-bind=verbose /usr/bin/echo $CUDA_AVAILABLE_DEVICES
nvidia-smi

export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3
export SLURM_CPU_BIND=verbose

source /usr/projects/artimis/mpmm/ryanw/Yoke/.venv/bin/activate

export OMP_NUM_THREADS=10

srun -vv --cpu-bind=verbose python -u train_LodeRunner_ddp_flyerplate.py @study001_START.input
