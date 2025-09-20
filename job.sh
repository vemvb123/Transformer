#!/bin/bash

#SBATCH -A master
#SBATCH -p normal
#SBATCH --output=output_3.out

# Use 4 nodes
##SBATCH --nodes=4

# 8 tasks in total, 2 per node (since each node has 2 GPUs)
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1 
#SBATCH --nodelist=hpc2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4  # Adjust CPU allocation per task
#SBATCH -t 10-00:00:00
#SBATCH --gpu-bind=none




# Load environment
source ../env/bin/activate

# Set JAX distributed environment variables
# export NCCL_DEBUG=INFO
# export JAX_USE_PJRT_CUDA=0   # Force JAX to use multi-node setup
# export JAX_PLATFORMS="cuda"

# Run script with SLURM's MPI support
python train_3.py
# python train.py








