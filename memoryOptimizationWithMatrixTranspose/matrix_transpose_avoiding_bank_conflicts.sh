#!/bin/bash -l
#SBATCH --job-name=matrix_transpose_avoidbc
#Project account
#SBATCH --account=nn9997k
#SBATCH --output=output_avoidbc.log
#SBATCH --error=error_avoidbc.log

#SBATCH --time=0:10:00
#SBATCH --partition=accel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --gpus=1
## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error
module --quiet purge  # Reset the modules to the system default
module load CUDA/12.4.0
module list  # List loaded modules for debugging


# Start GPU monitoring in the background
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv -l 1 > gpu_usage_avoidbc.csv & 
GPU_MONITOR_PID=$!

#directory where the program is located
cd /cluster/work/users/bba065/cudaprograms/memoryOptimizationWithMatrixTranspose

#complie the CUDA program
nvcc -arch=sm_60  matrix_transpose_avoiding_bank_conflicts.cu -o matrix_transpose_avoiding_bank_conflicts
# Run the script
./matrix_transpose_avoiding_bank_conflicts

#Stop GPU monitoring
kill $GPU_MONITOR_PID%