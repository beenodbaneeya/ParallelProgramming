cat run.sh
#!/bin/bash
#SBATCH --job-name=CUDA-test
#SBATCH --account=nn9997k
#SBATCH --time=05:00
#SBATCH --mem-per-cpu=1G
#SBATCH --qos=devel
#SBATCH --partition=accel
#SBATCH --gpus=1

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module load CUDA/12.4.0
module list

# Compile our code
nvcc vector_addition.cu -o vector_addition

# Setup monitoring
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory \
           --format=csv --loop=1 > "monitor-$SLURM_JOB_ID.csv" &
NVIDIA_MONITOR_PID=$!  # Capture PID of monitoring process

# Run our computation
./vector_addition

#After computation stop monitoring
kill -SIGINT "$NVIDIA_MONITOR_PID"
exit 0