#!/usr/bin/bash
#SBATCH -A naiss2026-4-5 -p alvis
#SBATCH --output=slurm_out.log
#SBATCH --error=slurm_out.log

# Load Python 3.9 module (required by SWIFT)
module load Python/3.9.6-GCCcore-11.2.0

# Activate SWIFT virtual environment
source /mimer/NOBACKUP/groups/naiss2025-22-1056/python3_9_6_envs/swift-research/bin/activate

# Print job info
echo "=========================================="
echo "SWIFT Experiment Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
echo "GPU IDs: $SLURM_GPU_IDS"
echo "=========================================="

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=$SLURM_GPU_IDS

# Check if remote.sh exists and source it
if [ ! -f remote.sh ]; then
    echo "ERROR: remote.sh not found. Copy remote.sh.example to remote.sh and edit it."
    exit 1
fi
source remote.sh

# Create logs directory if it doesn't exist
mkdir -p $SWIFT_LOGS

# Run SWIFT experiment
echo "Running SWIFT experiment: $SWIFT_EXPERIMENT_TYPE"
echo "Model: $SWIFT_MODEL"

# Main experiment script
python run_swift_baseline.py \
    --model $SWIFT_MODEL \
    --experiment-type $SWIFT_EXPERIMENT_TYPE \
    --num-runs $SWIFT_RUNS \
    --output-dir $SWIFT_LOGS/$SLURM_JOB_ID \
    --gpus $JWM_GPU_NUM

# Print completion message
echo "=========================================="
echo "SWIFT Experiment Completed"
echo "Results saved to: $SWIFT_LOGS/$SLURM_JOB_ID"
echo "=========================================="
