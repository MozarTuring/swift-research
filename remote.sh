# SWIFT Experiment Configuration
# Adjust these parameters for different runs

# GPU allocation
# Note: JWM_GPU_NUM=0 to let slurm.sh handle GPU spec (cluster uses --gres not --gpus-per-node)
export JWM_GPU_NUM=1
export JWM_NODES_NUM=1
export JWM_RUN_TIME="1-00:01:00"  # 7 days max on alvis partition
export CPUS_PER_TASK=4  # CPUs per task for slurm job

# Optional: meaningful tag to distinguish runs under the same commit
# If unset, meta_script.sh auto-generates a timestamp
# export JWM_RUN_TAG="swift_baseline_llama3"

# SWIFT experiment specific settings
export SWIFT_EXPERIMENT_TYPE="baseline"  # baseline, rl_policy, comparison
export SWIFT_MODEL="meta-llama/Llama-3-8B"  # or mistralai/Mistral-7B
export SWIFT_RUNS=3  # number of repeats for averaging

# Paths
export SWIFT_LOGS="/mimer/NOBACKUP/groups/naiss2025-22-1056/swift_logs"
export SWIFT_VENV="/mimer/NOBACKUP/groups/naiss2025-22-1056/python3_9_6_envs/swift-research"

module purge
module load Python/3.9.6-GCCcore-11.2.0

# echo "start del env"
# rm -rf ${SWIFT_VENV}
# echo "env deleted"
# python3.9 -m venv $SWIFT_VENV
# source $SWIFT_VENV/bin/activate
# pip install --upgrade pip

pip install -r requirements.txt

if false; then
    ssh alvis1
    cd /Users/maojingwei/baidu/project/ && source common_tools/meta_script.sh alvis1  swift-research remote_slurm

    # Check GPU status on alvis1
    # ssh alvis1 "nvidia-smi"
    # Check slurm queue
    # ssh alvis1 "squeue --me"
fi

if false; then
    exit
    ssh alvis1

fi
