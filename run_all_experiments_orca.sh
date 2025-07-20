#!/bin/bash

# This script automates the submission of Slurm jobs for a series of experiments.
# It acts as a queue manager to ensure a maximum number of concurrent jobs.
#
# - It iterates through all 120 possible jobs (40 configs x 3 runs).
# - Before submitting a new job, it checks if the user's job count is below the max limit.
# - If the queue is full, it waits and checks again until a slot is free.

# --- Configuration ---
MAX_JOBS=6
LOG_DIR="slurm_logs"
PYTHON_SCRIPT="main_concrete_baseline.py"
CONDA_ENV="torch"

# --- Create a directory for Slurm logs ---
mkdir -p "$LOG_DIR"

# --- Define iteration parameters ---
MODEL_ARCHS=(1 0)         # 1=VGG, 0=ResNet
SEARCH_DURS=(1 0)         # 1=Short, 0=Long
TRAIN_SCHEMES=(1 0)       # 1=Init, 0=Rewind
PRUNE_METHODS=(0 1 2 3 4) # 5 different pruning criteria
RUN_TAGS=(f s t)          # 3 replicate runs for each configuration

# --- Main loop for generating all 120 jobs ---
for method in "${PRUNE_METHODS[@]}"; do
  for scheme in "${TRAIN_SCHEMES[@]}"; do
    for duration in "${SEARCH_DURS[@]}"; do
      for arch in "${MODEL_ARCHS[@]}"; do
        for tag in "${RUN_TAGS[@]}"; do

          # --- QUEUE MANAGEMENT ---
          # This loop waits until there is a free slot in the queue.
          while true; do
            # Get the number of jobs currently running or pending for this user.
            current_jobs=$(squeue -u "$USER" -h | wc -l)
            if [[ "$current_jobs" -lt "$MAX_JOBS" ]]; then
              echo "Queue has space ($current_jobs/$MAX_JOBS jobs active). Proceeding."
              break # Exit the waiting loop
            else
              echo "Queue is full ($current_jobs/$MAX_JOBS jobs active). Waiting 30s..."
              sleep 30 # Wait before checking again
            fi
          done

          # --- Construct job name and options string ---
          options_str="$method,$scheme,$duration,$arch"
          job_name="concrete_evaluation_${tag}_${options_str//,/_}"
          log_file="$LOG_DIR/${job_name}.log"

          echo "--- Submitting job: $job_name ---"

          # --- Submit job ---
          sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --output=${log_file}

echo "Starting Slurm Job \$SLURM_JOB_ID: ${job_name}"
echo "Running on node: \$(hostname)"
echo "Timestamp: \$(date)"

# Activate the correct Conda environment
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Execute the python script
python -u ${PYTHON_SCRIPT} "concrete_evaluation_${tag}" "${options_str}" 1

echo "Job finished with exit code \$?"
EOF
          # Small pause to allow the Slurm controller to register the new job
          sleep 1

        done
      done
    done
  done
done

echo "============================================================"
echo "All 120 jobs have been submitted to the Slurm queue."
echo "The script will now exit. Use 'squeue -u \$USER' to monitor progress."
echo "============================================================"