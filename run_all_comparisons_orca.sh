#!/bin/bash

# --- Configuration ---
MAX_JOBS=6
LOG_DIR="slurm_logs"
PYTHON_SCRIPT="main_imp_comparison.py"
CONDA_ENV="torch"

# --- Create a directory for Slurm logs ---
mkdir -p "$LOG_DIR"

# --- Define iteration parameters ---
MODEL_ARCHS=(1 0)         # 1=VGG, 0=ResNet
PRUNE_METHODS=(0 1 2 3 4 5) # 6 pruning criteria
RUN_TAGS=(f s t)          # 3 replicate runs

# --- Main loop for generating all 240 jobs ---
for method in "${PRUNE_METHODS[@]}"; do
  for arch in "${MODEL_ARCHS[@]}"; do
    for tag in "${RUN_TAGS[@]}"; do

      # --- QUEUE MANAGEMENT ---
      while true; do
        current_jobs=$(squeue -u "$USER" -h | wc -l)
        if [[ "$current_jobs" -lt "$MAX_JOBS" ]]; then
          echo "Queue has space ($current_jobs/$MAX_JOBS jobs active). Proceeding."
          break
        else
          echo "Queue is full ($current_jobs/$MAX_JOBS jobs active). Waiting 60s..."
          sleep 60
        fi
      done

      # --- Construct job name and options ---
      options_str="$method,$arch"
      job_name="imp_backdrop_${tag}_${options_str//,/_}"
      log_file="$LOG_DIR/${job_name}.log"

      echo "--- Submitting job: $job_name ---"

      # --- Submit job using sbatch --wrap ---
      sbatch --job-name="${job_name}" \
              --gres=gpu:4 \
              --cpus-per-gpu=4 \
              --output="${log_file}" \
              --wrap="source \$(conda info --base)/etc/profile.d/conda.sh && \
                      conda activate ${CONDA_ENV} && \
                      echo 'Starting Slurm Job \$SLURM_JOB_ID: ${job_name}' && \
                      echo 'Running on node: \$(hostname)' && \
                      echo 'Timestamp: \$(date)' && \
                      python -u ${PYTHON_SCRIPT} 'imp_backdrop_${tag}' '${options_str}' 1 && \
                      echo 'Job finished with exit code \$?'"

      sleep 1
    done
  done
done

echo "============================================================"
echo "All 33 jobs have been submitted to the Slurm queue."
echo "Use 'squeue -u \$USER' to monitor job progress."
echo "============================================================"