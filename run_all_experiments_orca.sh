#!/bin/bash

# This script automates the submission of Slurm jobs for a series of experiments.
# It iterates through all 40 possible argument combinations for the Python script.
#
# For each combination, it queues 3 runs (f, s, t). It waits for each job
# to finish before submitting the next. If a job finishes with exit code 1,
# it skips the remaining runs for that combination and moves to the next set.

# --- Configuration ---
LOG_DIR="slurm_logs"
PYTHON_SCRIPT="main_concrete_baseline.py"
CONDA_ENV="torch"

# --- Create a directory for Slurm logs ---
mkdir -p "$LOG_DIR"

# --- Define iteration parameters ---
# These arrays define the parameter space for the experiment.
MODEL_ARCHS=(1 0)         # 1=VGG, 0=ResNet
SEARCH_DURS=(1 0)         # 1=Short, 0=Long
TRAIN_SCHEMES=(1 0)       # 1=Init, 0=Rewind
PRUNE_METHODS=(0 1 2 3 4) # 5 different pruning criteria
RUN_TAGS=(f s t)          # 3 replicate runs for each configuration

# --- Main loop for iterating through all 40 configurations ---
for method in "${PRUNE_METHODS[@]}"; do
  for scheme in "${TRAIN_SCHEMES[@]}"; do
    for duration in "${SEARCH_DURS[@]}"; do
      for arch in "${MODEL_ARCHS[@]}"; do

        options_str="$method,$scheme,$duration,$arch"
        echo "============================================================"
        echo "Starting new configuration set: $options_str"
        echo "(Method, Scheme, Duration, Arch)"
        echo "============================================================"

        # This inner loop handles the three runs (f, s, t) for the current configuration.
        for tag in "${RUN_TAGS[@]}"; do

          # --- Construct job name and options string ---
          job_name="concrete_evaluation_${tag}_${options_str//,/_}"
          log_file="$LOG_DIR/${job_name}.log"

          echo "--- Preparing job: $job_name ---"
          
          # --- Submit job and capture its ID ---
          # We use a "here document" to pass the script to sbatch.
          job_id=$(sbatch <<EOF | awk '{print $4}'
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
# The '-u' flag ensures unbuffered output, which is good for live logging.
python -u ${PYTHON_SCRIPT} "concrete_evaluation_${tag}" "${options_str}" 1

echo "Job finished with exit code \$?"
EOF
)
          # Check if the job submission was successful
          if [[ -z "$job_id" ]]; then
            echo "ERROR: Failed to submit job for configuration: ${options_str}, tag: ${tag}. Moving on."
            # Break the inner loop to move to the next configuration set
            break
          fi

          echo "Submitted job ${job_id}. Waiting for completion..."

          # --- Wait for the job to complete ---
          # This loop polls squeue until the job ID is no longer present.
          while squeue -j "${job_id}" 2>/dev/null | grep -q "${job_id}"; do
            sleep 30
          done

          # --- Check the exit code of the completed job ---
          # We use `sacct` to retrieve the final exit code.
          # The format can be 'ExitCode:Signal', so we parse just the code.
          exit_code=$(sacct -j "${job_id}" --format=ExitCode --noheader | head -n 1 | awk -F: '{print $1}')

          echo "Job ${job_id} finished with exit code: ${exit_code}"

          # --- Handle the specific exit code condition ---
          # If the exit code is exactly 1, break this inner loop and
          # move to the next configuration of 4 arguments.
          if [[ "${exit_code}" == "1" ]]; then
            echo "Exit code is 1. Skipping remaining runs for this configuration."
            break # Breaks out of the 'for tag in (f s t)' loop
          fi
        done
      done
    done
  done
done

echo "============================================================"
echo "All configurations have been processed. Script finished."
echo "============================================================"