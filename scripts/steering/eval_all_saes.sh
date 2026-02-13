#!/bin/bash

NUM_GPUS=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"

BASE_OUTPUT_DIR="steering/outputs"
LOG_DIR="logs/sae_protocol"

mkdir -p $LOG_DIR

# Define concepts and their eval prompts
declare -a CONCEPTS=(
    "piano|a piano song"
    "mood|a cheerful track"
    "tempo|a fast track"
    "female_vocals|This is a music of a female vocal singing"
    "drums|a drums song"
)

# Function to find the latest SAE output directory for a concept
find_latest_sae_output_dir() {
    local concept=$1
    local search_path="${BASE_OUTPUT_DIR}/${concept}/sae_*"

    # Find directories starting with sae_, sort by name (timestamp), get latest
    latest=$(ls -d ${search_path}/*/ 2>/dev/null | sort -r | head -1)
    if [ -n "$latest" ]; then
        # Remove trailing slash
        echo "${latest%/}"
    fi
}

# Build list of jobs
declare -a JOBS=()
job_idx=0

for concept_config in "${CONCEPTS[@]}"; do
    IFS='|' read -r concept eval_prompt <<< "$concept_config"

    output_dir=$(find_latest_sae_output_dir "$concept")

    if [ -z "$output_dir" ] || [ ! -d "$output_dir" ]; then
        echo "Warning: No SAE output found for concept=$concept, skipping..."
        continue
    fi

    # Check if alpha_0.0 exists (baseline required)
    if [ ! -d "${output_dir}/alpha_0.0" ]; then
        echo "Warning: No baseline (alpha_0.0) found in $output_dir, skipping..."
        continue
    fi

    # Build command (aesthetics_only mode - no eval_prompt needed)
    cmd="python steering/ace_steer/eval_steering_protocol.py --steering_dir \"$output_dir\" --eval_prompt \"$eval_prompt\""

    log_file="${LOG_DIR}/${concept}_sae_protocol.log"
    JOBS+=("$cmd|$log_file|$concept")
    job_idx=$((job_idx + 1))
done

echo "Found ${#JOBS[@]} SAE evaluation jobs to run"
echo "Using $NUM_GPUS GPUs"
echo ""

# Run jobs in parallel, distributing across GPUs
running_jobs=0
gpu_idx=2

for job_config in "${JOBS[@]}"; do
    IFS='|' read -r cmd log_file concept <<< "$job_config"

    gpu=$((gpu_idx % NUM_GPUS))

    echo "Starting: concept=$concept on GPU $gpu"
    echo "  Log: $log_file"

    CUDA_VISIBLE_DEVICES=$gpu eval $cmd > "$log_file" 2>&1 &

    gpu_idx=$((gpu_idx + 1))
    running_jobs=$((running_jobs + 1))

    # If we've launched NUM_GPUS jobs, wait for them to complete
    if [ $running_jobs -ge $NUM_GPUS ]; then
        echo "Waiting for batch of $NUM_GPUS jobs to complete..."
        wait
        running_jobs=0
        echo "Batch complete. Continuing..."
        echo ""
    fi
done

# Wait for remaining jobs
if [ $running_jobs -gt 0 ]; then
    echo "Waiting for final $running_jobs jobs to complete..."
    wait
fi

echo ""
echo "========================================="
echo "All SAE evaluation protocol jobs finished!"
echo "Results saved in respective output directories under 'protocol_results/'"
echo "Logs saved in $LOG_DIR"
echo "========================================="

# Print summary
echo ""
echo "Summary of results:"
for job_config in "${JOBS[@]}"; do
    IFS='|' read -r cmd log_file concept <<< "$job_config"
    output_dir=$(find_latest_sae_output_dir "$concept")
    summary_file="${output_dir}/protocol_results/summary.json"
done
