#!/bin/bash

STEER_MODE="cond_only"
NUM_GPUS=8  # Adjust based on available GPUs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"

BASE_OUTPUT_DIR="steering/outputs"
LOG_DIR="logs/protocol"

# Create log directory
mkdir -p $LOG_DIR

# Format: "concept|eval_prompt|negative_eval_prompt (optional)"
declare -a CONCEPTS=(
    "piano|a piano song|"
    "mood|a cheerful track|"
    "tempo|a fast track|"
    "female_vocals|This is a music of a female vocal singing|"
    "drums|a drums song|"
)

declare -a LAYERS_CONFIGS=(
    "all"
    "tf6"
    "tf7"
    "tf6tf7"
    "no_tf6tf7"
)

find_latest_output_dir() {
    local concept=$1
    local layers=$2
    local search_path="${BASE_OUTPUT_DIR}/${concept}/${STEER_MODE}"

    if [ -d "$search_path" ]; then
        latest=$(ls -d ${search_path}/${layers}_* 2>/dev/null | sort -r | head -1)
        echo "$latest"
    fi
}

# Build list of jobs
declare -a JOBS=()
job_idx=0

for concept_config in "${CONCEPTS[@]}"; do
    IFS='|' read -r concept eval_prompt negative_prompt <<< "$concept_config"

    for layers in "${LAYERS_CONFIGS[@]}"; do
        output_dir=$(find_latest_output_dir "$concept" "$layers")

        if [ -z "$output_dir" ] || [ ! -d "$output_dir" ]; then
            echo "Warning: No output found for concept=$concept, layers=$layers, skipping..."
            continue
        fi

        # Check if alpha_0.0 exists (baseline required)
        if [ ! -d "${output_dir}/alpha_0.0" ]; then
            echo "Warning: No baseline (alpha_0.0) found in $output_dir, skipping..."
            continue
        fi

        # Build command (aesthetics_only mode - no eval_prompt needed)
        cmd="python steering/ace_steer/eval_steering_protocol.py --steering_dir \"$output_dir\" --aesthetics_only"

        log_file="${LOG_DIR}/${concept}_${layers}_protocol.log"
        JOBS+=("$cmd|$log_file|$concept|$layers")
        job_idx=$((job_idx + 1))
    done
done

echo "Found ${#JOBS[@]} evaluation jobs to run"
echo "Using $NUM_GPUS GPUs"
echo ""

running_jobs=0
gpu_idx=0

for job_config in "${JOBS[@]}"; do
    IFS='|' read -r cmd log_file concept layers <<< "$job_config"

    gpu=$((gpu_idx % NUM_GPUS))

    echo "Starting: concept=$concept, layers=$layers on GPU $gpu"
    echo "  Log: $log_file"

    CUDA_VISIBLE_DEVICES=$gpu eval $cmd > "$log_file" 2>&1 &

    gpu_idx=$((gpu_idx + 1))
    running_jobs=$((running_jobs + 1))

    if [ $running_jobs -ge $NUM_GPUS ]; then
        echo "Waiting for batch of $NUM_GPUS jobs to complete..."
        wait
        running_jobs=0
        echo "Batch complete. Continuing..."
        echo ""
    fi
done

# Wait for any remaining jobs
if [ $running_jobs -gt 0 ]; then
    echo "Waiting for final $running_jobs jobs to complete..."
    wait
fi

echo ""
echo "========================================="
echo "All evaluation jobs finished!"
echo "Results saved in respective output directories under 'protocol_results/'"
echo "Logs saved in $LOG_DIR"
echo "========================================="

# Print summary of results
echo ""
echo "Summary of results:"
for job_config in "${JOBS[@]}"; do
    IFS='|' read -r cmd log_file concept layers <<< "$job_config"
    output_dir=$(find_latest_output_dir "$concept" "$layers")
    summary_file="${output_dir}/protocol_results/steering_summary.csv"

    if [ -f "$summary_file" ]; then
        echo ""
        echo "=== $concept / $layers ==="
        cat "$summary_file"
    fi
done
