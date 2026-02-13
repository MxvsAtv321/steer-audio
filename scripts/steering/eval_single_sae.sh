#!/bin/bash

# SAE steering evaluation script with timestamped output directories
# Usage:
#   ./helios_run_sae.sh                    # Auto-generate timestamp
#   ./helios_run_sae.sh 20260211143000     # Use specific timestamp

# =============================================================================
# CONFIGURATION - Modify these parameters as needed
# =============================================================================

# SAE model
SAE_PATH="<SAE_PATH>/transformer_blocks.7.cross_attn"
SAE_TYPE="sequence"

# Config source: true = load from {concept}_best_config.pkl, false = use params below
USE_CONFIG_FILE=false

# Feature selection (used when USE_CONFIG_FILE=false)
SELECTION_METHOD="tfidf"    # tfidf, diff, mean_pos, cohens_d, linear
TOP_K=20                    # Number of top features per timestep

# Weighting (used when USE_CONFIG_FILE=false, or set SKIP_WEIGHTING=true to override config)
SKIP_WEIGHTING=true         # true = simple feature selection, false = use weighting
WEIGHTING="none"            # none, raw, softmax, sqrt, log
WEIGHT_SOURCE=""            # tfidf, diff, mean_pos (empty = same as SELECTION_METHOD)

# Which concepts to run
RUN_PIANO=true
RUN_MOOD=false
RUN_TEMPO=false
RUN_FEMALE_VOCALS=false

# =============================================================================
# END CONFIGURATION
# =============================================================================

# Get timestamp (from argument or generate)
if [ -n "$1" ]; then
    TIMESTAMP="$1"
else
    TIMESTAMP=$(date +%Y%m%d%H%M%S)
fi

echo "=== SAE Steering Evaluation ==="
echo "Timestamp: $TIMESTAMP"
echo ""
echo "Configuration:"
echo "  SAE type: $SAE_TYPE"
if [ "$USE_CONFIG_FILE" = true ]; then
    echo "  Using config files from: steering/svs/sae_*/{concept}/{concept}_best_config.pkl"
else
    echo "  Selection: $SELECTION_METHOD"
    echo "  Top-k: $TOP_K"
fi
echo "  Skip weighting: $SKIP_WEIGHTING"
if [ "$SKIP_WEIGHTING" = false ] && [ "$USE_CONFIG_FILE" = false ]; then
    echo "  Weighting: $WEIGHTING"
    echo "  Weight source: ${WEIGHT_SOURCE:-$SELECTION_METHOD}"
fi
echo ""

# Base directories
SVS_DIR="steering/svs/sae_${TIMESTAMP:0:8}"  # Use date portion for svs
OUTPUT_BASE="steering/outputs/sae_${TIMESTAMP}"
LOG_DIR="logs/sae_steering/${TIMESTAMP}"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_BASE"

echo "Output: $OUTPUT_BASE"
echo "Logs: $LOG_DIR"
echo ""

# Function to run evaluation for a concept
run_concept() {
    local concept=$1

    local output_dir="${OUTPUT_BASE}/${concept}_${TIMESTAMP}"
    local config_path="${SVS_DIR}/${concept}/${concept}_best_config.pkl"
    local scores_path_all="${SVS_DIR}/${concept}/${concept}_all_scores.pkl"
    local scores_path_simple="${SVS_DIR}/${concept}/${concept}_scores.pkl"

    # Build command
    local cmd="python sae/scripts/eval_sae_steering.py"
    cmd="$cmd --concept $concept"
    cmd="$cmd --sae_path \"$SAE_PATH\""
    cmd="$cmd --sae_type $SAE_TYPE"
    cmd="$cmd --save_dir \"$output_dir\""

    # Config source
    if [ "$USE_CONFIG_FILE" = true ] && [ -f "$config_path" ]; then
        cmd="$cmd --config_path \"$config_path\""
        echo "[$concept] Using config file: $config_path"
    else
        cmd="$cmd --selection_method $SELECTION_METHOD"
        cmd="$cmd --top_k $TOP_K"

        if [ "$SKIP_WEIGHTING" = false ]; then
            cmd="$cmd --weighting $WEIGHTING"
            if [ -n "$WEIGHT_SOURCE" ]; then
                cmd="$cmd --weight_source $WEIGHT_SOURCE"
            fi
        fi
        echo "[$concept] Using params: method=$SELECTION_METHOD, k=$TOP_K"
    fi

    # Skip weighting (can override config file too)
    if [ "$SKIP_WEIGHTING" = true ]; then
        cmd="$cmd --skip_weighting"
    fi

    # Use cached scores if available
    if [ -f "$scores_path_all" ]; then
        cmd="$cmd --scores_cache_path \"$scores_path_all\""
        echo "[$concept] Using cached scores: $scores_path_all"
    elif [ -f "$scores_path_simple" ]; then
        cmd="$cmd --scores_cache_path \"$scores_path_simple\""
        echo "[$concept] Using cached scores: $scores_path_simple"
    else
        echo "[$concept] No cached scores found, will compute from scratch"
    fi

    echo "[$concept] Output: $output_dir"
    echo "[$concept] Running..."

    eval "$cmd" > "${LOG_DIR}/${concept}.log" 2>&1
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[$concept] Done!"
    else
        echo "[$concept] FAILED (exit: $exit_code). Check: ${LOG_DIR}/${concept}.log"
    fi
    echo ""
}

# Run selected concepts
echo "=== Running Evaluations ==="
[ "$RUN_PIANO" = true ] && run_concept "piano"
[ "$RUN_MOOD" = true ] && run_concept "mood"
[ "$RUN_TEMPO" = true ] && run_concept "tempo"
[ "$RUN_FEMALE_VOCALS" = true ] && run_concept "female_vocals"

echo "=== Complete ==="
echo "Results: $OUTPUT_BASE"
echo "Logs: $LOG_DIR"
