#!/usr/bin/env bash
# run_sae_scaling_real_small.sh
#
# Run the SAE scaling experiment on real ACE-Step activations (small preset).
#
# Prerequisites:
#   - $TADA_WORKDIR/cache/layer7 must exist (Arrow dataset produced by
#     sae/sae_src/sae/cache_activations_runner_ace.py on a Python 3.10-3.12 machine).
#
# See docs/scaling_real_runs.md for full instructions.

set -euo pipefail

# Default TADA_WORKDIR to the repository root if not set.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TADA_WORKDIR="${TADA_WORKDIR:-$REPO_ROOT}"

CACHE_DIR="$TADA_WORKDIR/cache/layer7"

if [ ! -d "$CACHE_DIR" ]; then
    echo ""
    echo "ERROR: Activation cache not found at: $CACHE_DIR"
    echo ""
    echo "To generate it, run the following on a machine with Python 3.10-3.12 and a GPU:"
    echo ""
    echo "  export TADA_WORKDIR=$TADA_WORKDIR"
    echo "  python sae/sae_src/sae/cache_activations_runner_ace.py"
    echo ""
    echo "Then copy the resulting directory to: $CACHE_DIR"
    echo "See docs/scaling_real_runs.md for full instructions."
    echo ""
    exit 1
fi

echo "Found activation cache at: $CACHE_DIR"
echo "Starting SAE scaling experiment (real small preset)..."

python "$REPO_ROOT/experiments/sae_scaling.py" \
    --preset-real-small \
    --activation-cache "$CACHE_DIR" \
    --out-dir "$REPO_ROOT/results/scaling_real_small"
