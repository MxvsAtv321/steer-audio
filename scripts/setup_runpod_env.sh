#!/usr/bin/env bash
# setup_runpod_env.sh — Install and verify dependencies for steer-audio on RunPod.
# Usage: bash scripts/setup_runpod_env.sh

set -euo pipefail

VENV=/workspace/steer-audio/.venv

echo "=== steer-audio RunPod environment setup ==="
echo "Activating venv: $VENV"
# shellcheck disable=SC1090
. "$VENV/bin/activate"
echo "Python: $(which python) — $(python --version)"
echo ""

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
echo "=== Installing dependencies ==="
pip install --quiet soundfile scipy muq fadtk laion-clap librosa tqdm pandas numpy
echo ""

# ---------------------------------------------------------------------------
# Verify imports
# ---------------------------------------------------------------------------
echo "=== Verifying imports ==="

declare -A RESULTS

run_check() {
    local name="$1"
    local code="$2"
    if python -c "$code" 2>/dev/null; then
        RESULTS[$name]="PASS"
    else
        RESULTS[$name]="FAIL"
        # Re-run without suppressing stderr so the error is visible
        python -c "$code" || true
    fi
}

run_check "soundfile"  "import soundfile as sf; print('soundfile OK')"
run_check "scipy"      "import scipy.signal; print('scipy OK')"
run_check "muq"        "from muq import MuQMuLan; print('muq OK')"
run_check "fadtk"      "import fadtk; print('fadtk OK')"
run_check "laion_clap" "import laion_clap; print('laion_clap OK')"
run_check "librosa"    "import librosa; print('librosa OK')"
run_check "tqdm"       "import tqdm; print('tqdm OK')"
run_check "pandas"     "import pandas; print('pandas OK')"
run_check "numpy"      "import numpy; print('numpy OK')"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Summary ==="
FAILURES=0
for pkg in soundfile scipy muq fadtk laion_clap librosa tqdm pandas numpy; do
    status="${RESULTS[$pkg]:-FAIL}"
    printf "  %-14s %s\n" "$pkg" "$status"
    if [ "$status" = "FAIL" ]; then
        FAILURES=$((FAILURES + 1))
    fi
done

echo ""
if [ "$FAILURES" -eq 0 ]; then
    echo "All checks PASSED."
    exit 0
else
    echo "$FAILURES check(s) FAILED."
    exit 1
fi
