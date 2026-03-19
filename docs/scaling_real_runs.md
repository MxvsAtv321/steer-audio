# SAE Scaling: Running Real ACE-Step Experiments

## Status

The SAE scaling law experiment (`experiments/sae_scaling.py`) is **fully implemented and
tested** using synthetic activations. All 55 tests pass and the complete output pipeline
(CSV, plots, LaTeX table) works on any CPU/MPS machine.

## Why Real Runs Require a Separate Environment

Real ACE-Step activation scaling requires generating activation caches with
`sae/sae_src/sae/cache_activations_runner_ace.py`. This script depends on `spacy==3.8.4`,
which is **incompatible with Python 3.13**. The primary development machine (MacBook Air
M4) runs Python 3.13 and therefore cannot install ACE-Step.

We deliberately avoid complex workarounds (vendoring ACE-Step, pinning an older Python,
or maintaining a parallel conda environment) to keep the repository environment simple and
stable. The synthetic dry-run mode provides a complete, tested code path that is
structurally identical to the real run.

## How to Generate Real Activations

On a machine with **Python 3.10–3.12** and a CUDA GPU:

```bash
# 1. Install the full environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements/requirements_1.txt
pip install -r requirements/requirements_2.txt --no-deps

# 2. Set the working directory
export TADA_WORKDIR=/path/to/tada_outputs

# 3. Cache activations from ACE-Step layer 7
python sae/sae_src/sae/cache_activations_runner_ace.py

# The cache is written to $TADA_WORKDIR/cache/layer7 (Arrow dataset format)
```

## Using the Cache on macOS / Python 3.13

Once `$TADA_WORKDIR/cache/layer7` exists (either generated locally or copied from the GPU
machine), pass `--activation-cache` to the scaling script:

```bash
export TADA_WORKDIR=/path/to/tada_outputs
python experiments/sae_scaling.py \
    --preset-real-small \
    --activation-cache $TADA_WORKDIR/cache/layer7 \
    --out-dir results/scaling_real_small
```

A convenience wrapper that checks for the cache first is available at:

```bash
bash scripts/run_sae_scaling_real_small.sh
```

## Available Run Modes (No Real Cache Required)

| Mode | Command | What it does |
|------|---------|--------------|
| Smoke test | `python experiments/sae_scaling.py --smoke-test` | Single config, ~5 s |
| Dry run (2×2) | `python experiments/sae_scaling.py --dry-run` | 12 configs, synthetic |
| Full dry-run grid | `python experiments/sae_scaling.py --dry-run --full-grid` | 300 configs, synthetic |
| Real small preset | `bash scripts/run_sae_scaling_real_small.sh` | 36 configs, real cache |

## Output Files

All runs write to `--out-dir` (default `results/scaling`):

- `all_results.csv` — full metric grid
- `fvu_vs_expansion.png` — FVU vs expansion factor (log-log)
- `alignment_vs_k.png` — ΔAlignment CLAP vs sparsity k
- `pareto_frontier.png` — Pareto-optimal configs on (FVU, dead%)
- `summary_table.md` — LaTeX-ready summary table
