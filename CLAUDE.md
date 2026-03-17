# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research code for the TADA paper: "Tuning Audio Diffusion Models through Activation Steering." The project analyzes and controls audio diffusion models (ACE-Step, AudioLDM2, Stable Audio) via three core techniques:
1. **Activation Patching** — localizes where semantic concepts (drums, tempo, mood) are processed
2. **CAA Steering Vectors** — computes and applies contrastive activation addition vectors at inference time
3. **Sparse Autoencoders (SAEs)** — learns interpretable feature decompositions of model activations

## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements/requirements_1.txt
pip install -r requirements/requirements_2.txt --no-deps
```

## Common Commands

**Activation Patching (localization):**
```bash
accelerate launch --num_processes 4 src/patch_layers.py \
    experiment=patch_ace/ace_drums \
    patch_layers=ace/tf7 \
    patch_config.path_with_results=outputs/ace/patching/drums/tf7
```

**Compute CAA Steering Vectors:**
```bash
python steering/ace_steer/compute_steering_vectors_caa.py \
    --concept piano \
    --num_inference_steps 30 \
    --save_dir steering_vectors
```

**Evaluate Steering Vectors:**
```bash
python steering/ace_steer/eval_steering_vectors.py \
    --sv_path steering_vectors/ace_piano_passes2_allTrue \
    --concept piano \
    --layers tf7 \
    --steer_mode cond_only
```

**Cache Activations + Train SAE:**
```bash
python sae/sae_src/sae/cache_activations_runner_ace.py
python sae/sae_src/scripts/train_ace.py
```

## Architecture

### Data Flow

```
Prompts → Activation Patching (src/) → localized layers
                                          ↓
                       CAA Steering (steering/) → steering vectors
                                          ↓
                          SAE Training (sae/) → interpretable features
                                          ↓
                              Controlled Audio Generation
```

### Key Components

**`src/`** — Activation patching pipeline
- `patch_layers.py` — Main entry point; uses Hydra config + Accelerate for distributed runs
- `models/ace_step/` — ACE-Step wrappers: `patchable_ace.py` (NNSight-based patching), `ace_steering/controller.py` (VectorStore hook interface)
- `models/audioldm/`, `models/stable_audio/` — Analogous wrappers for other models
- `metrics/` — FAD, CLAP, MUQ-T, alignment evaluation

**`steering/ace_steer/`** — CAA steering
- `compute_steering_vectors_caa.py` — Collects activations from positive/negative prompt pairs, saves difference vectors
- `eval_steering_vectors.py` — Evaluates steering quality across alpha values and layer configs
- `prompts.py` — Concept-to-prompt mappings for steering concepts

**`sae/sae_src/`** — SAE pipeline
- `sae/sae.py` — Encoder/decoder model with sparse activations
- `sae/trainer.py` — Distributed training loop
- `hooked_model/` — Model-specific activation hooks for ACE-Step, AudioLDM2, Stable Audio
- `configs/steer_prompts.py` — Concept prompt definitions

**`configs/`** — Hydra YAML configs organized by `patch_model/`, `patch_config/`, `patch_layers/`, `patch_data/`

### Configuration System

The entire patching pipeline is controlled by Hydra. Root configs are `configs/generate_audio_patch.yaml` and `configs/eval_audio.yaml`. Experiment variants (e.g., `experiment=patch_ace/ace_drums`) compose from multiple config groups.

### Supported Models / Concepts / Modes

- **Models:** ACE-Step, AudioLDM2, Stable Audio
- **Concepts:** drums, fast, female, happy, male, sad, slow, violin (plus others in `steer_prompts.py`)
- **Layers:** `tf6`, `tf7`, `tf6tf7`, `all`, `no_tf6tf7`
- **Steering modes:** `cond_only`, `separate`, `both_cond`, `uncond_only`
- **Alpha range:** `[-100, -90, ..., 100]`

---

## Working Style for Claude Code

When editing this repository, follow these rules:

- Read files before changing them; preserve public interfaces unless explicitly told otherwise.
- Prefer small, focused diffs that correspond to a single task.
- For any non-trivial change, add or update tests under `tests/` and ensure `pytest` passes.
- Do not hardcode paths; use `pathlib.Path` and environment variables (e.g., `TADA_WORKDIR`).
- Code should run on CPU but prefer GPU when available.
- Use type hints and docstrings on new Python functions.

When I say "don't ask for clarification for this task," pick the simplest reasonable assumption, document it in a comment or TODO, and continue. Only stop if you hit a hard blocker (missing file, unreadable config, missing dependency).

## Roadmap Integration

There is an additional file, `TADA_Build_Roadmap.md`, that defines detailed multi-week prompts (Phase 1–3).

When I reference a prompt, for example:

> Execute Phase 1, Prompt 1.1 from `TADA_Build_Roadmap.md` in this session.

You should:

1. Open `TADA_Build_Roadmap.md` and locate that prompt.
2. Follow its instructions exactly, using the "Working Style" above.
3. Apply changes directly without waiting for my per-file approval.
4. At the end of the session, summarize:
   - Files changed
   - Commands I should run (e.g., `pytest ...`, `python ...`) to verify.

If I ask you to execute several prompts in sequence (e.g., "Phase 1, Prompts 1.1–1.3"), complete them in order in a single session and give one final summary at the end.
