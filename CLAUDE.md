# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Title:** "When Does Audio Diffusion Commit? Compositional Concept Steering via Timestep-Aware Activation Algebra"
**Venue:** ISMIR 2026. Abstract lock: April 20, 2026. Paper deadline: April 27, 2026.
**Repo:** github.com/MxvsAtv321/steer-audio (fork of luk-st/steer-audio, TADA's repo)

Research code for the TADA paper. Analyzes and controls audio diffusion models (ACE-Step, AudioLDM2, Stable Audio) via three core techniques:
1. **Activation Patching** — localizes where semantic concepts (drums, tempo, mood) are processed
2. **CAA Steering Vectors** — computes and applies contrastive activation addition vectors at inference time
3. **Sparse Autoencoders (SAEs)** — learns interpretable feature decompositions of model activations

**Primary finding:** Exp 2 shows concept-dependent commitment timing in audio DiTs, independently validating PCI (Gorgun et al., arXiv:2512.08486, ICLR 2026) which found the same in image diffusion. Key advantage over PCI: internal activations (mechanistic) vs external prompts (behavioral).

## CRITICAL: Never Run Generation Experiments Locally

All generation experiments (anything calling ACE-Step or running steering) must run on the RunPod A40 at `/workspace/steer-audio`. Local machine has no GPU. Only run locally:
- Python syntax checks (`python -m py_compile`)
- Unit tests that don't need model weights
- Code edits and commits

## Environment

| Location | Path |
|----------|------|
| RunPod repo | `/workspace/steer-audio` |
| Local repo | `/Users/mxvsatv321/Documents/steer-audio` |
| RunPod venv | `/workspace/steer-audio/.venv` — activate with `. .venv/bin/activate` |

Key packages: `muq`, `fadtk`, `laion-clap`, `librosa`, `soundfile`, `scipy`

**DO NOT use `torchaudio.load()`** on the RunPod — it routes through TorchCodec and crashes with a `libnppicc.so.13` / FFmpeg error. Do not fix the system stack; use soundfile instead.

## Audio Loading Standard (always use this pattern)

```python
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
import torch

def load_audio_mono(path, target_sr=24000):
    data, sr = sf.read(str(path), always_2d=True)
    data = data.mean(axis=1)  # stereo → mono
    if sr != target_sr:
        g = gcd(target_sr, sr)
        data = resample_poly(data, target_sr // g, sr // g, axis=0)
    return torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, T)
```

Reference implementation: `scripts/score_human_eval_muq_mulan.py`.

## CLAP Scoring (always resample to 48 kHz)

ACE-Step outputs 44.1 kHz. `laion_clap` requires 48 kHz. Always resample before scoring. The fix is already in `scripts/run_paper_experiments.py`.

## MuQ-MuLan API

```python
from muq import MuQMuLan
mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
mulan = mulan.to(device).eval()
# audio: (1, T) float32 at mulan.sr (default 24000)
audio_embeds = mulan.extract_audio_latents(audio_tensor)
text_embeds  = mulan.extract_text_latents(["piano"])
similarity   = mulan.calc_similarity(audio_embeds, text_embeds)
```

**Do NOT call `muq.compute()`** — this method does not exist.

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

**Run paper experiments (RunPod only):**
```bash
python scripts/run_paper_experiments.py --experiments exp1 exp2 --concepts piano drums
python scripts/smoke_test.py          # quick end-to-end sanity check (< 5 min)
python scripts/score_human_eval_muq_mulan.py  # score existing WAVs with MuQ-MuLan
```

**FAD scoring:**
```bash
pip install fadtk
fadtk clap-laion-audio <baseline_dir> <eval_dir>
# baseline = unsteered generations, eval = steered generations
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
- `prompts.py` — Concept-to-prompt mappings

**`sae/sae_src/`** — SAE pipeline
- `sae/sae.py` — Encoder/decoder model with sparse activations
- `sae/trainer.py` — Distributed training loop
- `hooked_model/` — Model-specific activation hooks
- `configs/steer_prompts.py` — Concept prompt definitions

**`scripts/`** — Paper experiment runners
- `run_paper_experiments.py` — Single self-contained runner for all 7 paper experiments. CLI: `--experiments`, `--dry-run`, `--concepts`
- `smoke_test.py` — Minimal end-to-end test (< 5 min on A40); run before full experiments
- `score_human_eval_muq_mulan.py` — Scores `results/paper/human_eval/` WAVs with MuQ-MuLan

**`configs/`** — Hydra YAML configs organized by `patch_model/`, `patch_config/`, `patch_layers/`, `patch_data/`

### Configuration System

The entire patching pipeline is controlled by Hydra. Root configs are `configs/generate_audio_patch.yaml` and `configs/eval_audio.yaml`. Experiment variants (e.g., `experiment=patch_ace/ace_drums`) compose from multiple config groups.

### Supported Models / Concepts / Modes

- **Models:** ACE-Step, AudioLDM2, Stable Audio
- **Concepts:** drums, fast, female, happy, male, sad, slow, violin (plus others in `steer_prompts.py`)
- **Layers:** `tf6`, `tf7`, `tf6tf7`, `all`, `no_tf6tf7`
- **Steering modes:** `cond_only`, `separate`, `both_cond`, `uncond_only`
- **Alpha range:** `[-100, -90, ..., 100]`

## Results Structure

```
results/paper/
  exp1_*.csv ... exp7_*.csv       # all 7 experiment results
  human_eval/
    piano/ drums/ jazz/ mood/ tempo/ algebra/
    # each: pair_NN_steered.wav, pair_NN_unsteered.wav
  figures/                        # existing PNGs
  human_eval_muq_mulan.csv        # MuQ-MuLan scores (generated by scoring script)
outputs/vectors/                  # steering vectors for all 5 concepts
```

## sv.pkl Structure and Steering Setup

**sv.pkl format:** `{int_step: {layer_name: [np.ndarray]}}`. Real files have 30 integer keys (0–29) and layer keys `tf0`–`tf23`, all vectors shape `(2560,)`. Never assume string step keys or 1024-dim vectors.

**Canonical steering setup** (copy this exactly — deviations cause subtle bugs):
```python
num_cfg_passes = compute_num_cfg_passes(0.0, 0.0)
ctrl = VectorStore(device=device, save_only_cond=True, num_cfg_passes=num_cfg_passes)
ctrl.steer = True
ctrl.alpha = alpha
ctrl.steering_vectors = sv  # full raw dict, all steps × all layers
# explicit_layers handles filtering — controller only fires for hooked layers
register_vector_control(pipe.ace_step_transformer, ctrl, explicit_layers=["tf6", "tf7"])
audio = pipe.generate(..., return_type="audio", use_erg_lyric=False,
    guidance_scale_text=0.0, guidance_scale_lyric=0.0, guidance_scale=3.0,
    guidance_interval=1.0, guidance_interval_decay=0.0)
ctrl.reset()
```

**Controller is in:** `src/models/ace_step/ace_steering/controller.py`. The `_SafeSVDict` fix is already in place — do not revert it.

Key rules:
- **Always pass `explicit_layers`** to both collection and generation calls. Without it, all sub-blocks inside a transformer block are hooked (including 2560-dim ones), causing a size-mismatch RuntimeError.
- **Always register a fresh controller** before every `pipe.generate()` to prevent stale `actual_denoising_step` counters.
- **The pipeline makes a variable number of forward passes** beyond `infer_steps` (observed: +1 to +3). `_SafeSVDict` (in `run_paper_experiments.py`) handles this by falling back to the last step for any out-of-range integer key.
- **CFG:** `guidance_scale_text=0.0, guidance_scale_lyric=0.0` → `compute_num_cfg_passes` returns 2. `save_only_cond=True` stores only conditional activations.

## CSS Metric Definition

```
CSS(c, w) = Pr(CLAP_steered > CLAP_unsteered | concept=c, weight=w)
```

Estimated as fraction of pairs where steered wins. Report with binomial 95% CI. Analogous to PCI's CIS (Concept Insertion Success) metric.

## ReNorm Implementation

ReNorm normalizes steered activations to preserve the original norm:
```
h_steered_renorm = h_steered * (||h_original|| / ||h_steered||)
```
Implement in `src/models/ace_step/ace_steering/controller.py` as optional flag `--renorm` (default: False for backward compat).

## Correct Citations

- TADA: Staniszewski et al., arXiv:2602.11910, February 2026
- PCI: Gorgun et al., arXiv:2512.08486, ICLR 2026
- CaSteer: arXiv:2503.09630 (uses renormalized additive steering, NOT Householder)
- Linear Representation Hypothesis: Park et al., ICML 2024, arXiv:2311.03658
- CAE: Hao et al., arXiv:2505.03189, ICLR 2025 Workshop
- MuQ-Eval: not yet public as of March 2026

## What NOT to Do

- Do NOT use `torchaudio.load()` (crashes on RunPod)
- Do NOT call `muq.compute()` (doesn't exist)
- Do NOT modify `scripts/run_paper_experiments.py` unless fixing a verified bug
- Do NOT change CSV schemas of existing results files
- Do NOT post to arXiv before July 10, 2026 (ISMIR notification date)
- Do NOT use the generic LaTeX article class — use ISMIR 2026 template

## Working Style for Claude Code

- Read files before changing them; preserve public interfaces unless explicitly told otherwise.
- Prefer small, focused diffs that correspond to a single task.
- Do not hardcode paths; use `pathlib.Path` and environment variables (e.g., `TADA_WORKDIR`).
- Use type hints and docstrings on new Python functions.

When I say "don't ask for clarification for this task," pick the simplest reasonable assumption, document it in a comment or TODO, and continue. Only stop if you hit a hard blocker (missing file, unreadable config, missing dependency).

## Roadmap Integration

There is an additional file, `TADA_Build_Roadmap.md`, that defines detailed multi-week prompts (Phase 1–3).

When I reference a prompt (e.g., "Execute Phase 1, Prompt 1.1"), open `TADA_Build_Roadmap.md`, follow its instructions exactly, apply changes directly without waiting for per-file approval, and summarize files changed + commands to verify at the end.
