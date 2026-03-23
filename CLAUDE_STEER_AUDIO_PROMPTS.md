```markdown
# Claude Code Prompts — Steer-Audio / TADA! Build Roadmap

>Each prompt is self-contained with full context, exact file targets, and acceptance criteria. Do **not** skip phases — later phases depend on earlier ones.

---

## PHASE 1: Infrastructure Hardening

### Prompt 1.1 — Fix All Upstream Bugs + SAE Defaults

You are working in the `steer-audio` repository, a fork of `luk-st/steer-audio` implementing the TADA paper (arXiv 2602.11910): activation patching, Contrastive Activation Addition (CAA), and Sparse Autoencoder (SAE) steering on audio diffusion models targeting ACE-Step layers {6, 7}.

Fix all known divergences between this codebase and the TADA paper. Here is the exact list:

1. `sae/sae_src/sae/config.py` — `SaeConfig` defaults:
   - Change `expansion_factor` default from 32 → 4 (paper uses m=4)
   - Change `k` default from 32 → 64 (paper uses k=64)

2. `sae/sae_src/cache_activations_runner_ace.py` — `CacheActivationsRunnerConfig`:
   - Change `model_name` default from `"sd-legacy/stable-diffusion-v1-5"` → the ACE-Step model identifier
   - Change `guidance_scale` default from 9.0 → 7.0 (audio-appropriate default)

3. `sae/sae_src/hooks.py` — `SAEReconstructHook`:
   - Find the `np.sqrt` spatial reshape (written for 2D image tokens) and replace it with correct 1D audio token reshape. Audio activations from ACE-Step cross-attention layers are shape `(batch, seq_len, d_model)` — there is no spatial H×W grid.

4. `sae/sae_src/hooks.py` — `AblateHook` and `StableAudioAblateHook`:
   - Remove all `print()` debug statements
   - Remove all commented-out code blocks
   - `StableAudioAblateHook`: the SAE output is currently zeroed **after** decoding (wasteful). Move the zeroing to **before** the decoder call so the decode is skipped entirely.

5. `steering/ace_steer/compute_steering_vectors_caa.py` line ~27:
   - Replace the hardcoded `WORKDIR_PATH` string literal with:
     ```python
     WORKDIR_PATH = os.environ.get("TADA_WORKDIR", os.path.join(os.getcwd(), "outputs"))
     ```
   - Add `import os` at the top if not present.

6. `sae/scripts/eval_sae_steering.py` lines ~66–120:
   - Replace hardcoded `VECTORS_SEED`, `GENERATION_SEED`, and `MULTIPLIERS` list with function parameters that have those same values as defaults, so they remain overridable via CLI.

After all changes:
- Run the existing test suite (if any) and confirm no regressions.
- Add a brief docstring to each modified class/function explaining what it does and citing the relevant paper equation number.

Do **not** change any experiment logic, data flow, or model architecture. Only fix the listed bugs.

---

### Prompt 1.2 — Unified Hydra Config + Entry Points

You are working in the `steer-audio` repository. The upstream code uses hardcoded paths and separate scripts for each pipeline step. Your job is to create a unified Hydra configuration hierarchy and update all scripts to use it.

**CONTEXT** — The TADA pipeline has 3 steps:

1. Activation patching (`src/patch_layers.py`) → finds functional layers.
2. CAA vector computation (`steering/ace_steer/compute_steering_vectors_caa.py`) → produces `.safetensors` vectors.
3. Evaluation (`steering/ace_steer/eval_steering_vectors.py`) → measures CLAP/MuQ/LPAPS/FAD.

#### TASK 1 — Create `configs/` directory structure

Create:

```text
configs/
  base.yaml
  model/
    ace_step.yaml
    audioldm2.yaml
    stable_audio.yaml
  concept/
    tempo.yaml
    mood.yaml
    instruments.yaml
    vocal_gender.yaml
    genre.yaml
  experiment/
    patching.yaml
    caa_steering.yaml
    sae_training.yaml
```

`configs/base.yaml` must contain:

```yaml
workdir: ${oc.env:TADA_WORKDIR,${hydra:runtime.cwd}/outputs}
seed: 42
device: ${oc.env:TADA_DEVICE,cuda}
num_workers: 4
```

`configs/model/ace_step.yaml` must contain:

```yaml
name: ace_step
model_id: "ACE-Step/ACE-Step"
functional_layers: [audio-steering.github](https://audio-steering.github.io)
total_layers: 24
d_model: 1024
guidance_scale: 7.0
num_inference_steps: 60
audio_length_seconds: 30
```

Each concept yaml (e.g. `configs/concept/tempo.yaml`) must contain:

```yaml
name: tempo
positive_keyword: "fast tempo"
negative_keyword: "slow tempo"
similarity_metric: muq
tau: 20
alpha_range: [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100]
```

For `mood.yaml` set `tau: 40`. For `vocal_gender.yaml` set `similarity_metric: clap`, `tau: 20`.

#### TASK 2 — Update scripts to accept Hydra config

- Update `src/patch_layers.py` to read workdir, model config, and concept config from Hydra `cfg`.
- Update `steering/ace_steer/compute_steering_vectors_caa.py` to read all params from `cfg`.
- Update `sae/sae_src/cache_activations_runner_ace.py` to read model config from `cfg`.

Use `@hydra.main(config_path="../configs", config_name="base")` decorators. Do not change algorithm logic; only replace ad‑hoc argparse / hardcoded paths with Hydra config access.

#### TASK 3 — Write `tests/test_config.py`

Cover:

- `base.yaml` loads without errors via Hydra.
- Each model config has keys: `name`, `functional_layers`, `d_model`.
- Each concept config has keys: `name`, `alpha_range`, `tau`.
- `TADA_WORKDIR` env var override works (set env var in test and verify path).

All tests must pass with:

```bash
pytest tests/test_config.py
```

---

### Prompt 1.3 — Full pytest Test Suite (Core Modules)

You are working in the `steer-audio` repository. The upstream codebase has **zero** tests. Your job is to write a comprehensive pytest suite for all core numerical and structural operations.

Create the following test files. Each must be fully runnable with `pytest tests/ -x` **without** requiring GPU, ACE-Step weights, or real audio — use small random tensors and mock objects throughout.

#### FILE: `tests/test_sae.py`

Cover:

- SAE encode/decode roundtrip:
  - For a small SAE (`d_model=64`, `expansion=4`, `k=8`), verify that TopK produces exactly `k` non-zero activations.
- Reconstruction loss:
  - Verify `||h - decode(encode(h))||_2` decreases after a training step.
- TF‑IDF scoring (`sae/tfidf_utils.py` or `sae/sae_src/`):
  - Given mock activation means for positive/negative prompts, verify the TF‑IDF score formula
    \[
      \text{score}(j,c) = \mu_j(P_c) \cdot \log\left(1 + \frac{1}{\mu_j(P_{\tilde c}) + \epsilon}\right)
    \]
    matches a manually computed reference for 3 features.
- Dead feature detection:
  - After zeroing all activations for some features, verify dead feature count is correct.
- SAE steering vector construction:
  - Verify `v_c^SAE = Σ_{j∈F_c} W_dec[:,j]` produces the correct shape and values for a known feature set `F_c`.

#### FILE: `tests/test_patching.py`

Cover:

- Impact metric formula:

  \[
    \text{Impact}(l,c) =
    \frac{\text{sim}(l←c, l'←\tilde c) - \text{sim}(l←\tilde c, l'←\tilde c)}
         {\text{sim}(l←c, l'←c) - \text{sim}(l←\tilde c, l'←\tilde c)}
  \]

  Implement manually and compare against the codebase function for 3 known input values.

- Hook registration:
  - Mock a 2‑layer transformer, register a patch hook on layer 1, run forward pass, verify the activation was intercepted (use a side‑effect counter).
- Hook cleanup:
  - Verify that after deregistration, the forward pass is no longer intercepted.
- Functional layer identification:
  - Given a mock impact matrix (24 layers × 3 concepts), verify that the top‑2 layers are correctly identified as functional.

#### FILE: `tests/test_caa_steering.py`

Cover:

- CAA vector computation:
  - Given 5 positive and 5 negative activation vectors of shape `(seq_len=16, d=64)`, verify `v_c` and normalized `v_c^CAA` match manual computation.
- ReNorm:
  - Verify `ReNorm(h + α*v, h)` has the same L2 norm as `h` (within `1e-5` tolerance) for α in `{-100, 0, 50, 100}`.
- Steering injection:
  - Given a mock activation `h`, steering vector `v`, and α=30, verify the steered activation `h'` has correct shape and norm.
- Alpha sweep:
  - Verify that for α=0, the steered output equals the unsteered output exactly.

After writing all tests, run:

```bash
pytest tests/ -x -v
```

Fix any bugs you find in the source code **only** if the source code contradicts the paper formulas.

---

### Prompt 1.4 — `tada` CLI + `pyproject.toml`

You are working in the `steer-audio` repository. There is currently no installable package and no unified CLI. Create both.

#### TASK 1 — Create `pyproject.toml` in repo root

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "steer-audio"
version = "0.1.0"
description = "TADA! Tuning Audio Diffusion Models through Activation Steering"
requires-python = ">=3.10"
dependencies = [
  "torch>=2.1.0",
  "hydra-core>=1.3.0",
  "safetensors",
  "click>=8.0",
  "rich",
]

[project.scripts]
tada = "steer_audio.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-x --tb=short -q"
```

#### TASK 2 — Create `steer_audio/cli.py` using Click

Commands:

- `tada localize`  
  `--config-dir configs/ --model ace_step --concept tempo`  
  Runs `src/patch_layers.py` (activation patching), saves impact matrix to `$TADA_WORKDIR/patching/{concept}/`.

- `tada compute-vectors`  
  `--config-dir configs/ --model ace_step --concept tempo`  
  Runs `steering/ace_steer/compute_steering_vectors_caa.py`, saves vectors to `$TADA_WORKDIR/vectors/`.

- `tada train-sae`  
  `--config-dir configs/ --model ace_step --layer 7`  
  Runs `sae/sae_src/scripts/train_ace.py`, saves SAE checkpoint to `$TADA_WORKDIR/sae/`.

- `tada generate`  
  `--config-dir configs/ --model ace_step --concept tempo --alpha 50`  
  Generates steered audio; saves to `$TADA_WORKDIR/audio/{concept}/alpha_{alpha}/`.

- `tada evaluate`  
  `--config-dir configs/ --model ace_step --concept tempo`  
  Runs full alpha sweep evaluation; saves CSV + plots to `$TADA_WORKDIR/eval/{concept}/`.

- `tada list-vectors`  
  Lists all saved steering vectors in `$TADA_WORKDIR/vectors/` with a metadata table.

- `tada status`  
  Prints a summary table: which concepts have vectors, SAE checkpoints, and eval results.

All commands must:

- Load the appropriate Hydra config.
- Print rich progress/status messages.
- Exit with code 1 and a clear error message if required files are missing.
- Support `--dry-run` flag that prints what would be done without executing.

#### TASK 3 — Update `steer_audio/__init__.py`

Export:

```python
from steer_audio.vector_bank import SteeringVector, SteeringVectorBank
# (later phases will add more exports)
```

#### TASK 4 — Write `tests/test_cli.py`

Cover:

- `tada --help` exits 0.
- `tada status` exits 0 and prints a table (mock `$TADA_WORKDIR`).
- `tada list-vectors` on an empty directory prints “No vectors found”.
- `tada evaluate` without prior vectors prints an error and exits 1.

---

## PHASE 2: Advanced Steering Features

### Prompt 2.1 — `SteeringVector` & `SteeringVectorBank`

You are working in the `steer-audio` repository. Create `steer_audio/vector_bank.py` implementing two classes.

**CONTEXT (from TADA):**

- CAA vector: `v_c^CAA = v_c / ||v_c||_2`, stored per functional layer.
- SAE vector: `v_c^SAE = Σ_{j∈F_c} W_dec[:,j]`.
- Vectors are applied at layers l ∈ {6, 7}.

#### CLASS: `SteeringVector`

Fields:

- `concept: str`
- `method: str` — `"caa"` or `"sae"`
- `model: str` — e.g. `"ace_step"`
- `layers: List[int]` — e.g. `[6, 7]`
- `vector: Dict[int, Tensor]` — layer_idx → steering vector `(d_model,)`
- `alpha_range: List[float]`
- `metadata: Dict[str, Any]`

Methods:

- `save(path: str) -> None` — serialize to `.safetensors` + sidecar `.json`.
- `@classmethod load(path: str) -> SteeringVector`.
- `norm(layer: int) -> float` — L2 norm.
- `cosine_similarity(other, layer: int) -> float`.
- `__repr__` — human-readable summary.

#### CLASS: `SteeringVectorBank`

Methods:

- `add(vector: SteeringVector) -> None`
- `get(concept: str, method: str = "caa") -> SteeringVector`
- `list() -> List[str]` — e.g. `"tempo/caa"`.
- `save_all(directory: str) -> None`
- `@classmethod load_all(directory: str) -> SteeringVectorBank`
- `compose(concepts: List[str], method: str = "caa", orthogonalize: bool = True) -> Dict[int, Tensor]`
  - Returns `{layer: combined_vector}` using Gram-Schmidt if `orthogonalize=True`.
  - Gram-Schmidt: for each concept i, subtract projections onto all previous vectors.
  - If residual norm after orthogonalization is < 0.1, emit a warning:
    > Concept '{c}' has cosine similarity >{thresh} with a prior concept; its vector norm after orthogonalization is {norm:.4f}. Consider removing it or using a different concept.
- `interference_matrix(layer: int) -> np.ndarray` — N×N cosine similarity matrix.

#### `tests/test_vector_bank.py`

Cover:

- Save/load round-trip preserves all fields.
- Gram-Schmidt: two orthogonal vectors → unchanged.
- Gram-Schmidt degenerate: two identical vectors → second residual norm < 1e-5, warning raised.
- `interference_matrix` diagonal ≈ 1.0, correct shape.
- `compose` returns vectors of correct shape for each layer.

---

### Prompt 2.2 — `MultiConceptSteerer`

You are working in the `steer-audio` repository. Create `steer_audio/multi_steer.py`.

**CONTEXT** — Multi-concept steering sums multiple normalized vectors with independent alphas:

\[
h'_l = h_l + \sum_c \alpha_c v_c
\]

Optionally orthogonalized via Gram-Schmidt (Concept Sliders–style).

#### CLASS: `MultiConceptSteerer`

Constructor:

```python
def __init__(self, vector_bank: SteeringVectorBank, orthogonalize: bool = True): ...
```

Methods:

- `add_concept(concept: str, alpha: float, method: str = "caa") -> None`
- `remove_concept(concept: str) -> None`
- `set_alpha(concept: str, alpha: float) -> None`
- `get_combined_vectors(layer: int) -> Tensor`
  - Returns sum of active concept vectors for that layer.
  - Apply Gram-Schmidt if `orthogonalize=True`.
- `register_hooks(model: nn.Module, target_layers: List[int]) -> List[Any]`
  - Registers forward hooks on target layers that:
    1. Read current activation `h_l`.
    2. Get combined vector for this layer.
    3. Apply `h'_l = ReNorm(h_l + combined, h_l)`.
- `remove_hooks(handles: List[Any]) -> None`
- `interference_report() -> Dict[str, Any]`
  - Contains: `"cosine_matrix"`, `"max_cosine"`, `"warnings"`, `"clap_deltas"` (placeholder).

#### `tests/test_multi_steer.py`

Cover:

- Add/remove concept bookkeeping.
- `get_combined_vectors` norm behavior with orthogonal vs. parallel vectors.
- `alpha=0` → zero vector.
- Hook registration on a mock 2-layer module modifies values but not shape.
- `interference_report` warns when cosine > 0.5.
- Gram-Schmidt edge case for parallel vectors yields near-zero residual + warning.

---

### Prompt 2.3 — Timestep-Adaptive Steering

You are working in the `steer-audio` repository. Create `steer_audio/temporal_steering.py`.

**CONTEXT** — ACE-Step uses 60 diffusion steps. Early steps set global structure, late steps refine details.

#### FUNCTION: `get_schedule(schedule_type: str) -> Callable[[int, int], float]`

Return function `f(step, total_steps) -> float ∈ [0,1]` for:

- `"constant"`: 1.0
- `"linear"`: linearly decays 1.0 → 0.0 over steps
- `"cosine"`: `(1 + cos(pi * step / total_steps)) / 2`
- `"early_only"`: 1.0 if `step < total_steps * 0.4` else 0.0
- `"late_only"`: 0.0 if `step < total_steps * 0.6` else 1.0

#### CLASS: `TimestepAdaptiveSteerer`

Fields:

- Holds `MultiConceptSteerer` instance.
- Internal state dict `{"step": 0}`.
- Selected schedule function.

Methods:

- `step_alpha(base_alpha, step, total_steps) -> float`
- `register_scheduled_hooks(model, target_layers, total_steps) -> List`
- `advance_step() -> None`
- `reset() -> None`

#### `tests/test_temporal_steering.py`

Cover schedules at `step=0`, mid, last for `total=60`, plus:

- `early_only` and `late_only` boundaries.
- `step_alpha` scaling.
- `advance_step` / `reset`.
- Clamping when `step >= total_steps`.

---

### Prompt 2.4 — SAE Concept Algebra

You are working in the `steer-audio` repository. Create `steer_audio/concept_algebra.py`.

**CONTEXT** — From TADA:

- `v_c^SAE = Σ_{j∈F_c} W_dec[:,j]` (Eq. 7).
- TF‑IDF features: `score(j,c)` as in Eq. 6; τ=20 or 40.

#### CLASS: `ConceptFeatureSet`

Fields:

- `concept: str`
- `features: Set[int]`
- `scores: Dict[int, float]`
- `tau: int`

Methods:

- `union(other) -> ConceptFeatureSet`
- `subtract(other) -> ConceptFeatureSet`
- `intersect(other) -> ConceptFeatureSet`
- `weighted_blend(other, weight: float) -> ConceptFeatureSet`
- `to_steering_vector(W_dec: Tensor, layer: int) -> Tensor`
- `top_features(n: int = 10) -> List[Tuple[int, float]]`

#### CLASS: `ConceptAlgebra`

Constructor loads SAE decoder weights lazily from a checkpoint path.

Methods:

- `get_feature_set(concept: str, tau: int = 20) -> ConceptFeatureSet`
  - Load TF‑IDF scores from `$TADA_WORKDIR/tfidf/{concept}.json`.
- `parse_expression(expr: str) -> ConceptFeatureSet`
  - Supports `+`, `-`, `*` scalar multiply syntax like `"jazz + female_vocal - piano * 0.5"`.
- `build_vector(feature_set, layer: int) -> Tensor`.

#### CLASS: `AlgebraPresetBank`

Contains presets:

- `"jazz_no_piano": "jazz - piano"`
- `"upbeat_female": "upbeat + female_vocal"`
- `"mellow_guitar": "mood_mellow + guitar"`
- `"fast_techno": "tempo_fast + genre_techno"`
- `"slow_jazz_vocal": "tempo_slow + jazz + female_vocal"`

Methods: `get`, `list`, `add`.

#### `tests/test_concept_algebra.py`

Cover set operations, `to_steering_vector`, expression parsing, invalid syntax, preset access, and JSON round‑trip for `ConceptFeatureSet`.

---

### Prompt 2.5 — Self-Monitored Steering (Diffusion-Correct)

You are working in the `steer-audio` repository. Create `steer_audio/self_monitor.py`.

**CRITICAL CONTEXT** — SMITIN is for autoregressive MusicGen; ACE-Step is diffusion. Gating must work on diffusion steps, not tokens.

#### CLASS: `ConceptProbe`

Constructor:

```python
def __init__(self, concept: str, target_prompt: str, clap_model=None): ...
```

- Stub mode when `clap_model` is `None` → always return 0.5.

Methods:

- `score(audio_tensor, sample_rate=44100) -> float`
- `delta(current_score, previous_score) -> float`

#### CLASS: `SelfMonitoredSteerer`

Constructor:

```python
def __init__(
    self,
    multi_steerer: MultiConceptSteerer,
    probe: ConceptProbe,
    check_every_n_steps: int = 5,
    alpha_step: float = 5.0,
    max_alpha: float = 100.0,
    min_alpha: float = 0.0,
    convergence_threshold: float = 0.02,
): ...
```

Fields: `current_alpha`, `_step`, `_score_history`.

Methods:

- `should_check(step: int) -> bool`
- `update(partial_audio, sample_rate, step: int) -> float`
  - On check step: compute score, compare to last, adjust alpha:
    - Score ↓ (regressing) → `alpha += alpha_step`.
    - Score ↑ more than threshold → `alpha -= alpha_step`.
    - |Δ| < threshold → no change.
  - Clamp alpha to `[min_alpha, max_alpha]`.
- `reset() -> None`
- `get_history() -> Dict[str, List]`

#### `tests/test_self_monitor.py`

Cover stub behavior, `should_check`, alpha increase/decrease, clamping, reset, and history.

---

### Prompt 2.6 — Unified `SteeringPipeline` + Integration Tests

You are working in the `steer-audio` repository. Create `steer_audio/pipeline.py`.

#### CLASS: `SteeringPipeline`

Constructor:

```python
def __init__(
    self,
    vector_bank: SteeringVectorBank,
    model=None,
    target_layers: List[int] =, [audio-steering.github](https://audio-steering.github.io)
    orthogonalize: bool = True,
    schedule_type: str = "constant",
    self_monitor: bool = False,
    monitor_every_n_steps: int = 5,
): ...
```

Methods:

- `add_concept(concept, alpha, method="caa") -> SteeringPipeline`
- `remove_concept(concept) -> SteeringPipeline`
- `set_schedule(schedule_type) -> SteeringPipeline`
- `enable_self_monitoring(probe: ConceptProbe, **kwargs) -> SteeringPipeline`
- `generate(prompt, seed=42, num_inference_steps=60, audio_length_seconds=30.0, dry_run=False) -> Dict[str, Any]`
- `get_interference_report() -> Dict[str, Any]`
- Context manager `__enter__` / `__exit__` to ensure hooks removed.

#### `tests/test_integration_phase2.py`

Create `MockAceStep` with 2 linear layers for “layer 6/7”, support for hook registration, and simple forward.

Cover dry-run pipeline, context manager hook cleanup, chaining, interference report, and dry-run summary output.

Update `steer_audio/__init__.py` to export:

```python
from steer_audio.pipeline import SteeringPipeline
from steer_audio.vector_bank import SteeringVector, SteeringVectorBank
from steer_audio.multi_steer import MultiConceptSteerer
from steer_audio.temporal_steering import TimestepAdaptiveSteerer, get_schedule
from steer_audio.concept_algebra import ConceptAlgebra, ConceptFeatureSet, AlgebraPresetBank
from steer_audio.self_monitor import SelfMonitoredSteerer, ConceptProbe
```

---

## PHASE 3: Product & Research

### Prompt 3.1 — Gradio Audio Attribute Studio

You are working in the `steer-audio` repository. Create `demo/app.py`: a Gradio “Audio Attribute Studio” with three tabs.

The app must support:

- **STUB MODE** (no ACE-Step): pre-computed examples only; generation disabled.
- **FULL MODE**: ACE-Step installed → generation enabled.

Detect mode at startup by trying to import ACE-Step; fall back to stub mode gracefully.

#### Tab 1 — “🎵 Generate & Steer”

Components:

- `gr.Textbox` prompt.
- `gr.Number` seed.
- Five concept sliders with friendly labels and ranges -100→100.
- Schedule dropdown with schedule types.
- Generate button.
- Two `gr.Audio` outputs side-by-side (unsteered vs steered).
- CLAP alignment display (`gr.Textbox`).
- `gr.Examples` using example files under `demo/examples/`.

#### Tab 2 — “🔬 SAE Feature Explorer”

- Concept dropdown.
- Expression textbox.
- Evaluate button → Dataframe of top-10 features + heatmap image.
- In stub mode, uses JSON under `demo/examples/feature_table.json`.

#### Tab 3 — “📊 Batch Experiment”

- Concept selector (checkbox group).
- Alpha range inputs.
- Num prompts slider.
- Run Batch button, progress bar, CSV results + download button.

Add an “ℹ️ About” `gr.Markdown` explaining TADA briefly.

Create `demo/requirements_space.txt` with at least `gradio`, `torch`, `numpy`, `safetensors`.

Fix `README_SPACE.md` to use `app_file: demo/app.py`.

`tests/test_demo_app.py` should import the module (with heavy imports mocked), confirm the Blocks object exists, and assert three tabs and ≥5 examples.

---

### Prompt 3.2 — SAE Scaling Experiment

Create `experiments/sae_scaling.py` implementing `run_sae_scaling_experiment` with dry-run and real modes, plus `scripts/run_sae_scaling_real_small.sh` as described earlier (see previous answer for full details — keep the same arguments, outputs, and tests).

Key outputs:

- `experiments/results/scaling/all_results.csv`
- `power_law.json`
- `pareto.csv`
- `scaling_fvu.png`, `scaling_dead.png`

`tests/test_sae_scaling.py` covers dry-run, sentinels, file creation, and skip behavior when only sentinels exist.

---

### Prompt 3.3 — Vector Geometry Analysis

Create `experiments/vector_geometry.py` implementing `run_geometry_analysis` with dry-run support and five outputs:

- `cosine_heatmap.png` + `cosine_matrix.csv`
- `pca_2d.png`
- `probe_accuracy.csv` + `probe_results.md`
- `arithmetic_verification.csv`
- `layer_progression.png`

Save to `experiments/results/geometry/` and test via `tests/test_vector_geometry.py` as described earlier.

---

### Prompt 3.4 — Unified Eval Metrics + Alpha Sweep

Create `steer_audio/eval_metrics.py` with `CLAPBackend`, `MuQBackend`, `LPAPSBackend`, `FADBackend`, and `EvalSuite` implementing `evaluate_single`, `compute_alpha_sweep`, `plot_alpha_sweep`, and `summary_table`, including MuQ support and exact alpha range `{ -100 … 100 }`.

`tests/test_eval_metrics.py` should exercise stub mode and dry-run alpha sweep.

---

## PHASE 4: CI + Docs

### Prompt 4.1 — GitHub Actions CI

Add `.github/workflows/ci.yml`, `optional-dependencies` in `pyproject.toml`, `.coveragerc`, and `CONTRIBUTING.md` as detailed earlier (matrix on Python 3.10/3.11, pytest+coverage, Ruff lint).

---

### Prompt 4.2 — New README + HF Space Config

Rewrite `README.md` with:

- New title & badges.
- “What’s new in this fork” list.
- Quickstart commands.
- Pipeline overview.
- Module reference table.
- Links & citations.

Fix `README_SPACE.md` with proper metadata and `app_file: demo/app.py`. Create `demo/examples/` scaffolding.

---

## PHASE 5: Real Experiments (Requires ACE-Step + GPU or fast CPU)

### Prompt 5.1 — First Real Runs

Once ACE-Step is installable (Python 3.11):

- Generate real activation cache at layer 7.
- Compute real CAA vectors for 5 concepts.
- Run `vector_geometry.py` with real vectors.
- Run `sae_scaling.py --preset real_small`.
- Run `eval_sweep.py` for tempo with 20 prompts and subset alpha range.
- Persist results under `experiments/results/` and `results/eval/tempo/`.

Report key statistics (power law exponents, R², optimal alpha, CLAP deltas) as described earlier.

---

### Prompt 5.2 — Multi-Concept + Algebra + Schedule Experiments

Run `multi_concept_experiment.py`, `concept_algebra_demo.py`, and `timestep_schedule_experiment.py` using real vectors, saving CSVs and PNGs under `experiments/results/` as previously specified.

---

### Prompt 5.3 — `docs/results_summary.md`

Summarize all experiment results (scaling, geometry, multi-concept, schedules, eval metrics) into a single markdown doc in `docs/`.

---

## PHASE 6: Community & Hub

### Prompt 6.1 — Quickstart Notebook + HF Hub Upload

- Add `notebooks/quickstart.ipynb` demonstrating dry-run workflow.
- Add `steer_audio/hub.py` with `upload_vectors` and `download_vectors`.
- Extend `tada` CLI with `upload-vectors` and `download-vectors` commands that use `HF_TOKEN`.

---

## Meta Notes for Claude Code

- Always read `CLAUDE.md` (if present) before editing.
- After each prompt, run `pytest tests/ -x -q` and check `git diff --stat`.
- Never change paper equations or functional layers {6, 7}.
- Treat `-1.0` as the only sentinel for “no real metric yet”.

```