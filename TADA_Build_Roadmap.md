Paste this at the start of every Claude Code session:

text
You are a senior ML systems engineer and audio research scientist working on the TADA project — 
"Tuning Audio Diffusion Models through Activation Steering" (arXiv 2602.11910).

Repository: https://github.com/luk-st/steer-audio (already cloned locally)

Key facts you must internalize before writing any code:
- The codebase has three primary pipelines: (1) activation patching in `src/`, (2) CAA steering 
  in `steering/ace_steer/`, (3) SAE training in `sae/`
- The primary model is Ace-Step (DiT, 24 cross-attn layers); functional layers are {6, 7}
- CAA steering: h'_l = ReNorm(h_l + α·v_c, h_l) where v_c is mean-normalized contrastive vector
- SAE steering: h'_l = h_l + α·v_c^SAE where v_c^SAE = Σ_{j∈F_c} W_dec[:,j] (top-τ TF-IDF features)
- CRITICAL BUGS IN CURRENT CODEBASE:
    * SaeConfig defaults (expansion_factor=32, k=32) differ from paper values (m=4, k=64)
    * CacheActivationsRunnerConfig defaults point to SD 1.5 (image model), not audio
    * hooks.py uses np.sqrt spatial reshape (image-specific dead code)
    * StableAudioAblateHook:202 zeros SAE output after decoding (should be before)
    * WORKDIR_PATH is hardcoded as "<WORKDIR_PATH>" placeholder
- There are ZERO unit tests and ZERO CI/CD pipelines
- `notebooks/` and `data/` directories are empty
- `gradio` is in requirements but no demo exists

Always write type-annotated Python. Always add docstrings. Never hardcode paths — use pathlib 
and environment variables. When modifying existing files, preserve all existing functionality 
unless explicitly told to remove it.
Phase 1 — Foundation (Weeks 1–2)
Goal: A stable, reproducible, well-tested codebase where every existing experiment runs from a single command. No new features — only hardening.

Sprint 1A: Fix the Bugs and Config System
Claude Code Prompt 1.1 — Critical Bug Fixes

text
TASK: Fix all critical bugs in the steer-audio codebase identified in the engineering audit.

Fix the following issues IN ORDER:

1. `sae/sae_src/sae/config.py` — SaeConfig defaults:
   Change `expansion_factor` default from 32 to 4
   Change `k` default from 32 to 64
   Add a comment: # Paper values: m=4, k=64 (arXiv 2602.11910 Table 3)

2. `sae/sae_src/cache_activations_runner_ace.py` — CacheActivationsRunnerConfig defaults:
   Change `model_name` default away from `sd-legacy/stable-diffusion-v1-5`
   Change `guidance_scale` default from 9.0 to 7.5 (standard audio diffusion default)
   Add a docstring explaining this config is for ACE-Step only

3. `sae/sae_src/sae/hooks.py` — SAEReconstructHook:
   Find and remove the `np.sqrt` spatial reshape logic (this is image-specific dead code)
   Replace with a comment: # Audio activations are 1D sequences; no spatial reshape needed
   Ensure the hook handles 1D sequence activations correctly

4. Wherever StableAudioAblateHook:202 zeros the SAE output after decoding:
   Move the zeroing operation to BEFORE the decode step
   Add a comment explaining why: # Zero features before decoding to avoid wasted computation

5. In ALL scripts that contain the literal string `"<WORKDIR_PATH>"`:
   Replace with: `os.environ.get("TADA_WORKDIR", str(Path.home() / "tada_outputs"))`
   Import os and pathlib.Path at the top if not already present

After each fix, write a one-line comment in the file: # BUG FIX: [description] — [date]

DONE WHEN: `grep -r "<WORKDIR_PATH>" .` returns nothing; `python -c "from sae.sae_src.sae.config 
import SaeConfig; c = SaeConfig(); assert c.expansion_factor == 4 and c.k == 64"` passes.
Claude Code Prompt 1.2 — Unified Config System

text
TASK: Replace all hardcoded parameters across the codebase with a unified Hydra config system.

1. Create `configs/` directory with the following hierarchy:
   configs/
   ├── base.yaml              # Global paths, seeds, device settings
   ├── model/
   │   ├── ace_step.yaml      # ACE-Step specific settings (layers, hidden dim, etc.)
   │   ├── stable_audio.yaml  # Stable Audio Open settings
   │   └── audioldm2.yaml     # AudioLDM2 settings
   ├── concept/
   │   ├── tempo.yaml         # Concept-specific: prompt pairs, tau, alpha range
   │   ├── mood.yaml
   │   ├── vocal_gender.yaml
   │   ├── instruments.yaml
   │   └── genre.yaml
   └── experiment/
       ├── patching.yaml      # Activation patching experiment config
       ├── caa_steering.yaml  # CAA steering config
       └── sae_training.yaml  # SAE training config

2. `base.yaml` must contain:
   workdir: ${oc.env:TADA_WORKDIR,${hydra:runtime.cwd}/outputs}
   seed: 42
   device: cuda
   num_workers: 4

3. `model/ace_step.yaml` must contain:
   name: ace-step
   hidden_dim: 3072
   num_layers: 24
   functional_layers: [6, 7]
   cross_attn_heads: 24
   sample_rate: 44100
   max_duration: 240

4. Each `concept/*.yaml` must contain: name, positive_keywords, negative_keywords, 
   tau (number of SAE features), alpha_range, similarity_metric (muq or clap)

5. Add `@hydra.main` decorator to the three main entry point scripts:
   - `src/patch_layers.py`
   - `steering/ace_steer/compute_steering_vectors_caa.py`
   - `sae/scripts/train_ace.py`

6. Create `configs/concept/tempo.yaml` as a template with full comments explaining each field.

DONE WHEN: `python src/patch_layers.py model=ace_step concept=tempo experiment=patching` 
runs without errors (even if it fails at model loading due to missing weights).
Claude Code Prompt 1.3 — Comprehensive Test Suite

text
TASK: Create a complete pytest test suite for the steer-audio codebase. 
This is the most critical infrastructure task — nothing ships without tests.

Create `tests/` directory with the following files:

FILE: tests/conftest.py
- Fixtures: small_activation_tensor (batch=2, seq=16, dim=64), 
  mock_sae_config (expansion_factor=4, k=4), mock_ace_step_config
- Use torch.manual_seed(42) for all random tensors
- Provide a fixture that creates a minimal SAE model without GPU

FILE: tests/test_sae.py — Test the SAE model
Tests to write:
1. test_sae_encode_shape: encoder output shape is (batch, expansion_factor * dim)
2. test_topk_sparsity: exactly k values are nonzero after TopK
3. test_sae_reconstruct_shape: decoded output matches input shape
4. test_reconstruction_improves_with_epochs: loss decreases over 3 training steps
5. test_bpre_bias_applied: pre-bias is subtracted before encoding
6. test_decoder_columns_unit_norm: each decoder column has unit L2 norm after init

FILE: tests/test_patching.py — Test activation patching logic
Tests to write:
1. test_impact_metric_range: Impact(l,c) is in [0,1] for valid inputs
2. test_impact_metric_zeros_for_no_change: if patched and unpatched are identical, impact=0
3. test_impact_metric_one_for_full_change: if unpatched matches target, impact=1
4. test_hook_registers_and_deregisters: hook count on model is 0 after context manager exits

FILE: tests/test_caa_steering.py — Test CAA vector computation and application
Tests to write:
1. test_caa_vector_unit_norm: computed v_c has ||v_c||_2 = 1.0 (within 1e-6)
2. test_renorm_preserves_magnitude: ||ReNorm(h', h)||_2 == ||h||_2 (within 1e-6)
3. test_steering_changes_activation: steered h != original h when alpha != 0
4. test_steering_identity_at_zero_alpha: steered h == original h when alpha == 0
5. test_multi_concept_steering_shape: output shape matches input for simultaneous steering

FILE: tests/test_tfidf_scoring.py — Test SAE feature scoring
Tests to write:
1. test_tfidf_score_positive: score is always non-negative
2. test_tfidf_higher_for_exclusive_features: feature activating only for P_c scores higher
   than one activating for both P_c and P_c_bar
3. test_top_tau_features_are_highest_scored: returned feature indices match argsort(scores)[-tau:]

FILE: tests/test_config.py — Test config system
Tests to write:
1. test_sae_config_paper_defaults: expansion_factor==4, k==64
2. test_functional_layers_ace_step: functional_layers == [6, 7]
3. test_workdir_env_override: setting TADA_WORKDIR env var changes output path

Run all tests with: `pytest tests/ -v --tb=short`
Target: 100% pass rate, >80% line coverage on core modules.

DONE WHEN: `pytest tests/ -v` shows all tests passing with no warnings about missing modules.
Sprint 1B: CLI and Developer Experience
Claude Code Prompt 1.4 — Unified CLI

text
TASK: Create a single unified CLI entry point that replaces the current 3-step manual pipeline.

Create `cli.py` at the repository root using Python's `click` library (add to requirements if needed).

The CLI must support these commands:

COMMAND: tada localize
Usage: tada localize --model ace-step --concepts tempo mood vocal_gender --n-prompts 256 --seeds 8
- Runs the activation patching pipeline from src/patch_layers.py
- Outputs: configs/cache/{model}/{concept}/impact_scores.json
- Shows a progress bar per concept
- Prints a summary table: concept → top-3 layers by Impact score

COMMAND: tada compute-vectors
Usage: tada compute-vectors --model ace-step --concept tempo --method caa
Usage: tada compute-vectors --model ace-step --concept tempo --method sae --tau 20
- Runs compute_steering_vectors_caa.py or sae path
- Outputs: vectors/{model}/{concept}/{method}_vector.safetensors with metadata

COMMAND: tada generate
Usage: tada generate --prompt "a jazz piano trio" --concept tempo --alpha 60 --output out.wav
Usage: tada generate --prompt "..." --concept tempo --alpha 60 --concept mood --alpha 40 --output out.wav
- Loads pre-computed vectors from vectors/ directory
- Applies steering during Ace-Step inference
- Saves output WAV file + prints CLAP alignment score

COMMAND: tada evaluate
Usage: tada evaluate --concept tempo --method caa --alpha-range "-100,100,10"
- Runs the full evaluation protocol
- Outputs: results/{concept}/{method}/metrics.json + metrics.csv

COMMAND: tada list-vectors
- Prints a table of all available pre-computed vectors (concept, method, model, date, N-prompts)

COMMAND: tada status
- Prints environment check: GPU available, models cached, vectors computed, dependencies installed

Each command must:
- Log to file at outputs/logs/{command}_{timestamp}.log
- Handle keyboard interrupt gracefully
- Print colored output using `rich` (add to requirements)
- Validate that required model weights exist before starting any long computation

DONE WHEN: `tada status` runs and prints a valid environment check table.
Claude Code Prompt 1.5 — Steering Vector Bank

text
TASK: Create a SteeringVectorBank class for serialization, loading, and management of 
steering vectors as a first-class data type.

Create `steer_audio/vector_bank.py`:

class SteeringVector:
    """A computed steering vector with full provenance metadata."""
    concept: str          # e.g., "tempo"
    method: str           # "caa" or "sae"
    model_name: str       # e.g., "ace-step"
    layers: list[int]     # e.g., [6, 7]
    vector: torch.Tensor  # shape: (hidden_dim,) — unit norm for CAA, unnormalized for SAE
    alpha_range: tuple    # (min, max) empirically validated
    n_prompt_pairs: int   # N used in CAA computation
    tau: int | None       # top-tau features for SAE (None for CAA)
    created_at: str       # ISO 8601 timestamp
    clap_delta: float     # ΔAlignment at alpha=50 (quality signal)
    lpaps_at_50: float    # LPAPS at alpha=50 (preservation signal)

class SteeringVectorBank:
    """Registry and I/O for steering vectors."""
    
    def save(self, sv: SteeringVector, path: Path) -> None:
        # Save as .safetensors with metadata dict serialized as JSON in metadata field
        
    def load(self, path: Path) -> SteeringVector:
        # Load from .safetensors, deserialize metadata
        
    def load_all(self, directory: Path) -> dict[str, SteeringVector]:
        # Load all vectors in a directory, keyed by "{concept}_{method}"
        
    def compose(self, vectors: list[tuple[SteeringVector, float]]) -> dict[int, torch.Tensor]:
        # Compose multiple (vector, alpha) pairs into a per-layer activation delta dict
        # Returns: {layer_idx: Σ_c α_c · v_c} for all affected layers
        # Applies Gram-Schmidt orthogonalization if vectors share layers
        
    def summary_table(self) -> str:
        # Returns a rich-formatted table of all loaded vectors with metadata

Add a download function:
    def download_pretrained(concept: str, model: str = "ace-step") -> SteeringVector:
        # Downloads from HuggingFace Hub: "tada-steering-vectors/{model}/{concept}"
        # (Stub implementation — raises NotImplementedError with instructions for now)

DONE WHEN: 
  sv = SteeringVector(concept="tempo", method="caa", model_name="ace-step", 
                      layers=[6,7], vector=torch.randn(3072), ...)
  bank = SteeringVectorBank()
  bank.save(sv, Path("test_vector.safetensors"))
  sv2 = bank.load(Path("test_vector.safetensors"))
  assert sv2.concept == "tempo"
runs without error. Add a test in tests/test_vector_bank.py.
Phase 2 — Core Features (Weeks 3–5)
Goal: Implement the three highest-impact novel capabilities: multi-concept steering, timestep-adaptive steering, and the SAE concept algebra system. These are the features that differentiate this project from the original paper.

Sprint 2A: Multi-Concept Steering Engine
Claude Code Prompt 2.1 — Multi-Concept Steering with Orthogonalization

text
TASK: Implement simultaneous multi-concept steering with optional Gram-Schmidt 
orthogonalization and interference measurement.

Create `steer_audio/multi_steer.py`:

class MultiConceptSteerer:
    """
    Apply multiple steering vectors simultaneously during diffusion inference.
    
    Mathematical foundation:
    h'_l = ReNorm(h_l + Σ_c α_c · v_c, h_l)   [CAA multi-steer]
    h'_l = h_l + Σ_c α_c · v_c^SAE              [SAE multi-steer, no ReNorm]
    
    When orthogonalize=True, applies Gram-Schmidt to {v_c} before summation
    to minimize concept interference.
    """
    
    def __init__(self, vectors: dict[str, SteeringVector], orthogonalize: bool = False):
        self.vectors = vectors
        if orthogonalize:
            self._apply_gram_schmidt()
    
    def _apply_gram_schmidt(self) -> None:
        # In-place Gram-Schmidt orthogonalization of self.vectors
        # Process in order of decreasing clap_delta (steer most reliable concept first)
        # v_i ← v_i - Σ_{j<i} (v_i · v_j) * v_j, then renormalize
    
    def interference_matrix(self) -> torch.Tensor:
        # Returns (N_concepts × N_concepts) matrix of cosine similarities
        # Values near ±1 indicate high interference; near 0 is orthogonal
    
    def get_hooks(self, alphas: dict[str, float]) -> list[callable]:
        # Returns list of PyTorch forward hooks for injection into the model
        # Each hook is keyed to a specific layer index
        # Hook logic: h' = ReNorm(h + Σ_c alpha_c * v_c, h) for layers in any vector
    
    def steer(
        self, 
        model: any,                    # Ace-Step model instance
        prompt: str, 
        alphas: dict[str, float],      # {"tempo": 60, "mood": 40}
        duration: float = 30.0,
        seed: int = 42
    ) -> tuple[np.ndarray, int]:       # (audio_array, sample_rate)
        # Register hooks, run inference, deregister hooks, return audio

Create `experiments/multi_concept_experiment.py`:
- Test all 10 concept pairs from {tempo, mood, vocal_gender, guitar, drums, jazz, techno}
- For each pair: generate with individual steering vs. joint steering
- Compute interference matrix before and after orthogonalization
- Measure: CLAP per concept in joint vs. individual condition
- Save results to results/multi_concept/interference_matrix.csv

DONE WHEN: 
  steerer = MultiConceptSteerer({"tempo": sv_tempo, "mood": sv_mood})
  print(steerer.interference_matrix())  # prints 2x2 tensor
  audio, sr = steerer.steer(model, "a jazz song", {"tempo": 60, "mood": 40})
  assert audio.shape[0] > 0
Claude Code Prompt 2.2 — Timestep-Adaptive Steering

text
TASK: Implement timestep-adaptive steering schedules — apply different alpha values 
at different diffusion timesteps.

Scientific motivation: Early diffusion timesteps (high noise, t close to T) determine 
global structure (genre, mood, tempo). Later timesteps (low noise, t close to 0) refine 
fine details (specific instrument timbre, production quality). A cosine schedule that 
applies maximum alpha during early steps and tapers to zero should improve preservation 
metrics while maintaining concept alignment.

Create `steer_audio/temporal_steering.py`:

class TimestepSchedule(Protocol):
    """Protocol for alpha schedules over diffusion timesteps."""
    def __call__(self, t: int, T: int) -> float: ...

def constant_schedule(alpha: float) -> TimestepSchedule:
    return lambda t, T: alpha

def cosine_schedule(alpha_max: float, alpha_min: float = 0.0) -> TimestepSchedule:
    # alpha decreases from alpha_max to alpha_min following cosine curve
    # Peak at t/T = 1.0 (start of diffusion), trough at t/T = 0.0 (end)
    return lambda t, T: alpha_min + (alpha_max - alpha_min) * 0.5 * (1 + math.cos(math.pi * (1 - t/T)))

def early_only_schedule(alpha: float, cutoff: float = 0.5) -> TimestepSchedule:
    # Apply alpha only for t/T > cutoff (first half of diffusion process)
    return lambda t, T: alpha if t / T > cutoff else 0.0

def late_only_schedule(alpha: float, cutoff: float = 0.5) -> TimestepSchedule:
    # Apply alpha only for t/T <= cutoff (second half of diffusion process)
    return lambda t, T: alpha if t / T <= cutoff else 0.0

class TimestepAdaptiveSteerer:
    """
    Apply steering vectors with timestep-dependent alpha schedules.
    The hook reads the current diffusion timestep from the model's scheduler
    and applies the schedule function to determine the effective alpha.
    """
    
    def __init__(
        self, 
        vector: SteeringVector, 
        schedule: TimestepSchedule,
        layers: list[int] | None = None  # If None, use vector.layers
    ): ...
    
    def steer(self, model, prompt: str, duration: float, seed: int) -> tuple[np.ndarray, int]: ...

Create `experiments/timestep_schedule_experiment.py`:
- For each concept in {tempo, mood, instruments}:
  - Generate 32 samples each with: constant, cosine, early_only, late_only schedules
  - Evaluate LPAPS, ΔAlignment CLAP, smoothness (std), CE/CU/PC/PQ
  - Plot: alpha value vs. timestep step for each schedule
  - Plot: metric value vs. schedule type (bar chart, save as PNG)
- Hypothesis to test: cosine_schedule achieves better preservation (lower LPAPS) 
  than constant_schedule at same mean alpha

DONE WHEN: 
  steerer = TimestepAdaptiveSteerer(sv_tempo, cosine_schedule(alpha_max=80))
  audio, sr = steerer.steer(model, "a jazz song", duration=30.0, seed=42)
  sf.write("test_cosine.wav", audio, sr)
  # File exists and is valid audio
Sprint 2B: SAE Concept Algebra
Claude Code Prompt 2.3 — SAE Concept Algebra System

text
TASK: Implement SAE-based concept arithmetic — add, subtract, and intersect 
musical concepts via feature set operations on the SAE.

Mathematical foundation:
- Concept c is represented by its top-τ feature set: F_c ⊆ {1...|features|}
- Addition (A + B): F_A ∪ F_B → v = Σ_{j ∈ F_A ∪ F_B} W_dec[:,j]
- Subtraction (A - B): F_A \ F_B → v = Σ_{j ∈ F_A \ F_B} W_dec[:,j]
- Intersection (A ∩ B): F_A ∩ F_B → v = Σ_{j ∈ F_A ∩ F_B} W_dec[:,j]
- Weighted blend (0.7*A + 0.3*B): weight W_dec columns proportionally before summing

Create `steer_audio/concept_algebra.py`:

class ConceptFeatureSet:
    """
    Sparse representation of a concept as a set of SAE feature indices + scores.
    """
    concept: str
    feature_indices: np.ndarray      # top-τ feature indices, shape (τ,)
    tfidf_scores: np.ndarray         # corresponding TF-IDF scores, shape (τ,)
    decoder_matrix: torch.Tensor     # W_dec from trained SAE, shape (hidden_dim, num_features)
    
    def to_steering_vector(self, weight: float = 1.0) -> torch.Tensor:
        # Returns Σ_{j ∈ self.feature_indices} weight * W_dec[:,j]
    
    def overlap(self, other: 'ConceptFeatureSet') -> float:
        # Jaccard similarity |F_a ∩ F_b| / |F_a ∪ F_b|
    
    def __add__(self, other):    # Union of feature sets
    def __sub__(self, other):    # Set difference
    def __and__(self, other):    # Intersection
    def __mul__(self, scalar):   # Scale TF-IDF weights by scalar

class ConceptAlgebra:
    """
    Perform concept arithmetic and produce steering vectors from SAE feature algebra.
    """
    
    def __init__(self, sae_model: SAE, concept_features: dict[str, ConceptFeatureSet]):
        self.sae = sae_model
        self.features = concept_features
    
    def expr(self, expression: str) -> ConceptFeatureSet:
        """
        Parse and evaluate a concept algebra expression.
        
        Examples:
          algebra.expr("jazz + female_vocal - piano")
          algebra.expr("0.7 * jazz + 0.3 * techno")
          algebra.expr("fast_tempo & energetic_mood")  # intersection
        
        Grammar:
          expr := term (('+' | '-') term)*
          term := scalar? '*'? concept | '(' expr ')' | concept '&' concept
          scalar := float
          concept := identifier  # must be a key in self.features
        """
    
    def to_steering_vector(self, expr_result: ConceptFeatureSet) -> SteeringVector:
        # Convert algebra result to a SteeringVector for use in inference
    
    def feature_overlap_heatmap(self) -> plt.Figure:
        # Returns a matplotlib heatmap of pairwise Jaccard overlaps between all concepts

Create `experiments/concept_algebra_demo.py`:
- Load all concept feature sets for ACE-Step layer 7 SAE
- Demonstrate 5 algebra expressions with audio outputs for each:
  1. "jazz + female_vocal"            → jazz with female vocals
  2. "fast_tempo - drums"             → fast but without drums
  3. "0.5 * jazz + 0.5 * reggae"     → jazz-reggae hybrid
  4. "energetic_mood & guitar"        → energetic guitar music
  5. "slow_tempo - sad_mood"          → slow but not sad (e.g., peaceful ballad)
- For each: generate 4 samples, run CLAP evaluation against descriptive text prompt
- Save feature overlap heatmap as experiments/results/feature_overlap.png

DONE WHEN: All 5 demo expressions run without error and produce valid WAV files.
Sprint 2C: Self-Monitoring Controller
Claude Code Prompt 2.4 — SMITIN-Style Self-Monitoring

text
TASK: Implement a self-monitoring steering controller that prevents over-steering 
by measuring concept presence during inference and adaptively reducing alpha.

Scientific motivation: At high alpha values, CAA steering degrades audio quality 
(LPAPS increases, CE/PQ decrease). A controller that detects when the target concept 
is sufficiently present and reduces alpha prevents this degradation — similar to 
SMITIN (Koo et al. 2025, arXiv 2404.02252).

Create `steer_audio/self_monitor.py`:

class ConceptProbe:
    """
    Lightweight linear probe trained to detect concept presence from CLAP embeddings.
    Training data: CLAP embeddings of generated audio with positive/negative concept prompts.
    """
    
    def __init__(self, concept: str, clap_model: any):
        self.concept = concept
        self.classifier = LogisticRegression(max_iter=1000)  # sklearn
    
    def train(
        self, 
        positive_audio_paths: list[Path],  # Audio generated with positive prompts
        negative_audio_paths: list[Path]   # Audio generated with negative prompts
    ) -> float:
        # Extract CLAP embeddings, fit logistic regression, return train accuracy
    
    def predict_proba(self, audio: np.ndarray, sample_rate: int) -> float:
        # Returns P(concept present) in [0, 1]
    
    def save(self, path: Path) -> None: ...
    def load(cls, path: Path) -> 'ConceptProbe': ...

class SelfMonitoredSteerer:
    """
    Adaptive steering that reduces alpha when concept is detected.
    
    Algorithm (per diffusion step):
    1. Every `check_every` steps, decode partial latent → run CLAP → run probe
    2. If P(concept) > threshold_high: set effective_alpha = alpha * decay_factor
    3. If P(concept) < threshold_low: restore effective_alpha to original alpha
    4. Apply effective_alpha this step
    """
    
    def __init__(
        self,
        vector: SteeringVector,
        probe: ConceptProbe,
        alpha: float,
        threshold_high: float = 0.85,    # Stop boosting above this
        threshold_low: float = 0.40,     # Resume boosting below this
        decay_factor: float = 0.5,       # Halve alpha when concept detected
        check_every: int = 5             # Check every N diffusion steps
    ): ...
    
    def steer(self, model, prompt: str, duration: float, seed: int) -> tuple[np.ndarray, int]: ...
    
    def get_monitoring_trace(self) -> pd.DataFrame:
        # Returns dataframe: [step, effective_alpha, concept_probability, decoded_clap_score]
        # For visualization and debugging

Create `experiments/self_monitor_experiment.py`:
- Train probes for: tempo, mood, vocal_gender (use cached generated audio)
- Compare: fixed alpha vs. self-monitored for alpha in {50, 75, 100}
- Primary claim: self-monitoring achieves same ΔAlignment as fixed alpha=75 
  but with lower LPAPS and higher CE/PQ
- Save monitoring trace plots as PNG (timestep vs. effective_alpha)

DONE WHEN: ConceptProbe.train() achieves >75% accuracy on held-out set for 'tempo'.
Phase 3 — Product + Research (Weeks 6–8)
Goal: Deploy the Audio Attribute Studio and run the experiments that constitute a publishable research contribution.

Sprint 3A: The Audio Attribute Studio (App)
Claude Code Prompt 3.1 — Gradio Audio Attribute Studio

text
TASK: Build a production-quality Gradio application — the "Audio Attribute Studio" — 
that exposes all TADA capabilities through an intuitive browser interface.

This is the project's public face. Quality standard: HuggingFace Space deployment.

Create `app/studio.py`:

LAYOUT:
The app must have three tabs:

TAB 1: "Generate & Steer"
- Text prompt input (large, placeholder: "a jazz piano trio at a small club")
- Duration slider: 10–60 seconds
- Generation seed input
- Concept steering panel (shown as sliders):
  - Tempo: range -100 to +100, step 10, default 0
  - Mood (valence): range -100 to +100
  - Vocal Gender (feminine ← → masculine): range -100 to +100
  - Instruments row: Guitar / Drums / Flute / Violin / Trumpet (each -100 to +100)
  - Genre row: Jazz / Reggae / Techno (each -100 to +100)
- "Advanced" accordion:
  - Steering method: dropdown ["CAA", "SAE", "Multi-layer CAA"]
  - Orthogonalize concepts: checkbox (default True)
  - Schedule: dropdown ["Constant", "Cosine Decay", "Early Steps Only", "Late Steps Only"]
- Generate button
- Output: audio player (waveform visualization), spectrogram image, 
  auto-computed CLAP alignment score, generation time
- A/B comparison: side-by-side player (steered vs. unsteered baseline, computed simultaneously)

TAB 2: "SAE Feature Explorer"
- Concept selector: dropdown of all available concepts
- Feature importance chart: horizontal bar chart of top-20 TF-IDF scores
- Feature activation audio: click any feature bar → play the top-3 max-activating clips
- Concept algebra expression input: text field (e.g., "jazz + female_vocal - piano")
- "Evaluate Expression" button → generates audio from algebra result
- Feature overlap heatmap: precomputed image

TAB 3: "Batch Experiment"
- Upload a CSV of prompts (concept, prompt, alpha columns)
- Run batch evaluation
- Download results as CSV with all metrics (CLAP, FAD, LPAPS, CE, CU, PC, PQ)
- Progress bar

TECHNICAL REQUIREMENTS:
- Load all available SteeringVectors at startup using SteeringVectorBank.load_all()
- Use gr.State for model instance (load once, reuse)
- Wrap all generation in try/except with user-friendly error messages
- Show GPU memory usage in the footer
- All generations must be reproducible (deterministic given seed)
- Add `examples/` with 5 pre-defined prompts + alpha combinations for the examples panel

DEPLOYMENT CONFIG:
Create `app/requirements_space.txt` with pinned versions for HuggingFace Spaces.
Create `README_SPACE.md` with the HF Space metadata block:
  ---
  title: Audio Attribute Studio
  emoji: 🎵
  colorFrom: purple
  colorTo: blue
  sdk: gradio
  sdk_version: 5.x
  app_file: app/studio.py
  pinned: true
  ---

DONE WHEN: `python app/studio.py` launches locally, all three tabs render, 
and a generation with tempo=50 produces audio that is perceptibly faster than 
the unsteered baseline.
Sprint 3B: Research Experiments
Claude Code Prompt 3.2 — Scaling Laws for SAE Features

text
TASK: Implement and run the SAE scaling law experiment — the most publishable 
research contribution of this project.

Scientific question: How does SAE quality (reconstruction, feature interpretability, 
steering effectiveness) scale with expansion factor m, sparsity k, and training data size?

This is novel: no prior work has published scaling curves for SAEs in audio diffusion models.

Create `experiments/sae_scaling.py`:

EXPERIMENT GRID:
expansion_factors = [2, 4, 8, 16, 32]
k_values = [8, 16, 32, 64, 128]
data_sizes = [100, 500, 1000, 5000]  # Number of MusicCaps prompts
seeds = [42, 123, 456]  # 3 seeds per config for error bars

METRICS TO COLLECT for each (m, k, data_size, seed) configuration:
1. Fraction of Variance Unexplained (FVU): 1 - R² of reconstruction
2. Dead feature % (features that never activate over validation set)
3. Mean sparsity (actual average nonzero activations)
4. Top-concept steering ΔAlignment CLAP (use tempo concept, alpha=50, 32 samples)
5. Top-concept LPAPS (preservation at alpha=50)
6. Human-interpretable feature % (requires automated labeling — use CLAP similarity)

AUTOMATED INTERPRETABILITY SCORING:
For each SAE feature (up to top-50 by activation frequency):
1. Find top-5 max-activating audio clips
2. Compute CLAP similarity between each clip and 20 candidate concept labels
3. If max CLAP similarity > 0.4: feature is "interpretable"
4. interpretability_score = interpretable_features / total_active_features

ANALYSIS:
1. Fit power law: FVU ~ m^{-a} * k^{-b} → estimate a, b exponents
2. Find the Pareto frontier: configs that are non-dominated on (FVU, ΔAlignment, dead%)
3. Confirm paper's choice (m=4, k=64) is near-Pareto-optimal

OUTPUTS:
- experiments/results/scaling/all_results.csv (full grid)
- experiments/results/scaling/fvu_vs_expansion.png
- experiments/results/scaling/alignment_vs_k.png
- experiments/results/scaling/pareto_frontier.png
- experiments/results/scaling/summary_table.md (LaTeX-ready)

DONE WHEN: The smallest grid point (m=2, k=8, n=100) trains and evaluates successfully. 
Full grid estimated time: log to stdout ("Estimated: X GPU-hours").
Claude Code Prompt 3.3 — Steering Vector Geometry Analysis

text
TASK: Perform a geometric analysis of all computed steering vectors to characterize 
the structure of the musical concept subspace in Ace-Step's cross-attention activations.

Scientific question: Are musical concepts encoded as orthogonal linear directions 
(the "linear representation hypothesis") or as a tangled, non-linear subspace?

Create `experiments/vector_geometry.py`:

ANALYSIS 1 — Pairwise Cosine Similarities
- Load all available CAA vectors and SAE vectors for ACE-Step layer 7
- Compute pairwise cosine similarity matrix (all concepts × all concepts)
- Expected: orthogonal concepts (e.g., tempo vs. vocal_gender) ≈ 0
             related concepts (e.g., jazz vs. guitar) might be positively correlated
- Plot: annotated heatmap, save as vector_geometry/cosine_heatmap.png

ANALYSIS 2 — PCA of Concept Subspace
- Stack all concept vectors as rows: V ∈ R^{N_concepts × hidden_dim}
- Run PCA, plot explained variance ratio (scree plot)
- Plot first 2 PCs colored by concept category (tempo/mood/instrument/genre)
- Test: do instruments cluster together? Do tempo/mood concepts separate?
- Save: vector_geometry/pca_2d.png, vector_geometry/pca_variance.png

ANALYSIS 3 — Linear Probing
For each concept c:
  - Generate 256 audio samples with positive prompts, 256 with negative
  - Extract layer-7 cross-attention activations (mean-pooled over sequence and timesteps)
  - Train logistic regression probe on 80% train / 20% test split
  - Measure: probe accuracy, probe weight direction vs. v_c^CAA cosine similarity
- If probe weight aligns well with v_c: concept is linearly encoded
- Save: vector_geometry/probing_accuracy.csv

ANALYSIS 4 — Concept Arithmetic Verification
Test: v_fast_tempo + v_female_vocal ≈ direction toward "fast tempo with female vocal"?
- Generate 32 samples with combined prompt "fast female vocal music"
- Extract activations, project onto v_fast + v_female (cosine similarity)
- Compare: composed vs. individually steered samples
- Save: vector_geometry/arithmetic_verification.csv

ANALYSIS 5 — Layer Progression
For each concept: compute the CAA vector at ALL 24 layers (not just functional {6,7})
- Plot: cosine similarity between each layer's concept vector and layer 7's vector
- Expected: layers {6,7} have the "cleanest" concept encoding

All plots saved to experiments/results/geometry/.
Full analysis summary saved as experiments/results/geometry/report.md (markdown).

DONE WHEN: PCA plot generates and shows at least 2 distinct clusters.
Sprint 3C: Final Polish and Documentation
Claude Code Prompt 3.4 — Complete README and Tutorial Notebooks

text
TASK: Write a comprehensive README.md and three tutorial Jupyter notebooks.

README.md requirements:
- Header with: project title, description, paper citation (arXiv 2602.11910), 
  badges (Python 3.10+, PyTorch 2.x, HF Space, License: MIT)
- One-sentence hook: "TADA lets you steer any aspect of generated music — 
  tempo, mood, instruments, genre — without retraining, using activation-space 
  steering vectors discovered through mechanistic interpretability."
- Quick start (5 commands: git clone, pip install, download model, compute vector, generate)
- Architecture diagram (ASCII art showing the 3-pipeline structure)
- Available concepts table (concept, model, method, tau, CLAP delta)
- Link to HuggingFace Space demo
- Link to 3 tutorial notebooks
- Contribution guide

Create `notebooks/01_quickstart.ipynb`:
A 10-cell notebook that takes a complete beginner from zero to steered audio:
1. Environment check (GPU, dependencies)
2. Load ACE-Step model
3. Generate baseline (unsteered)
4. Load pre-computed tempo steering vector
5. Generate with CAA steering at alpha=50
6. Generate with CAA steering at alpha=-50
7. A/B audio comparison
8. Try 3 concepts with alpha sliders using ipywidgets
9. Multi-concept steering (tempo + mood)
10. Save to WAV

Create `notebooks/02_compute_your_own_vectors.ipynb`:
A 12-cell notebook walking through vector computation:
1–3: Activation patching to find functional layers (visualize impact scores)
4–6: Build contrastive prompt dataset for a custom concept
7–9: Compute CAA vector
10: Train SAE (small, 100 steps demo)
11: Compute SAE vector (TF-IDF scoring)
12: Compare CAA vs. SAE on custom concept

Create `notebooks/03_concept_algebra.ipynb`:
A 8-cell notebook on concept arithmetic:
1: Load SAE feature sets
2: Visualize feature overlap heatmap
3: "jazz + female_vocal" example
4: "fast_tempo - drums" example
5: "0.7*jazz + 0.3*reggae" blend
6: Interactive algebra expression evaluator (ipywidgets text input)
7: Evaluation with CLAP scores
8: Export algebra result as reusable SteeringVector

DONE WHEN: All three notebooks run cell-by-cell without errors on a machine 
with ACE-Step weights downloaded.
The Complete Build Order
text
WEEK 1:  Prompt 1.1 (Bug Fixes) → Prompt 1.2 (Config System) → Prompt 1.3 (Tests)
WEEK 2:  Prompt 1.4 (CLI) → Prompt 1.5 (Vector Bank)
WEEK 3:  Prompt 2.1 (Multi-Concept) → Prompt 2.2 (Timestep Adaptive)
WEEK 4:  Prompt 2.3 (Concept Algebra) → Prompt 2.4 (Self-Monitor)
WEEK 5:  Buffer / integrate and test all Phase 2 components end-to-end
WEEK 6:  Prompt 3.1 (Gradio Studio)
WEEK 7:  Prompt 3.2 (Scaling Laws) → Prompt 3.3 (Geometry Analysis)
WEEK 8:  Prompt 3.4 (README + Notebooks) → HF Space deployment → paper draft outline



Claude Code Meta-Instructions
These rules apply to EVERY session, regardless of which prompt you're running.

When Claude Code Gets Stuck
If Claude Code generates code that fails, use this recovery prompt:

text
The previous attempt failed with this error: [PASTE ERROR]

Before retrying, do the following:
1. Read the exact file that caused the error: [filename]
2. Identify the root cause (import error, shape mismatch, wrong API, etc.)
3. State your fix in one sentence
4. Make ONLY the minimum change required to fix this specific error
5. Do not refactor or improve anything else in the same change
When You Want Better Code Quality
text
Review the code you just wrote for `[filename]` against these standards:
1. Every function has a docstring with Args, Returns, Raises
2. Every tensor operation includes a comment with the expected shape
3. No magic numbers — all constants have named variables with explanatory comments
4. Error messages include the actual values that caused the error (f-strings)
5. Logging uses the `logging` module, not print()
6. All file paths use pathlib.Path, never os.path.join strings

List every violation, then fix them all.
When You Want Tests Written First (TDD)
text
Before writing any implementation for [feature], write the tests first.

Tests must:
1. Cover the happy path (correct input → correct output)
2. Cover at least 2 edge cases (empty input, extreme values, wrong types)
3. Cover at least 1 failure case (invalid input raises the right exception)
4. Use only the public interface — no testing of private methods

Show me the tests. I will review them before you write any implementation.
Committing Work Between Sessions
End every Claude Code session with:

text
Create a git commit for all changes made in this session.
Commit message format: "[Phase X.Y] Description — closes Prompt [N]"
Also update CHANGELOG.md with:
- What was added
- What was fixed  
- What tests now pass
- Any known limitations or TODOs remaining from this prompt
Definition of "Done" for the Entire Project
The project is complete when all of the following are true:

Criterion	Verification
Criterion	Verification
All existing experiments reproduce	pytest tests/ -v → 100% pass
Single-command generation works	tada generate --prompt "jazz" --concept tempo --alpha 60 --output out.wav
Multi-concept steering works	2+ concepts applied simultaneously, interference matrix computed
Concept algebra system works	5 demo expressions produce valid audio
Gradio app runs locally	python app/studio.py → browser opens, audio generates
HF Space deployed	Public URL accessible, GPU inference working
Scaling law data collected	experiments/results/scaling/all_results.csv exists with >50 configs
Geometry analysis complete	cosine heatmap, PCA plot, probing accuracy all generated
Tutorial notebooks run clean	All 3 notebooks execute without errors from top to bottom
README complete	All sections filled, quick start works for a new user
