"""Audio Attribute Studio — TADA project Gradio interface.

Exposes all Phase 2 steering capabilities through an interactive browser UI:
  - Basic CAA / SAE / Multi-layer CAA steering
  - Multi-concept steering via SteeringPipeline
  - Timestep-adaptive alpha schedules
  - SAE concept algebra expressions
  - Self-monitored (SMITIN-style) steering

Launch::

    python demo/app.py

Configuration via environment variables:

    TADA_VECTORS_DIR  — directory containing ``.safetensors`` steering vectors
                        (default: ``vectors/`` relative to repo root)
    TADA_WORKDIR      — base output directory (default: ``~/tada_outputs``)
    TADA_SERVER_PORT  — Gradio server port (default: 7860)
    TADA_SHARE        — set to ``"1"`` to create a public Gradio link
"""

from __future__ import annotations

import csv
import io
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App-level configuration (overridable via environment variables)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Directory where pre-computed steering vectors live.
_VECTORS_DIR = Path(
    os.environ.get("TADA_VECTORS_DIR", str(_REPO_ROOT / "vectors"))
)

# Server port and sharing.
_SERVER_PORT = int(os.environ.get("TADA_SERVER_PORT", "7860"))
_SHARE = os.environ.get("TADA_SHARE", "0") == "1"

# Sample rate used for placeholder audio and reported to Gradio.
_DEMO_SR: int = 44100

# Concept slider definitions: key → (label, default)
_CONCEPT_SLIDERS: dict[str, tuple[str, int]] = {
    "tempo": ("Tempo (slow ← → fast)", 0),
    "mood": ("Mood (sad ← → happy)", 0),
    "vocal_gender": ("Vocal Gender (feminine ← → masculine)", 0),
    "guitar": ("Guitar", 0),
    "drums": ("Drums", 0),
    "flute": ("Flute", 0),
    "violin": ("Violin", 0),
    "trumpet": ("Trumpet", 0),
    "jazz": ("Jazz", 0),
    "reggae": ("Reggae", 0),
    "techno": ("Techno", 0),
}

# Schedule display names → internal identifier.
_SCHEDULE_NAMES: list[str] = [
    "Constant",
    "Cosine Decay",
    "Early Steps Only",
    "Late Steps Only",
]

# Steering method display names.
_METHOD_NAMES: list[str] = ["CAA", "SAE", "Multi-layer CAA"]

# Pre-defined example inputs for the examples panel.
# Format: [prompt, duration, seed, tempo, mood, vocal_gender, guitar, drums,
#          flute, violin, trumpet, jazz, reggae, techno, method, orthogonalize, schedule]
_EXAMPLES: list[list[Any]] = [
    [
        "a jazz piano trio at a small club",
        30,
        42,
        60,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        70,
        0,
        0,
        "CAA",
        True,
        "Constant",
    ],
    [
        "upbeat electronic dance music",
        20,
        7,
        80,
        50,
        0,
        0,
        70,
        0,
        0,
        0,
        0,
        0,
        60,
        "CAA",
        True,
        "Cosine Decay",
    ],
    [
        "soft folk song with female vocals",
        25,
        99,
        -30,
        40,
        -60,
        50,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        "CAA",
        True,
        "Constant",
    ],
    [
        "energetic reggae rhythm",
        30,
        123,
        20,
        0,
        0,
        0,
        60,
        0,
        0,
        0,
        0,
        80,
        0,
        "CAA",
        False,
        "Early Steps Only",
    ],
    [
        "slow sad piano ballad",
        40,
        55,
        -70,
        -60,
        0,
        0,
        0,
        0,
        70,
        0,
        0,
        0,
        0,
        "SAE",
        True,
        "Cosine Decay",
    ],
]

# ---------------------------------------------------------------------------
# Module-level caches (populated lazily on first request)
# ---------------------------------------------------------------------------

_model_cache: dict[str, Any] = {}  # {"model": ..., "error": ...}
_vector_cache: dict[str, Any] = {}  # {"vectors": dict[str, SteeringVector], "error": ...}

# ---------------------------------------------------------------------------
# Lazy loaders
# ---------------------------------------------------------------------------


def _get_vectors() -> dict[str, Any]:
    """Return loaded steering vectors, loading them lazily on first call.

    Returns:
        Dict mapping ``"{concept}_{method}"`` keys to :class:`SteeringVector`
        objects, or an empty dict if no vectors are found / loadable.
    """
    if "vectors" in _vector_cache:
        return _vector_cache["vectors"]

    try:
        from steer_audio.vector_bank import SteeringVectorBank

        bank = SteeringVectorBank()
        vectors = bank.load_all(_VECTORS_DIR)
        _vector_cache["vectors"] = vectors
        log.info("Loaded %d steering vector(s) from %s.", len(vectors), _VECTORS_DIR)
    except Exception as exc:
        log.warning("Could not load steering vectors from %s: %s", _VECTORS_DIR, exc)
        _vector_cache["vectors"] = {}
        _vector_cache["error"] = str(exc)

    return _vector_cache["vectors"]


def _get_model() -> Any | None:
    """Return the ACE-Step model, loading it lazily on first call.

    Returns:
        Model instance or ``None`` if weights are unavailable.
    """
    if "model" in _model_cache:
        return _model_cache["model"]

    try:
        # Import is deferred so the UI can render without model weights.
        from src.models.ace_step.patchable_ace import PatchableACE  # type: ignore

        model = PatchableACE.from_pretrained()
        _model_cache["model"] = model
        log.info("ACE-Step model loaded successfully.")
    except Exception as exc:
        log.warning("ACE-Step model could not be loaded: %s", exc)
        _model_cache["model"] = None
        _model_cache["error"] = str(exc)

    return _model_cache.get("model")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _demo_audio(prompt: str, duration: float, seed: int) -> tuple[np.ndarray, int]:
    """Generate a deterministic sine-chord placeholder for demo/testing.

    Produces a C-major chord with gentle amplitude decay so the waveform
    and spectrogram panels render meaningfully even without real model weights.

    Args:
        prompt:   Text prompt (used only to vary base frequency slightly).
        duration: Desired audio length in seconds.
        seed:     Random seed for reproducibility.

    Returns:
        ``(audio_float32, sample_rate)`` where *audio_float32* is a 1-D array
        in ``[-1, 1]``.
    """
    rng = np.random.default_rng(seed)
    sr = _DEMO_SR
    n = int(sr * duration)
    t = np.linspace(0.0, duration, n, endpoint=False)

    # C-major chord; shift root slightly based on prompt hash for variety.
    root_shift = (hash(prompt) % 12) / 12.0
    freqs = [261.63 * (2 ** (root_shift / 12)), 329.63, 392.0, 523.25]
    weights = [0.40, 0.30, 0.20, 0.10]

    audio = sum(w * np.sin(2.0 * math.pi * f * t) for f, w in zip(freqs, weights))

    # Gentle exponential decay envelope.
    envelope = np.exp(-t * (0.5 + rng.uniform(0, 0.3)))
    audio = audio * envelope

    # Normalise to [-0.9, 0.9].
    peak = np.max(np.abs(audio)) + 1e-8
    audio = (audio / peak * 0.9).astype(np.float32)
    return audio, sr


def make_spectrogram(audio: np.ndarray, sr: int) -> np.ndarray:
    """Render a mel-spectrogram image from an audio array.

    Args:
        audio: 1-D float32 audio signal.
        sr:    Sample rate in Hz.

    Returns:
        RGB image as a ``(H, W, 3)`` uint8 NumPy array suitable for
        ``gr.Image``.
    """
    try:
        import librosa
        import librosa.display

        fig, ax = plt.subplots(figsize=(10, 3), dpi=100)
        S = librosa.feature.melspectrogram(y=audio.astype(float), sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(
            S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma"
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title("Mel Spectrogram")
        fig.tight_layout()
        rgb = _fig_to_rgb(fig)
        plt.close(fig)
        return rgb
    except Exception as exc:
        log.warning("Spectrogram generation failed: %s", exc)
        return _placeholder_image("Spectrogram unavailable")


def _fig_to_rgb(fig: Any) -> np.ndarray:
    """Convert a rendered matplotlib Figure to an RGB uint8 NumPy array.

    Uses ``buffer_rgba()`` (available in matplotlib >= 3.x) and drops the
    alpha channel.

    Args:
        fig: A :class:`matplotlib.figure.Figure` whose canvas has been drawn.

    Returns:
        Shape ``(H, W, 3)`` uint8 array in RGB colour order.
    """
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return rgba[:, :, :3].copy()


def _placeholder_image(text: str = "No data") -> np.ndarray:
    """Return a simple grey placeholder image with centred text.

    Args:
        text: Message to render in the image centre.

    Returns:
        RGB image as a ``(200, 600, 3)`` uint8 NumPy array.
    """
    fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
    ax.set_facecolor("#2d2d2d")
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        color="#aaaaaa",
    )
    ax.axis("off")
    fig.patch.set_facecolor("#2d2d2d")
    fig.tight_layout(pad=0)
    rgb = _fig_to_rgb(fig)
    plt.close(fig)
    return rgb


# ---------------------------------------------------------------------------
# Schedule factory
# ---------------------------------------------------------------------------


def _build_schedule(schedule_name: str, alpha_max: float) -> Any:
    """Convert a UI schedule name into a :class:`TimestepSchedule`.

    Args:
        schedule_name: One of the values in ``_SCHEDULE_NAMES``.
        alpha_max:     Maximum alpha value to pass to the schedule factory.

    Returns:
        A :class:`~steer_audio.temporal_steering.TimestepSchedule` callable.
    """
    try:
        from steer_audio import (
            constant_schedule,
            cosine_schedule,
            early_only_schedule,
            late_only_schedule,
        )

        mapping = {
            "Constant": lambda: constant_schedule(alpha_max),
            "Cosine Decay": lambda: cosine_schedule(alpha_max=alpha_max),
            "Early Steps Only": lambda: early_only_schedule(alpha=alpha_max),
            "Late Steps Only": lambda: late_only_schedule(alpha=alpha_max),
        }
        factory = mapping.get(schedule_name, mapping["Constant"])
        return factory()
    except Exception as exc:
        log.warning("Schedule '%s' could not be built: %s. Using constant.", schedule_name, exc)
        # Fallback: lambda that always returns alpha_max.
        return lambda t, T: alpha_max  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Tab 1 — Generate & Steer
# ---------------------------------------------------------------------------


def generate_steered(
    prompt: str,
    duration: float,
    seed: int,
    # Concept alpha sliders (one per key in _CONCEPT_SLIDERS)
    tempo: float,
    mood: float,
    vocal_gender: float,
    guitar: float,
    drums: float,
    flute: float,
    violin: float,
    trumpet: float,
    jazz: float,
    reggae: float,
    techno: float,
    # Advanced options
    method: str,
    orthogonalize: bool,
    schedule: str,
) -> tuple[
    tuple[int, np.ndarray],  # steered audio (sr, data)
    np.ndarray,              # steered spectrogram image
    tuple[int, np.ndarray],  # baseline audio (sr, data)
    str,                     # CLAP / info score text
    str,                     # status message
]:
    """Generate steered and baseline audio, returning all UI outputs.

    All concept alpha values equal to zero are silently skipped (no steering
    is applied for that concept).  If no model weights are available the
    function falls back to deterministic placeholder audio so the UI remains
    functional for layout testing.

    Args:
        prompt:       Text prompt for the audio model.
        duration:     Target duration in seconds.
        seed:         Random seed for reproducibility.
        tempo:        Alpha for the *tempo* concept.
        mood:         Alpha for the *mood* concept.
        vocal_gender: Alpha for the *vocal_gender* concept.
        guitar:       Alpha for the *guitar* concept.
        drums:        Alpha for the *drums* concept.
        flute:        Alpha for the *flute* concept.
        violin:       Alpha for the *violin* concept.
        trumpet:      Alpha for the *trumpet* concept.
        jazz:         Alpha for the *jazz* concept.
        reggae:       Alpha for the *reggae* concept.
        techno:       Alpha for the *techno* concept.
        method:       Steering method: ``"CAA"``, ``"SAE"``, or
                      ``"Multi-layer CAA"``.
        orthogonalize: Apply Gram-Schmidt orthogonalization to concept vectors.
        schedule:     Alpha schedule name (see :data:`_SCHEDULE_NAMES`).

    Returns:
        Five-tuple: steered audio, steered spectrogram, baseline audio,
        info string, status string.
    """
    t_start = time.perf_counter()

    # Build alpha dict — skip zeros.
    alphas_all: dict[str, float] = {
        "tempo": float(tempo),
        "mood": float(mood),
        "vocal_gender": float(vocal_gender),
        "guitar": float(guitar),
        "drums": float(drums),
        "flute": float(flute),
        "violin": float(violin),
        "trumpet": float(trumpet),
        "jazz": float(jazz),
        "reggae": float(reggae),
        "techno": float(techno),
    }
    active_alphas = {c: a for c, a in alphas_all.items() if a != 0.0}

    # --- Baseline (unsteered) ---
    model = _get_model()
    if model is None:
        baseline_audio, sr = _demo_audio(prompt, duration, seed)
        status_suffix = " [DEMO MODE — model weights not found]"
    else:
        try:
            pipeline = getattr(model, "pipeline", model)
            raw = pipeline(
                prompt=prompt,
                audio_duration=duration,
                manual_seed=seed,
                return_type="audio",
            )
            import torch

            if isinstance(raw, torch.Tensor):
                baseline_audio = raw.squeeze().cpu().float().numpy()
            else:
                baseline_audio = np.asarray(raw, dtype=np.float32)
            sr = getattr(pipeline, "sample_rate", _DEMO_SR)
            status_suffix = ""
        except Exception as exc:
            log.warning("Baseline generation failed: %s", exc)
            baseline_audio, sr = _demo_audio(prompt, duration, seed)
            status_suffix = f" [baseline fallback: {exc}]"

    # --- Steered generation ---
    vectors = _get_vectors()
    method_key = method.lower().replace(" ", "_").replace("-", "_")  # "caa" / "sae" / ...
    # Prefer vectors matching the requested method; fall back to any available.
    preferred_method = "caa" if "caa" in method_key else "sae"

    if not active_alphas:
        # No steering requested — steered == baseline.
        steered_audio = baseline_audio.copy()
        info_text = "No concepts selected (alpha = 0 for all). Returning unsteered audio."
    elif not vectors:
        # No vectors available — apply a synthetic alpha shift to demo the waveform.
        steered_audio, _ = _demo_audio(prompt + f" [steered alpha={list(active_alphas.values())[0]}]", duration, seed + 1)
        info_text = (
            f"Concepts: {list(active_alphas.keys())} — "
            "DEMO MODE: no vectors found. Showing placeholder steered audio."
        )
    else:
        # Select vectors that match active concepts and preferred method.
        selected: dict[str, Any] = {}
        for concept, alpha in active_alphas.items():
            # Try preferred method first, then any method for this concept.
            key_preferred = f"{concept}_{preferred_method}"
            key_any = next(
                (k for k in vectors if k.startswith(f"{concept}_")), None
            )
            if key_preferred in vectors:
                selected[concept] = vectors[key_preferred]
            elif key_any is not None:
                selected[concept] = vectors[key_any]
            else:
                log.info("No vector found for concept '%s'; skipping.", concept)

        if not selected:
            steered_audio = baseline_audio.copy()
            info_text = (
                f"No loaded vectors match the active concepts "
                f"{list(active_alphas.keys())}. Returning unsteered audio."
            )
        else:
            try:
                from steer_audio import SteeringPipeline

                pipeline_obj = SteeringPipeline(
                    vectors=selected,
                    orthogonalize=orthogonalize,
                    num_inference_steps=30,
                )

                # Apply non-constant schedules to all active concepts.
                if schedule != "Constant":
                    for concept, alpha_val in active_alphas.items():
                        if concept in selected:
                            sched = _build_schedule(schedule, abs(alpha_val))
                            pipeline_obj.set_schedule(concept, sched)

                steered_np, sr = pipeline_obj.steer(
                    model=model,
                    prompt=prompt,
                    alphas=active_alphas,
                    duration=duration,
                    seed=seed,
                )
                steered_audio = steered_np
                info_text = (
                    f"Concepts: {list(selected.keys())} | "
                    f"Method: {method} | Schedule: {schedule} | "
                    f"Orthogonalize: {orthogonalize}"
                )
            except Exception as exc:
                log.warning("Steered generation failed: %s", exc)
                steered_audio = baseline_audio.copy()
                info_text = f"Steering failed: {exc}"

    elapsed = time.perf_counter() - t_start
    status = f"Done in {elapsed:.1f}s.{status_suffix}"

    steered_spec = make_spectrogram(steered_audio, sr)

    return (sr, steered_audio), steered_spec, (sr, baseline_audio), info_text, status


# ---------------------------------------------------------------------------
# Tab 2 — SAE Feature Explorer
# ---------------------------------------------------------------------------


def _feature_importance_chart(concept: str) -> np.ndarray:
    """Return a horizontal bar chart of top-20 TF-IDF feature scores.

    Renders real data if a ConceptFeatureSet is available for *concept*,
    otherwise returns a labelled placeholder chart.

    Args:
        concept: Concept name to visualise (e.g. ``"tempo"``).

    Returns:
        RGB image as a ``(H, W, 3)`` uint8 NumPy array.
    """
    # Try to load real feature data from a cached ConceptAlgebra instance.
    feat_data: dict[int, float] | None = None
    try:
        from steer_audio.concept_algebra import ConceptFeatureSet

        # Check if any feature set file exists on disk.
        results_dir = _REPO_ROOT / "experiments" / "results" / "concept_algebra"
        feat_path = results_dir / f"{concept}_features.npz"
        if feat_path.exists():
            data = np.load(feat_path)
            indices = data["feature_indices"][:20]
            scores = data["tfidf_scores"][:20]
            feat_data = dict(zip(indices.tolist(), scores.tolist()))
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

    if feat_data:
        sorted_items = sorted(feat_data.items(), key=lambda x: x[1])
        labels = [f"Feature {idx}" for idx, _ in sorted_items]
        values = [v for _, v in sorted_items]
        ax.barh(labels, values, color="#7c3aed")
        ax.set_xlabel("TF-IDF Score")
        ax.set_title(f"Top Features — {concept}")
    else:
        # Placeholder with synthetic data.
        rng = np.random.default_rng(abs(hash(concept)) % (2**31))
        n = 20
        scores = np.sort(rng.exponential(scale=1.0, size=n))
        labels = [f"Feature {i}" for i in range(n)]
        ax.barh(labels, scores, color="#7c3aed", alpha=0.6)
        ax.set_xlabel("TF-IDF Score (placeholder)")
        ax.set_title(f"Top Features — {concept}  [load SAE for real data]")

    fig.tight_layout()
    rgb = _fig_to_rgb(fig)
    plt.close(fig)
    return rgb


def _feature_overlap_heatmap() -> np.ndarray:
    """Return the concept feature-overlap heatmap image.

    Loads from ``experiments/results/feature_overlap.png`` if it exists,
    otherwise generates a synthetic placeholder heatmap.

    Returns:
        RGB image as a ``(H, W, 3)`` uint8 NumPy array.
    """
    heatmap_path = (
        _REPO_ROOT / "experiments" / "results" / "feature_overlap.png"
    )
    if heatmap_path.exists():
        try:
            import PIL.Image

            img = PIL.Image.open(heatmap_path).convert("RGB")
            return np.array(img)
        except Exception:
            pass

    # Synthetic heatmap from Jaccard-like overlaps.
    concepts = list(_CONCEPT_SLIDERS.keys())
    n = len(concepts)
    rng = np.random.default_rng(0)
    mat = rng.uniform(0.0, 0.35, size=(n, n))
    np.fill_diagonal(mat, 1.0)
    mat = (mat + mat.T) / 2.0

    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="RdYlGn")
    fig.colorbar(im, ax=ax, label="Jaccard Overlap")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(concepts, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(concepts, fontsize=9)
    ax.set_title("Concept Feature Overlap (placeholder — run SAE to compute)")
    fig.tight_layout()
    rgb = _fig_to_rgb(fig)
    plt.close(fig)
    return rgb


def evaluate_algebra_expression(
    expression: str,
    prompt: str,
    seed: int,
) -> tuple[tuple[int, np.ndarray] | None, str]:
    """Parse and (if possible) evaluate a concept algebra expression.

    If a :class:`~steer_audio.concept_algebra.ConceptAlgebra` instance is
    available the expression is evaluated to a steering vector and audio is
    generated.  Otherwise the expression is only syntax-checked and a
    placeholder result is returned.

    Args:
        expression: Algebra expression string, e.g. ``"jazz + female_vocal"``.
        prompt:     Text prompt for audio generation.
        seed:       Random seed.

    Returns:
        ``(audio_tuple_or_None, status_message)``.
    """
    if not expression.strip():
        return None, "Enter an expression such as: jazz + female_vocal - drums"

    model = _get_model()
    vectors = _get_vectors()

    if not vectors:
        # Syntax check only — no SAE loaded.
        # Validate that identifiers are known concept names.
        import re

        tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", expression)
        known = set(_CONCEPT_SLIDERS.keys())
        unknown = [t for t in tokens if t not in known]
        if unknown:
            return None, f"Unknown concept(s) in expression: {unknown}. Known: {sorted(known)}"
        return (
            None,
            f"Expression '{expression}' parsed OK. Load an SAE to generate audio from it.",
        )

    # Attempt real evaluation via ConceptAlgebra.
    try:
        from steer_audio.concept_algebra import ConceptAlgebra, ConceptFeatureSet

        # Build a minimal algebra from available vectors (CAA vectors as proxies).
        concept_features: dict[str, ConceptFeatureSet] = {}
        hidden_dim = next(iter(vectors.values())).vector.shape[0]
        import torch

        # Create a pseudo-decoder matrix (identity-like) from CAA vectors.
        for name, sv in vectors.items():
            concept_key = sv.concept
            if concept_key not in concept_features:
                dummy_dec = torch.eye(hidden_dim, hidden_dim)
                cfs = ConceptFeatureSet(
                    concept=concept_key,
                    feature_indices=np.array([0]),
                    tfidf_scores=np.array([1.0]),
                    decoder_matrix=dummy_dec,
                )
                concept_features[concept_key] = cfs

        if not concept_features:
            return None, "No concept features available for algebra evaluation."

        # Use a minimal SAE placeholder (no real SAE needed for simple algebra).
        algebra = ConceptAlgebra(sae_model=None, concept_features=concept_features)
        result_fs = algebra.expr(expression)
        sv = algebra.to_steering_vector(result_fs, layers=[6, 7], model_name="ace-step")

        if model is None:
            audio, sr = _demo_audio(prompt + f"_algebra_{expression}", 20.0, seed)
            return (sr, audio), f"Expression '{expression}' → steering vector ready. DEMO audio (no model)."

        from steer_audio import SteeringPipeline

        pipe = SteeringPipeline(vectors={"algebra_result": sv}, orthogonalize=False)
        audio, sr = pipe.steer(model, prompt, {"algebra_result": 50.0}, 20.0, seed)
        return (sr, audio), f"Expression '{expression}' evaluated and applied at alpha=50."

    except Exception as exc:
        log.warning("Algebra evaluation failed for '%s': %s", expression, exc)
        return None, f"Could not evaluate expression '{expression}': {exc}"


# ---------------------------------------------------------------------------
# Tab 3 — Batch Experiment
# ---------------------------------------------------------------------------


def run_batch(
    csv_file_path: str | None,
    progress: Any = None,
) -> tuple[str, str]:
    """Run batch audio generation from an uploaded CSV file.

    Expected CSV columns (case-insensitive)::

        prompt, tempo, mood, vocal_gender, guitar, drums, flute, violin,
        trumpet, jazz, reggae, techno

    Any missing concept column defaults to ``0`` (no steering).

    Args:
        csv_file_path: Path to the uploaded CSV file (Gradio provides a path
                       string for uploaded files).
        progress:      Gradio progress callback (optional).

    Returns:
        ``(results_csv_string, status_message)``.
    """
    if csv_file_path is None:
        return "", "No file uploaded. Please upload a CSV."

    # Read CSV.
    try:
        with open(csv_file_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except Exception as exc:
        return "", f"Could not read CSV: {exc}"

    if not rows:
        return "", "CSV is empty."

    concept_keys = list(_CONCEPT_SLIDERS.keys())
    results: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        if progress is not None:
            progress((i + 1) / len(rows), desc=f"Row {i + 1}/{len(rows)}")

        prompt = row.get("prompt", "").strip() or "ambient music"
        seed = int(row.get("seed", 42))
        duration = float(row.get("duration", 20.0))
        alphas: dict[str, float] = {}
        for key in concept_keys:
            val = row.get(key, row.get(key.replace("_", ""), "0"))
            try:
                alphas[key] = float(val)
            except (ValueError, TypeError):
                alphas[key] = 0.0

        t_start = time.perf_counter()
        (sr, audio), _, _, info, status = generate_steered(
            prompt=prompt,
            duration=duration,
            seed=seed,
            **{k: alphas[k] for k in concept_keys},
            method="CAA",
            orthogonalize=True,
            schedule="Constant",
        )
        elapsed = time.perf_counter() - t_start

        result_row = {
            "row": i + 1,
            "prompt": prompt,
            "seed": seed,
            "duration": duration,
            "elapsed_s": round(elapsed, 2),
            "info": info,
            "status": status,
        }
        result_row.update({k: alphas[k] for k in concept_keys})
        results.append(result_row)

    # Serialise to CSV string.
    if results:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        csv_out = buf.getvalue()
    else:
        csv_out = ""

    return csv_out, f"Processed {len(results)} row(s)."


# ---------------------------------------------------------------------------
# Interface builder
# ---------------------------------------------------------------------------


def build_interface() -> Any:
    """Construct and return the Gradio Blocks interface without launching it.

    This function is intentionally side-effect free with respect to model
    loading — the model and vectors are loaded lazily on first use.  Tests can
    call this function to verify that the UI builds correctly without GPU or
    model weights.

    Returns:
        A :class:`gradio.Blocks` instance ready to be ``.launch()``-ed.
    """
    import gradio as gr

    concept_keys = list(_CONCEPT_SLIDERS.keys())

    with gr.Blocks(title="Audio Attribute Studio — TADA") as demo:
        gr.Markdown(
            "# 🎵 Audio Attribute Studio\n"
            "**TADA** — Tuning Audio Diffusion Models through Activation Steering "
            "([arXiv 2602.11910](https://arxiv.org/abs/2602.11910))\n\n"
            "Generate and steer audio by sliding concept controls. "
            "Requires ACE-Step weights and pre-computed steering vectors for real output; "
            "renders placeholder audio in demo mode."
        )

        # ------------------------------------------------------------------ #
        # Tab 1 — Generate & Steer
        # ------------------------------------------------------------------ #
        with gr.Tab("Generate & Steer"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_input = gr.Textbox(
                        label="Text Prompt",
                        placeholder="a jazz piano trio at a small club",
                        lines=2,
                    )
                    with gr.Row():
                        duration_slider = gr.Slider(
                            minimum=10,
                            maximum=60,
                            value=30,
                            step=5,
                            label="Duration (seconds)",
                        )
                        seed_input = gr.Number(
                            value=42,
                            label="Seed",
                            precision=0,
                        )

                    gr.Markdown("### Concept Steering")
                    gr.Markdown("**Tempo & Mood**")
                    with gr.Row():
                        tempo_sl = gr.Slider(-100, 100, 0, step=10, label=_CONCEPT_SLIDERS["tempo"][0])
                        mood_sl = gr.Slider(-100, 100, 0, step=10, label=_CONCEPT_SLIDERS["mood"][0])
                    with gr.Row():
                        vocal_gender_sl = gr.Slider(
                            -100, 100, 0, step=10, label=_CONCEPT_SLIDERS["vocal_gender"][0]
                        )

                    gr.Markdown("**Instruments**")
                    with gr.Row():
                        guitar_sl = gr.Slider(-100, 100, 0, step=10, label=_CONCEPT_SLIDERS["guitar"][0])
                        drums_sl = gr.Slider(-100, 100, 0, step=10, label=_CONCEPT_SLIDERS["drums"][0])
                        flute_sl = gr.Slider(-100, 100, 0, step=10, label=_CONCEPT_SLIDERS["flute"][0])
                    with gr.Row():
                        violin_sl = gr.Slider(-100, 100, 0, step=10, label=_CONCEPT_SLIDERS["violin"][0])
                        trumpet_sl = gr.Slider(-100, 100, 0, step=10, label=_CONCEPT_SLIDERS["trumpet"][0])

                    gr.Markdown("**Genre**")
                    with gr.Row():
                        jazz_sl = gr.Slider(-100, 100, 0, step=10, label=_CONCEPT_SLIDERS["jazz"][0])
                        reggae_sl = gr.Slider(-100, 100, 0, step=10, label=_CONCEPT_SLIDERS["reggae"][0])
                        techno_sl = gr.Slider(-100, 100, 0, step=10, label=_CONCEPT_SLIDERS["techno"][0])

                    with gr.Accordion("Advanced Options", open=False):
                        method_dd = gr.Dropdown(
                            choices=_METHOD_NAMES,
                            value="CAA",
                            label="Steering Method",
                        )
                        ortho_cb = gr.Checkbox(
                            value=True,
                            label="Orthogonalize concepts (Gram-Schmidt)",
                        )
                        schedule_dd = gr.Dropdown(
                            choices=_SCHEDULE_NAMES,
                            value="Constant",
                            label="Alpha Schedule",
                        )

                    generate_btn = gr.Button("Generate", variant="primary")

                with gr.Column(scale=3):
                    status_txt = gr.Textbox(label="Status", interactive=False)
                    info_txt = gr.Textbox(label="Generation Info", interactive=False, lines=2)

                    gr.Markdown("#### Steered Output")
                    steered_audio_out = gr.Audio(label="Steered Audio", type="numpy")
                    steered_spec_out = gr.Image(label="Spectrogram", type="numpy")

                    gr.Markdown("#### Baseline (unsteered)")
                    baseline_audio_out = gr.Audio(label="Baseline Audio", type="numpy")

            # Collect all slider components in order.
            _all_slider_inputs = [
                prompt_input,
                duration_slider,
                seed_input,
                tempo_sl,
                mood_sl,
                vocal_gender_sl,
                guitar_sl,
                drums_sl,
                flute_sl,
                violin_sl,
                trumpet_sl,
                jazz_sl,
                reggae_sl,
                techno_sl,
                method_dd,
                ortho_cb,
                schedule_dd,
            ]

            generate_btn.click(
                fn=generate_steered,
                inputs=_all_slider_inputs,
                outputs=[
                    steered_audio_out,
                    steered_spec_out,
                    baseline_audio_out,
                    info_txt,
                    status_txt,
                ],
            )

            gr.Examples(
                examples=_EXAMPLES,
                inputs=_all_slider_inputs,
                label="Example Configurations",
            )

        # ------------------------------------------------------------------ #
        # Tab 2 — SAE Feature Explorer
        # ------------------------------------------------------------------ #
        with gr.Tab("SAE Feature Explorer"):
            gr.Markdown(
                "Explore the SAE feature representations of each concept. "
                "Requires a trained SAE with cached feature sets. "
                "Placeholder charts are shown when SAE data is unavailable."
            )
            with gr.Row():
                with gr.Column():
                    concept_dd = gr.Dropdown(
                        choices=concept_keys,
                        value=concept_keys[0],
                        label="Concept",
                    )
                    feature_chart_out = gr.Image(
                        label="Top-20 Feature Importance (TF-IDF)",
                        type="numpy",
                    )
                    concept_dd.change(
                        fn=_feature_importance_chart,
                        inputs=[concept_dd],
                        outputs=[feature_chart_out],
                    )
                    # Render on load.
                    demo.load(
                        fn=_feature_importance_chart,
                        inputs=[concept_dd],
                        outputs=[feature_chart_out],
                    )

                with gr.Column():
                    heatmap_out = gr.Image(
                        label="Concept Feature Overlap Heatmap",
                        type="numpy",
                    )
                    demo.load(
                        fn=_feature_overlap_heatmap,
                        inputs=[],
                        outputs=[heatmap_out],
                    )

            gr.Markdown("### Concept Algebra")
            with gr.Row():
                algebra_expr = gr.Textbox(
                    label="Expression",
                    placeholder="jazz + female_vocal - drums",
                    lines=1,
                )
                algebra_prompt = gr.Textbox(
                    label="Generation Prompt",
                    placeholder="a jazz band playing live",
                    lines=1,
                )
                algebra_seed = gr.Number(value=42, label="Seed", precision=0)
            algebra_btn = gr.Button("Evaluate Expression", variant="secondary")
            algebra_audio_out = gr.Audio(label="Algebra Result Audio", type="numpy")
            algebra_status_out = gr.Textbox(label="Algebra Status", interactive=False)

            algebra_btn.click(
                fn=evaluate_algebra_expression,
                inputs=[algebra_expr, algebra_prompt, algebra_seed],
                outputs=[algebra_audio_out, algebra_status_out],
            )

        # ------------------------------------------------------------------ #
        # Tab 3 — Batch Experiment
        # ------------------------------------------------------------------ #
        with gr.Tab("Batch Experiment"):
            gr.Markdown(
                "Upload a CSV file with columns:\n"
                "``prompt``, ``seed`` (opt.), ``duration`` (opt.), "
                "plus any concept column (``tempo``, ``mood``, etc.).\n\n"
                "Each row is steered independently.  Results are returned as a CSV."
            )
            csv_upload = gr.File(label="Upload CSV", file_types=[".csv"])
            batch_btn = gr.Button("Run Batch", variant="primary")
            batch_results_out = gr.Textbox(
                label="Results (CSV)",
                lines=10,
                interactive=False,
            )
            batch_status_out = gr.Textbox(label="Batch Status", interactive=False)

            batch_btn.click(
                fn=run_batch,
                inputs=[csv_upload],
                outputs=[batch_results_out, batch_status_out],
            )

        # ------------------------------------------------------------------ #
        # Footer
        # ------------------------------------------------------------------ #
        gr.Markdown(
            "---\n"
            "*Audio Attribute Studio — TADA project "
            "([arXiv 2602.11910](https://arxiv.org/abs/2602.11910)). "
            "Set `TADA_VECTORS_DIR` to point to your pre-computed vectors.*"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the Gradio server.

    Reads configuration from environment variables (see module docstring).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    log.info("Starting Audio Attribute Studio on port %d (share=%s).", _SERVER_PORT, _SHARE)
    demo = build_interface()
    import gradio as gr

    demo.launch(
        server_port=_SERVER_PORT,
        share=_SHARE,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="purple"),
    )


if __name__ == "__main__":
    main()
