#!/usr/bin/env python3
"""
Phase 5.2 — Multi-Concept + Algebra + Schedule Experiments (CUDA GPU required).

Runs three complementary experiment groups using pre-computed CAA steering
vectors from Phase 5.1:

  1. **Multi-concept steering** — apply two CAA vectors simultaneously via
     linear combination (piano+mood, tempo+mood).
  2. **Concept algebra** — construct custom vectors through arithmetic:
       ``v_piano - v_drums``  (remove drum texture while keeping piano)
       ``0.7*v_mood + 0.3*v_tempo``  (weighted blend)
  3. **Timestep schedule comparison** — scale alpha per denoising step using
     ``constant``, ``cosine``, ``early_only``, ``late_only`` schedules on the
     ``tempo`` concept.

All three groups generate audio, compute CLAP alignment, and write results to
``results/eval/`` and ``experiments/results/``.

Usage (on RunPod A40 or any CUDA machine):
  export TADA_WORKDIR=/workspace/steer-audio/outputs
  export ACEMODEL_PATH=/workspace/ACE-Step
  python scripts/run_phase5_2.py [--dry-run] [--skip-multi] [--skip-algebra] [--skip-schedule]

Requirements:
  - Python 3.10–3.12
  - CUDA GPU
  - /workspace/ACE-Step/ containing ACE-Step model weights
  - Pre-computed vectors from Phase 5.1 in $TADA_WORKDIR/vectors/

Reference: TADA roadmap Prompt 5.2 (arXiv 2602.11910).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repo path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAE_ROOT = _REPO_ROOT / "sae"
_SRC_ROOT = _REPO_ROOT / "src"
_ACE_ROOT = _SRC_ROOT / "models" / "ace_step"

for _p in [str(_REPO_ROOT), str(_SAE_ROOT), str(_SAE_ROOT / "sae_src"), str(_SRC_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ACE_SUBMODULE = _ACE_ROOT / "ACE"
if _ACE_SUBMODULE.exists() and str(_ACE_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(_ACE_SUBMODULE))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Concepts with pre-computed vectors from Phase 5.1.
ALL_CONCEPTS: list[str] = ["piano", "tempo", "mood", "female_vocals", "drums"]

# Multi-concept pairs: (concept_a, concept_b, alpha_a, alpha_b)
MULTI_CONCEPT_PAIRS: list[tuple[str, str, float, float]] = [
    ("piano", "mood", 1.0, 1.0),
    ("tempo", "mood", 1.0, 1.0),
]

# Concept algebra expressions as (label, concept_a, weight_a, concept_b, weight_b)
# weight < 0 means subtract; None for concept_b means single concept scaled.
ALGEBRA_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "label": "piano_minus_drums",
        "expr": "piano - drums",
        "concept_a": "piano",
        "weight_a": 1.0,
        "concept_b": "drums",
        "weight_b": -1.0,
    },
    {
        "label": "0p7_mood_plus_0p3_tempo",
        "expr": "0.7*mood + 0.3*tempo",
        "concept_a": "mood",
        "weight_a": 0.7,
        "concept_b": "tempo",
        "weight_b": 0.3,
    },
]

# Schedule experiment config.
SCHEDULE_CONCEPT: str = "tempo"
SCHEDULE_ALPHA: float = 1.0  # base alpha (scaled by schedule at each step)
SCHEDULE_TYPES: list[str] = ["constant", "cosine", "early_only", "late_only"]

# Shared generation parameters (match Phase 5.1 for comparability).
AUDIO_DURATION: float = 12.0
INFER_STEPS: int = 30
SAMPLE_RATE: int = 44100
FUNCTIONAL_LAYERS: list[str] = ["tf6", "tf7"]

TEST_PROMPTS: list[str] = [
    "an upbeat electronic track with synths",
    "a calm acoustic guitar melody",
    "a fast-paced jazz piano trio",
]


# ---------------------------------------------------------------------------
# Schedule functions
# ---------------------------------------------------------------------------


def _get_schedule_fn(schedule_type: str):
    """Return f(step_idx, total_steps) -> float ∈ [0, 1]."""
    if schedule_type == "constant":
        return lambda step, total: 1.0
    elif schedule_type == "cosine":
        def _cosine(step: int, total: int) -> float:
            if total <= 0:
                return 1.0
            return (1.0 + math.cos(math.pi * min(step, total) / total)) / 2.0
        return _cosine
    elif schedule_type == "early_only":
        return lambda step, total: 1.0 if (total > 0 and step < total * 0.4) else 0.0
    elif schedule_type == "late_only":
        return lambda step, total: 0.0 if (total > 0 and step < total * 0.6) else 1.0
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type!r}")


# ---------------------------------------------------------------------------
# Vector loading
# ---------------------------------------------------------------------------


def load_sv_pkl(sv_dir: Path) -> dict | None:
    """Load a sv.pkl steering-vectors dict from a Phase 5.1 vector directory.

    Returns None when the file is missing or unreadable.

    Args:
        sv_dir: Directory containing ``sv.pkl`` (e.g.
                ``$TADA_WORKDIR/vectors/ace_piano_passes2_allTrue/``).

    Returns:
        The pickled steering-vectors dict, or ``None`` on failure.
    """
    sv_path = sv_dir / "sv.pkl"
    if not sv_path.exists():
        log.warning("sv.pkl not found at %s", sv_path)
        return None
    try:
        with open(sv_path, "rb") as f:
            sv = pickle.load(f)
        log.debug("Loaded sv.pkl from %s  (%d steps)", sv_path, len(sv))
        return sv
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to load sv.pkl from %s: %s", sv_path, exc)
        return None


def build_combined_sv(
    sv_a: dict,
    weight_a: float,
    sv_b: dict,
    weight_b: float,
) -> dict:
    """Combine two step-keyed steering-vector dicts via linear algebra.

    Computes ``weight_a * sv_a + weight_b * sv_b`` at every (step, layer)
    entry.  Missing entries in either dict are treated as zero.

    Args:
        sv_a:     Phase 5.1 sv.pkl dict for concept A.
        weight_a: Scalar multiplier for concept A.
        sv_b:     Phase 5.1 sv.pkl dict for concept B.
        weight_b: Scalar multiplier for concept B.

    Returns:
        Combined steering-vectors dict with the same structure.
    """
    combined: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    all_step_keys = sorted(set(list(sv_a.keys()) + list(sv_b.keys())))

    for step_key in all_step_keys:
        layers_a = sv_a.get(step_key, {})
        layers_b = sv_b.get(step_key, {})
        all_layers = sorted(set(list(layers_a.keys()) + list(layers_b.keys())))
        for layer_name in all_layers:
            vecs_a = layers_a.get(layer_name, [])
            vecs_b = layers_b.get(layer_name, [])

            # Each entry is a list with one numpy array.
            arr_a = vecs_a[0] if vecs_a else None
            arr_b = vecs_b[0] if vecs_b else None

            if arr_a is not None and arr_b is not None:
                combined_arr = weight_a * arr_a + weight_b * arr_b
            elif arr_a is not None:
                combined_arr = weight_a * arr_a
            elif arr_b is not None:
                combined_arr = weight_b * arr_b
            else:
                continue

            # Normalize the combined vector.
            norm = np.linalg.norm(combined_arr)
            if norm > 0:
                combined_arr = combined_arr / norm
            combined[step_key][layer_name] = [combined_arr]

    return dict(combined)


def apply_schedule_to_sv(
    sv: dict,
    schedule_type: str,
    total_steps: int,
) -> dict:
    """Pre-scale a steering-vector dict by a timestep schedule.

    Each step's vectors are multiplied by ``schedule_fn(step_idx, total_steps)``
    so that the downstream VectorStore controller (which applies a fixed alpha)
    effectively uses a time-varying alpha.

    Args:
        sv:           Phase 5.1 sv.pkl dict.
        schedule_type: One of ``"constant"``, ``"cosine"``, ``"early_only"``,
                        ``"late_only"``.
        total_steps:  Total denoising steps (used to normalise the schedule).

    Returns:
        New steering-vectors dict with per-step scale factors applied.
    """
    schedule_fn = _get_schedule_fn(schedule_type)
    step_keys = sorted(sv.keys())
    n_steps = len(step_keys)

    scaled: dict[str, dict[str, list[Any]]] = {}
    for step_idx, step_key in enumerate(step_keys):
        scale = schedule_fn(step_idx, n_steps)
        layers = sv[step_key]
        scaled[step_key] = {}
        for layer_name, vecs in layers.items():
            arr = vecs[0] if vecs else None
            if arr is not None:
                scaled[step_key][layer_name] = [arr * scale]
    return scaled


# ---------------------------------------------------------------------------
# Audio generation (single steering-vector dict)
# ---------------------------------------------------------------------------


def generate_with_sv(
    pipe,
    sv_dict: dict,
    label: str,
    out_dir: Path,
    device: str,
    alpha: float = 1.0,
    test_prompts: list[str] = TEST_PROMPTS,
    infer_steps: int = INFER_STEPS,
    audio_duration: float = AUDIO_DURATION,
    layers: list[str] = FUNCTIONAL_LAYERS,
    steer_mode: str = "cond_only",
    dry_run: bool = False,
) -> list[Path]:
    """Generate audio steered by *sv_dict* and save WAVs under *out_dir*.

    Args:
        pipe:         Loaded SimpleACEStepPipeline.
        sv_dict:      Pre-computed (and possibly combined/scaled) steering-
                      vectors dict in the Phase 5.1 format.
        label:        Short label used in log messages.
        out_dir:      Directory to write generated WAV files.
        device:       Torch device string.
        alpha:        Steering strength multiplier passed to VectorStore.
        test_prompts: Prompts to generate audio for.
        infer_steps:  Number of diffusion steps.
        audio_duration: Audio clip length in seconds.
        layers:       Functional layer names to steer (e.g. ``["tf6","tf7"]``).
        steer_mode:   VectorStore steering mode.
        dry_run:      If ``True``, create empty placeholder WAV files.

    Returns:
        List of WAV paths (one per prompt).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        paths = []
        for p_idx in range(len(test_prompts)):
            wav = out_dir / f"p{p_idx}.wav"
            wav.touch()
            paths.append(wav)
        log.info("[dry-run] Skipped audio generation for '%s'", label)
        return paths

    from src.models.ace_step.ace_steering.controller import (  # type: ignore
        VectorStore,
        compute_num_cfg_passes,
        register_vector_control,
    )

    num_cfg_passes = compute_num_cfg_passes(0.0, 0.0)
    paths: list[Path] = []

    for p_idx, prompt in enumerate(test_prompts):
        wav_path = out_dir / f"p{p_idx}.wav"
        if wav_path.exists() and wav_path.stat().st_size > 0:
            log.info("  [skip] %s already exists.", wav_path.name)
            paths.append(wav_path)
            continue

        controller = VectorStore(
            device=device,
            save_only_cond=(steer_mode == "cond_only"),
            num_cfg_passes=num_cfg_passes,
        )
        controller.steer = True
        controller.alpha = alpha
        controller.steering_vectors = sv_dict

        register_vector_control(
            pipe.ace_step_transformer,
            controller,
            explicit_layers=layers,
        )

        audio_output = pipe.generate(
            prompt=prompt,
            audio_duration=audio_duration,
            infer_step=infer_steps,
            manual_seed=42,
            return_type="audio",
            use_erg_lyric=False,
            guidance_scale_text=0.0,
            guidance_scale_lyric=0.0,
            guidance_scale=3.0,
            guidance_interval=1.0,
            guidance_interval_decay=0.0,
        )
        controller.reset()

        import torchaudio  # type: ignore

        audio_tensor = audio_output.cpu()
        if audio_tensor.ndim == 3:
            audio_tensor = audio_tensor.squeeze(0)
        torchaudio.save(str(wav_path), audio_tensor, SAMPLE_RATE)
        log.info("  Saved %s (label=%s)", wav_path.name, label)
        paths.append(wav_path)

    return paths


# ---------------------------------------------------------------------------
# CLAP evaluation
# ---------------------------------------------------------------------------


def evaluate_clap_paths(
    paths: list[Path],
    test_prompts: list[str],
    dry_run: bool = False,
) -> float:
    """Compute mean CLAP alignment over *paths*.

    Falls back to a stub value of ``-1.0`` when ``laion_clap`` is not
    installed or when *dry_run* is ``True``.

    Args:
        paths:        List of WAV file paths (aligned with *test_prompts*).
        test_prompts: Text prompts corresponding to each WAV.
        dry_run:      Return stub value without loading the model.

    Returns:
        Mean CLAP cosine similarity, or ``-1.0`` on failure / dry-run.
    """
    if dry_run:
        return -1.0

    try:
        import laion_clap  # type: ignore
        import torchaudio  # type: ignore

        model_clap = laion_clap.CLAP_Module(enable_fusion=False)
        model_clap.load_ckpt()

        clap_vals: list[float] = []
        for wav_path, prompt in zip(paths, test_prompts):
            if not wav_path.exists() or wav_path.stat().st_size == 0:
                continue
            try:
                waveform, _ = torchaudio.load(str(wav_path))
                audio_data = waveform.mean(0).numpy()
                audio_embed = model_clap.get_audio_embedding_from_data(
                    [audio_data], use_tensor=False
                )
                text_embed = model_clap.get_text_embedding([prompt])
                cos = float(
                    np.dot(audio_embed[0], text_embed[0])
                    / (np.linalg.norm(audio_embed[0]) * np.linalg.norm(text_embed[0]) + 1e-8)
                )
                clap_vals.append(cos)
            except Exception as exc:  # noqa: BLE001
                log.warning("CLAP eval failed for %s: %s", wav_path, exc)

        return float(np.mean(clap_vals)) if clap_vals else -1.0

    except ImportError:
        log.warning("laion_clap not installed — returning stub CLAP score.")
        return -1.0


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write *rows* to a CSV file at *path*.

    Args:
        path:       Destination file path (parent directory must exist).
        rows:       List of row dicts.
        fieldnames: Column order for the header.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), path)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Experiment 1: Multi-concept steering
# ---------------------------------------------------------------------------


def run_multi_concept_experiments(
    pipe,
    vectors_dir: Path,
    eval_root: Path,
    device: str,
    dry_run: bool = False,
) -> list[dict]:
    """Run multi-concept steering experiments and return result rows.

    For each concept pair in ``MULTI_CONCEPT_PAIRS``:
      - Load pre-computed sv.pkl for each concept.
      - Combine vectors: ``weight_a * sv_a + weight_b * sv_b``.
      - Generate audio with the combined vector.
      - Evaluate CLAP.

    Args:
        pipe:        Loaded pipeline (or ``None`` in dry-run).
        vectors_dir: Root directory of Phase 5.1 vectors
                     (e.g. ``$TADA_WORKDIR/vectors/``).
        eval_root:   Root for saving generated audio and CSVs.
        device:      Torch device.
        dry_run:     Skip real inference.

    Returns:
        List of result-row dicts with keys
        ``[concept_a, concept_b, alpha_a, alpha_b, clap, wav_dir]``.
    """
    log.info("=== Experiment 1: Multi-Concept Steering ===")
    rows: list[dict] = []

    for concept_a, concept_b, alpha_a, alpha_b in MULTI_CONCEPT_PAIRS:
        label = f"{concept_a}_plus_{concept_b}"
        log.info("  Pair: %s  (α_a=%.2f  α_b=%.2f)", label, alpha_a, alpha_b)

        sv_dir_a = vectors_dir / f"ace_{concept_a}_passes2_allTrue"
        sv_dir_b = vectors_dir / f"ace_{concept_b}_passes2_allTrue"

        sv_a = load_sv_pkl(sv_dir_a)
        sv_b = load_sv_pkl(sv_dir_b)

        if sv_a is None or sv_b is None:
            if not dry_run:
                log.warning("Missing vectors for pair %s — skipping.", label)
                continue
            # In dry-run, create stub dicts.
            sv_a = sv_a or {}
            sv_b = sv_b or {}

        combined_sv = build_combined_sv(sv_a, alpha_a, sv_b, alpha_b)

        out_dir = eval_root / "multi_concept" / label
        wav_paths = generate_with_sv(
            pipe=pipe,
            sv_dict=combined_sv,
            label=label,
            out_dir=out_dir,
            device=device,
            alpha=1.0,  # alpha already baked into combined_sv
            dry_run=dry_run,
        )

        clap = evaluate_clap_paths(wav_paths, TEST_PROMPTS, dry_run=dry_run)
        log.info("  %s  mean_CLAP=%.4f", label, clap)

        rows.append(
            {
                "concept_a": concept_a,
                "concept_b": concept_b,
                "alpha_a": alpha_a,
                "alpha_b": alpha_b,
                "label": label,
                "clap": f"{clap:.6f}",
                "wav_dir": str(out_dir),
            }
        )

    # Save CSV.
    csv_path = eval_root / "multi_concept" / "results.csv"
    write_csv(
        csv_path,
        rows,
        ["concept_a", "concept_b", "alpha_a", "alpha_b", "label", "clap", "wav_dir"],
    )
    return rows


# ---------------------------------------------------------------------------
# Experiment 2: Concept algebra
# ---------------------------------------------------------------------------


def run_concept_algebra_experiments(
    pipe,
    vectors_dir: Path,
    eval_root: Path,
    device: str,
    dry_run: bool = False,
) -> list[dict]:
    """Run concept algebra steering experiments and return result rows.

    For each entry in ``ALGEBRA_EXPERIMENTS``:
      - Load sv.pkl for concepts A and B.
      - Combine: ``weight_a * sv_a + weight_b * sv_b``.
      - Generate audio and evaluate CLAP.

    Args:
        pipe:        Loaded pipeline (or ``None`` in dry-run).
        vectors_dir: Phase 5.1 vectors root dir.
        eval_root:   Root for generated audio and CSVs.
        device:      Torch device.
        dry_run:     Skip real inference.

    Returns:
        List of result-row dicts with keys
        ``[label, expr, clap, wav_dir]``.
    """
    log.info("=== Experiment 2: Concept Algebra ===")
    rows: list[dict] = []

    for exp in ALGEBRA_EXPERIMENTS:
        label = exp["label"]
        expr = exp["expr"]
        concept_a = exp["concept_a"]
        weight_a = exp["weight_a"]
        concept_b = exp["concept_b"]
        weight_b = exp["weight_b"]

        log.info("  Algebra: %s  (expr='%s')", label, expr)

        sv_dir_a = vectors_dir / f"ace_{concept_a}_passes2_allTrue"
        sv_dir_b = vectors_dir / f"ace_{concept_b}_passes2_allTrue"

        sv_a = load_sv_pkl(sv_dir_a)
        sv_b = load_sv_pkl(sv_dir_b)

        if sv_a is None or sv_b is None:
            if not dry_run:
                log.warning("Missing vectors for algebra experiment %s — skipping.", label)
                continue
            sv_a = sv_a or {}
            sv_b = sv_b or {}

        combined_sv = build_combined_sv(sv_a, weight_a, sv_b, weight_b)

        out_dir = eval_root / "concept_algebra" / label
        wav_paths = generate_with_sv(
            pipe=pipe,
            sv_dict=combined_sv,
            label=label,
            out_dir=out_dir,
            device=device,
            alpha=1.0,
            dry_run=dry_run,
        )

        clap = evaluate_clap_paths(wav_paths, TEST_PROMPTS, dry_run=dry_run)
        log.info("  %s  mean_CLAP=%.4f", label, clap)

        rows.append(
            {
                "label": label,
                "expr": expr,
                "concept_a": concept_a,
                "weight_a": weight_a,
                "concept_b": concept_b,
                "weight_b": weight_b,
                "clap": f"{clap:.6f}",
                "wav_dir": str(out_dir),
            }
        )

    csv_path = eval_root / "concept_algebra" / "results.csv"
    write_csv(
        csv_path,
        rows,
        ["label", "expr", "concept_a", "weight_a", "concept_b", "weight_b", "clap", "wav_dir"],
    )
    return rows


# ---------------------------------------------------------------------------
# Experiment 3: Timestep schedule comparison
# ---------------------------------------------------------------------------


def run_schedule_experiments(
    pipe,
    vectors_dir: Path,
    eval_root: Path,
    device: str,
    concept: str = SCHEDULE_CONCEPT,
    base_alpha: float = SCHEDULE_ALPHA,
    dry_run: bool = False,
) -> list[dict]:
    """Run timestep schedule comparison experiments.

    Applies four different schedules to the *concept* steering vector and
    evaluates CLAP for each.

    The schedule is baked into the steering-vectors dict by pre-scaling
    each step's vector by ``schedule_fn(step_idx, total_steps)``.  The
    VectorStore then applies the scaled vectors with a fixed alpha of 1.0.

    Args:
        pipe:       Loaded pipeline (or ``None`` in dry-run).
        vectors_dir: Phase 5.1 vectors root dir.
        eval_root:  Root for generated audio and CSVs.
        device:     Torch device.
        concept:    Concept to steer (default: ``"tempo"``).
        base_alpha: Steering strength before schedule scaling.
        dry_run:    Skip real inference.

    Returns:
        List of result-row dicts with keys
        ``[schedule, concept, base_alpha, clap, wav_dir]``.
    """
    log.info("=== Experiment 3: Timestep Schedule Comparison (concept=%s) ===", concept)
    rows: list[dict] = []

    sv_dir = vectors_dir / f"ace_{concept}_passes2_allTrue"
    base_sv = load_sv_pkl(sv_dir)

    if base_sv is None and not dry_run:
        log.warning("No vectors for '%s' — skipping schedule experiments.", concept)
        return rows

    base_sv = base_sv or {}

    for schedule_type in SCHEDULE_TYPES:
        label = f"{concept}_{schedule_type}"
        log.info("  Schedule: %s", schedule_type)

        # Pre-scale per-step vectors by the schedule function.
        n_steps = max(1, len(base_sv))
        scaled_sv = apply_schedule_to_sv(base_sv, schedule_type, n_steps)

        out_dir = eval_root / "schedules" / label
        wav_paths = generate_with_sv(
            pipe=pipe,
            sv_dict=scaled_sv,
            label=label,
            out_dir=out_dir,
            device=device,
            alpha=base_alpha,
            dry_run=dry_run,
        )

        clap = evaluate_clap_paths(wav_paths, TEST_PROMPTS, dry_run=dry_run)
        log.info("  %s  mean_CLAP=%.4f", label, clap)

        rows.append(
            {
                "schedule": schedule_type,
                "concept": concept,
                "base_alpha": base_alpha,
                "clap": f"{clap:.6f}",
                "wav_dir": str(out_dir),
            }
        )

    csv_path = eval_root / "schedules" / "results.csv"
    write_csv(
        csv_path,
        rows,
        ["schedule", "concept", "base_alpha", "clap", "wav_dir"],
    )
    return rows


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _save_schedule_plot(rows: list[dict], out_path: Path) -> None:
    """Save a bar chart of CLAP scores per schedule type.

    Args:
        rows:     Schedule experiment result rows.
        out_path: Destination PNG path.
    """
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        schedules = [r["schedule"] for r in rows]
        clap_vals = [float(r["clap"]) for r in rows]

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
        bars = ax.bar(schedules, clap_vals, color=colors[: len(schedules)])

        ax.set_xlabel("Schedule type")
        ax.set_ylabel("Mean CLAP alignment")
        ax.set_title(f"Timestep Schedule Comparison — concept: {SCHEDULE_CONCEPT}")
        ax.set_ylim(0, max(0.01, max(clap_vals) * 1.2) if clap_vals else 1.0)

        for bar, val in zip(bars, clap_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        log.info("Saved schedule comparison plot to %s", out_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not save schedule plot: %s", exc)


def _save_multi_concept_plot(rows: list[dict], out_path: Path) -> None:
    """Save a bar chart of CLAP scores per multi-concept pair.

    Args:
        rows:     Multi-concept experiment result rows.
        out_path: Destination PNG path.
    """
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        labels = [r["label"] for r in rows]
        clap_vals = [float(r["clap"]) for r in rows]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, clap_vals, color="#4e79a7")
        ax.set_xlabel("Concept pair")
        ax.set_ylabel("Mean CLAP alignment")
        ax.set_title("Multi-Concept Steering Results")
        ax.set_ylim(0, max(0.01, max(clap_vals) * 1.2) if clap_vals else 1.0)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        log.info("Saved multi-concept plot to %s", out_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not save multi-concept plot: %s", exc)


# ---------------------------------------------------------------------------
# docs/results_summary.md update
# ---------------------------------------------------------------------------


def update_results_summary(
    multi_rows: list[dict],
    algebra_rows: list[dict],
    schedule_rows: list[dict],
    docs_dir: Path,
) -> None:
    """Append a Phase 5.2 section to docs/results_summary.md.

    Skips the update if the section already exists to avoid duplicates.

    Args:
        multi_rows:    Multi-concept experiment results.
        algebra_rows:  Concept algebra experiment results.
        schedule_rows: Schedule comparison results.
        docs_dir:      Directory containing ``results_summary.md``.
    """
    summary_path = docs_dir / "results_summary.md"
    if not summary_path.exists():
        log.warning("docs/results_summary.md not found at %s — skipping update.", summary_path)
        return

    existing = summary_path.read_text()
    if "## Real Run Results (Phase 5.2)" in existing:
        log.info("docs/results_summary.md already contains Phase 5.2 section — skipping update.")
        return

    lines: list[str] = [
        "",
        "---",
        "",
        "## Real Run Results (Phase 5.2)",
        "",
        "Multi-concept, concept algebra, and timestep schedule experiments on ACE-Step.",
        "",
        "### Multi-Concept Steering",
        "",
        "| Concept pair | α_a | α_b | mean CLAP |",
        "|--------------|-----|-----|-----------|",
    ]
    for r in multi_rows:
        clap_val = float(r["clap"])
        clap_str = f"{clap_val:.3f}" if clap_val >= 0 else "n/a"
        lines.append(f"| {r['concept_a']} + {r['concept_b']} | {r['alpha_a']} | {r['alpha_b']} | {clap_str} |")

    lines += [
        "",
        "### Concept Algebra",
        "",
        "| Expression | mean CLAP |",
        "|-----------|-----------|",
    ]
    for r in algebra_rows:
        clap_val = float(r["clap"])
        clap_str = f"{clap_val:.3f}" if clap_val >= 0 else "n/a"
        lines.append(f"| `{r['expr']}` | {clap_str} |")

    lines += [
        "",
        "### Timestep Schedule Comparison",
        "",
        f"Concept: `{SCHEDULE_CONCEPT}`  base_alpha: `{SCHEDULE_ALPHA}`",
        "",
        "| Schedule | mean CLAP |",
        "|----------|-----------|",
    ]
    for r in schedule_rows:
        clap_val = float(r["clap"])
        clap_str = f"{clap_val:.3f}" if clap_val >= 0 else "n/a"
        lines.append(f"| {r['schedule']} | {clap_str} |")

    lines += [
        "",
        "> Results generated by `scripts/run_phase5_2.py` on ACE-Step (CUDA).",
        "> CLAP scores via `laion_clap`; n/a = library not installed or dry-run.",
        "",
    ]

    with open(summary_path, "a") as f:
        f.write("\n".join(lines) + "\n")
    log.info("Updated %s with Phase 5.2 results.", summary_path)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def check_environment() -> dict[str, Any]:
    """Verify CUDA, Python version, and model weights availability.

    Returns:
        Dict with keys ``cuda``, ``device``, ``gpu_name``, ``vram_gb``,
        ``model_path``, ``model_exists``, ``python_version``, ``python_ok``.
    """
    info: dict[str, Any] = {}
    info["cuda"] = torch.cuda.is_available()
    info["device"] = "cuda" if info["cuda"] else "cpu"
    if info["cuda"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        info["gpu_name"] = "n/a"
        info["vram_gb"] = 0.0
    log.info("GPU: %s  VRAM: %.1f GB  CUDA: %s", info["gpu_name"], info["vram_gb"], info["cuda"])

    model_path = Path(os.environ.get("ACEMODEL_PATH", "/workspace/ACE-Step"))
    info["model_path"] = model_path
    info["model_exists"] = model_path.exists()
    if not info["model_exists"]:
        log.warning("ACE-Step model not found at %s; set ACEMODEL_PATH.", model_path)

    py = sys.version_info
    info["python_version"] = f"{py.major}.{py.minor}.{py.micro}"
    info["python_ok"] = py < (3, 13)
    if not info["python_ok"]:
        log.error("Python %s — ACE-Step requires < 3.13.", info["python_version"])

    return info


def load_ace_pipeline(device: str, model_path: Path):
    """Load and return a SimpleACEStepPipeline.

    Args:
        device:     Torch device string (``"cuda"`` or ``"cpu"``).
        model_path: Path to ACE-Step model weights.

    Returns:
        Loaded pipeline object.
    """
    from src.models.ace_step.pipeline_ace import SimpleACEStepPipeline  # type: ignore

    log.info("Loading ACE-Step pipeline from %s ...", model_path)
    t0 = time.time()
    pipe = SimpleACEStepPipeline(device=device)
    pipe.load()
    log.info("ACE-Step loaded in %.1f s", time.time() - t0)
    return pipe


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Phase 5.2."""
    p = argparse.ArgumentParser(
        description="Phase 5.2 — Multi-Concept + Algebra + Schedule Experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip actual model inference; write placeholder files.",
    )
    p.add_argument("--skip-multi", action="store_true", help="Skip multi-concept experiments.")
    p.add_argument("--skip-algebra", action="store_true", help="Skip concept algebra experiments.")
    p.add_argument("--skip-schedule", action="store_true", help="Skip schedule experiments.")
    p.add_argument(
        "--alpha",
        type=float,
        default=SCHEDULE_ALPHA,
        help="Base alpha for schedule and single-concept experiments.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for Phase 5.2."""
    args = parse_args()

    workdir = Path(os.environ.get("TADA_WORKDIR", str(_REPO_ROOT / "outputs")))
    model_path = Path(os.environ.get("ACEMODEL_PATH", "/workspace/ACE-Step"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("=== Phase 5.2 — Multi-Concept + Algebra + Schedule Experiments ===")
    log.info("TADA_WORKDIR : %s", workdir)
    log.info("ACEMODEL_PATH: %s", model_path)
    log.info("device       : %s", device)
    log.info("dry_run      : %s", args.dry_run)

    env_info = check_environment()
    if not args.dry_run:
        if not env_info["cuda"]:
            log.error("No CUDA device — aborting. Use --dry-run for a test run.")
            sys.exit(1)
        if not env_info["model_exists"]:
            log.error(
                "ACE-Step weights not found at %s. Set ACEMODEL_PATH or use --dry-run.",
                model_path,
            )
            sys.exit(1)
        if not env_info["python_ok"]:
            log.error("Python %s — ACE-Step requires Python < 3.13.", env_info["python_version"])
            sys.exit(1)

    vectors_dir = workdir / "vectors"
    eval_root = _REPO_ROOT / "results" / "eval"
    experiments_results = _REPO_ROOT / "experiments" / "results"

    # Verify that Phase 5.1 vectors exist (real run only).
    if not args.dry_run:
        found = list(vectors_dir.glob("ace_*_passes2_allTrue/sv.pkl"))
        if not found:
            log.error(
                "No Phase 5.1 steering vectors found under %s. "
                "Run scripts/run_phase5_1.py first, or use --dry-run.",
                vectors_dir,
            )
            sys.exit(1)
        log.info("Found %d concept vector dir(s) under %s.", len(found), vectors_dir)

    # Load pipeline once (reused for all experiments).
    pipe = None
    if not args.dry_run:
        pipe = load_ace_pipeline(device=device, model_path=model_path)

    multi_rows: list[dict] = []
    algebra_rows: list[dict] = []
    schedule_rows: list[dict] = []

    # ---- Experiment 1: Multi-concept ----------------------------------------
    if not args.skip_multi:
        multi_rows = run_multi_concept_experiments(
            pipe=pipe,
            vectors_dir=vectors_dir,
            eval_root=eval_root,
            device=device,
            dry_run=args.dry_run,
        )
        if multi_rows:
            _save_multi_concept_plot(
                multi_rows,
                eval_root / "multi_concept" / "multi_concept_clap.png",
            )

    # ---- Experiment 2: Concept algebra --------------------------------------
    if not args.skip_algebra:
        algebra_rows = run_concept_algebra_experiments(
            pipe=pipe,
            vectors_dir=vectors_dir,
            eval_root=eval_root,
            device=device,
            dry_run=args.dry_run,
        )

    # ---- Experiment 3: Schedule comparison ----------------------------------
    if not args.skip_schedule:
        schedule_rows = run_schedule_experiments(
            pipe=pipe,
            vectors_dir=vectors_dir,
            eval_root=eval_root,
            device=device,
            concept=SCHEDULE_CONCEPT,
            base_alpha=args.alpha,
            dry_run=args.dry_run,
        )
        if schedule_rows:
            _save_schedule_plot(
                schedule_rows,
                eval_root / "schedules" / "schedule_comparison.png",
            )

    # ---- Update docs/results_summary.md -------------------------------------
    update_results_summary(
        multi_rows=multi_rows,
        algebra_rows=algebra_rows,
        schedule_rows=schedule_rows,
        docs_dir=_REPO_ROOT / "docs",
    )

    # ---- Print summary -------------------------------------------------------
    log.info("\n=== Phase 5.2 Summary ===")

    if multi_rows:
        log.info("\nMulti-Concept Steering:")
        for r in multi_rows:
            log.info("  %-35s  CLAP=%s", r["label"], r["clap"])

    if algebra_rows:
        log.info("\nConcept Algebra:")
        for r in algebra_rows:
            log.info("  %-35s  CLAP=%s  expr='%s'", r["label"], r["clap"], r["expr"])

    if schedule_rows:
        log.info("\nTimestep Schedules (concept=%s):", SCHEDULE_CONCEPT)
        for r in schedule_rows:
            log.info("  %-15s  CLAP=%s", r["schedule"], r["clap"])

    log.info("\nOutput directories:")
    log.info("  Multi-concept : %s", eval_root / "multi_concept")
    log.info("  Algebra       : %s", eval_root / "concept_algebra")
    log.info("  Schedules     : %s", eval_root / "schedules")
    log.info("  Docs summary  : %s", _REPO_ROOT / "docs" / "results_summary.md")
    log.info("\nDone.")


if __name__ == "__main__":
    main()
