"""
Timestep Schedule Experiment — Phase 2, Prompt 2.2.

Compares four alpha schedules (constant, cosine, early_only, late_only)
across concepts {tempo, mood, instruments} and reports CLAP ΔAlignment and
LPAPS preservation metrics.

Hypothesis: ``cosine_schedule`` achieves better audio preservation (lower LPAPS)
than ``constant_schedule`` at the same mean alpha while maintaining comparable
concept alignment (ΔAlignment CLAP).

Usage (requires ACE-Step weights and computed steering vectors)::

    python experiments/timestep_schedule_experiment.py \
        --concept tempo \
        --alpha 80 \
        --n-samples 32 \
        --vectors-dir vectors/ \
        --output-dir experiments/results/timestep/

When model weights are unavailable, the script generates synthetic schedule
visualisation plots and exits gracefully.  Set ``--dry-run`` to force this.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np

from steer_audio.temporal_steering import (
    TimestepAdaptiveSteerer,
    TimestepSchedule,
    constant_schedule,
    cosine_schedule,
    early_only_schedule,
    late_only_schedule,
)
from steer_audio.vector_bank import SteeringVector, SteeringVectorBank

log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONCEPTS = ["tempo", "mood", "instruments"]
SCHEDULE_NAMES = ["constant", "cosine", "early_only", "late_only"]
NUM_INFERENCE_STEPS = 30


# ---------------------------------------------------------------------------
# Schedule builders
# ---------------------------------------------------------------------------


def build_schedules(alpha: float) -> dict[str, TimestepSchedule]:
    """Instantiate the four baseline schedules for a given peak alpha.

    Args:
        alpha: Peak (or constant) alpha value.

    Returns:
        Mapping ``schedule_name → TimestepSchedule``.
    """
    return {
        "constant": constant_schedule(alpha),
        "cosine": cosine_schedule(alpha_max=alpha, alpha_min=0.0),
        "early_only": early_only_schedule(alpha, cutoff=0.5),
        "late_only": late_only_schedule(alpha, cutoff=0.5),
    }


# ---------------------------------------------------------------------------
# Schedule visualisation (runs without model weights)
# ---------------------------------------------------------------------------


def plot_schedule_curves(
    alpha: float,
    num_steps: int,
    output_dir: Path,
) -> None:
    """Plot alpha vs. diffusion step for all four schedules.

    Args:
        alpha:      Peak/constant alpha value.
        num_steps:  Total number of diffusion steps.
        output_dir: Directory where ``alpha_vs_step.png`` is saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    schedules = build_schedules(alpha)
    steps = list(range(num_steps))

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"constant": "#1f77b4", "cosine": "#ff7f0e",
               "early_only": "#2ca02c", "late_only": "#d62728"}

    for name, sched in schedules.items():
        # Use TimestepAdaptiveSteerer.schedule_values() helper indirectly.
        T = num_steps
        values = [sched(max(1, T - k), T) for k in steps]
        ax.plot(steps, values, label=name, color=colors[name], linewidth=2)

    ax.set_xlabel("Diffusion step (0 = start, T−1 = end)")
    ax.set_ylabel("Effective alpha")
    ax.set_title(f"Alpha schedules — peak α={alpha}, T={num_steps}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = output_dir / "alpha_vs_step.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    log.info("Saved schedule curve plot → %s", out_path)


def plot_mean_alpha_comparison(
    alpha: float,
    num_steps: int,
    output_dir: Path,
) -> None:
    """Bar chart comparing the mean effective alpha across schedules.

    A ``constant_schedule`` has mean = alpha; others have lower means.
    This quantifies the expected regularisation from each schedule.

    Args:
        alpha:      Peak/constant alpha value.
        num_steps:  Total diffusion steps.
        output_dir: Output directory.
    """
    schedules = build_schedules(alpha)
    T = num_steps

    means: dict[str, float] = {}
    for name, sched in schedules.items():
        values = [sched(max(1, T - k), T) for k in range(T)]
        means[name] = float(np.mean(values))

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    ax.bar(list(means.keys()), list(means.values()), color=colors)
    ax.set_ylabel("Mean effective alpha")
    ax.set_title(f"Mean alpha by schedule — peak α={alpha}")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    out_path = output_dir / "mean_alpha_comparison.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    log.info("Saved mean-alpha comparison → %s", out_path)


# ---------------------------------------------------------------------------
# Model-based evaluation (requires weights)
# ---------------------------------------------------------------------------


def evaluate_schedule(
    steerer: TimestepAdaptiveSteerer,
    model: Any,
    prompts: list[str],
    seed_base: int = 42,
) -> dict[str, float]:
    """Generate audio for each prompt and compute stub evaluation metrics.

    Args:
        steerer:    Configured :class:`TimestepAdaptiveSteerer`.
        model:      ACE-Step model instance.
        prompts:    List of text prompts.
        seed_base:  Base random seed (seed = seed_base + i per prompt).

    Returns:
        Dict with keys ``"clap_delta"``, ``"lpaps"``, ``"n_samples"``.
    """
    import soundfile as sf

    audio_outputs: list[np.ndarray] = []
    sample_rates: list[int] = []

    for i, prompt in enumerate(prompts):
        try:
            audio, sr = steerer.steer(
                model=model,
                prompt=prompt,
                duration=10.0,
                seed=seed_base + i,
                num_inference_steps=NUM_INFERENCE_STEPS,
            )
            audio_outputs.append(audio)
            sample_rates.append(sr)
        except Exception as exc:  # noqa: BLE001
            log.warning("Inference failed for prompt %d: %s", i, exc)

    if not audio_outputs:
        return {"clap_delta": float("nan"), "lpaps": float("nan"), "n_samples": 0}

    # TODO: replace stub metrics with real CLAP / LPAPS computation.
    # Stub: use RMS energy as a proxy for "something changed".
    rms_values = [float(np.sqrt(np.mean(a**2))) for a in audio_outputs]
    mean_rms = float(np.mean(rms_values))

    return {
        "clap_delta": mean_rms,   # placeholder — replace with real CLAP ΔAlignment
        "lpaps": 0.0,             # placeholder — replace with real LPAPS
        "n_samples": len(audio_outputs),
    }


def load_vector(vectors_dir: Path, concept: str) -> SteeringVector | None:
    """Load a pre-computed steering vector for *concept* from *vectors_dir*.

    Looks for files matching ``{concept}_caa.safetensors`` or any
    ``*.safetensors`` file whose loaded ``concept`` field matches.

    Args:
        vectors_dir: Directory of ``.safetensors`` files.
        concept:     Concept name (e.g. "tempo").

    Returns:
        Loaded :class:`SteeringVector` or ``None`` if not found.
    """
    bank = SteeringVectorBank()
    candidates = sorted(vectors_dir.glob("*.safetensors"))
    for p in candidates:
        try:
            sv = bank.load(p)
            if sv.concept == concept:
                return sv
        except Exception as exc:  # noqa: BLE001
            log.debug("Skipping %s: %s", p, exc)
    log.warning("No steering vector found for concept '%s' in %s", concept, vectors_dir)
    return None


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------


def save_results(
    results: dict[str, dict[str, float]],
    output_dir: Path,
    concept: str,
    alpha: float,
) -> None:
    """Persist experiment results as JSON and a bar-chart PNG.

    Args:
        results:    ``{schedule_name: {metric: value}}`` mapping.
        output_dir: Target directory.
        concept:    Concept name (used in filenames).
        alpha:      Peak alpha (used in filenames).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON dump.
    json_path = output_dir / f"{concept}_alpha{int(alpha)}_results.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved results JSON → %s", json_path)

    # Bar chart: ΔAlignment CLAP per schedule.
    sched_names = list(results.keys())
    clap_vals = [results[s].get("clap_delta", 0.0) for s in sched_names]
    lpaps_vals = [results[s].get("lpaps", 0.0) for s in sched_names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(sched_names)]

    axes[0].bar(sched_names, clap_vals, color=colors)
    axes[0].set_title(f"CLAP ΔAlignment — {concept} α={alpha}")
    axes[0].set_ylabel("ΔAlignment (higher = more concept present)")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(sched_names, lpaps_vals, color=colors)
    axes[1].set_title(f"LPAPS — {concept} α={alpha}")
    axes[1].set_ylabel("LPAPS (lower = better preservation)")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    png_path = output_dir / f"{concept}_alpha{int(alpha)}_metrics.png"
    fig.savefig(str(png_path), dpi=150)
    plt.close(fig)
    log.info("Saved metrics bar chart → %s", png_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Timestep schedule comparison experiment (Phase 2.2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--concept",
        choices=CONCEPTS + ["all"],
        default="tempo",
        help="Concept to evaluate, or 'all' to run all three.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=80.0,
        help="Peak alpha (constant for 'constant' schedule, max for others).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=4,
        help="Number of audio samples per schedule (use 32 for full experiment).",
    )
    parser.add_argument(
        "--vectors-dir",
        type=Path,
        default=Path(os.environ.get("TADA_WORKDIR", "outputs")) / "vectors",
        help="Directory containing pre-computed .safetensors steering vectors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("TADA_WORKDIR", "experiments/results")) / "timestep",
        help="Directory for output plots and JSON results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model inference; only generate schedule visualisation plots.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=NUM_INFERENCE_STEPS,
        help="Diffusion denoising steps (= T).",
    )
    return parser.parse_args()


def run_for_concept(
    concept: str,
    alpha: float,
    n_samples: int,
    vectors_dir: Path,
    output_dir: Path,
    num_inference_steps: int,
    dry_run: bool,
) -> None:
    """Run the full schedule comparison for one concept.

    Args:
        concept:             Concept name (e.g. "tempo").
        alpha:               Peak alpha.
        n_samples:           Audio samples per schedule.
        vectors_dir:         Directory of precomputed vectors.
        output_dir:          Output directory for this run.
        num_inference_steps: T for diffusion.
        dry_run:             Skip inference if ``True``.
    """
    concept_out = output_dir / concept
    concept_out.mkdir(parents=True, exist_ok=True)

    # 1. Always generate schedule visualisation (no model needed).
    plot_schedule_curves(alpha, num_inference_steps, concept_out)
    plot_mean_alpha_comparison(alpha, num_inference_steps, concept_out)

    if dry_run:
        log.info("[%s] Dry-run: skipping inference.", concept)
        return

    # 2. Load steering vector.
    sv = load_vector(vectors_dir, concept)
    if sv is None:
        log.warning(
            "[%s] No vector found — skipping inference. "
            "Run compute_steering_vectors_caa.py first.",
            concept,
        )
        return

    # 3. Load model (lazy import to allow dry-run without heavy deps).
    try:
        # TODO: replace with actual ACE-Step model loading once weights are available.
        # Assumption: model is loaded via a helper in the TADA codebase.
        raise ImportError("Model loading not yet configured for this experiment.")
    except ImportError as exc:
        log.warning(
            "[%s] Cannot load ACE-Step model (%s). "
            "Skipping inference; only schedule plots were generated.",
            concept,
            exc,
        )
        return

    # 4. Build schedules and run inference.
    schedules = build_schedules(alpha)
    prompts_map = {
        "tempo": ["upbeat dance music", "slow ambient melody"] * (n_samples // 2 + 1),
        "mood": ["happy cheerful music", "calm peaceful melody"] * (n_samples // 2 + 1),
        "instruments": ["guitar solo", "piano melody"] * (n_samples // 2 + 1),
    }
    prompts = (prompts_map.get(concept, ["ambient music"] * n_samples))[:n_samples]

    results: dict[str, dict[str, float]] = {}
    for sched_name, sched in schedules.items():
        log.info("[%s] Evaluating schedule: %s (α_peak=%.0f)", concept, sched_name, alpha)
        steerer = TimestepAdaptiveSteerer(
            vector=sv,
            schedule=sched,
            layers=sv.layers,
        )
        metrics = evaluate_schedule(steerer, None, prompts)  # model=None placeholder
        results[sched_name] = metrics
        log.info("[%s] %s → %s", concept, sched_name, metrics)

    save_results(results, concept_out, concept, alpha)


def main() -> None:
    """Entry point for the timestep schedule experiment."""
    args = parse_args()

    concepts = CONCEPTS if args.concept == "all" else [args.concept]

    for concept in concepts:
        log.info("=" * 60)
        log.info("Running timestep schedule experiment for concept: %s", concept)
        run_for_concept(
            concept=concept,
            alpha=args.alpha,
            n_samples=args.n_samples,
            vectors_dir=args.vectors_dir,
            output_dir=args.output_dir,
            num_inference_steps=args.num_inference_steps,
            dry_run=args.dry_run,
        )

    log.info("=" * 60)
    log.info("Experiment complete.  Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()
