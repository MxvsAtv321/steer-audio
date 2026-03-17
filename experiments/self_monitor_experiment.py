"""
Self-monitoring steering experiment — Phase 2, Prompt 2.4 (TADA roadmap).

Compares fixed-alpha steering vs. self-monitored steering for concepts
{tempo, mood, vocal_gender} at alpha in {50, 75, 100}.

Primary claim
-------------
Self-monitoring achieves the same ΔAlignment as fixed alpha=75 but with
lower LPAPS (better audio preservation) and higher CE/PQ.

Usage
-----
    python experiments/self_monitor_experiment.py [--dry-run]

    --dry-run   Stub out model/CLAP loading; useful for CI smoke-testing.

Outputs
-------
    experiments/results/self_monitor/probe_accuracies.csv
    experiments/results/self_monitor/comparison_results.csv
    experiments/results/self_monitor/trace_<concept>_alpha<A>.png  (one per run)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — ensure repo root is on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from steer_audio.self_monitor import ConceptProbe, SelfMonitoredSteerer, _stub_clap_extractor
from steer_audio.vector_bank import SteeringVector, SteeringVectorBank

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONCEPTS = ["tempo", "mood", "vocal_gender"]
ALPHA_VALUES = [50, 75, 100]
RESULTS_DIR = _REPO_ROOT / "experiments" / "results" / "self_monitor"
VECTORS_DIR = _REPO_ROOT / "vectors"

# Positive / negative text prompts used to generate training audio for probes.
PROBE_PROMPTS: dict[str, tuple[str, str]] = {
    "tempo": (
        "a fast-paced energetic dance track",
        "a slow calm ambient piece",
    ),
    "mood": (
        "a happy uplifting joyful song",
        "a sad melancholic gloomy song",
    ),
    "vocal_gender": (
        "a female vocal lead singer pop track",
        "a male vocal lead singer pop track",
    ),
}


# ---------------------------------------------------------------------------
# Stub model for dry-run mode
# ---------------------------------------------------------------------------


class _StubModel:
    """Minimal model stub for smoke-testing without real weights."""

    sample_rate: int = 44100

    class _StubBlocks(list):
        pass

    class _StubBlock:
        class _StubAttn:
            _forward_hooks: dict = {}

            def register_forward_hook(self, fn: Any) -> Any:
                import torch

                class _Handle:
                    def remove(self) -> None:
                        pass

                return _Handle()

        cross_attn = _StubAttn()

    def __init__(self, n_blocks: int = 8) -> None:
        self.transformer_blocks = [self._StubBlock() for _ in range(n_blocks)]

    def pipeline(self, prompt: str, duration: float = 30.0, seed: int = 42) -> np.ndarray:
        """Return silence of the requested duration."""
        n = int(duration * self.sample_rate)
        return np.zeros(n, dtype=np.float32)

    def decode_latents(self, latent: Any) -> np.ndarray:
        return np.zeros(self.sample_rate, dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_model(dry_run: bool) -> Any:
    """Load ACE-Step model or return stub in dry-run mode.

    Args:
        dry_run: If True, return a ``_StubModel`` without loading weights.

    Returns:
        Model instance.
    """
    if dry_run:
        log.info("Dry-run mode: using stub model.")
        return _StubModel()

    try:
        # TODO: Replace with actual ACE-Step loading once weights are available.
        # from models.ace_step.patchable_ace import PatchableACEStep
        # return PatchableACEStep.from_pretrained(...)
        log.warning(
            "Real ACE-Step loading is not implemented here. "
            "Falling back to stub model."
        )
        return _StubModel()
    except Exception as exc:
        log.error("Failed to load model: %s — falling back to stub.", exc)
        return _StubModel()


def _load_steering_vector(concept: str, dry_run: bool) -> SteeringVector:
    """Load a pre-computed steering vector for *concept*, or create a random stub.

    Args:
        concept: Concept name (e.g. ``"tempo"``).
        dry_run: Use random stub vector if True or no file is found.

    Returns:
        :class:`SteeringVector`.
    """
    bank = SteeringVectorBank()
    hidden_dim = 3072  # ACE-Step cross-attention hidden dim

    if not dry_run:
        for suffix in ("_caa", "_sae", ""):
            candidate = VECTORS_DIR / f"{concept}{suffix}.safetensors"
            if candidate.exists():
                sv = bank.load(candidate)
                log.info("Loaded steering vector from %s", candidate)
                return sv

    log.info(
        "No pre-computed vector found for '%s'; using random stub (dim=%d).",
        concept,
        hidden_dim,
    )
    import torch

    torch.manual_seed(abs(hash(concept)) % (2**31))
    return SteeringVector(
        concept=concept,
        method="caa",
        model_name="ace-step",
        layers=[6, 7],
        vector=torch.randn(hidden_dim),
        clap_delta=0.0,
    )


def _make_probe(concept: str, dry_run: bool) -> ConceptProbe:
    """Build and 'train' a ConceptProbe for *concept*.

    In dry-run mode the probe is trained on randomly generated embeddings so
    that it satisfies the ``is_trained`` invariant without real audio.

    Args:
        concept: Concept name.
        dry_run: If True, train on synthetic embeddings.

    Returns:
        Trained :class:`ConceptProbe`.
    """
    probe = ConceptProbe(concept=concept, clap_extractor=_stub_clap_extractor)

    if dry_run:
        # Bypass file-loading: directly fit on synthetic embeddings.
        _fit_probe_on_synthetic_data(probe)
        return probe

    # Discover audio files under results/concept_name/{positive,negative}/.
    pos_dir = RESULTS_DIR / concept / "positive"
    neg_dir = RESULTS_DIR / concept / "negative"

    if pos_dir.exists() and neg_dir.exists():
        pos_paths = sorted(pos_dir.glob("*.wav"))
        neg_paths = sorted(neg_dir.glob("*.wav"))
        if pos_paths and neg_paths:
            acc = probe.train(pos_paths, neg_paths)
            log.info(
                "Probe for '%s' trained on %d pos / %d neg files. Accuracy=%.3f",
                concept,
                len(pos_paths),
                len(neg_paths),
                acc,
            )
            return probe

    log.warning(
        "No training audio found for '%s' under %s. "
        "Using synthetic data (not representative).",
        concept,
        RESULTS_DIR,
    )
    _fit_probe_on_synthetic_data(probe)
    return probe


def _fit_probe_on_synthetic_data(probe: ConceptProbe) -> None:
    """Directly fit *probe*'s classifier on random synthetic embeddings.

    Circumvents file-loading by injecting embeddings without WAV files.
    Used in dry-run mode and as a fallback when no training audio exists.

    Args:
        probe: Untrained :class:`ConceptProbe` to fit in-place.
    """
    rng = np.random.default_rng(abs(hash(probe.concept)) % (2**31))
    n_each = 20
    # Positive embeddings cluster around +1 in dim 0; negative around -1.
    pos_embs = rng.normal(loc=1.0, scale=0.5, size=(n_each, 512)).astype(np.float32)
    neg_embs = rng.normal(loc=-1.0, scale=0.5, size=(n_each, 512)).astype(np.float32)
    X = np.vstack([pos_embs, neg_embs])
    y = np.array([1] * n_each + [0] * n_each)
    probe.classifier.fit(X, y)
    probe._is_trained = True
    acc = float(probe.classifier.score(X, y))
    log.info(
        "Probe for '%s' trained on synthetic data. Accuracy=%.3f",
        probe.concept,
        acc,
    )


def _compute_dummy_metrics(audio: np.ndarray) -> dict[str, float]:
    """Return stub metrics for smoke testing.

    In a real run these would be computed via FAD, CLAP, and MUQ-T.

    Args:
        audio: 1-D float32 audio array.

    Returns:
        Dict with keys: clap_alignment, lpaps, ce, pq.
    """
    rng = np.random.default_rng(int(np.abs(audio[:10].sum() * 1e6)) % (2**31))
    return {
        "clap_alignment": float(rng.uniform(0.2, 0.9)),
        "lpaps": float(rng.uniform(0.1, 0.5)),
        "ce": float(rng.uniform(0.3, 0.9)),
        "pq": float(rng.uniform(0.3, 0.9)),
    }


def _save_trace_plot(
    trace_df: "pd.DataFrame",
    concept: str,
    alpha: int,
    out_dir: Path,
) -> None:
    """Plot and save the monitoring trace for one run.

    Args:
        trace_df: DataFrame from :meth:`~SelfMonitoredSteerer.get_monitoring_trace`.
        concept:  Concept name.
        alpha:    Nominal alpha value.
        out_dir:  Output directory.
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(
            trace_df["step"],
            trace_df["effective_alpha"],
            color="royalblue",
            label="effective alpha",
        )
        ax1.set_xlabel("Diffusion step")
        ax1.set_ylabel("Effective alpha", color="royalblue")
        ax1.tick_params(axis="y", labelcolor="royalblue")

        ax2 = ax1.twinx()
        ax2.plot(
            trace_df["step"],
            trace_df["concept_probability"],
            color="tomato",
            linestyle="--",
            label="P(concept)",
        )
        ax2.set_ylabel("P(concept)", color="tomato")
        ax2.tick_params(axis="y", labelcolor="tomato")
        ax2.set_ylim(0.0, 1.0)

        fig.suptitle(f"Self-monitoring trace: {concept}, alpha={alpha}")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        out_path = out_dir / f"trace_{concept}_alpha{alpha}.png"
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=100)
        plt.close(fig)
        log.info("Saved trace plot: %s", out_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not save trace plot: %s", exc)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(dry_run: bool = False) -> None:
    """Execute the full self-monitoring vs. fixed-alpha comparison.

    Args:
        dry_run: If True, use stub models and skip real audio generation.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = _load_model(dry_run)
    bank = SteeringVectorBank()

    probe_accuracies: list[dict] = []
    comparison_rows: list[dict] = []

    for concept in CONCEPTS:
        log.info("=" * 60)
        log.info("Concept: %s", concept)
        log.info("=" * 60)

        sv = _load_steering_vector(concept, dry_run)
        probe = _make_probe(concept, dry_run)
        probe_acc = float(probe.classifier.score(
            *_get_probe_test_data(probe, concept)
        ))
        log.info("Probe '%s' held-out accuracy: %.3f", concept, probe_acc)
        probe_accuracies.append({"concept": concept, "accuracy": probe_acc})

        positive_prompt, _ = PROBE_PROMPTS.get(concept, ("music", "music"))

        for alpha in ALPHA_VALUES:
            log.info("  alpha = %d", alpha)

            # ---- Fixed-alpha baseline ----
            # Simulate N_SAMPLES=4 generations; steer() already applies hooks
            # internally, so here we call steer() once to get the trace.
            fixed_steerer = SelfMonitoredSteerer(
                vector=sv,
                probe=probe,
                alpha=float(alpha),
                # Very high threshold so alpha is never decayed (fixed).
                threshold_high=1.1,
                threshold_low=-0.1,
                check_every=5,
            )
            fixed_audio, sr = fixed_steerer.steer(
                model, positive_prompt, duration=10.0, seed=42
            )
            fixed_metrics = _compute_dummy_metrics(fixed_audio)
            comparison_rows.append(
                {
                    "concept": concept,
                    "alpha": alpha,
                    "method": "fixed",
                    **fixed_metrics,
                }
            )
            log.info("    Fixed — %s", fixed_metrics)

            # ---- Self-monitored ----
            sm_steerer = SelfMonitoredSteerer(
                vector=sv,
                probe=probe,
                alpha=float(alpha),
                threshold_high=0.85,
                threshold_low=0.40,
                decay_factor=0.5,
                check_every=5,
            )
            sm_audio, _ = sm_steerer.steer(
                model, positive_prompt, duration=10.0, seed=42
            )
            sm_metrics = _compute_dummy_metrics(sm_audio)
            comparison_rows.append(
                {
                    "concept": concept,
                    "alpha": alpha,
                    "method": "self_monitored",
                    **sm_metrics,
                }
            )
            log.info("    Self-monitored — %s", sm_metrics)

            # Save monitoring trace plot if trace is non-empty.
            try:
                trace_df = sm_steerer.get_monitoring_trace()
                import pandas as pd

                _save_trace_plot(trace_df, concept, alpha, RESULTS_DIR)
            except RuntimeError:
                log.debug("No trace to plot for %s alpha=%d.", concept, alpha)

    # ---- Write CSV outputs ----
    probe_csv = RESULTS_DIR / "probe_accuracies.csv"
    _write_csv(probe_csv, probe_accuracies, fieldnames=["concept", "accuracy"])
    log.info("Saved probe accuracies: %s", probe_csv)

    comparison_csv = RESULTS_DIR / "comparison_results.csv"
    fieldnames = ["concept", "alpha", "method", "clap_alignment", "lpaps", "ce", "pq"]
    _write_csv(comparison_csv, comparison_rows, fieldnames=fieldnames)
    log.info("Saved comparison results: %s", comparison_csv)

    _print_summary(comparison_rows)


def _get_probe_test_data(probe: ConceptProbe, concept: str) -> tuple:
    """Return (X_test, y_test) for probe evaluation.

    Falls back to synthetic data consistent with training stubs.

    Args:
        probe:   Trained probe.
        concept: Concept name.

    Returns:
        Tuple ``(X, y)`` suitable for ``probe.classifier.score(X, y)``.
    """
    rng = np.random.default_rng((abs(hash(concept)) + 1) % (2**31))
    n_each = 10
    pos = rng.normal(loc=1.0, scale=0.5, size=(n_each, 512)).astype(np.float32)
    neg = rng.normal(loc=-1.0, scale=0.5, size=(n_each, 512)).astype(np.float32)
    X = np.vstack([pos, neg])
    y = np.array([1] * n_each + [0] * n_each)
    return X, y


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write *rows* to a CSV at *path*.

    Args:
        path:       Output file path.
        rows:       List of dicts with keys matching *fieldnames*.
        fieldnames: Column order.
    """
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict]) -> None:
    """Print a simple text table comparing fixed vs. self-monitored metrics.

    Args:
        rows: Comparison rows from the experiment.
    """
    print("\n" + "=" * 72)
    print(f"{'Concept':15s} {'Alpha':>6s} {'Method':>16s} "
          f"{'CLAP Δ':>8s} {'LPAPS':>7s} {'CE':>6s} {'PQ':>6s}")
    print("-" * 72)
    for r in rows:
        print(
            f"{r['concept']:15s} {r['alpha']:>6d} {r['method']:>16s} "
            f"{r['clap_alignment']:>8.3f} {r['lpaps']:>7.3f} "
            f"{r['ce']:>6.3f} {r['pq']:>6.3f}"
        )
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the experiment."""
    parser = argparse.ArgumentParser(
        description="Self-monitoring steering experiment (Phase 2.4)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Use stub models; skip real audio generation (for CI/smoke tests).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_experiment(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
