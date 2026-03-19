#!/usr/bin/env python3
"""Steering evaluation sweep — Phase 3.4-pre.

Evaluates pre-generated steered audio across all alpha values, producing a
unified metrics CSV and two plots (CLAP alignment vs α, FAD/LPAPS vs α).

Reads the ``alpha_*/`` directory structure written by
``steering/ace_steer/eval_steering_vectors.py`` and runs all available metric
backends (CLAP, FAD, LPAPS) via :class:`steer_audio.eval_metrics.EvalSuite`.

Scientific purpose:
    Bridges geometry analysis (Phase 3.3) with empirical steering quality.
    The resulting CLAP-vs-alpha curve shows the "steering range" for each
    concept; the FAD/LPAPS curves show whether audio quality degrades at
    high alpha.  Together these support the paper's claim that CAA steers
    effectively without excessive quality loss.

Outputs
-------
  <out-dir>/
    metrics.csv           — one row per alpha, columns: alpha, clap, fad, lpaps
    clap_vs_alpha.png     — alignment curve
    fad_vs_alpha.png      — FAD curve (if FAD backend available)
    lpaps_vs_alpha.png    — LPAPS curve (if LPAPS backend available)

Usage
-----
  # Dry-run — generates synthetic audio, uses stub backends, no models needed:
  python experiments/eval_sweep.py --dry-run --concept tempo

  # Real run — evaluate pre-generated steered audio:
  python experiments/eval_sweep.py \\
      --steered-dir outputs/ace/steering/tempo/tf7 \\
      --reference-dir outputs/ace/baseline \\
      --concept tempo \\
      --prompt "fast tempo music" \\
      --out-dir results/eval/tempo
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from steer_audio.eval_metrics import (
    EvalSuite,
    compute_alpha_sweep,
    plot_alpha_sweep,
)

# ---------------------------------------------------------------------------
# Synthetic WAV generation for dry-run
# ---------------------------------------------------------------------------

# Alpha values used in the synthetic dry-run grid
_DRY_ALPHAS: List[float] = [-100, -50, -20, 0, 20, 50, 100]
_DRY_N_WAVS: int = 4  # WAV files per alpha directory
_DRY_SAMPLE_RATE: int = 44100
_DRY_DURATION_S: float = 1.0  # 1-second clips — enough to test the pipeline


def _write_sine_wav(path: Path, freq: float, sr: int, duration: float) -> None:
    """Write a single-channel sine-wave WAV file.

    Args:
        path:     Destination file path (parent must exist).
        freq:     Sine frequency in Hz.
        sr:       Sample rate in Hz.
        duration: Duration in seconds.
    """
    import struct
    import wave

    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def make_dry_run_dirs(base_dir: Path, alphas: List[float] = _DRY_ALPHAS) -> Path:
    """Create synthetic ``alpha_*/`` audio directories for dry-run mode.

    Each directory contains :data:`_DRY_N_WAVS` sine-wave WAV files whose
    frequency is modulated by alpha (higher alpha → slightly higher pitch)
    to create a detectable signal without real model inference.

    Args:
        base_dir: Root directory under which ``alpha_*/`` are created.
        alphas:   Alpha values to generate directories for.

    Returns:
        Path to the created root directory.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    base_freq = 440.0  # A4

    for alpha in alphas:
        alpha_dir = base_dir / f"alpha_{alpha:.0f}"
        alpha_dir.mkdir(exist_ok=True)
        # Pitch shift proportional to alpha so metrics can detect structure
        freq = base_freq * (1.0 + alpha / 500.0)
        for i in range(_DRY_N_WAVS):
            wav_path = alpha_dir / f"sample_{i:02d}.wav"
            _write_sine_wav(wav_path, freq=freq + i * 5,
                            sr=_DRY_SAMPLE_RATE, duration=_DRY_DURATION_S)

    # Write a reference (alpha=0) directory
    ref_dir = base_dir / "reference"
    ref_dir.mkdir(exist_ok=True)
    for i in range(_DRY_N_WAVS):
        _write_sine_wav(ref_dir / f"ref_{i:02d}.wav",
                        freq=base_freq + i * 5,
                        sr=_DRY_SAMPLE_RATE, duration=_DRY_DURATION_S)

    log.info(
        "Dry-run: created %d alpha dirs + reference in %s", len(alphas), base_dir
    )
    return base_dir


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------


def run_eval_sweep(
    steered_dir: Path,
    out_dir: Path,
    concept: str,
    prompt: Optional[str],
    reference_dir: Optional[Path],
    stub: bool,
    backends: List[str],
) -> "pd.DataFrame":
    """Run the full evaluation sweep and write outputs.

    Args:
        steered_dir:   Root of ``alpha_*/`` directories.
        out_dir:       Directory for CSV and plots.
        concept:       Concept name (for plot titles).
        prompt:        Text prompt forwarded to CLAP.
        reference_dir: Baseline directory for FAD and LPAPS.
        stub:          If True, all backends return fixed stub values.
        backends:      List of backend names to enable.

    Returns:
        The metrics DataFrame.
    """
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    suite = EvalSuite(backends=backends, stub=stub)

    # Log backend availability
    avail = suite.availability()
    for name, ok in avail.items():
        status = "available" if ok else "unavailable (will return NaN)"
        log.info("Backend %-8s: %s", name, status)

    log.info("Running alpha sweep in %s", steered_dir)
    df = compute_alpha_sweep(
        steered_dir=steered_dir,
        suite=suite,
        prompt=prompt,
        reference_dir=reference_dir,
    )

    if df.empty:
        log.error("No alpha directories found — nothing to evaluate.")
        return df

    # Write CSV
    csv_path = out_dir / "metrics.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    log.info("Saved metrics → %s", csv_path)

    # Write plots
    plot_alpha_sweep(df, out_dir, concept=concept)

    # Print summary table to stdout
    print(f"\n{'─'*60}")
    print(f"Eval sweep: concept={concept}, {len(df)} alpha values")
    print(f"{'─'*60}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"{'─'*60}")
    print(f"Outputs → {out_dir.resolve()}")
    print()

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Steering evaluation sweep (Phase 3.4-pre)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Generate synthetic audio and use stub backends. "
            "No model weights or ACE-Step required."
        ),
    )
    parser.add_argument(
        "--steered-dir",
        type=Path,
        default=None,
        help="Root directory containing alpha_*/ subdirectories (real run).",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Unsteered baseline audio directory (used by FAD and LPAPS).",
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="concept",
        help="Concept name for plot titles and CSV metadata.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for CLAP alignment scoring.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/eval"),
        help="Output directory for metrics.csv and plots.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["clap", "fad", "lpaps"],
        choices=["clap", "fad", "lpaps"],
        help="Metric backends to enable.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the eval sweep script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args(argv)

    if args.dry_run:
        with tempfile.TemporaryDirectory() as tmp:
            dry_dir = Path(tmp) / "dry_steered"
            make_dry_run_dirs(dry_dir)
            reference_dir = dry_dir / "reference"
            out_dir = args.out_dir / args.concept
            run_eval_sweep(
                steered_dir=dry_dir,
                out_dir=out_dir,
                concept=args.concept,
                prompt=args.prompt or f"music with {args.concept}",
                reference_dir=reference_dir,
                stub=True,  # dry-run always uses stubs
                backends=args.backends,
            )
    else:
        if args.steered_dir is None:
            log.error("--steered-dir is required for real runs (or use --dry-run).")
            sys.exit(1)
        out_dir = args.out_dir / args.concept
        run_eval_sweep(
            steered_dir=args.steered_dir,
            out_dir=out_dir,
            concept=args.concept,
            prompt=args.prompt,
            reference_dir=args.reference_dir,
            stub=False,
            backends=args.backends,
        )


if __name__ == "__main__":
    main()
