#!/usr/bin/env python3
"""
compute_fad_human_eval.py — Compute FAD between steered and unsteered
human eval WAVs for each concept.

Uses fadtk with encodec-emb and clap-laion-music embeddings.

Usage:
    python scripts/compute_fad_human_eval.py \
        [--root results/paper/human_eval] \
        [--output_csv results/paper/human_eval_fad.csv] \
        [--tmp_dir /tmp]
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FAD_EMBEDDINGS = ["encodec-emb", "clap-laion-music"]
CONCEPTS = ["drums", "jazz", "mood", "piano", "tempo"]
FAD_WARN_THRESHOLD = 10.0
MIN_RELIABLE_SAMPLES = 50


# ---------------------------------------------------------------------------
# FAD computation helpers
# ---------------------------------------------------------------------------


def run_fad_python_api(embedding: str, baseline_dir: Path, eval_dir: Path) -> float | None:
    """Try to compute FAD using the fadtk Python API.

    Returns FAD score or None if the API is unavailable.
    """
    try:
        from fadtk import FrechetAudioDistance  # type: ignore

        fad = FrechetAudioDistance(embedding, audio_load_worker=4)
        score = fad.score(str(baseline_dir), str(eval_dir))
        return float(score)
    except Exception as e:
        print(f"    Python API failed ({e}), falling back to subprocess …")
        return None


def run_fad_subprocess(embedding: str, baseline_dir: Path, eval_dir: Path) -> float | None:
    """Compute FAD via fadtk CLI subprocess.

    Returns FAD score parsed from stdout, or None on failure.
    """
    try:
        result = subprocess.run(
            ["fadtk", embedding, str(baseline_dir), str(eval_dir)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"    CLI error: {result.stderr.strip()}", file=sys.stderr)
            return None
        # Parse FAD score from stdout — fadtk prints something like "FAD: 3.141592"
        for line in result.stdout.splitlines():
            line_lower = line.lower()
            if "fad" in line_lower or any(char.isdigit() for char in line):
                tokens = line.split()
                for tok in reversed(tokens):
                    try:
                        return float(tok)
                    except ValueError:
                        continue
        print(f"    Could not parse FAD from output: {result.stdout!r}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("    ERROR: 'fadtk' CLI not found in PATH.", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print("    ERROR: fadtk subprocess timed out.", file=sys.stderr)
        return None


def compute_fad(embedding: str, baseline_dir: Path, eval_dir: Path) -> float | None:
    """Compute FAD using Python API, falling back to CLI subprocess."""
    score = run_fad_python_api(embedding, baseline_dir, eval_dir)
    if score is None:
        score = run_fad_subprocess(embedding, baseline_dir, eval_dir)
    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute FAD between steered and unsteered human eval WAVs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/paper/human_eval"),
        help="Root directory with concept subdirectories.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("results/paper/human_eval_fad.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--tmp_dir",
        type=Path,
        default=Path("/tmp"),
        help="Base directory for temporary WAV staging.",
    )
    args = parser.parse_args()

    root: Path = args.root
    output_csv: Path = args.output_csv
    tmp_base: Path = args.tmp_dir

    # Verify fadtk is available
    try:
        import fadtk  # noqa: F401
        print("fadtk: ok\n")
    except ImportError:
        print("ERROR: fadtk not installed. Run: pip install fadtk", file=sys.stderr)
        sys.exit(1)

    if not root.exists():
        print(f"ERROR: root directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []

    for concept in CONCEPTS:
        concept_dir = root / concept
        if not concept_dir.exists():
            print(f"Skipping {concept}: directory not found at {concept_dir}")
            continue

        print(f"\nConcept: {concept}")

        # Separate WAVs into steered / unsteered
        all_wavs = sorted(concept_dir.glob("*.wav"))
        unsteered_wavs = [w for w in all_wavs if "unsteered" in w.name]
        steered_wavs = [w for w in all_wavs if "steered" in w.name and "unsteered" not in w.name]

        n_unsteered = len(unsteered_wavs)
        n_steered = len(steered_wavs)
        print(f"  Found {n_unsteered} unsteered, {n_steered} steered WAVs")

        if n_unsteered == 0 or n_steered == 0:
            print(f"  WARNING: Missing files for {concept}, skipping.")
            continue

        if n_unsteered < MIN_RELIABLE_SAMPLES or n_steered < MIN_RELIABLE_SAMPLES:
            print(
                f"  WARNING: n={min(n_unsteered, n_steered)} per condition — "
                f"FAD estimates are unreliable at this sample size (< {MIN_RELIABLE_SAMPLES})."
            )

        # Stage files in temporary directories
        tmp_unsteered = tmp_base / f"fad_{concept}_unsteered"
        tmp_steered = tmp_base / f"fad_{concept}_steered"
        tmp_unsteered.mkdir(parents=True, exist_ok=True)
        tmp_steered.mkdir(parents=True, exist_ok=True)

        for w in unsteered_wavs:
            dst = tmp_unsteered / w.name
            if not dst.exists():
                shutil.copy2(w, dst)

        for w in steered_wavs:
            dst = tmp_steered / w.name
            if not dst.exists():
                shutil.copy2(w, dst)

        # Compute FAD for each embedding
        fad_scores: dict[str, float | None] = {}
        for emb in FAD_EMBEDDINGS:
            print(f"  Computing FAD ({emb}) …")
            score = compute_fad(emb, tmp_unsteered, tmp_steered)
            fad_scores[emb] = score
            if score is not None:
                flag = " *** HIGH FAD ***" if score > FAD_WARN_THRESHOLD else ""
                print(f"    FAD ({emb}) = {score:.4f}{flag}")
            else:
                print(f"    FAD ({emb}) = FAILED")

        rows.append(
            {
                "concept": concept,
                "n_unsteered": n_unsteered,
                "n_steered": n_steered,
                "fad_encodec": fad_scores.get("encodec-emb"),
                "fad_clap": fad_scores.get("clap-laion-music"),
            }
        )

    if not rows:
        print("\nERROR: No FAD scores computed.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["concept", "n_unsteered", "n_steered", "fad_encodec", "fad_clap"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved FAD results to {output_csv}")

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FAD RESULTS TABLE  (*** = FAD > 10, potential degradation)")
    print("=" * 60)
    print(f"{'concept':<10} {'n_un':>6} {'n_st':>6} {'FAD_encodec':>13} {'FAD_clap':>12}")
    print("-" * 60)
    for r in rows:
        enc = r["fad_encodec"]
        clp = r["fad_clap"]
        enc_str = f"{enc:.4f}" if enc is not None else "  FAILED"
        clp_str = f"{clp:.4f}" if clp is not None else "  FAILED"
        enc_flag = " ***" if enc is not None and enc > FAD_WARN_THRESHOLD else "    "
        clp_flag = " ***" if clp is not None and clp > FAD_WARN_THRESHOLD else "    "
        print(
            f"{r['concept']:<10} {r['n_unsteered']:>6} {r['n_steered']:>6} "
            f"{enc_str:>13}{enc_flag} {clp_str:>12}{clp_flag}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
