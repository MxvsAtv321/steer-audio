#!/usr/bin/env python3
"""
compute_css_metric.py — Compute Concept Steering Success (CSS) from Exp 2 results.

CSS(c, w) = Pr(CLAP_steered > CLAP_unsteered | concept=c, window=w)
Estimated as fraction of pairs where steered wins. Reported with binomial 95% CI.
Analogous to PCI's CIS metric (Gorgun et al., arXiv:2512.08486, ICLR 2026).

Usage:
    python scripts/compute_css_metric.py \
        [--input results/paper/exp2_timestep_commitment.csv] \
        [--output_csv results/paper/css_curves.csv] \
        [--output_fig results/paper/figures/css_curves.png]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONCEPTS = ["piano", "tempo", "mood", "drums", "jazz"]
CONCEPT_COLORS = {
    "piano": "#4C72B0",
    "tempo": "#DD8452",
    "mood": "#55A868",
    "drums": "#C44E52",
    "jazz": "#8172B3",
}


# ---------------------------------------------------------------------------
# Core CSS computation
# ---------------------------------------------------------------------------


def compute_css(
    clap_steered: np.ndarray,
    clap_unsteered: np.ndarray,
) -> Tuple[float, float, float, float, int, int]:
    """Compute CSS with Wilson 95% CI.

    Args:
        clap_steered:   array of steered CLAP scores
        clap_unsteered: array of unsteered CLAP scores (same length)

    Returns:
        css, ci_lower, ci_upper, pvalue, n, n_wins
    """
    wins = clap_steered > clap_unsteered
    n = len(wins)
    k = int(wins.sum())

    if n == 0:
        return 0.5, 0.0, 1.0, 1.0, 0, 0

    result = binomtest(k=k, n=n, p=0.5, alternative="greater")
    ci = result.proportion_ci(confidence_level=0.95, method="wilson")
    css = k / n
    return css, float(ci.low), float(ci.high), float(result.pvalue), n, k


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CSS metric from Exp 2 timestep commitment results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/paper/exp2_timestep_commitment.csv"),
        help="Path to exp2 CSV.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("results/paper/css_curves.csv"),
        help="Output CSS curves CSV.",
    )
    parser.add_argument(
        "--output_fig",
        type=Path,
        default=Path("results/paper/figures/css_curves.png"),
        help="Output figure path.",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    output_csv: Path = args.output_csv
    output_fig: Path = args.output_fig

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Columns: {list(df.columns)}\n")

    # Detect column names flexibly
    col_steered = next(
        (c for c in df.columns if "steered" in c.lower() and "un" not in c.lower()),
        None,
    )
    col_unsteered = next(
        (c for c in df.columns if "unsteered" in c.lower()),
        None,
    )
    col_concept = next(
        (c for c in df.columns if "concept" in c.lower()),
        "concept",
    )
    col_window = next(
        (c for c in df.columns if c.lower() in ("window", "window_label", "win")),
        "window",
    )
    col_window_start = next(
        (c for c in df.columns if "window_start" in c.lower() or c.lower() == "start"),
        None,
    )

    if col_steered is None or col_unsteered is None:
        print("ERROR: Could not detect clap_steered/clap_unsteered columns.", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    if col_window not in df.columns and "window" in df.columns:
        col_window = "window"

    print(f"Using columns: concept={col_concept!r}, window={col_window!r}, "
          f"steered={col_steered!r}, unsteered={col_unsteered!r}\n")

    # ------------------------------------------------------------------
    # Compute CSS per (concept, window)
    # ------------------------------------------------------------------
    rows: list[dict] = []

    grouped = df.groupby([col_concept, col_window], sort=False)

    for (concept, window), grp in grouped:
        steered_arr = grp[col_steered].values.astype(float)
        unsteered_arr = grp[col_unsteered].values.astype(float)

        css, ci_lo, ci_hi, pval, n, n_wins = compute_css(steered_arr, unsteered_arr)

        # Derive window_start from label (e.g. "0.0-0.2" → 0.0)
        if col_window_start and col_window_start in grp.columns:
            w_start = float(grp[col_window_start].iloc[0])
        else:
            try:
                w_start = float(str(window).split("-")[0])
            except (ValueError, IndexError):
                w_start = float("nan")

        rows.append(
            {
                "concept": concept,
                "window": window,
                "window_start": w_start,
                "css": css,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "pvalue": pval,
                "n_prompts": n,
                "n_wins": n_wins,
                "significant": pval < 0.05,
            }
        )

    css_df = pd.DataFrame(rows).sort_values(["concept", "window_start"])

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    css_df.to_csv(output_csv, index=False)
    print(f"Saved CSS curves to {output_csv} ({len(css_df)} rows)\n")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("=" * 72)
    print("CSS SUMMARY TABLE  (* = p < 0.05)")
    print("=" * 72)

    windows_sorted = sorted(css_df["window"].unique(), key=lambda w: float(str(w).split("-")[0]))
    concepts_present = sorted(css_df["concept"].unique())

    # Header
    header = f"{'concept':<10}" + "".join(f"{w:>12}" for w in windows_sorted)
    print(header)
    print("-" * len(header))

    for concept in concepts_present:
        sub = css_df[css_df["concept"] == concept].set_index("window")
        row_str = f"{concept:<10}"
        for w in windows_sorted:
            if w in sub.index:
                r = sub.loc[w]
                star = "*" if r["significant"] else " "
                row_str += f"  {r['css']:.3f}{star}     "
            else:
                row_str += f"  {'N/A':<9}"
        print(row_str)

    print("=" * 72)
    print()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    output_fig.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for concept in concepts_present:
        sub = css_df[css_df["concept"] == concept].sort_values("window_start")
        xs = sub["window_start"].values
        ys = sub["css"].values
        lo = sub["ci_lower"].values
        hi = sub["ci_upper"].values
        color = CONCEPT_COLORS.get(concept, None)

        ax.plot(xs, ys, marker="o", linewidth=2, label=concept, color=color)
        ax.fill_between(xs, lo, hi, alpha=0.15, color=color)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="Chance (0.5)")
    ax.set_xlabel("Denoising Window Start", fontsize=12)
    ax.set_ylabel("CSS (Pr[steered > unsteered])", fontsize=12)
    ax.set_title("Concept Steering Success by Denoising Window", fontsize=13)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.05, 0.95)
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_fig}")


if __name__ == "__main__":
    main()
