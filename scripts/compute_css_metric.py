#!/usr/bin/env python3
"""
compute_css_metric.py — Compute CSS (Concept Steering Success) from Exp 2 results.

CSS(c, w) = Pr(CLAP_steered > CLAP_unsteered | concept=c, window=w)
Estimated as fraction of pairs (rows) where steered wins.
With the current aggregated CSV, n=1 per (concept, window) — this is a prototype
that will produce real curves once exp2_timestep_commitment_raw.csv is available.

Usage:
    python scripts/compute_css_metric.py \
        [--input results/paper/exp2_timestep_commitment.csv] \
        [--output_csv results/paper/css_curves.csv] \
        [--output_fig results/paper/figures/css_curves.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import binomtest
except ImportError:
    print(
        "ERROR: scipy is not installed. Run: pip install scipy",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONCEPTS = ["piano", "tempo", "mood", "drums", "jazz"]
CONCEPT_COLORS = {
    "piano": "#4C72B0",
    "tempo": "#DD8452",
    "mood":  "#55A868",
    "drums": "#C44E52",
    "jazz":  "#8172B3",
}
_FALLBACK_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _fallback_color(i: int) -> str:
    return _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------


def _detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """Return a mapping of role → actual column name."""
    cols = {c: c.lower() for c in df.columns}
    found: dict[str, str] = {}

    for col, lc in cols.items():
        if "concept" in lc and "col_concept" not in found:
            found["col_concept"] = col
        if lc in {"window", "window_label", "win"} and "col_window" not in found:
            found["col_window"] = col
        if ("window_start" in lc or lc == "start") and "col_win_start" not in found:
            found["col_win_start"] = col
        if ("window_end" in lc or lc == "end") and "col_win_end" not in found:
            found["col_win_end"] = col
        if "steered" in lc and "unsteered" not in lc and "col_steered" not in found:
            found["col_steered"] = col
        if "unsteered" in lc and "col_unsteered" not in found:
            found["col_unsteered"] = col

    return found


# ---------------------------------------------------------------------------
# CSS computation
# ---------------------------------------------------------------------------


def compute_css(
    steered: np.ndarray,
    unsteered: np.ndarray,
) -> tuple[float, float, float, float, int, int]:
    """Return css, ci_lower, ci_upper, pvalue, n, n_wins."""
    wins = steered > unsteered
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
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("results/paper/css_curves.csv"),
    )
    parser.add_argument(
        "--output_fig",
        type=Path,
        default=Path("results/paper/figures/css_curves.png"),
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load + detect columns
    # ------------------------------------------------------------------
    df = pd.read_csv(args.input)
    print(f"Read {len(df)} rows from {args.input}")

    col_map = _detect_columns(df)

    for required in ("col_concept", "col_window", "col_steered", "col_unsteered"):
        if required not in col_map:
            print(
                f"ERROR: could not detect required column '{required}' "
                f"in columns: {list(df.columns)}",
                file=sys.stderr,
            )
            sys.exit(1)

    col_concept   = col_map["col_concept"]
    col_window    = col_map["col_window"]
    col_win_start = col_map.get("col_win_start")
    col_steered   = col_map["col_steered"]
    col_unsteered = col_map["col_unsteered"]

    print(
        f"Columns: concept={col_concept!r}, window={col_window!r}, "
        f"steered={col_steered!r}, unsteered={col_unsteered!r}, "
        f"window_start={col_win_start!r}"
    )

    # ------------------------------------------------------------------
    # Compute CSS per (concept, window)
    # ------------------------------------------------------------------
    rows: list[dict] = []

    for (concept, window), grp in df.groupby([col_concept, col_window], sort=False):
        steered_arr   = grp[col_steered].values.astype(float)
        unsteered_arr = grp[col_unsteered].values.astype(float)

        css, ci_lo, ci_hi, pval, n, n_wins = compute_css(steered_arr, unsteered_arr)

        # window_start
        if col_win_start and col_win_start in grp.columns:
            w_start = float(grp[col_win_start].mean())
        else:
            try:
                w_start = float(str(window).split("-")[0])
            except (ValueError, IndexError):
                w_start = float("nan")

        rows.append(
            {
                "concept":      concept,
                "window":       window,
                "window_start": w_start,
                "css":          css,
                "ci_lower":     ci_lo,
                "ci_upper":     ci_hi,
                "pvalue":       pval,
                "n_prompts":    n,
                "n_wins":       n_wins,
                "significant":  bool(pval < 0.05 and css > 0.5),
            }
        )

    css_df = pd.DataFrame(rows).sort_values(["concept", "window_start"])

    n_concepts = css_df["concept"].nunique()
    n_windows  = css_df["window"].nunique()
    print(f"Computed CSS for {n_concepts} concepts × {n_windows} windows = {len(css_df)} rows")

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    css_df.to_csv(args.output_csv, index=False)
    print(f"Saved {args.output_csv}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, concept in enumerate(css_df["concept"].unique()):
        sub   = css_df[css_df["concept"] == concept].sort_values("window_start")
        xs    = sub["window_start"].values
        ys    = sub["css"].values
        lo    = sub["ci_lower"].values
        hi    = sub["ci_upper"].values
        color = CONCEPT_COLORS.get(concept, _fallback_color(i))

        ax.plot(xs, ys, marker="o", linewidth=2, label=concept, color=color)
        ax.fill_between(xs, lo, hi, alpha=0.15, color=color)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="Chance (0.5)")
    ax.set_xlabel("Denoising Window Start", fontsize=12)
    ax.set_ylabel("CSS  Pr[steered > unsteered]", fontsize=12)
    ax.set_title("Concept Steering Success (CSS) by Denoising Window", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=10)

    args.output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output_fig}")


if __name__ == "__main__":
    main()
