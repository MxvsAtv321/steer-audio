#!/usr/bin/env python3
"""
analyze_concept_geometry.py — Geometry analysis of concept steering vectors.

Loads pre-computed CAA steering vectors from outputs/vectors/ (sv.pkl files)
and computes:
  1. Cosine similarity matrix (5×5)
  2. Vector norms
  3. Bootstrap stability (if raw activation pairs available)
  4. PCA scatter + scree plot

Output:
  results/paper/figures/concept_geometry.png  — 3-panel figure
  results/paper/concept_geometry.csv          — per-concept stats

Usage:
    python scripts/analyze_concept_geometry.py \
        [--vectors_dir outputs/vectors] \
        [--output_dir results/paper]
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONCEPTS = ["piano", "tempo", "mood", "drums", "jazz"]
# Sub-directory name patterns to check (RunPod may have drums/jazz)
VECTOR_DIR_PATTERNS = [
    "ace_{concept}_passes2_allTrue",
    "ace_{concept}",
    "{concept}",
]
LAYER_PRIORITY = ["tf7", "tf6"]  # preferred layers for representative vector
N_BOOTSTRAP = 100
BOOTSTRAP_FRAC = 0.5


# ---------------------------------------------------------------------------
# Vector loading
# ---------------------------------------------------------------------------


def find_sv_pkl(vectors_dir: Path, concept: str) -> Path | None:
    """Search for sv.pkl for a given concept under vectors_dir."""
    for pattern in VECTOR_DIR_PATTERNS:
        candidate = vectors_dir / pattern.format(concept=concept) / "sv.pkl"
        if candidate.exists():
            return candidate
    # Fallback: recursive search
    for p in vectors_dir.rglob("sv.pkl"):
        if concept in p.parts:
            return p
    return None


def extract_mean_vector(sv: dict, layer: str) -> np.ndarray | None:
    """Extract and average the steering vectors across all timesteps for a layer.

    sv.pkl format: {int_step: {layer_name: [np.ndarray]}}
    """
    vecs = []
    for step_key, layer_dict in sv.items():
        if layer in layer_dict:
            arr = layer_dict[layer]
            if isinstance(arr, list):
                arr = arr[0]
            if hasattr(arr, "cpu"):
                arr = arr.cpu().float().numpy()
            vecs.append(np.asarray(arr, dtype=np.float32).ravel())
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def load_concept_vector(vectors_dir: Path, concept: str) -> np.ndarray | None:
    """Load a single representative steering vector for *concept*."""
    pkl_path = find_sv_pkl(vectors_dir, concept)
    if pkl_path is None:
        print(f"  WARNING: sv.pkl not found for concept '{concept}'", file=sys.stderr)
        return None

    with open(pkl_path, "rb") as f:
        sv = pickle.load(f)

    # Try preferred layers
    for layer in LAYER_PRIORITY:
        vec = extract_mean_vector(sv, layer)
        if vec is not None:
            print(f"  {concept}: loaded from {pkl_path} (layer={layer}, dim={vec.shape[0]})")
            return vec

    # Fallback: any available layer
    all_layers = set()
    for step_dict in sv.values():
        all_layers.update(step_dict.keys())

    for layer in sorted(all_layers):
        vec = extract_mean_vector(sv, layer)
        if vec is not None:
            print(f"  {concept}: loaded from {pkl_path} (layer={layer} [fallback], dim={vec.shape[0]})")
            return vec

    print(f"  WARNING: No usable vectors found in {pkl_path}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Bootstrap stability (only if multiple sv.pkl variants available)
# ---------------------------------------------------------------------------


def bootstrap_stability(
    vectors_dir: Path,
    concept: str,
    n_bootstrap: int = N_BOOTSTRAP,
    frac: float = BOOTSTRAP_FRAC,
) -> float | None:
    """Compute bootstrap cosine-similarity stability for *concept*.

    Looks for multiple sv.pkl files (e.g., different seeds/subsets).
    Returns mean pairwise cosine similarity across bootstrap vectors,
    or None if not enough data.
    """
    # Collect all sv.pkl paths mentioning this concept
    sv_paths = list(vectors_dir.rglob(f"*{concept}*/sv.pkl"))
    if len(sv_paths) < 2:
        return None

    # For each path, extract a mean vector and use it as a "bootstrap replicate"
    bootstrap_vecs = []
    for p in sv_paths[:n_bootstrap]:
        try:
            with open(p, "rb") as f:
                sv = pickle.load(f)
            for layer in LAYER_PRIORITY:
                vec = extract_mean_vector(sv, layer)
                if vec is not None:
                    bootstrap_vecs.append(vec)
                    break
        except Exception:
            continue

    if len(bootstrap_vecs) < 2:
        return None

    mat = np.stack(bootstrap_vecs)
    cos_mat = cosine_similarity(mat)
    # Mean of off-diagonal elements
    n = len(bootstrap_vecs)
    off_diag = cos_mat[np.triu_indices(n, k=1)]
    return float(off_diag.mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concept geometry analysis of CAA steering vectors."
    )
    parser.add_argument(
        "--vectors_dir",
        type=Path,
        default=Path("outputs/vectors"),
        help="Directory containing per-concept sv.pkl files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/paper"),
        help="Output directory for CSV and figure.",
    )
    args = parser.parse_args()

    vectors_dir: Path = args.vectors_dir
    output_dir: Path = args.output_dir

    if not vectors_dir.exists():
        print(f"ERROR: vectors_dir not found: {vectors_dir}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load vectors
    # ------------------------------------------------------------------
    print("Loading concept vectors …")
    concept_vecs: dict[str, np.ndarray] = {}
    for concept in CONCEPTS:
        vec = load_concept_vector(vectors_dir, concept)
        if vec is not None:
            concept_vecs[concept] = vec

    if len(concept_vecs) < 2:
        print(
            "ERROR: fewer than 2 concept vectors loaded. "
            "Ensure sv.pkl files are present under vectors_dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    loaded_concepts = list(concept_vecs.keys())
    V = np.stack([concept_vecs[c] for c in loaded_concepts])  # (n_concepts, hidden_dim)
    print(f"\nLoaded {len(loaded_concepts)} concepts: {loaded_concepts}")
    print(f"Vector shape: {V.shape}\n")

    # ------------------------------------------------------------------
    # 2. Cosine similarity matrix
    # ------------------------------------------------------------------
    cos_sim = cosine_similarity(V)  # (n_concepts, n_concepts)

    # ------------------------------------------------------------------
    # 3. Vector norms
    # ------------------------------------------------------------------
    norms = np.linalg.norm(V, axis=1)

    # ------------------------------------------------------------------
    # 4. Bootstrap stability
    # ------------------------------------------------------------------
    print("Computing bootstrap stability …")
    stabilities: dict[str, float | None] = {}
    for concept in loaded_concepts:
        stab = bootstrap_stability(vectors_dir, concept)
        if stab is None:
            print(f"  {concept}: insufficient data for bootstrap, skipping.")
        stabilities[concept] = stab

    # ------------------------------------------------------------------
    # 5. PCA
    # ------------------------------------------------------------------
    n_components = min(len(loaded_concepts), V.shape[1])
    pca2 = PCA(n_components=min(2, n_components))
    coords_2d = pca2.fit_transform(V)

    pca_full = PCA(n_components=n_components)
    pca_full.fit(V)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)

    # Components needed for 90% variance
    n_for_90 = int(np.searchsorted(cum_var, 0.90)) + 1

    # ------------------------------------------------------------------
    # 6. Figure — 3 panels
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Concept Steering Vector Geometry", fontsize=14, y=1.01)

    # Panel a: Cosine similarity heatmap
    ax = axes[0]
    df_cos = pd.DataFrame(cos_sim, index=loaded_concepts, columns=loaded_concepts)
    sns.heatmap(
        df_cos,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("(a) Cosine Similarity", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)

    # Panel b: PCA 2D scatter
    ax = axes[1]
    if coords_2d.shape[1] >= 2:
        xv = pca2.explained_variance_ratio_[0] * 100
        yv = pca2.explained_variance_ratio_[1] * 100
        for i, concept in enumerate(loaded_concepts):
            marker = "X" if concept == "tempo" else "o"
            ax.scatter(coords_2d[i, 0], coords_2d[i, 1], marker=marker,
                       s=150, zorder=3, label=concept)
            ax.annotate(concept, (coords_2d[i, 0], coords_2d[i, 1]),
                        textcoords="offset points", xytext=(6, 4), fontsize=10)
        ax.set_xlabel(f"PC1 ({xv:.1f}% var)", fontsize=11)
        ax.set_ylabel(f"PC2 ({yv:.1f}% var)", fontsize=11)
        ax.legend(fontsize=9, loc="best")
    else:
        ax.text(0.5, 0.5, "Only 1 component\n(need ≥2 concepts)", ha="center", va="center",
                transform=ax.transAxes, fontsize=11)
    ax.set_title("(b) PCA 2D Projection", fontsize=12)
    ax.tick_params(labelsize=10)

    # Panel c: Scree plot
    ax = axes[2]
    component_indices = np.arange(1, len(cum_var) + 1)
    ax.plot(component_indices, cum_var, marker="o", linewidth=2, color="#4C72B0")
    ax.axhline(0.90, color="red", linestyle="--", linewidth=1.0, label="90% variance")
    ax.axvline(n_for_90, color="red", linestyle=":", linewidth=1.0,
               label=f"PC{n_for_90} achieves 90%")
    ax.set_xlabel("Number of Principal Components", fontsize=11)
    ax.set_ylabel("Cumulative Explained Variance", fontsize=11)
    ax.set_title("(c) PCA Scree Plot", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    output_dir.joinpath("figures").mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "figures" / "concept_geometry.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {fig_path}")

    # ------------------------------------------------------------------
    # 7. Save CSV
    # ------------------------------------------------------------------
    tempo_idx = loaded_concepts.index("tempo") if "tempo" in loaded_concepts else None
    piano_idx = loaded_concepts.index("piano") if "piano" in loaded_concepts else None

    csv_rows = []
    for i, concept in enumerate(loaded_concepts):
        csv_rows.append(
            {
                "concept": concept,
                "vector_norm": float(norms[i]),
                "bootstrap_stability": stabilities.get(concept),
                "pca_pc1": float(coords_2d[i, 0]) if coords_2d.shape[1] >= 1 else float("nan"),
                "pca_pc2": float(coords_2d[i, 1]) if coords_2d.shape[1] >= 2 else float("nan"),
                "cosine_to_tempo": float(cos_sim[i, tempo_idx]) if tempo_idx is not None else float("nan"),
            }
        )

    csv_path = output_dir / "concept_geometry.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Saved geometry CSV to {csv_path}")

    # ------------------------------------------------------------------
    # 8. Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("CONCEPT GEOMETRY SUMMARY")
    print("=" * 72)

    header = f"{'concept':<10} {'norm':>8} {'stability':>11}"
    for other in loaded_concepts:
        header += f" {'cos_' + other:>12}"
    print(header)
    print("-" * len(header))

    for i, concept in enumerate(loaded_concepts):
        stab = stabilities.get(concept)
        stab_str = f"{stab:.3f}" if stab is not None else "   N/A"
        row_str = f"{concept:<10} {norms[i]:>8.2f} {stab_str:>11}"
        for j in range(len(loaded_concepts)):
            row_str += f" {cos_sim[i, j]:>12.4f}"
        print(row_str)

    print("=" * 72)
    print(f"\nPCA: {n_for_90} component(s) needed for 90% explained variance")
    print()


if __name__ == "__main__":
    main()
