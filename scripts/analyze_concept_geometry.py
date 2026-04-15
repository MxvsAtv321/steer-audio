#!/usr/bin/env python3
"""
analyze_concept_geometry.py — Geometry analysis of concept steering vectors.

Discovers vector files for the 5 concepts (piano, tempo, mood, drums, jazz)
under outputs/vectors/, loads them with a multi-format loader
(torch → numpy → pickle), then computes:
  1. 5×5 cosine similarity matrix
  2. Vector norms
  3. PCA (up to 3 components)

Outputs:
  results/paper/vector_cosine_matrix.csv
  results/paper/vector_pca.csv
  results/paper/figures/fig2_concept_geometry.png  (+ .pdf)

Paths are resolved relative to the repo root (parent of this script's
directory), so the script works on Mac (/Users/.../steer-audio) and in
Colab (/content/steer-audio) without any changes.

Usage:
    python scripts/analyze_concept_geometry.py [--vectors_dir ...] [--output_dir ...]
"""

from __future__ import annotations

import argparse
import os
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
# Repo-relative paths  (work on Mac *and* Colab without modification)
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_VECTORS_DIR: Path = REPO_ROOT / "outputs" / "vectors"
DEFAULT_OUTPUT_DIR: Path = REPO_ROOT / "results" / "paper"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONCEPTS = ["piano", "tempo", "mood", "drums", "jazz"]
LAYER_PRIORITY = ["tf7", "tf6"]   # preferred layers when sv.pkl is a step-dict


# ---------------------------------------------------------------------------
# Multi-format loader
# ---------------------------------------------------------------------------

def _try_loaders(path: Path) -> tuple[object, str] | tuple[None, None]:
    """Try torch → numpy → pickle in order.  Return (obj, loader_name) or (None, None)."""
    # 1. torch.load
    try:
        import torch
        obj = torch.load(str(path), weights_only=False)
        return obj, "torch.load"
    except Exception:
        pass

    # 2. np.load
    try:
        obj = np.load(str(path), allow_pickle=True)
        return obj, "np.load"
    except Exception:
        pass

    # 3. pickle
    try:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        return obj, "pickle.load"
    except Exception:
        pass

    return None, None


def _to_1d_float32(obj: object) -> np.ndarray | None:
    """Flatten any supported object to a 1-D float32 numpy array."""
    # torch Tensor
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().float().numpy().ravel()
    except ImportError:
        pass

    # plain ndarray or numpy scalar
    if isinstance(obj, np.ndarray):
        return obj.astype(np.float32).ravel()

    # numpy 0-d array wrapped in object array (np.load result)
    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        inner = obj.item()
        return _to_1d_float32(inner)

    # sv.pkl format: {int_step: {layer_name: [np.ndarray]}}
    if isinstance(obj, dict):
        return _extract_sv_pkl(obj)

    # list / tuple of arrays
    if isinstance(obj, (list, tuple)) and len(obj) > 0:
        first = obj[0]
        try:
            import torch
            if isinstance(first, torch.Tensor):
                return first.detach().cpu().float().numpy().ravel()
        except ImportError:
            pass
        if isinstance(first, np.ndarray):
            return first.astype(np.float32).ravel()

    return None


def _extract_sv_pkl(sv: dict) -> np.ndarray | None:
    """Average steering vectors across timesteps from sv.pkl step-dict format."""
    vecs: list[np.ndarray] = []
    for step_val in sv.values():
        # step_val is either {layer: [...]} or a direct array
        if isinstance(step_val, dict):
            layer_dict = step_val
        else:
            layer_dict = {"_": step_val}

        # prefer tf7 / tf6, then anything
        for layer in LAYER_PRIORITY + sorted(
            k for k in layer_dict if k not in LAYER_PRIORITY
        ):
            if layer not in layer_dict:
                continue
            arr = layer_dict[layer]
            if isinstance(arr, list):
                arr = arr[0]
            try:
                import torch
                if isinstance(arr, torch.Tensor):
                    arr = arr.detach().cpu().float().numpy()
            except ImportError:
                pass
            if isinstance(arr, np.ndarray) and arr.size > 0:
                vecs.append(arr.astype(np.float32).ravel())
                break   # one layer per step is enough

    if not vecs:
        return None
    return np.mean(vecs, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_concept_files(vectors_dir: Path) -> dict[str, Path]:
    """
    Walk *vectors_dir* recursively.  For each file whose name (or any parent
    directory name inside vectors_dir) contains a concept substring, record it.
    Files are filtered to common vector extensions; directories named after a
    concept are searched for the first loadable file inside them.
    """
    EXTENSIONS = {".pkl", ".pt", ".pth", ".npy", ".npz", ""}

    print(f"\nScanning {vectors_dir} …")
    concept_files: dict[str, Path] = {}

    for path in sorted(vectors_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in EXTENSIONS:
            continue

        size_kb = path.stat().st_size / 1024
        # Build a searchable string from the path relative to vectors_dir
        rel = str(path.relative_to(vectors_dir)).lower()
        print(f"  {path.relative_to(vectors_dir)}  ({size_kb:.1f} KB)")

        for concept in CONCEPTS:
            if concept in concept_files:
                continue   # already found one for this concept
            if concept in rel:
                concept_files[concept] = path
                break

    return concept_files


# ---------------------------------------------------------------------------
# Load one concept vector
# ---------------------------------------------------------------------------

def load_vector(path: Path, concept: str) -> np.ndarray | None:
    obj, loader = _try_loaders(path)
    if obj is None:
        print(f"  ERROR: all loaders failed for {path}", file=sys.stderr)
        return None

    vec = _to_1d_float32(obj)
    if vec is None:
        print(f"  ERROR: could not extract 1-D array from {path} (type={type(obj).__name__})",
              file=sys.stderr)
        return None

    print(f"  {concept}: loader={loader}, shape={vec.shape}, dtype={vec.dtype}")
    return vec


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(
    concepts: list[str],
    cos_sim: np.ndarray,
    coords_2d: np.ndarray,
    ev_ratio: np.ndarray,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Concept Steering Vector Geometry", fontsize=13, y=1.02)

    # --- Panel 1: cosine similarity heatmap ---
    ax = axes[0]
    df_cos = pd.DataFrame(cos_sim, index=concepts, columns=concepts)
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
    ax.set_title("Concept vector cosine similarity", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    # --- Panel 2: PCA scatter ---
    ax = axes[1]
    palette = sns.color_palette("tab10", n_colors=len(concepts))
    color_map = {c: palette[i] for i, c in enumerate(concepts)}
    color_map["tempo"] = "crimson"   # distinct color for tempo per spec

    if coords_2d.shape[1] >= 2:
        xv = ev_ratio[0] * 100
        yv = ev_ratio[1] * 100
        for i, concept in enumerate(concepts):
            ax.scatter(
                coords_2d[i, 0], coords_2d[i, 1],
                color=color_map[concept],
                marker="X" if concept == "tempo" else "o",
                s=160, zorder=3,
            )
            ax.annotate(
                concept,
                (coords_2d[i, 0], coords_2d[i, 1]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=10,
            )
        ax.set_xlabel(f"PC1 ({xv:.1f}% var)", fontsize=10)
        ax.set_ylabel(f"PC2 ({yv:.1f}% var)", fontsize=10)
        ax.text(
            0.02, 0.97,
            f"Explained var: PC1={xv:.1f}%, PC2={yv:.1f}%",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            color="gray",
        )
    else:
        ax.text(0.5, 0.5, "Need ≥ 2 concepts for scatter",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title("PCA (PC1 vs PC2)", fontsize=11)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    png_path = figures_dir / "fig2_concept_geometry.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure → {png_path}")

    try:
        pdf_path = figures_dir / "fig2_concept_geometry.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved figure → {pdf_path}")
    except Exception as exc:
        warnings.warn(f"PDF save failed ({exc}); PNG still saved.")

    plt.close(fig)


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
        default=DEFAULT_VECTORS_DIR,
        help=f"Directory containing per-concept vector files (default: {DEFAULT_VECTORS_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for CSVs and figure (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    vectors_dir: Path = args.vectors_dir
    output_dir: Path = args.output_dir

    print(f"REPO_ROOT   : {REPO_ROOT}")
    print(f"vectors_dir : {vectors_dir}")
    print(f"output_dir  : {output_dir}")

    if not vectors_dir.exists():
        print(f"\nERROR: vectors_dir not found: {vectors_dir}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Discover files
    # ------------------------------------------------------------------
    concept_files = discover_concept_files(vectors_dir)
    missing = [c for c in CONCEPTS if c not in concept_files]
    if missing:
        print(f"\nERROR: vectors not found for: {missing}", file=sys.stderr)
        print("Ensure outputs/vectors/ contains files (or sub-dirs) whose names "
              "include the concept substring.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load vectors
    # ------------------------------------------------------------------
    print("\nLoading vectors …")
    concept_vecs: dict[str, np.ndarray] = {}
    for concept in CONCEPTS:
        vec = load_vector(concept_files[concept], concept)
        if vec is None:
            print(f"\nERROR: could not load vector for '{concept}'", file=sys.stderr)
            sys.exit(1)
        concept_vecs[concept] = vec

    V = np.stack([concept_vecs[c] for c in CONCEPTS])   # (5, d)
    print(f"\nMatrix V shape: {V.shape}")

    # ------------------------------------------------------------------
    # 3. L2 norms
    # ------------------------------------------------------------------
    norms = np.linalg.norm(V, axis=1)
    print("\nL2 norms:")
    for concept, norm in zip(CONCEPTS, norms):
        print(f"  {concept:<8} {norm:.4f}")

    # ------------------------------------------------------------------
    # 4. Cosine similarity matrix
    # ------------------------------------------------------------------
    cos_sim = cosine_similarity(V)   # (5, 5)

    df_cos = pd.DataFrame(cos_sim, index=CONCEPTS, columns=CONCEPTS)
    print("\nCosine similarity matrix:")
    print(df_cos.round(4).to_string())

    output_dir.mkdir(parents=True, exist_ok=True)
    cos_csv = output_dir / "vector_cosine_matrix.csv"
    df_cos.to_csv(cos_csv)
    print(f"\nSaved → {cos_csv}")

    # ------------------------------------------------------------------
    # 5. PCA
    # ------------------------------------------------------------------
    n_components = min(3, len(CONCEPTS) - 1)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(V)    # (5, n_components)
    ev = pca.explained_variance_ratio_

    print(f"\nPCA explained variance ratios ({n_components} components):")
    for i, r in enumerate(ev):
        print(f"  PC{i+1}: {r*100:.2f}%")

    pca_rows = []
    for i, concept in enumerate(CONCEPTS):
        row: dict = {"concept": concept}
        for k in range(n_components):
            row[f"PC{k+1}"] = float(coords[i, k])
        for k, r in enumerate(ev):
            row[f"explained_var_ratio_pc{k+1}"] = float(r)
        pca_rows.append(row)

    df_pca = pd.DataFrame(pca_rows)
    pca_csv = output_dir / "vector_pca.csv"
    df_pca.to_csv(pca_csv, index=False)
    print(f"Saved → {pca_csv}")

    # ------------------------------------------------------------------
    # 6. Figure
    # ------------------------------------------------------------------
    make_figure(CONCEPTS, cos_sim, coords[:, :2], ev[:2], output_dir)

    # ------------------------------------------------------------------
    # 7. Textual report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONCEPT GEOMETRY REPORT")
    print("=" * 60)

    # highest cosine similarity (off-diagonal)
    cos_off = cos_sim.copy()
    np.fill_diagonal(cos_off, -np.inf)
    i, j = np.unravel_index(cos_off.argmax(), cos_off.shape)
    print(f"Most similar pair  : {CONCEPTS[i]} & {CONCEPTS[j]}  "
          f"(cosine = {cos_sim[i, j]:.4f})")

    # most orthogonal (closest to 0)
    cos_abs = np.abs(cos_off)
    np.fill_diagonal(cos_abs, np.inf)
    pi, pj = np.unravel_index(cos_abs.argmin(), cos_abs.shape)
    print(f"Most orthogonal pair: {CONCEPTS[pi]} & {CONCEPTS[pj]}  "
          f"(cosine = {cos_sim[pi, pj]:.4f})")

    # smallest norm
    weakest = CONCEPTS[int(norms.argmin())]
    print(f"Smallest L2 norm   : {weakest}  (norm = {norms.min():.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
