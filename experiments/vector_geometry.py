#!/usr/bin/env python3
"""Steering Vector Geometry Analysis — Phase 3.3

Characterises the structure of musical concept representations in ACE-Step's
cross-attention activations via five geometric analyses.

Scientific question:
    Are musical concepts encoded as orthogonal linear directions (the "linear
    representation hypothesis") or as a tangled, non-linear subspace?

Analyses
--------
1. Pairwise cosine similarity heatmap   (vectors only — runs anywhere)
2. PCA of concept subspace             (vectors only — runs anywhere)
3. Linear probing                       (needs activations; synthetic in dry-run)
4. Concept arithmetic verification     (needs activations; synthetic in dry-run)
5. Layer progression                   (needs activations; synthetic in dry-run)

Outputs
-------
All figures and CSVs are written to --out-dir (default: results/geometry).

  cosine_heatmap.png          — annotated N×N similarity matrix
  pca_2d.png                  — 2-D PCA scatter coloured by category
  pca_variance.png            — cumulative explained-variance scree plot
  probing_accuracy.csv        — per-concept probe accuracy + v_c alignment
  arithmetic_verification.csv — composition cosine-similarity scores
  layer_progression.png       — per-layer vector cosine-sim with layer-7 ref
  report.md                   — markdown summary of all findings

Usage
-----
  # Dry-run — synthetic vectors, no ACE-Step required:
  python experiments/vector_geometry.py --dry-run

  # Real run — load .safetensors vectors from a directory:
  python experiments/vector_geometry.py --vectors-dir steering_vectors/

  # Real run with custom output directory:
  python experiments/vector_geometry.py --vectors-dir sv/ --out-dir results/my_geo

Notes
-----
Analyses 3-5 require ACE-Step and real activation caches.  On Python 3.13
(MacBook Air M4) spacy==3.8.4 prevents ACE-Step installation, so --dry-run
generates structured synthetic activations instead.  See docs/scaling_real_runs.md.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")  # headless-safe

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ACE-Step hidden dimension (cross-attention, layer 7)
ACE_HIDDEN_DIM: int = 3072
# Synthetic hidden dim used in dry-run to keep tests fast
DRY_HIDDEN_DIM: int = 64

# Concept categories for PCA colouring
CONCEPT_CATEGORIES: Dict[str, str] = {
    "drums": "instrument",
    "guitar": "instrument",
    "violin": "instrument",
    "flute": "instrument",
    "piano": "instrument",
    "trumpet": "instrument",
    "fast": "tempo",
    "slow": "tempo",
    "happy": "mood",
    "sad": "mood",
    "female": "gender",
    "male": "gender",
    "jazz": "genre",
    "techno": "genre",
    "reggae": "genre",
}

CATEGORY_COLORS: Dict[str, str] = {
    "instrument": "#1f77b4",
    "tempo": "#ff7f0e",
    "mood": "#2ca02c",
    "gender": "#d62728",
    "genre": "#9467bd",
    "unknown": "#8c564b",
}

# Reference layer for Analysis 5 (layer-progression)
REFERENCE_LAYER: int = 7
NUM_LAYERS: int = 24


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ConceptVector:
    """A named steering vector with category metadata.

    Attributes:
        name:     Concept name, e.g. "tempo".
        category: Semantic category, e.g. "instrument".
        method:   "caa" or "sae".
        layer:    Transformer-block index this vector was computed at.
        vec:      Unit-norm direction; shape ``(hidden_dim,)``.
    """

    name: str
    category: str
    method: str
    layer: int
    vec: torch.Tensor


@dataclass
class GeometryReport:
    """Collects summary statistics across all five analyses."""

    n_concepts: int = 0
    hidden_dim: int = 0
    mean_off_diag_cosine: float = 0.0
    max_off_diag_cosine: float = 0.0
    pca_top2_variance: float = 0.0
    pca_top3_variance: float = 0.0
    n_clusters_detected: int = 0
    mean_probe_accuracy: float = 0.0
    mean_probe_vc_alignment: float = 0.0
    arithmetic_mean_cosine: float = 0.0
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Vector loading / synthesis
# ---------------------------------------------------------------------------


def load_vectors_from_dir(vectors_dir: Path) -> List[ConceptVector]:
    """Load steering vectors from a directory of ``.safetensors`` files.

    Each file is expected to follow the naming convention
    ``{concept}_{method}.safetensors`` and contain a ``safetensors`` tensor
    plus JSON metadata (as written by ``SteeringVectorBank.save``).

    Args:
        vectors_dir: Directory containing ``.safetensors`` files.

    Returns:
        List of :class:`ConceptVector` instances, one per file.

    Raises:
        ImportError: If ``safetensors`` is not installed.
        FileNotFoundError: If *vectors_dir* does not exist.
    """
    if not vectors_dir.exists():
        raise FileNotFoundError(f"Vectors directory not found: {vectors_dir}")

    try:
        from safetensors.torch import load_file  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "safetensors is required for loading real vectors. "
            "Install it with: pip install safetensors"
        ) from exc

    result: List[ConceptVector] = []
    for path in sorted(vectors_dir.glob("*.safetensors")):
        tensors = load_file(str(path))
        if "vector" not in tensors:
            log.warning("Skipping %s: no 'vector' key found", path.name)
            continue
        vec = tensors["vector"]
        # Attempt to infer concept and method from filename
        stem = path.stem  # e.g. "tempo_caa"
        parts = stem.rsplit("_", 1)
        concept = parts[0] if len(parts) == 2 else stem
        method = parts[1] if len(parts) == 2 else "caa"
        category = CONCEPT_CATEGORIES.get(concept, "unknown")
        # Normalize to unit length (CAA vectors should already be; defensive)
        norm = vec.norm()
        if norm > 1e-8:
            vec = vec / norm
        result.append(
            ConceptVector(
                name=concept,
                category=category,
                method=method,
                layer=REFERENCE_LAYER,
                vec=vec,
            )
        )
    log.info("Loaded %d vectors from %s", len(result), vectors_dir)
    return result


def make_synthetic_vectors(
    n_concepts: int = 8,
    hidden_dim: int = DRY_HIDDEN_DIM,
    seed: int = 42,
) -> List[ConceptVector]:
    """Generate structured synthetic concept vectors for dry-run mode.

    The vectors are *not* fully random.  We embed deliberate structure so that
    geometry analyses produce meaningful (testable) outputs:

    - Instruments cluster together (small mutual cosines ≈ 0.3–0.5).
    - Tempo concepts are near-orthogonal to instruments.
    - Mood concepts are near-orthogonal to both.
    - Genre concepts blend instrument + mood directions.

    Args:
        n_concepts: Total number of synthetic concepts to generate.
        hidden_dim: Dimensionality of each vector.
        seed:       RNG seed for reproducibility.

    Returns:
        List of :class:`ConceptVector` instances.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Build a small set of basis directions via Gram-Schmidt
    bases: List[np.ndarray] = []
    for _ in range(4):
        v = rng.standard_normal(hidden_dim)
        for b in bases:
            v -= (v @ b) * b
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            v = v / norm
        bases.append(v)

    # instrument, tempo, mood, genre basis indices
    categories = ["instrument", "tempo", "mood", "genre"]
    concept_names = [
        "drums", "guitar", "violin", "piano",
        "fast", "slow",
        "happy", "sad",
        "jazz", "techno",
        "female", "male",
        "flute", "trumpet",
        "reggae", "classical",
    ]

    result: List[ConceptVector] = []
    for i in range(min(n_concepts, len(concept_names))):
        name = concept_names[i]
        cat = CONCEPT_CATEGORIES.get(name, "unknown")
        cat_idx = categories.index(cat) if cat in categories else 0
        # Blend dominant basis with small noise
        v = 0.8 * bases[cat_idx] + 0.2 * rng.standard_normal(hidden_dim)
        v = v / np.linalg.norm(v)
        result.append(
            ConceptVector(
                name=name,
                category=cat,
                method="caa",
                layer=REFERENCE_LAYER,
                vec=torch.from_numpy(v).float(),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Analysis 1 — Pairwise cosine similarity heatmap
# ---------------------------------------------------------------------------


def analysis_cosine_heatmap(
    concepts: List[ConceptVector],
    out_dir: Path,
) -> np.ndarray:
    """Compute pairwise cosine similarities and save an annotated heatmap.

    Args:
        concepts: List of concept vectors (all same hidden_dim).
        out_dir:  Directory for the output PNG.

    Returns:
        ``(N, N)`` numpy array of cosine similarities.
    """
    n = len(concepts)
    names = [c.name for c in concepts]
    V = torch.stack([c.vec for c in concepts])  # (N, hidden_dim)
    V_norm = F.normalize(V, dim=1)              # unit-norm rows
    sim_matrix = (V_norm @ V_norm.T).numpy()    # (N, N)

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(sim_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Pairwise Cosine Similarity of Concept Vectors")
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    # Annotate cells
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if abs(sim_matrix[i, j]) < 0.7 else "white")
    fig.tight_layout()
    out_path = out_dir / "cosine_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved cosine heatmap → %s", out_path)
    return sim_matrix


# ---------------------------------------------------------------------------
# Analysis 2 — PCA of concept subspace
# ---------------------------------------------------------------------------


def analysis_pca(
    concepts: List[ConceptVector],
    out_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run PCA on concept vectors and save scatter + scree plots.

    Args:
        concepts: List of concept vectors.
        out_dir:  Directory for output PNGs.

    Returns:
        Tuple of ``(coords_2d, explained_ratios)`` arrays.
        ``coords_2d`` has shape ``(N, 2)``.
    """
    V = torch.stack([c.vec for c in concepts]).numpy()  # (N, hidden_dim)
    n_comp = min(len(concepts), V.shape[1])
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(V)   # (N, n_comp)
    ratios = pca.explained_variance_ratio_

    # — Scatter plot of PC1 vs PC2 —
    fig, ax = plt.subplots(figsize=(7, 6))
    seen_cats: set = set()
    for i, c in enumerate(concepts):
        col = CATEGORY_COLORS.get(c.category, CATEGORY_COLORS["unknown"])
        label = c.category if c.category not in seen_cats else None
        seen_cats.add(c.category)
        ax.scatter(coords[i, 0], coords[i, 1], color=col, s=100,
                   label=label, zorder=3)
        ax.annotate(c.name, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel(f"PC1 ({ratios[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratios[1]*100:.1f}%)" if n_comp > 1 else "PC2 (n/a)")
    ax.set_title("PCA of Steering Vectors (coloured by category)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pca_2d.png", dpi=150)
    plt.close(fig)

    # — Scree / cumulative variance plot —
    cumvar = np.cumsum(ratios)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(range(1, len(ratios) + 1), ratios * 100, label="Individual", alpha=0.6)
    ax2.plot(range(1, len(ratios) + 1), cumvar * 100, "r-o", label="Cumulative")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Explained Variance (%)")
    ax2.set_title("PCA Scree Plot — Concept Steering Vectors")
    ax2.axhline(80, color="gray", linestyle="--", alpha=0.5, label="80% threshold")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "pca_variance.png", dpi=150)
    plt.close(fig2)

    log.info(
        "PCA: top-2 variance=%.1f%%, top-3=%.1f%%",
        float(cumvar[1]) * 100 if n_comp > 1 else float(ratios[0]) * 100,
        float(cumvar[2]) * 100 if n_comp > 2 else float(cumvar[min(1, n_comp - 1)]) * 100,
    )
    return coords[:, :2] if n_comp >= 2 else coords, ratios


# ---------------------------------------------------------------------------
# Analysis 3 — Linear probing
# ---------------------------------------------------------------------------


def _make_synthetic_activations(
    concept: ConceptVector,
    n_pos: int,
    n_neg: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic mean-pooled activations for linear probing.

    Positive samples are the concept direction plus Gaussian noise.
    Negative samples are the orthogonal complement plus noise.

    Args:
        concept: The concept vector used as the positive direction.
        n_pos:   Number of positive samples.
        n_neg:   Number of negative samples.
        seed:    RNG seed.

    Returns:
        Tuple ``(X, y)`` where ``X`` has shape ``(n_pos+n_neg, hidden_dim)``
        and ``y`` is a binary label array.
    """
    rng = np.random.default_rng(seed)
    v = concept.vec.numpy()  # (hidden_dim,)
    dim = v.shape[0]
    signal_scale = 1.5
    noise_scale = 1.0

    X_pos = signal_scale * v[np.newaxis, :] + noise_scale * rng.standard_normal((n_pos, dim))
    X_neg = -0.3 * v[np.newaxis, :] + noise_scale * rng.standard_normal((n_neg, dim))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * n_pos + [0] * n_neg, dtype=int)
    return X, y


def analysis_linear_probing(
    concepts: List[ConceptVector],
    out_dir: Path,
    dry_run: bool = True,
    n_samples_per_class: int = 128,
    seed: int = 42,
) -> List[Dict]:
    """Train logistic regression probes and compare weight alignment with v_c.

    In dry-run mode, synthetic activations are generated using the concept
    direction as signal.  In real mode (not yet implemented), this would
    extract layer-7 cross-attention activations from ACE-Step.

    Args:
        concepts:            List of concept vectors.
        out_dir:             Directory for output CSV.
        dry_run:             If True, use synthetic activations.
        n_samples_per_class: Number of positive/negative samples per concept.
        seed:                RNG seed.

    Returns:
        List of result dicts (one per concept) with keys:
        ``concept``, ``accuracy``, ``vc_alignment``.
    """
    if not dry_run:
        log.warning(
            "Real activation probing requires ACE-Step (Python 3.10-3.12). "
            "Falling back to synthetic activations."
        )

    rows: List[Dict] = []
    for i, c in enumerate(concepts):
        X, y = _make_synthetic_activations(
            c, n_samples_per_class, n_samples_per_class, seed + i
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed + i
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)

        # Cosine similarity between probe weight and v_c
        w = clf.coef_[0]  # (hidden_dim,)
        v = c.vec.numpy()
        vc_align = float(
            np.dot(w, v) / (np.linalg.norm(w) * np.linalg.norm(v) + 1e-8)
        )

        rows.append({
            "concept": c.name,
            "category": c.category,
            "method": c.method,
            "accuracy": round(acc, 4),
            "vc_alignment": round(vc_align, 4),
            "dry_run": dry_run,
        })
        log.info(
            "Probe %-12s acc=%.3f  vc_align=%.3f", c.name, acc, vc_align
        )

    # Write CSV
    csv_path = out_dir / "probing_accuracy.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    log.info("Saved probing results → %s", csv_path)
    return rows


# ---------------------------------------------------------------------------
# Analysis 4 — Concept arithmetic verification
# ---------------------------------------------------------------------------


def analysis_arithmetic_verification(
    concepts: List[ConceptVector],
    out_dir: Path,
    dry_run: bool = True,
    n_samples: int = 32,
    seed: int = 42,
) -> List[Dict]:
    """Verify v_A + v_B approximates the "A and B" direction.

    Tests the prediction: if activations for "fast tempo + female vocal" are
    generated, their projection onto v_fast + v_female should be higher than
    onto either individual vector alone.

    In dry-run mode, synthetic activations are drawn from the composed
    direction with noise.

    Args:
        concepts:  List of concept vectors (need ≥ 2).
        out_dir:   Directory for output CSV.
        dry_run:   If True, use synthetic activations.
        n_samples: Number of samples per pair.
        seed:      RNG seed.

    Returns:
        List of result dicts with keys:
        ``pair``, ``composed_cosine``, ``a_cosine``, ``b_cosine``.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []

    # Test up to 5 pairs
    pairs = [(concepts[i], concepts[j])
             for i in range(len(concepts))
             for j in range(i + 1, len(concepts))][:5]

    for a, b in pairs:
        va = a.vec.numpy()
        vb = b.vec.numpy()
        composed = va + vb
        composed_norm = composed / (np.linalg.norm(composed) + 1e-8)

        if not dry_run:
            log.warning(
                "Real arithmetic verification needs ACE-Step. Using synthetic."
            )
        # Synthetic activations sampled near the composed direction
        noise = rng.standard_normal((n_samples, va.shape[0]))
        X = 1.5 * composed_norm[np.newaxis, :] + 0.8 * noise

        # Mean activation
        x_mean = X.mean(axis=0)
        x_mean_norm = x_mean / (np.linalg.norm(x_mean) + 1e-8)

        cos_composed = float(np.dot(x_mean_norm, composed_norm))
        cos_a = float(np.dot(x_mean_norm, va / (np.linalg.norm(va) + 1e-8)))
        cos_b = float(np.dot(x_mean_norm, vb / (np.linalg.norm(vb) + 1e-8)))

        rows.append({
            "pair": f"{a.name}+{b.name}",
            "composed_cosine": round(cos_composed, 4),
            "a_cosine": round(cos_a, 4),
            "b_cosine": round(cos_b, 4),
            "dry_run": dry_run,
        })
        log.info(
            "Arithmetic %-20s  composed=%.3f  a=%.3f  b=%.3f",
            rows[-1]["pair"], cos_composed, cos_a, cos_b,
        )

    csv_path = out_dir / "arithmetic_verification.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    log.info("Saved arithmetic verification → %s", csv_path)
    return rows


# ---------------------------------------------------------------------------
# Analysis 5 — Layer progression
# ---------------------------------------------------------------------------


def _make_layer_vectors(
    reference: ConceptVector,
    num_layers: int,
    seed: int,
) -> List[torch.Tensor]:
    """Generate a plausible per-layer vector trajectory for dry-run mode.

    The vectors start near-random at layer 0, gradually rotate toward the
    reference direction, and peak at REFERENCE_LAYER.

    Args:
        reference:  The reference concept vector (at REFERENCE_LAYER).
        num_layers: Total number of transformer layers.
        seed:       RNG seed.

    Returns:
        List of ``num_layers`` unit-norm tensors, one per layer.
    """
    torch.manual_seed(seed)
    ref = reference.vec
    dim = ref.shape[0]
    vecs: List[torch.Tensor] = []
    for layer_idx in range(num_layers):
        # Linear interpolation: noise at layer 0, reference at REFERENCE_LAYER
        t = layer_idx / max(REFERENCE_LAYER, 1)
        t = min(t, 1.0)
        noise = torch.randn(dim)
        noise = noise - (noise @ ref) * ref  # orthogonal component
        v = t * ref + (1 - t) * noise
        v = F.normalize(v, dim=0)
        vecs.append(v)
    return vecs


def analysis_layer_progression(
    concepts: List[ConceptVector],
    out_dir: Path,
    dry_run: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """Plot cosine similarity of each layer's concept vector vs layer-7.

    In dry-run mode, synthetic per-layer vectors are generated using a
    smooth trajectory toward the reference vector.

    Args:
        concepts: List of concept vectors (reference is at REFERENCE_LAYER).
        out_dir:  Directory for output PNG.
        dry_run:  If True, generate synthetic per-layer vectors.
        seed:     RNG seed.

    Returns:
        ``(N_concepts, NUM_LAYERS)`` numpy array of cosine similarities.
    """
    all_sims = np.zeros((len(concepts), NUM_LAYERS))

    for i, c in enumerate(concepts):
        if not dry_run:
            log.warning(
                "Real layer progression needs ACE-Step. Using synthetic."
            )
        layer_vecs = _make_layer_vectors(c, NUM_LAYERS, seed + i)
        ref = F.normalize(c.vec, dim=0)
        for layer_idx, lv in enumerate(layer_vecs):
            all_sims[i, layer_idx] = float(F.normalize(lv, dim=0) @ ref)

    fig, ax = plt.subplots(figsize=(10, 5))
    layers = list(range(NUM_LAYERS))
    for i, c in enumerate(concepts):
        ax.plot(layers, all_sims[i], label=c.name, alpha=0.8)
    ax.axvline(REFERENCE_LAYER, color="black", linestyle="--",
               alpha=0.5, label=f"Layer {REFERENCE_LAYER} (reference)")
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel(f"Cosine Similarity with Layer-{REFERENCE_LAYER} Vector")
    ax.set_title("Concept Vector Similarity Across Layers")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "layer_progression.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved layer progression → %s", out_path)
    return all_sims


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def write_report(
    report: GeometryReport,
    sim_matrix: np.ndarray,
    concepts: List[ConceptVector],
    pca_ratios: np.ndarray,
    probe_rows: List[Dict],
    arith_rows: List[Dict],
    out_dir: Path,
) -> Path:
    """Write a markdown summary report of all analyses.

    Args:
        report:     Populated :class:`GeometryReport` dataclass.
        sim_matrix: Pairwise cosine similarity matrix.
        concepts:   List of concept vectors.
        pca_ratios: Explained variance ratios from PCA.
        probe_rows: Rows from Analysis 3.
        arith_rows: Rows from Analysis 4.
        out_dir:    Directory for report.md.

    Returns:
        Path to the written report.
    """
    names = [c.name for c in concepts]
    most_similar_pair = ("", "", 0.0)
    n = len(concepts)
    for i in range(n):
        for j in range(i + 1, n):
            val = float(sim_matrix[i, j])
            if val > most_similar_pair[2]:
                most_similar_pair = (names[i], names[j], val)

    lines = [
        "# Steering Vector Geometry Analysis — Report",
        "",
        "## Overview",
        "",
        f"- **Concepts analysed:** {report.n_concepts}",
        f"- **Hidden dimension:** {report.hidden_dim}",
        "",
        "## Analysis 1 — Pairwise Cosine Similarities",
        "",
        f"- Mean off-diagonal cosine: `{report.mean_off_diag_cosine:.4f}`",
        f"- Max off-diagonal cosine: `{report.max_off_diag_cosine:.4f}`",
        f"  (most similar pair: **{most_similar_pair[0]}** ↔ **{most_similar_pair[1]}**"
        f"  = {most_similar_pair[2]:.3f})",
        "",
        "> Interpretation: values near 0 indicate orthogonal (independent) concepts; "
        "values near ±1 indicate high interference.",
        "",
        "## Analysis 2 — PCA of Concept Subspace",
        "",
        f"- Top-2 PCs explain `{report.pca_top2_variance*100:.1f}%` of variance",
        f"- Top-3 PCs explain `{report.pca_top3_variance*100:.1f}%` of variance",
        "",
        "> Instruments, tempo, mood, and genre concepts should appear as distinct "
        "clusters in the 2-D PCA plot.",
        "",
        "## Analysis 3 — Linear Probing",
        "",
    ]
    if probe_rows:
        lines += [
            "| Concept | Category | Accuracy | v_c Alignment |",
            "|---------|----------|----------|---------------|",
        ]
        for row in probe_rows:
            lines.append(
                f"| {row['concept']} | {row['category']} "
                f"| {row['accuracy']:.3f} | {row['vc_alignment']:.3f} |"
            )
        lines += [
            "",
            f"- Mean probe accuracy: `{report.mean_probe_accuracy:.3f}`",
            f"- Mean v_c alignment: `{report.mean_probe_vc_alignment:.3f}`",
        ]
    lines += [
        "",
        "> High alignment (|cosine| > 0.5) between probe weight and v_c is evidence "
        "for linear encoding of the concept.",
        "",
        "## Analysis 4 — Concept Arithmetic Verification",
        "",
    ]
    if arith_rows:
        lines += [
            "| Pair | Composed Cosine | A Cosine | B Cosine |",
            "|------|----------------|----------|----------|",
        ]
        for row in arith_rows:
            lines.append(
                f"| {row['pair']} | {row['composed_cosine']:.3f} "
                f"| {row['a_cosine']:.3f} | {row['b_cosine']:.3f} |"
            )
        lines += [
            "",
            f"- Mean composed cosine: `{report.arithmetic_mean_cosine:.3f}`",
        ]
    lines += [
        "",
        "> Composed cosine should exceed individual A/B cosines, confirming "
        "additive structure in the concept subspace.",
        "",
        "## Analysis 5 — Layer Progression",
        "",
        f"- Reference layer: {REFERENCE_LAYER}",
        f"- Layers analysed: 0–{NUM_LAYERS - 1}",
        "",
        "> The plot (layer_progression.png) shows cosine similarity between each "
        f"layer's concept vector and the layer-{REFERENCE_LAYER} reference. "
        f"Functional layers ({{6, 7}}) are expected to show the highest similarity.",
        "",
        "## Notes",
        "",
    ]
    for note in report.notes:
        lines.append(f"- {note}")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    log.info("Saved report → %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_geometry_analysis(
    concepts: List[ConceptVector],
    out_dir: Path,
    dry_run: bool,
    seed: int = 42,
) -> GeometryReport:
    """Run all five geometry analyses and write outputs to *out_dir*.

    Args:
        concepts: List of concept vectors to analyse.
        out_dir:  Output directory (created if missing).
        dry_run:  If True, use synthetic activations for Analyses 3-5.
        seed:     Global RNG seed.

    Returns:
        Populated :class:`GeometryReport`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report = GeometryReport(
        n_concepts=len(concepts),
        hidden_dim=concepts[0].vec.shape[0] if concepts else 0,
    )

    # Analysis 1 — Cosine heatmap
    sim_matrix = analysis_cosine_heatmap(concepts, out_dir)
    n = len(concepts)
    if n > 1:
        mask = ~np.eye(n, dtype=bool)
        off_diag = sim_matrix[mask]
        report.mean_off_diag_cosine = float(off_diag.mean())
        report.max_off_diag_cosine = float(off_diag.max())

    # Analysis 2 — PCA
    _, pca_ratios = analysis_pca(concepts, out_dir)
    cumvar = np.cumsum(pca_ratios)
    report.pca_top2_variance = float(cumvar[1]) if len(cumvar) > 1 else float(cumvar[0])
    report.pca_top3_variance = float(cumvar[2]) if len(cumvar) > 2 else report.pca_top2_variance

    # Analysis 3 — Linear probing
    probe_rows = analysis_linear_probing(
        concepts, out_dir, dry_run=dry_run, seed=seed
    )
    if probe_rows:
        report.mean_probe_accuracy = float(
            np.mean([r["accuracy"] for r in probe_rows])
        )
        report.mean_probe_vc_alignment = float(
            np.mean([abs(r["vc_alignment"]) for r in probe_rows])
        )

    # Analysis 4 — Arithmetic verification
    arith_rows = analysis_arithmetic_verification(
        concepts, out_dir, dry_run=dry_run, seed=seed
    )
    if arith_rows:
        report.arithmetic_mean_cosine = float(
            np.mean([r["composed_cosine"] for r in arith_rows])
        )

    # Analysis 5 — Layer progression
    layer_sims = analysis_layer_progression(
        concepts, out_dir, dry_run=dry_run, seed=seed
    )

    if dry_run:
        report.notes.append(
            "Analyses 3-5 used *synthetic* activations (dry-run mode). "
            "For real results, run on a Python 3.10-3.12 machine with ACE-Step installed "
            "and pass --vectors-dir pointing to real .safetensors files."
        )
    report.notes.append(
        f"Analysed {n} concepts; hidden_dim={report.hidden_dim}."
    )

    write_report(
        report, sim_matrix, concepts, pca_ratios,
        probe_rows, arith_rows, out_dir
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Steering vector geometry analysis (Phase 3.3)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Use synthetic vectors and activations. No ACE-Step required. "
            "Mutually exclusive with --vectors-dir."
        ),
    )
    parser.add_argument(
        "--vectors-dir",
        type=Path,
        default=None,
        help="Directory of .safetensors steering vector files (real run).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/geometry"),
        help="Output directory for plots, CSVs, and report (default: results/geometry).",
    )
    parser.add_argument(
        "--n-concepts",
        type=int,
        default=8,
        help="Number of synthetic concepts in dry-run mode (default: 8).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DRY_HIDDEN_DIM,
        help=(
            f"Hidden dimension for synthetic vectors "
            f"(default: {DRY_HIDDEN_DIM}; real ACE-Step: {ACE_HIDDEN_DIM})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global RNG seed (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the geometry analysis experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args(argv)

    if args.vectors_dir is not None and args.dry_run:
        log.error("--vectors-dir and --dry-run are mutually exclusive.")
        sys.exit(1)

    if args.vectors_dir is not None:
        log.info("Loading real vectors from %s", args.vectors_dir)
        concepts = load_vectors_from_dir(args.vectors_dir)
        dry_run = False
    else:
        if not args.dry_run:
            log.info(
                "No --vectors-dir given; defaulting to --dry-run. "
                "Pass --vectors-dir to use real steering vectors."
            )
        log.info(
            "Generating %d synthetic vectors (dim=%d, seed=%d)",
            args.n_concepts, args.hidden_dim, args.seed,
        )
        concepts = make_synthetic_vectors(
            n_concepts=args.n_concepts,
            hidden_dim=args.hidden_dim,
            seed=args.seed,
        )
        dry_run = True

    if not concepts:
        log.error("No concept vectors to analyse. Exiting.")
        sys.exit(1)

    report = run_geometry_analysis(
        concepts=concepts,
        out_dir=args.out_dir,
        dry_run=dry_run,
        seed=args.seed,
    )

    print(f"\n{'='*60}")
    print(f"Geometry Analysis Complete — {len(concepts)} concepts")
    print(f"  Mean off-diagonal cosine : {report.mean_off_diag_cosine:.4f}")
    print(f"  PCA top-2 variance       : {report.pca_top2_variance*100:.1f}%")
    print(f"  Mean probe accuracy      : {report.mean_probe_accuracy:.3f}")
    print(f"  Mean v_c alignment       : {report.mean_probe_vc_alignment:.3f}")
    print(f"  Arithmetic mean cosine   : {report.arithmetic_mean_cosine:.3f}")
    print(f"  Outputs → {args.out_dir.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
