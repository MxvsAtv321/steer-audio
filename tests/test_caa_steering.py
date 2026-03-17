"""
Unit tests for CAA (Contrastive Activation Addition) steering utilities.

Tests the mathematical properties of:
  - compute_sv: unit-norm steering vector from positive/negative activations
  - renorm: magnitude-preserving renormalization
  - apply_caa_steering: full CAA steering step

Reference: arXiv 2602.11910 — Section 3.2.
"""

import sys
from pathlib import Path

import torch
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from steering.caa_utils import compute_sv, renorm, apply_caa_steering


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_activations(n: int, dim: int, seed: int) -> list:
    """Create a list of n random activation tensors of shape (dim,)."""
    torch.manual_seed(seed)
    return [torch.randn(dim) for _ in range(n)]


# ---------------------------------------------------------------------------
# 1. Computed steering vector has unit L2 norm
# ---------------------------------------------------------------------------


def test_caa_vector_unit_norm():
    """compute_sv returns a unit-norm vector (||v_c||₂ = 1.0 within 1e-6)."""
    dim = 64
    pos = _make_activations(8, dim, seed=0)
    neg = _make_activations(8, dim, seed=1)

    v_c = compute_sv(pos, neg)

    norm = v_c.norm().item()
    assert abs(norm - 1.0) < 1e-6, (
        f"Expected unit-norm steering vector, got ||v_c||₂ = {norm:.8f}"
    )


# ---------------------------------------------------------------------------
# 2. ReNorm preserves original magnitude
# ---------------------------------------------------------------------------


def test_renorm_preserves_magnitude():
    """||ReNorm(h', h)||₂ == ||h||₂ (within 1e-6) for each token."""
    torch.manual_seed(42)
    h = torch.randn(4, 16, 64)   # (batch, seq, dim)
    h_steered = h + 5.0           # arbitrary perturbation

    h_renormed = renorm(h_steered, h)

    orig_norms = h.float().norm(dim=-1)
    renormed_norms = h_renormed.float().norm(dim=-1)

    assert torch.allclose(orig_norms, renormed_norms, atol=1e-6), (
        "ReNorm should preserve per-token magnitude; "
        f"max deviation: {(orig_norms - renormed_norms).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# 3. Steering changes the activation when alpha != 0
# ---------------------------------------------------------------------------


def test_steering_changes_activation():
    """apply_caa_steering with alpha != 0 produces a different tensor."""
    torch.manual_seed(42)
    h = torch.randn(2, 16, 64)
    v_c = torch.randn(64)
    v_c = v_c / v_c.norm()  # unit norm

    h_steered = apply_caa_steering(h, v_c, alpha=50.0)

    assert not torch.allclose(h.float(), h_steered, atol=1e-5), (
        "Steered activation should differ from original when alpha != 0."
    )


# ---------------------------------------------------------------------------
# 4. Steering is identity when alpha = 0
# ---------------------------------------------------------------------------


def test_steering_identity_at_zero_alpha():
    """apply_caa_steering with alpha=0 returns a tensor equal to the input."""
    torch.manual_seed(42)
    h = torch.randn(2, 16, 64)
    v_c = torch.randn(64)
    v_c = v_c / v_c.norm()

    h_steered = apply_caa_steering(h, v_c, alpha=0.0)

    assert torch.allclose(h.float(), h_steered, atol=1e-6), (
        "Steering with alpha=0 should be a no-op (identity)."
    )


# ---------------------------------------------------------------------------
# 5. Multi-concept steering output shape matches input
# ---------------------------------------------------------------------------


def test_multi_concept_steering_shape():
    """Applying multiple steering vectors sequentially preserves activation shape."""
    torch.manual_seed(42)
    dim = 64
    h = torch.randn(2, 16, dim)

    # Two independent steering vectors (e.g., tempo + mood)
    concepts = {
        "tempo": (torch.randn(dim) / torch.randn(dim).norm(), 50.0),
        "mood": (torch.randn(dim) / torch.randn(dim).norm(), 30.0),
    }

    h_out = h.float()
    for name, (v_c, alpha) in concepts.items():
        v_c = v_c / v_c.norm()  # ensure unit norm
        h_out = apply_caa_steering(h_out, v_c, alpha)

    assert h_out.shape == h.shape, (
        f"Multi-concept steering changed activation shape: "
        f"expected {h.shape}, got {h_out.shape}"
    )
