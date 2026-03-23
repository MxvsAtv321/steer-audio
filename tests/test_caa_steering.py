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


# ---------------------------------------------------------------------------
# 6. CAA vector manual computation: 5 pos / 5 neg activations, shape (16, 64)
# ---------------------------------------------------------------------------


def test_caa_vector_manual_computation():
    """compute_sv matches manual (mean_pos - mean_neg) / ||diff||₂.

    Uses 5 positive and 5 negative 2-D activation tensors of shape (16, 64)
    so that the input format mirrors real cross-attention activations.
    """
    seq_len, dim = 16, 64
    n = 5
    torch.manual_seed(7)

    pos = [torch.randn(seq_len, dim) for _ in range(n)]
    neg = [torch.randn(seq_len, dim) for _ in range(n)]

    # Manual computation
    pos_mean = torch.stack([v.float() for v in pos]).mean(dim=0)   # (16, 64)
    neg_mean = torch.stack([v.float() for v in neg]).mean(dim=0)   # (16, 64)
    diff = pos_mean - neg_mean
    v_manual = diff / diff.norm()

    v_c = compute_sv(pos, neg)

    assert v_c.shape == (seq_len, dim), (
        f"Expected shape ({seq_len}, {dim}), got {v_c.shape}"
    )
    assert torch.allclose(v_c, v_manual, atol=1e-6), (
        "compute_sv does not match manual (mean_pos - mean_neg) / ||diff||₂"
    )

    # Unit norm
    assert abs(v_c.norm().item() - 1.0) < 1e-6, (
        f"CAA vector should have unit norm; got {v_c.norm().item():.8f}"
    )


# ---------------------------------------------------------------------------
# 7. ReNorm preserves magnitude for specific alpha values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [-100.0, 0.0, 50.0, 100.0])
def test_renorm_preserves_magnitude_for_alpha(alpha):
    """||ReNorm(h + α·v, h)||₂ == ||h||₂ within 1e-5 for each token.

    Tests the ReNorm operation for α ∈ {-100, 0, 50, 100}.
    At α=0 the steered tensor equals h, so renorm is trivially correct;
    for other alphas we verify magnitude preservation.
    """
    torch.manual_seed(13)
    dim = 64
    h = torch.randn(4, 16, dim)
    v = torch.randn(dim)
    v = v / v.norm()

    h_perturbed = h.float() + alpha * v.float()
    h_renormed = renorm(h_perturbed, h)

    orig_norms = h.float().norm(dim=-1)     # (4, 16)
    renormed_norms = h_renormed.norm(dim=-1)  # (4, 16)

    assert torch.allclose(orig_norms, renormed_norms, atol=1e-5), (
        f"α={alpha}: ReNorm failed to preserve magnitude; "
        f"max deviation={( orig_norms - renormed_norms).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# 8. Steering injection: correct shape and norm for α=30
# ---------------------------------------------------------------------------


def test_steering_injection_shape_and_norm():
    """apply_caa_steering(h, v, α=30) returns correct shape and preserves norm."""
    torch.manual_seed(99)
    h = torch.randn(3, 16, 64)
    v = torch.randn(64)
    v = v / v.norm()

    h_steered = apply_caa_steering(h, v, alpha=30.0)

    # Shape must match
    assert h_steered.shape == h.shape, (
        f"Steered shape {h_steered.shape} != input shape {h.shape}"
    )

    # ReNorm: per-token norm is preserved
    orig_norms = h.float().norm(dim=-1)
    steered_norms = h_steered.norm(dim=-1)

    assert torch.allclose(orig_norms, steered_norms, atol=1e-5), (
        f"α=30: per-token norm not preserved after steering; "
        f"max deviation={( orig_norms - steered_norms).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# 9. Alpha sweep: α=0 gives exact unsteered output
# ---------------------------------------------------------------------------


def test_alpha_sweep_zero_is_identity():
    """For every α in {-100, 0, 50, 100}, α=0 returns an output equal to h."""
    torch.manual_seed(55)
    h = torch.randn(2, 16, 64)
    v = torch.randn(64)
    v = v / v.norm()

    h_zero = apply_caa_steering(h, v, alpha=0.0)

    assert torch.allclose(h.float(), h_zero, atol=1e-6), (
        "apply_caa_steering with α=0 should equal the unsteered input exactly."
    )
