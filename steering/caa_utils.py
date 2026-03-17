"""
Utility functions for Contrastive Activation Addition (CAA) steering.

Implements the core CAA formulas used in:
  h'_l = ReNorm(h_l + α·v_c, h_l)

where v_c is the unit-norm contrastive steering vector for concept c.

Reference: arXiv 2602.11910 — Section 3.2 (CAA Steering Vectors).
"""

from typing import Sequence

import torch
from torch import Tensor


# Epsilon to avoid division by zero in renorm
_RENORM_EPS: float = 1e-8


def compute_sv(
    pos_vectors: Sequence[Tensor],
    neg_vectors: Sequence[Tensor],
) -> Tensor:
    """Compute a unit-norm CAA steering vector from positive/negative activations.

    v_c = (mean(pos) - mean(neg)) / ||mean(pos) - mean(neg)||₂

    Args:
        pos_vectors: List of activation tensors from positive-concept prompts.
            Each tensor has the same shape.
        neg_vectors: List of activation tensors from negative-concept prompts.
            Same shape as elements of ``pos_vectors``.

    Returns:
        Unit-norm steering vector with the same shape as each input tensor.

    Raises:
        ValueError: If either list is empty or shapes do not match.
    """
    if not pos_vectors or not neg_vectors:
        raise ValueError("pos_vectors and neg_vectors must both be non-empty.")

    pos_stack = torch.stack([v.float() for v in pos_vectors])
    neg_stack = torch.stack([v.float() for v in neg_vectors])

    pos_avg = pos_stack.mean(dim=0)
    neg_avg = neg_stack.mean(dim=0)

    diff = pos_avg - neg_avg
    norm = diff.norm()
    if norm < _RENORM_EPS:
        raise ValueError(
            f"Steering vector norm is near zero ({norm:.2e}); "
            "positive and negative activations may be identical."
        )
    return diff / norm


def renorm(
    h_steered: Tensor,
    h_orig: Tensor,
    eps: float = _RENORM_EPS,
) -> Tensor:
    """Renormalize a steered activation to preserve the original magnitude.

    ReNorm(h', h) = h' / ||h'|| * ||h||   (per-token, last dim)

    This ensures the magnitude of each token's hidden state is unchanged after
    adding the steering vector, preventing energy amplification.

    Args:
        h_steered: Activation after adding the steering vector.
            Shape: (..., dim).
        h_orig: Original (unsteered) activation tensor.
            Same shape as ``h_steered``.
        eps: Small constant to avoid division by zero.

    Returns:
        Renormalized activation with same shape and L2 norm (per token) as
        ``h_orig``.
    """
    orig_norm = h_orig.float().norm(dim=-1, keepdim=True)
    steered_norm = h_steered.float().norm(dim=-1, keepdim=True)
    return (h_steered.float() / (steered_norm + eps)) * orig_norm


def apply_caa_steering(
    h: Tensor,
    v_c: Tensor,
    alpha: float,
) -> Tensor:
    """Apply a CAA steering vector to an activation tensor.

    h'_l = ReNorm(h_l + α·v_c, h_l)

    Args:
        h: Activation tensor to steer.  Shape: (..., dim).
        v_c: Unit-norm steering vector.  Shape: (dim,) or broadcastable.
        alpha: Steering strength.  Positive strengthens concept, negative suppresses.

    Returns:
        Steered activation with same shape as ``h``.
    """
    h_steered = h.float() + alpha * v_c.float()
    if alpha == 0.0:
        return h.float()
    return renorm(h_steered, h)
