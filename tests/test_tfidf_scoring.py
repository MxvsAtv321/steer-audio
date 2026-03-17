"""
Unit tests for SAE feature TF-IDF scoring.

Tests the ``compute_tfidf_scores`` and ``top_tau_features`` utilities
from ``sae.tfidf_utils``.
"""

import sys
from pathlib import Path

import torch
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAE_ROOT = _REPO_ROOT / "sae"
for _p in [str(_REPO_ROOT), str(_SAE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tfidf_utils import compute_tfidf_scores, top_tau_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_activations(n_samples: int, n_features: int, seed: int) -> torch.Tensor:
    """Return non-negative activations (ReLU output) of shape (n_samples, n_features)."""
    torch.manual_seed(seed)
    return torch.relu(torch.randn(n_samples, n_features))


# ---------------------------------------------------------------------------
# 1. TF-IDF scores are always non-negative
# ---------------------------------------------------------------------------


def test_tfidf_score_positive():
    """compute_tfidf_scores returns non-negative values for any valid input."""
    pos = _make_activations(n_samples=16, n_features=32, seed=0)
    neg = _make_activations(n_samples=16, n_features=32, seed=1)

    scores = compute_tfidf_scores(pos, neg)

    assert (scores >= 0).all(), (
        f"All TF-IDF scores should be non-negative; "
        f"got min={scores.min().item():.4f}"
    )


# ---------------------------------------------------------------------------
# 2. Exclusive features score higher than shared features
# ---------------------------------------------------------------------------


def test_tfidf_higher_for_exclusive_features():
    """A feature that only activates for positive prompts scores higher
    than one that activates equally for positive and negative prompts.

    'Exclusive' feature: high pos activation, zero neg activation.
    'Shared'   feature: equal pos and neg activation.
    """
    n_samples = 32
    n_features = 2  # feature 0 = exclusive, feature 1 = shared

    # Exclusive: pos=1.0, neg=0.0
    pos = torch.zeros(n_samples, n_features)
    neg = torch.zeros(n_samples, n_features)
    pos[:, 0] = 1.0   # exclusive feature
    pos[:, 1] = 1.0   # shared feature
    neg[:, 1] = 1.0   # also active in negative class

    scores = compute_tfidf_scores(pos, neg)

    assert scores[0] > scores[1], (
        f"Exclusive feature (score={scores[0].item():.4f}) should score higher "
        f"than shared feature (score={scores[1].item():.4f})."
    )


# ---------------------------------------------------------------------------
# 3. top_tau_features returns the highest-scored indices
# ---------------------------------------------------------------------------


def test_top_tau_features_are_highest_scored():
    """top_tau_features(scores, tau) returns indices matching argsort(scores)[-tau:]."""
    torch.manual_seed(42)
    scores = torch.rand(100)
    tau = 10

    returned_indices = top_tau_features(scores, tau)

    # Ground truth: top tau indices by score value
    expected_indices = set(scores.argsort(descending=True)[:tau].tolist())
    returned_set = set(returned_indices.tolist())

    assert returned_set == expected_indices, (
        f"top_tau_features returned wrong indices.\n"
        f"Expected: {sorted(expected_indices)}\n"
        f"Got:      {sorted(returned_set)}"
    )
