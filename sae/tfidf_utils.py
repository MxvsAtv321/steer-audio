"""
TF-IDF feature scoring utilities for SAE concept analysis.

Computes concept-discriminative feature scores using a TF-IDF analogy:
  - TF (term frequency)  = mean activation on positive-concept prompts
  - IDF (inverse document freq) = log(1 + 1 / (mean_neg + ε))

Score = TF × IDF

Features with high score activate strongly for the concept (high TF) but
rarely for other prompts (high IDF), making them discriminative.

Reference: arXiv 2602.11910 — Section 3.3 (SAE Feature Scoring).
"""

import torch
from torch import Tensor


def compute_tfidf_scores(
    pos_activations: Tensor,
    neg_activations: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Compute TF-IDF importance scores for all SAE features.

    Args:
        pos_activations: SAE latent activations on positive-concept prompts.
            Shape: (n_pos_samples, num_features).
        neg_activations: SAE latent activations on negative-concept prompts.
            Shape: (n_neg_samples, num_features).
        eps: Small constant added to mean_neg before log to prevent division
            by zero and ensure IDF is finite.

    Returns:
        Score tensor of shape (num_features,).  All values are ≥ 0
        because TF = mean_pos ≥ 0 (SAE uses ReLU) and IDF > 0.
    """
    mean_pos = pos_activations.float().mean(dim=0)  # (num_features,)
    mean_neg = neg_activations.float().mean(dim=0)  # (num_features,)

    tf = mean_pos
    idf = torch.log(1.0 + 1.0 / (mean_neg + eps))

    return tf * idf  # (num_features,)


def top_tau_features(scores: Tensor, tau: int) -> Tensor:
    """Return indices of the top-τ features by TF-IDF score.

    Args:
        scores: 1-D tensor of feature scores, shape (num_features,).
        tau: Number of top features to return.

    Returns:
        1-D tensor of ``tau`` feature indices corresponding to the highest
        scores (order within the result is not guaranteed to be sorted).

    Raises:
        ValueError: If ``tau > len(scores)``.
    """
    if tau > len(scores):
        raise ValueError(
            f"tau={tau} exceeds num_features={len(scores)}."
        )
    return torch.topk(scores, tau).indices
