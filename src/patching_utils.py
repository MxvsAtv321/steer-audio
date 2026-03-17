"""
Utility functions for activation patching impact analysis.

The impact metric measures how much patching a given layer l moved the
model output toward the target concept distribution.

Reference: arXiv 2602.11910 — Section 3.1 (Activation Patching).
"""

from typing import Callable, Generator
import contextlib

import torch
import torch.nn as nn
from torch import Tensor


def compute_impact(
    patched: Tensor,
    source: Tensor,
    target: Tensor,
) -> float:
    """Compute the activation patching impact score ∈ [0, 1].

    Measures the scalar projection of (patched - source) onto the direction
    (target - source), clamped to [0, 1].

    - Impact = 0  when ``patched == source``  (no effect from patching).
    - Impact = 1  when ``patched == target``  (full effect from patching).

    Args:
        patched: Model output after patching layer l.
            Any shape; will be flattened.
        source: Baseline model output (no concept active).
            Same shape as ``patched``.
        target: Target model output (concept fully active).
            Same shape as ``patched``.

    Returns:
        Scalar float in [0, 1].
    """
    diff_patched = (patched - source).float().flatten()
    diff_target = (target - source).float().flatten()

    den = diff_target.pow(2).sum()
    if den < 1e-10:
        # target == source: degenerate case, no concept signal to measure
        return 0.0

    # Scalar projection of diff_patched onto diff_target, normalized by ||diff_target||²
    sim = torch.dot(diff_patched, diff_target) / den
    return float(sim.clamp(0.0, 1.0))


@contextlib.contextmanager
def hook_context(
    module: nn.Module,
    hook_fn: Callable,
) -> Generator[None, None, None]:
    """Register a forward hook and automatically remove it on exit.

    Args:
        module: The ``nn.Module`` on which to register the hook.
        hook_fn: A callable with the standard PyTorch hook signature
            ``hook_fn(module, input, output) -> output | None``.

    Yields:
        Nothing; the hook is active for the duration of the ``with`` block.
    """
    handle = module.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()
