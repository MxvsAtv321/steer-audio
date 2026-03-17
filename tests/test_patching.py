"""
Unit tests for the activation patching impact metric and hook utilities.

Tests the ``compute_impact`` function from ``src.patching_utils`` and
the ``hook_context`` context manager.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import pytest

# Ensure src/ is importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.patching_utils import compute_impact, hook_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_tensor(seed: int = 0, shape: tuple = (4, 16)) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(*shape)


# ---------------------------------------------------------------------------
# 1. Impact metric stays in [0, 1]
# ---------------------------------------------------------------------------


def test_impact_metric_range():
    """Impact(l, c) ∈ [0, 1] for arbitrary valid inputs."""
    torch.manual_seed(42)
    for _ in range(20):
        source = torch.randn(8, 32)
        target = torch.randn(8, 32)
        patched = torch.randn(8, 32)
        impact = compute_impact(patched, source, target)
        assert 0.0 <= impact <= 1.0, (
            f"Impact out of [0, 1]: got {impact}"
        )


# ---------------------------------------------------------------------------
# 2. Impact = 0 when patched equals source (no change)
# ---------------------------------------------------------------------------


def test_impact_metric_zeros_for_no_change():
    """If patched == source (no patch effect), impact = 0."""
    source = _random_tensor(seed=1)
    target = _random_tensor(seed=2)

    impact = compute_impact(
        patched=source.clone(),  # identical to source
        source=source,
        target=target,
    )
    assert impact == pytest.approx(0.0, abs=1e-6), (
        f"Expected impact=0 when patched==source, got {impact}"
    )


# ---------------------------------------------------------------------------
# 3. Impact = 1 when patched equals target (full patch effect)
# ---------------------------------------------------------------------------


def test_impact_metric_one_for_full_change():
    """If patched == target (full patch effect), impact = 1."""
    source = _random_tensor(seed=3)
    target = _random_tensor(seed=4)

    impact = compute_impact(
        patched=target.clone(),  # identical to target
        source=source,
        target=target,
    )
    assert impact == pytest.approx(1.0, abs=1e-5), (
        f"Expected impact=1 when patched==target, got {impact}"
    )


# ---------------------------------------------------------------------------
# 4. Hooks are removed after context manager exits
# ---------------------------------------------------------------------------


def test_hook_registers_and_deregisters():
    """Forward hook count on a module is 0 after hook_context exits."""
    module = nn.Linear(4, 4)

    # No hooks before entering context
    assert len(module._forward_hooks) == 0

    with hook_context(module, lambda m, i, o: o):
        # Hook is registered inside the context
        assert len(module._forward_hooks) == 1

    # Hook is removed after context exits
    assert len(module._forward_hooks) == 0, (
        "Forward hooks should be empty after hook_context exits; "
        f"found {len(module._forward_hooks)} hook(s) remaining."
    )
