"""
Unit tests for the activation patching impact metric and hook utilities.

Tests the ``compute_impact`` function from ``src.patching_utils``,
the ``hook_context`` context manager, and functional layer identification.

Reference: arXiv 2602.11910 — Section 3.1 (Activation Patching, Eq. 9).
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


# ---------------------------------------------------------------------------
# 5. Impact formula: 3 known input values with manual reference
# ---------------------------------------------------------------------------


def test_impact_formula_manual_reference():
    """compute_impact matches a manually computed scalar projection for 3 inputs.

    The implementation computes:
        impact = clamp(dot(patched - source, target - source) / ||target - source||², 0, 1)

    We construct inputs where the result is analytically known.
    """
    # Case A: patched is halfway between source and target → impact = 0.5
    source = torch.zeros(4)
    target = torch.ones(4)
    patched_half = 0.5 * torch.ones(4)

    diff_p = (patched_half - source).float().flatten()
    diff_t = (target - source).float().flatten()
    expected_a = float((diff_p @ diff_t) / (diff_t @ diff_t))  # 0.5

    assert compute_impact(patched_half, source, target) == pytest.approx(expected_a, abs=1e-6), (
        f"Case A: expected {expected_a:.4f}"
    )

    # Case B: patched is 1/4 of the way from source to target → impact = 0.25
    patched_quarter = 0.25 * torch.ones(4)
    diff_p2 = (patched_quarter - source).float().flatten()
    expected_b = float((diff_p2 @ diff_t) / (diff_t @ diff_t))  # 0.25

    assert compute_impact(patched_quarter, source, target) == pytest.approx(expected_b, abs=1e-6), (
        f"Case B: expected {expected_b:.4f}"
    )

    # Case C: source == target → degenerate, impact = 0.0
    assert compute_impact(patched_half, source, source.clone()) == pytest.approx(0.0, abs=1e-6), (
        "Case C: degenerate (source==target) should return 0."
    )


# ---------------------------------------------------------------------------
# 6. Hook registration intercepts activations (side-effect counter)
# ---------------------------------------------------------------------------


def test_hook_intercepts_activation():
    """A forward hook registered on layer 1 of a 2-layer model is called
    exactly once per forward pass; the side-effect counter increments."""

    class TwoLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = nn.Linear(8, 8, bias=False)
            self.layer1 = nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.layer1(self.layer0(x))

    model = TwoLayerModel()
    counter = {"calls": 0}

    def counting_hook(module, inp, out):
        counter["calls"] += 1
        return out

    x = torch.randn(2, 8)

    # Hook on layer1 only
    with hook_context(model.layer1, counting_hook):
        model(x)
        assert counter["calls"] == 1, (
            f"Hook should be called exactly once; got {counter['calls']}"
        )
        model(x)
        assert counter["calls"] == 2, (
            f"Hook should have been called twice total; got {counter['calls']}"
        )


# ---------------------------------------------------------------------------
# 7. Hook cleanup: after deregistration, forward pass is not intercepted
# ---------------------------------------------------------------------------


def test_hook_cleanup_stops_interception():
    """After hook_context exits, further forward passes are NOT intercepted."""
    module = nn.Linear(4, 4, bias=False)
    counter = {"calls": 0}

    def counting_hook(m, inp, out):
        counter["calls"] += 1
        return out

    x = torch.randn(2, 4)

    with hook_context(module, counting_hook):
        module(x)

    # After the context exits the hook is removed; this call should NOT increment
    module(x)
    assert counter["calls"] == 1, (
        f"Hook should only fire inside the context; "
        f"total calls={counter['calls']}, expected 1."
    )


# ---------------------------------------------------------------------------
# 8. Functional layer identification from a mock impact matrix
# ---------------------------------------------------------------------------


def _top_k_layers(impact_matrix: torch.Tensor, k: int = 2) -> list:
    """Return the indices of the top-k layers by mean impact across concepts."""
    mean_impact = impact_matrix.mean(dim=1)  # (num_layers,)
    return torch.topk(mean_impact, k).indices.tolist()


def test_functional_layer_identification():
    """Given a 24×3 impact matrix, the top-2 layers are correctly identified.

    We set layers 6 and 7 to have the highest impact values and verify
    that _top_k_layers returns exactly {6, 7}.
    """
    torch.manual_seed(0)
    num_layers, num_concepts = 24, 3

    impact_matrix = torch.rand(num_layers, num_concepts) * 0.3  # low baseline

    # Inject high impact at layers 6 and 7
    impact_matrix[6, :] = 0.9
    impact_matrix[7, :] = 0.8

    top2 = _top_k_layers(impact_matrix, k=2)

    assert set(top2) == {6, 7}, (
        f"Expected functional layers {{6, 7}}, got {set(top2)}"
    )
