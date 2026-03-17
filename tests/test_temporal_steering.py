"""
Tests for steer_audio.temporal_steering — Phase 2, Prompt 2.2.

Covers:
- All four built-in schedule functions
- Linear schedule (bonus)
- TimestepAdaptiveSteerer (hook registration, per-step alpha dispatch,
  hook cleanup, CAA vs. SAE path)
- Schedule symmetry and boundary conditions
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from steer_audio.temporal_steering import (
    TimestepAdaptiveSteerer,
    TimestepSchedule,
    constant_schedule,
    cosine_schedule,
    early_only_schedule,
    late_only_schedule,
    linear_schedule,
)
from steer_audio.vector_bank import SteeringVector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def caa_vector() -> SteeringVector:
    """Minimal CAA SteeringVector for layers [0, 1], hidden_dim=64."""
    torch.manual_seed(0)
    return SteeringVector(
        concept="tempo",
        method="caa",
        model_name="ace-step",
        layers=[0, 1],
        vector=torch.randn(64),
        clap_delta=0.15,
    )


@pytest.fixture
def sae_vector() -> SteeringVector:
    """Minimal SAE SteeringVector for layer [0], hidden_dim=64."""
    torch.manual_seed(1)
    return SteeringVector(
        concept="mood",
        method="sae",
        model_name="ace-step",
        layers=[0],
        vector=torch.randn(64),
        clap_delta=0.10,
    )


class _SimpleBlock(nn.Module):
    """Minimal transformer block stub with a cross_attn sub-module."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.cross_attn = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cross_attn(x)


def _make_model(n_blocks: int = 4, dim: int = 64) -> Any:
    """Return a stub model whose transformer_blocks are _SimpleBlocks."""
    model = MagicMock()
    blocks = nn.ModuleList([_SimpleBlock(dim) for _ in range(n_blocks)])
    model.transformer_blocks = blocks
    # pipeline mock: returns a zero audio tensor.
    pipeline_mock = MagicMock()
    pipeline_mock.return_value = torch.zeros(dim)
    pipeline_mock.sample_rate = 44100
    model.pipeline = pipeline_mock
    return model


# ---------------------------------------------------------------------------
# constant_schedule
# ---------------------------------------------------------------------------


class TestConstantSchedule:
    def test_returns_alpha_at_every_timestep(self) -> None:
        sched = constant_schedule(60.0)
        for t in [0, 1, 15, 29, 30]:
            assert sched(t, 30) == pytest.approx(60.0)

    def test_zero_alpha(self) -> None:
        sched = constant_schedule(0.0)
        assert sched(10, 30) == pytest.approx(0.0)

    def test_negative_alpha(self) -> None:
        sched = constant_schedule(-50.0)
        assert sched(5, 30) == pytest.approx(-50.0)


# ---------------------------------------------------------------------------
# cosine_schedule
# ---------------------------------------------------------------------------


class TestCosineSchedule:
    def test_peak_at_start(self) -> None:
        """t = T → t/T = 1 → alpha = alpha_max."""
        sched = cosine_schedule(80.0)
        assert sched(30, 30) == pytest.approx(80.0, abs=1e-6)

    def test_trough_at_end(self) -> None:
        """t = 0 → alpha = alpha_min = 0.0."""
        sched = cosine_schedule(80.0)
        assert sched(0, 30) == pytest.approx(0.0, abs=1e-6)

    def test_alpha_min_offset(self) -> None:
        """alpha_min shifts the trough."""
        sched = cosine_schedule(80.0, alpha_min=20.0)
        assert sched(0, 30) == pytest.approx(20.0, abs=1e-6)
        assert sched(30, 30) == pytest.approx(80.0, abs=1e-6)

    def test_midpoint_is_halfway(self) -> None:
        """t = T/2 → factor ≈ 0.5 → alpha ≈ (alpha_max + alpha_min) / 2."""
        sched = cosine_schedule(80.0, alpha_min=0.0)
        mid = sched(15, 30)
        expected = 0.5 * (1.0 + math.cos(math.pi * (1.0 - 15 / 30))) * 80.0
        assert mid == pytest.approx(expected, abs=1e-5)

    def test_monotonically_decreasing(self) -> None:
        """Alpha should be non-increasing as t decreases from T to 0."""
        sched = cosine_schedule(80.0)
        T = 30
        values = [sched(t, T) for t in range(T, -1, -1)]  # T down to 0
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1] - 1e-9

    def test_zero_total_steps_safe(self) -> None:
        """T = 0 must not raise ZeroDivisionError."""
        sched = cosine_schedule(80.0)
        result = sched(0, 0)
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# early_only_schedule
# ---------------------------------------------------------------------------


class TestEarlyOnlySchedule:
    def test_active_above_cutoff(self) -> None:
        sched = early_only_schedule(60.0, cutoff=0.5)
        assert sched(20, 30) == pytest.approx(60.0)  # t/T ≈ 0.67 > 0.5

    def test_inactive_at_or_below_cutoff(self) -> None:
        sched = early_only_schedule(60.0, cutoff=0.5)
        assert sched(15, 30) == pytest.approx(0.0)   # t/T = 0.5, not > 0.5
        assert sched(5, 30) == pytest.approx(0.0)    # t/T ≈ 0.17 ≤ 0.5

    def test_full_range_active_at_cutoff_0(self) -> None:
        """With cutoff=0 every step is 'early' (t/T > 0 for t >= 1)."""
        sched = early_only_schedule(60.0, cutoff=0.0)
        assert sched(1, 30) == pytest.approx(60.0)

    def test_none_active_at_cutoff_1(self) -> None:
        """With cutoff=1 no step satisfies t/T > 1."""
        sched = early_only_schedule(60.0, cutoff=1.0)
        assert sched(30, 30) == pytest.approx(0.0)  # t/T = 1, not > 1


# ---------------------------------------------------------------------------
# late_only_schedule
# ---------------------------------------------------------------------------


class TestLateOnlySchedule:
    def test_active_at_or_below_cutoff(self) -> None:
        sched = late_only_schedule(60.0, cutoff=0.5)
        assert sched(10, 30) == pytest.approx(60.0)  # t/T ≈ 0.33 ≤ 0.5
        assert sched(15, 30) == pytest.approx(60.0)  # t/T = 0.5 ≤ 0.5

    def test_inactive_above_cutoff(self) -> None:
        sched = late_only_schedule(60.0, cutoff=0.5)
        assert sched(20, 30) == pytest.approx(0.0)   # t/T ≈ 0.67 > 0.5

    def test_complementary_to_early_only(self) -> None:
        """early_only + late_only should equal constant at every step."""
        alpha = 60.0
        early = early_only_schedule(alpha, cutoff=0.5)
        late = late_only_schedule(alpha, cutoff=0.5)
        T = 30
        for t in range(1, T + 1):
            # They are NOT strictly complementary due to the strict inequality in early.
            combined = early(t, T) + late(t, T)
            # At the boundary t/T == 0.5 only late is active.
            assert combined == pytest.approx(alpha)


# ---------------------------------------------------------------------------
# linear_schedule
# ---------------------------------------------------------------------------


class TestLinearSchedule:
    def test_peak_at_start(self) -> None:
        sched = linear_schedule(60.0)
        assert sched(30, 30) == pytest.approx(60.0, abs=1e-6)

    def test_trough_at_end(self) -> None:
        sched = linear_schedule(60.0)
        assert sched(0, 30) == pytest.approx(0.0, abs=1e-6)

    def test_linear_midpoint(self) -> None:
        sched = linear_schedule(60.0, alpha_end=0.0)
        assert sched(15, 30) == pytest.approx(30.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TimestepSchedule protocol conformance
# ---------------------------------------------------------------------------


class TestTimestepScheduleProtocol:
    def test_all_schedules_conform_to_protocol(self) -> None:
        schedules = [
            constant_schedule(50.0),
            cosine_schedule(50.0),
            early_only_schedule(50.0),
            late_only_schedule(50.0),
            linear_schedule(50.0),
        ]
        for sched in schedules:
            assert isinstance(sched, TimestepSchedule), (
                f"{sched} does not implement TimestepSchedule protocol"
            )
            result = sched(15, 30)
            assert isinstance(result, float)


# ---------------------------------------------------------------------------
# TimestepAdaptiveSteerer
# ---------------------------------------------------------------------------


class TestTimestepAdaptiveSteerer:
    def test_init_defaults_to_vector_layers(self, caa_vector: SteeringVector) -> None:
        steerer = TimestepAdaptiveSteerer(caa_vector, constant_schedule(60.0))
        assert steerer.layers == caa_vector.layers

    def test_init_custom_layers(self, caa_vector: SteeringVector) -> None:
        steerer = TimestepAdaptiveSteerer(caa_vector, constant_schedule(60.0), layers=[2])
        assert steerer.layers == [2]

    def test_schedule_values_length(self, caa_vector: SteeringVector) -> None:
        steerer = TimestepAdaptiveSteerer(caa_vector, constant_schedule(60.0))
        vals = steerer.schedule_values(30)
        assert len(vals) == 30

    def test_schedule_values_constant(self, caa_vector: SteeringVector) -> None:
        steerer = TimestepAdaptiveSteerer(caa_vector, constant_schedule(60.0))
        vals = steerer.schedule_values(30)
        assert all(v == pytest.approx(60.0) for v in vals)

    def test_schedule_values_cosine_decreasing(self, caa_vector: SteeringVector) -> None:
        steerer = TimestepAdaptiveSteerer(caa_vector, cosine_schedule(80.0))
        vals = steerer.schedule_values(30)
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1] - 1e-9

    def test_hooks_registered_and_removed(self, caa_vector: SteeringVector) -> None:
        """Hook count on the target module is 0 after the context manager exits."""
        steerer = TimestepAdaptiveSteerer(
            caa_vector, constant_schedule(60.0), layers=[0]
        )
        model = _make_model(n_blocks=4, dim=64)
        target = model.transformer_blocks[0].cross_attn

        with steerer._hooked(model.transformer_blocks, total_T=30):
            assert len(target._forward_hooks) == 1

        assert len(target._forward_hooks) == 0

    def test_hooks_registered_for_multiple_layers(
        self, caa_vector: SteeringVector
    ) -> None:
        steerer = TimestepAdaptiveSteerer(
            caa_vector, constant_schedule(60.0), layers=[0, 1]
        )
        model = _make_model(n_blocks=4, dim=64)
        t0 = model.transformer_blocks[0].cross_attn
        t1 = model.transformer_blocks[1].cross_attn

        with steerer._hooked(model.transformer_blocks, total_T=30):
            assert len(t0._forward_hooks) == 1
            assert len(t1._forward_hooks) == 1

        assert len(t0._forward_hooks) == 0
        assert len(t1._forward_hooks) == 0

    def test_out_of_range_layer_skipped(self, caa_vector: SteeringVector) -> None:
        """Out-of-range layer index should be skipped without error."""
        steerer = TimestepAdaptiveSteerer(
            caa_vector, constant_schedule(60.0), layers=[99]
        )
        model = _make_model(n_blocks=4, dim=64)
        # Should not raise.
        with steerer._hooked(model.transformer_blocks, total_T=30):
            pass

    def test_caa_hook_modifies_activation(self, caa_vector: SteeringVector) -> None:
        """A non-zero CAA hook must change the activation tensor."""
        steerer = TimestepAdaptiveSteerer(
            caa_vector, constant_schedule(60.0), layers=[0]
        )
        model = _make_model(n_blocks=4, dim=64)
        x = torch.ones(1, 8, 64)  # (batch, seq, dim)
        x_orig = x.clone()

        # Manually trigger one forward pass through the hooked block.
        with steerer._hooked(model.transformer_blocks, total_T=30):
            out = model.transformer_blocks[0](x)

        assert not torch.allclose(out, x_orig), (
            "CAA hook at α=60 must change the activation"
        )

    def test_zero_alpha_hook_leaves_activation_unchanged(
        self, caa_vector: SteeringVector
    ) -> None:
        """A zero-alpha hook must leave activation identical."""
        steerer = TimestepAdaptiveSteerer(
            caa_vector, constant_schedule(0.0), layers=[0]
        )
        model = _make_model(n_blocks=4, dim=64)
        x = torch.ones(1, 8, 64)
        x_orig = x.clone()

        with steerer._hooked(model.transformer_blocks, total_T=30):
            out = model.transformer_blocks[0](x)

        assert torch.allclose(out, x_orig), (
            "Zero-alpha hook must not change the activation"
        )

    def test_sae_hook_modifies_activation_without_renorm(
        self, sae_vector: SteeringVector
    ) -> None:
        """SAE hook applies additive delta without renorm; output must differ."""
        steerer = TimestepAdaptiveSteerer(
            sae_vector, constant_schedule(60.0), layers=[0]
        )
        model = _make_model(n_blocks=4, dim=64)
        x = torch.ones(1, 8, 64)
        x_orig = x.clone()

        with steerer._hooked(model.transformer_blocks, total_T=30):
            out = model.transformer_blocks[0](x)

        assert not torch.allclose(out, x_orig)

    def test_early_only_hook_inactive_at_late_steps(
        self, caa_vector: SteeringVector
    ) -> None:
        """With early_only_schedule, the last-step hook must not modify activations."""
        T = 10
        # Late step: t = max(1, T - (T-1)) = 1; t/T = 0.1 ≤ 0.5 → inactive.
        steerer = TimestepAdaptiveSteerer(
            caa_vector, early_only_schedule(80.0, cutoff=0.5), layers=[0]
        )
        model = _make_model(n_blocks=4, dim=64)
        target = model.transformer_blocks[0].cross_attn
        x = torch.ones(1, 8, 64)

        outputs = []
        with steerer._hooked(model.transformer_blocks, total_T=T):
            # Simulate T forward passes; capture each output.
            for _ in range(T):
                outputs.append(target(x))

        # Last output (step T-1 → t=1 → t/T=0.1 ≤ 0.5) should be unchanged.
        x_orig = x  # Identity cross_attn returns x unchanged by design.
        assert torch.allclose(outputs[-1], x_orig), (
            "early_only hook must be inactive at the last diffusion step"
        )

    def test_steerer_get_transformer_blocks_fallback(
        self, caa_vector: SteeringVector
    ) -> None:
        """_get_transformer_blocks should find blocks at model.transformer_blocks."""
        steerer = TimestepAdaptiveSteerer(caa_vector, constant_schedule(60.0))

        # Use a plain object so MagicMock does not auto-create patchable_model.
        class _StubModel:
            def __init__(self, b: nn.ModuleList) -> None:
                self.transformer_blocks = b

        blocks_list = nn.ModuleList([_SimpleBlock() for _ in range(4)])
        stub = _StubModel(blocks_list)
        found = steerer._get_transformer_blocks(stub)
        assert len(found) == 4

    def test_steerer_get_transformer_blocks_not_found(
        self, caa_vector: SteeringVector
    ) -> None:
        """Should raise AttributeError if no recognized path exists."""
        steerer = TimestepAdaptiveSteerer(caa_vector, constant_schedule(60.0))
        bad_model = MagicMock(spec=[])  # no attributes at all
        with pytest.raises(AttributeError):
            steerer._get_transformer_blocks(bad_model)


# ---------------------------------------------------------------------------
# Schedule interaction tests
# ---------------------------------------------------------------------------


class TestScheduleInteractions:
    def test_cosine_mean_less_than_constant(self) -> None:
        """The mean alpha of cosine_schedule is less than constant_schedule."""
        T = 30
        alpha = 80.0
        const = constant_schedule(alpha)
        cosine = cosine_schedule(alpha)
        mean_const = np.mean([const(max(1, T - k), T) for k in range(T)])
        mean_cosine = np.mean([cosine(max(1, T - k), T) for k in range(T)])
        assert mean_cosine < mean_const, (
            "cosine schedule mean should be less than constant at same peak alpha"
        )

    def test_early_only_deactivates_second_half(self) -> None:
        T = 30
        sched = early_only_schedule(80.0, cutoff=0.5)
        # Second half: steps T//2 .. T-1 (t ≈ T/2 down to 1, t/T ≤ 0.5)
        second_half = [sched(max(1, T - k), T) for k in range(T // 2, T)]
        assert all(v == pytest.approx(0.0) for v in second_half)

    def test_late_only_deactivates_first_half(self) -> None:
        T = 30
        sched = late_only_schedule(80.0, cutoff=0.5)
        # First half: steps 0 .. T//2-1 (t ≈ T down to T/2+1, t/T > 0.5)
        first_half = [sched(max(1, T - k), T) for k in range(T // 2)]
        assert all(v == pytest.approx(0.0) for v in first_half)
