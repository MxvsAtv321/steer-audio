"""
Tests for steer_audio.temporal_steering — Prompt 2.3: Timestep-Adaptive Steering.

Covers:
- get_schedule: all 5 types at step=0, mid, final for total_steps=60
- early_only / late_only boundaries
- step_alpha scaling examples
- advance_step / reset behavior
- Clamping when step >= total_steps
- register_scheduled_hooks: hook registration, scale interacts with step state
- Invalid schedule_type raises ValueError
All tests are CPU-only and independent of ACE-Step / real audio.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from steer_audio.multi_steer import MultiConceptSteerer
from steer_audio.temporal_steering import TimestepAdaptiveSteerer, get_schedule
from steer_audio.vector_bank import SteeringVector, SteeringVectorBank


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

TOTAL = 60  # canonical ACE-Step step count


def _make_bank(dim: int = 8, alpha: float = 50.0) -> tuple[SteeringVectorBank, MultiConceptSteerer]:
    """Return a (bank, steerer) pair with one active concept."""
    torch.manual_seed(42)
    sv = SteeringVector(
        concept="tempo",
        method="caa",
        model_name="ace-step",
        layers=[0, 1],
        vector=torch.randn(dim),
    )
    bank = SteeringVectorBank()
    bank.add(sv)
    multi = MultiConceptSteerer(bank)
    multi.add_concept("tempo", alpha=alpha)
    return bank, multi


class _SimpleModel(nn.Module):
    """2-layer model whose children can be hooked."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.layer0 = nn.Linear(dim, dim, bias=False)
        self.layer1 = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer1(self.layer0(x))


# ---------------------------------------------------------------------------
# get_schedule — constant
# ---------------------------------------------------------------------------


class TestGetScheduleConstant:
    def test_step0(self) -> None:
        f = get_schedule("constant")
        assert f(0, TOTAL) == pytest.approx(1.0)

    def test_mid(self) -> None:
        f = get_schedule("constant")
        assert f(30, TOTAL) == pytest.approx(1.0)

    def test_final(self) -> None:
        f = get_schedule("constant")
        assert f(TOTAL, TOTAL) == pytest.approx(1.0)

    def test_beyond_total(self) -> None:
        f = get_schedule("constant")
        assert f(TOTAL + 10, TOTAL) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# get_schedule — linear
# ---------------------------------------------------------------------------


class TestGetScheduleLinear:
    def test_step0_is_one(self) -> None:
        f = get_schedule("linear")
        assert f(0, TOTAL) == pytest.approx(1.0)

    def test_mid_is_half(self) -> None:
        f = get_schedule("linear")
        assert f(30, TOTAL) == pytest.approx(0.5)

    def test_final_is_zero(self) -> None:
        f = get_schedule("linear")
        assert f(TOTAL, TOTAL) == pytest.approx(0.0)

    def test_clamped_beyond_total(self) -> None:
        """step > total_steps should clamp to 0.0."""
        f = get_schedule("linear")
        assert f(TOTAL + 5, TOTAL) == pytest.approx(0.0)

    def test_values_in_unit_interval(self) -> None:
        f = get_schedule("linear")
        for step in range(TOTAL + 1):
            v = f(step, TOTAL)
            assert 0.0 <= v <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# get_schedule — cosine
# ---------------------------------------------------------------------------


class TestGetScheduleCosine:
    def test_step0_is_one(self) -> None:
        """(1 + cos(0)) / 2 = 1.0."""
        f = get_schedule("cosine")
        assert f(0, TOTAL) == pytest.approx(1.0)

    def test_mid_is_half(self) -> None:
        """(1 + cos(π/2)) / 2 = 0.5."""
        f = get_schedule("cosine")
        assert f(30, TOTAL) == pytest.approx(0.5, abs=1e-6)

    def test_final_is_zero(self) -> None:
        """(1 + cos(π)) / 2 = 0.0."""
        f = get_schedule("cosine")
        assert f(TOTAL, TOTAL) == pytest.approx(0.0, abs=1e-6)

    def test_formula(self) -> None:
        """Verify exact formula at an arbitrary step."""
        f = get_schedule("cosine")
        step, total = 15, 60
        expected = (1.0 + math.cos(math.pi * step / total)) / 2.0
        assert f(step, total) == pytest.approx(expected, abs=1e-8)

    def test_clamped_beyond_total(self) -> None:
        f = get_schedule("cosine")
        assert f(TOTAL + 10, TOTAL) == pytest.approx(0.0, abs=1e-6)

    def test_monotonically_non_increasing(self) -> None:
        f = get_schedule("cosine")
        vals = [f(s, TOTAL) for s in range(TOTAL + 1)]
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1] - 1e-9


# ---------------------------------------------------------------------------
# get_schedule — early_only
# ---------------------------------------------------------------------------


class TestGetScheduleEarlyOnly:
    # Boundary: TOTAL * 0.4 = 24.  Active when step < 24.

    def test_step0_active(self) -> None:
        f = get_schedule("early_only")
        assert f(0, TOTAL) == pytest.approx(1.0)

    def test_step_just_before_boundary_active(self) -> None:
        f = get_schedule("early_only")
        assert f(23, TOTAL) == pytest.approx(1.0)  # 23 < 24

    def test_step_at_boundary_inactive(self) -> None:
        f = get_schedule("early_only")
        assert f(24, TOTAL) == pytest.approx(0.0)  # 24 is not < 24

    def test_mid_step_inactive(self) -> None:
        f = get_schedule("early_only")
        assert f(30, TOTAL) == pytest.approx(0.0)

    def test_final_step_inactive(self) -> None:
        f = get_schedule("early_only")
        assert f(TOTAL, TOTAL) == pytest.approx(0.0)

    def test_beyond_total_inactive(self) -> None:
        f = get_schedule("early_only")
        assert f(TOTAL + 5, TOTAL) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_schedule — late_only
# ---------------------------------------------------------------------------


class TestGetScheduleLateOnly:
    # Boundary: TOTAL * 0.6 = 36.  Active when step >= 36.

    def test_step0_inactive(self) -> None:
        f = get_schedule("late_only")
        assert f(0, TOTAL) == pytest.approx(0.0)

    def test_step_just_before_boundary_inactive(self) -> None:
        f = get_schedule("late_only")
        assert f(35, TOTAL) == pytest.approx(0.0)  # 35 < 36

    def test_step_at_boundary_active(self) -> None:
        f = get_schedule("late_only")
        assert f(36, TOTAL) == pytest.approx(1.0)  # 36 is not < 36

    def test_mid_step_inactive(self) -> None:
        f = get_schedule("late_only")
        assert f(30, TOTAL) == pytest.approx(0.0)  # 30 < 36

    def test_final_step_active(self) -> None:
        f = get_schedule("late_only")
        assert f(TOTAL, TOTAL) == pytest.approx(1.0)

    def test_beyond_total_active(self) -> None:
        f = get_schedule("late_only")
        assert f(TOTAL + 5, TOTAL) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# get_schedule — invalid type
# ---------------------------------------------------------------------------


def test_get_schedule_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Unknown schedule type"):
        get_schedule("unknown_type")


# ---------------------------------------------------------------------------
# TimestepAdaptiveSteerer — step_alpha
# ---------------------------------------------------------------------------


class TestStepAlpha:
    def test_constant_preserves_base_alpha(self) -> None:
        _, multi = _make_bank()
        steerer = TimestepAdaptiveSteerer(multi, schedule_type="constant")
        assert steerer.step_alpha(50.0, 0, TOTAL) == pytest.approx(50.0)
        assert steerer.step_alpha(50.0, 30, TOTAL) == pytest.approx(50.0)
        assert steerer.step_alpha(50.0, TOTAL, TOTAL) == pytest.approx(50.0)

    def test_linear_scales_correctly(self) -> None:
        _, multi = _make_bank()
        steerer = TimestepAdaptiveSteerer(multi, schedule_type="linear")
        assert steerer.step_alpha(50.0, 0, TOTAL) == pytest.approx(50.0)
        assert steerer.step_alpha(50.0, 30, TOTAL) == pytest.approx(25.0)
        assert steerer.step_alpha(50.0, TOTAL, TOTAL) == pytest.approx(0.0)

    def test_cosine_at_step0(self) -> None:
        _, multi = _make_bank()
        steerer = TimestepAdaptiveSteerer(multi, schedule_type="cosine")
        assert steerer.step_alpha(100.0, 0, TOTAL) == pytest.approx(100.0)

    def test_cosine_at_final(self) -> None:
        _, multi = _make_bank()
        steerer = TimestepAdaptiveSteerer(multi, schedule_type="cosine")
        assert steerer.step_alpha(100.0, TOTAL, TOTAL) == pytest.approx(0.0, abs=1e-6)

    def test_early_only_zero_after_boundary(self) -> None:
        _, multi = _make_bank()
        steerer = TimestepAdaptiveSteerer(multi, schedule_type="early_only")
        assert steerer.step_alpha(80.0, 24, TOTAL) == pytest.approx(0.0)

    def test_late_only_zero_before_boundary(self) -> None:
        _, multi = _make_bank()
        steerer = TimestepAdaptiveSteerer(multi, schedule_type="late_only")
        assert steerer.step_alpha(80.0, 35, TOTAL) == pytest.approx(0.0)

    def test_late_only_full_at_boundary(self) -> None:
        _, multi = _make_bank()
        steerer = TimestepAdaptiveSteerer(multi, schedule_type="late_only")
        assert steerer.step_alpha(80.0, 36, TOTAL) == pytest.approx(80.0)

    def test_negative_base_alpha_scales_correctly(self) -> None:
        _, multi = _make_bank()
        steerer = TimestepAdaptiveSteerer(multi, schedule_type="linear")
        assert steerer.step_alpha(-50.0, 30, TOTAL) == pytest.approx(-25.0)


# ---------------------------------------------------------------------------
# TimestepAdaptiveSteerer — advance_step / reset
# ---------------------------------------------------------------------------


class TestAdvanceStepAndReset:
    def test_initial_step_is_zero(self) -> None:
        _, multi = _make_bank()
        steerer = TimestepAdaptiveSteerer(multi, schedule_type="constant")
        assert steerer._state["step"] == 0

    def test_advance_step_increments(self) -> None:
        _, multi = _make_bank()
        steerer = TimestepAdaptiveSteerer(multi, schedule_type="constant")
        steerer.advance_step()
        assert steerer._state["step"] == 1
        steerer.advance_step()
        assert steerer._state["step"] == 2

    def test_reset_goes_to_zero(self) -> None:
        _, multi = _make_bank()
        steerer = TimestepAdaptiveSteerer(multi, schedule_type="constant")
        for _ in range(10):
            steerer.advance_step()
        steerer.reset()
        assert steerer._state["step"] == 0

    def test_step_alpha_reflects_advance_step(self) -> None:
        """Calling step_alpha with current step should match the state."""
        _, multi = _make_bank()
        steerer = TimestepAdaptiveSteerer(multi, schedule_type="linear")
        # At step 0, scale = 1.0
        assert steerer.step_alpha(100.0, steerer._state["step"], TOTAL) == pytest.approx(100.0)
        steerer.advance_step()
        # At step 1, scale = 1 - 1/60 ≈ 0.983
        expected = 100.0 * (1 - 1 / TOTAL)
        assert steerer.step_alpha(100.0, steerer._state["step"], TOTAL) == pytest.approx(expected, rel=1e-5)

    def test_reset_after_full_run(self) -> None:
        _, multi = _make_bank()
        steerer = TimestepAdaptiveSteerer(multi, schedule_type="linear")
        for _ in range(TOTAL):
            steerer.advance_step()
        assert steerer._state["step"] == TOTAL
        steerer.reset()
        assert steerer._state["step"] == 0

    def test_step_beyond_total_clamped_by_schedule(self) -> None:
        """When step > total_steps the linear schedule should clamp to 0."""
        _, multi = _make_bank()
        steerer = TimestepAdaptiveSteerer(multi, schedule_type="linear")
        # Advance past total_steps
        for _ in range(TOTAL + 10):
            steerer.advance_step()
        val = steerer.step_alpha(50.0, steerer._state["step"], TOTAL)
        assert val == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TimestepAdaptiveSteerer — register_scheduled_hooks
# ---------------------------------------------------------------------------


class TestRegisterScheduledHooks:
    def _model_and_steerer(
        self, schedule_type: str = "constant", alpha: float = 50.0, dim: int = 8
    ) -> tuple[_SimpleModel, TimestepAdaptiveSteerer]:
        _, multi = _make_bank(dim=dim, alpha=alpha)
        model = _SimpleModel(dim=dim)
        steerer = TimestepAdaptiveSteerer(multi, schedule_type=schedule_type)
        return model, steerer

    def test_returns_correct_number_of_handles(self) -> None:
        model, steerer = self._model_and_steerer()
        handles = steerer.register_scheduled_hooks(model, target_layers=[0, 1], total_steps=TOTAL)
        assert len(handles) == 2
        # Cleanup
        steerer.multi_steerer.remove_hooks(handles)

    def test_single_layer_hook(self) -> None:
        model, steerer = self._model_and_steerer()
        handles = steerer.register_scheduled_hooks(model, target_layers=[0], total_steps=TOTAL)
        assert len(handles) == 1
        steerer.multi_steerer.remove_hooks(handles)

    def test_hook_modifies_output_at_nonzero_scale(self) -> None:
        """With constant schedule and nonzero alpha, output should differ from unhooked."""
        model, steerer = self._model_and_steerer(schedule_type="constant", alpha=50.0)
        x = torch.ones(1, 8)
        # Unhooked output
        unhooked = model.layer0(x).detach().clone()
        # Hooked output
        handles = steerer.register_scheduled_hooks(model, target_layers=[0], total_steps=TOTAL)
        hooked = model.layer0(x).detach().clone()
        steerer.multi_steerer.remove_hooks(handles)
        # The hook adds a scaled steering vector, so outputs must differ
        assert not torch.allclose(hooked, unhooked), (
            "Hook must change the activation when alpha != 0 and scale != 0"
        )

    def test_hook_no_change_when_scale_is_zero(self) -> None:
        """At step=total_steps with linear schedule, scale=0 → output unchanged."""
        dim = 8
        model, steerer = self._model_and_steerer(schedule_type="linear", alpha=50.0, dim=dim)
        # Advance to total_steps so schedule returns 0
        for _ in range(TOTAL):
            steerer.advance_step()
        x = torch.ones(1, dim)
        unhooked = model.layer0(x).detach().clone()
        handles = steerer.register_scheduled_hooks(model, target_layers=[0], total_steps=TOTAL)
        hooked = model.layer0(x).detach().clone()
        steerer.multi_steerer.remove_hooks(handles)
        assert torch.allclose(hooked, unhooked, atol=1e-6), (
            "Hook with scale=0 must leave activation unchanged"
        )

    def test_hooks_removed_cleanly(self) -> None:
        """After remove_hooks, the output matches the unhooked baseline."""
        model, steerer = self._model_and_steerer(schedule_type="constant", alpha=50.0)
        x = torch.ones(1, 8)
        unhooked = model.layer0(x).detach().clone()
        handles = steerer.register_scheduled_hooks(model, target_layers=[0], total_steps=TOTAL)
        steerer.multi_steerer.remove_hooks(handles)
        after_removal = model.layer0(x).detach().clone()
        assert torch.allclose(after_removal, unhooked, atol=1e-6), (
            "After hook removal, output must match pre-hook baseline"
        )

    def test_output_shape_unchanged(self) -> None:
        """Hook must not change tensor shape."""
        model, steerer = self._model_and_steerer(schedule_type="constant", alpha=50.0)
        x = torch.ones(2, 8)  # batch=2
        handles = steerer.register_scheduled_hooks(model, target_layers=[0], total_steps=TOTAL)
        out = model.layer0(x)
        steerer.multi_steerer.remove_hooks(handles)
        assert out.shape == x.shape

    def test_scale_decreases_effect_over_steps(self) -> None:
        """With linear schedule, effect at step 0 should be larger than at step 30."""
        dim = 8
        _, multi = _make_bank(dim=dim, alpha=50.0)
        model = _SimpleModel(dim=dim)
        x = torch.ones(1, dim)
        baseline = model.layer0(x).detach().clone()

        # Measure effect at step=0
        steerer0 = TimestepAdaptiveSteerer(multi, schedule_type="linear")
        handles0 = steerer0.register_scheduled_hooks(model, [0], TOTAL)
        out0 = model.layer0(x).detach().clone()
        steerer0.multi_steerer.remove_hooks(handles0)
        diff0 = (out0 - baseline).norm().item()

        # Measure effect at step=30 (scale=0.5)
        steerer30 = TimestepAdaptiveSteerer(multi, schedule_type="linear")
        for _ in range(30):
            steerer30.advance_step()
        handles30 = steerer30.register_scheduled_hooks(model, [0], TOTAL)
        out30 = model.layer0(x).detach().clone()
        steerer30.multi_steerer.remove_hooks(handles30)
        diff30 = (out30 - baseline).norm().item()

        assert diff0 > diff30, "Linear schedule: effect at step 0 > effect at step 30"

    def test_out_of_range_layer_hooks_model_root(self) -> None:
        """A layer index beyond len(children) should hook the model root without error."""
        model, steerer = self._model_and_steerer()
        handles = steerer.register_scheduled_hooks(model, target_layers=[99], total_steps=TOTAL)
        assert len(handles) == 1  # registered on model root
        steerer.multi_steerer.remove_hooks(handles)

    def test_hooks_live_update_with_advance_step(self) -> None:
        """advance_step() between forward passes changes effective scale."""
        dim = 8
        _, multi = _make_bank(dim=dim, alpha=100.0)
        model = _SimpleModel(dim=dim)
        x = torch.ones(1, dim)
        baseline = model.layer0(x).detach().clone()
        steerer = TimestepAdaptiveSteerer(multi, schedule_type="linear")
        handles = steerer.register_scheduled_hooks(model, [0], TOTAL)

        # Step 0 → scale = 1.0
        out_step0 = model.layer0(x).detach().clone()
        diff0 = (out_step0 - baseline).norm().item()

        # Advance to total_steps → scale = 0.0
        for _ in range(TOTAL):
            steerer.advance_step()
        out_last = model.layer0(x).detach().clone()
        diff_last = (out_last - baseline).norm().item()

        steerer.multi_steerer.remove_hooks(handles)
        assert diff0 > diff_last, (
            "After advancing to total_steps, linear-schedule effect should be ~0"
        )


# ---------------------------------------------------------------------------
# get_schedule — return type and value range
# ---------------------------------------------------------------------------


class TestScheduleValueRange:
    @pytest.mark.parametrize("stype", ["constant", "linear", "cosine", "early_only", "late_only"])
    def test_values_in_unit_interval(self, stype: str) -> None:
        f = get_schedule(stype)
        for step in range(0, TOTAL + 5):
            v = f(step, TOTAL)
            assert 0.0 <= v <= 1.0 + 1e-9, f"Schedule {stype} returned {v} at step {step}"

    @pytest.mark.parametrize("stype", ["constant", "linear", "cosine", "early_only", "late_only"])
    def test_returns_float(self, stype: str) -> None:
        f = get_schedule(stype)
        assert isinstance(f(0, TOTAL), float)
