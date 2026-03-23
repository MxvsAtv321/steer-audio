"""
Timestep-adaptive steering schedules for diffusion model inference.

At each denoising step the steering alpha is modulated by a schedule
function f(step, total_steps) ∈ [0, 1], giving fine-grained control
over *when* in the diffusion process a concept is applied.

ACE-Step uses 60 diffusion steps (configurable).  Early steps set global
structure; late steps refine details.  Schedules allow concentrating
steering in the most effective phase.

Reference: arXiv 2602.11910 §3.2 — TADA timestep-adaptive steering.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Protocol, runtime_checkable

import torch
import torch.nn as nn

from steer_audio.multi_steer import MultiConceptSteerer, _renorm
from steer_audio.vector_bank import SteeringVector

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legacy schedule API (kept for backward compatibility with pipeline.py and
# test_integration_phase2.py which were implemented against the Phase 2.2 API)
# ---------------------------------------------------------------------------


@runtime_checkable
class TimestepSchedule(Protocol):
    """Protocol for alpha schedules over diffusion timesteps.

    Args:
        t: Current noise-level timestep (descends from T to 0).
        T: Total number of inference timesteps.

    Returns:
        Effective alpha at this timestep.
    """

    def __call__(self, t: int, T: int) -> float:  # noqa: D102
        ...


def constant_schedule(alpha: float) -> TimestepSchedule:
    """Return *alpha* regardless of the current timestep."""

    def _schedule(t: int, T: int) -> float:
        return alpha

    return _schedule  # type: ignore[return-value]


def cosine_schedule(alpha_max: float, alpha_min: float = 0.0) -> TimestepSchedule:
    """Cosine decay from *alpha_max* (t=T) to *alpha_min* (t=0)."""

    def _schedule(t: int, T: int) -> float:
        if T == 0:
            return float(alpha_max)
        factor = 0.5 * (1.0 + math.cos(math.pi * (1.0 - t / T)))
        return float(alpha_min + (alpha_max - alpha_min) * factor)

    return _schedule  # type: ignore[return-value]


def early_only_schedule(alpha: float, cutoff: float = 0.5) -> TimestepSchedule:
    """Apply *alpha* only during the early (high-noise) portion of diffusion."""

    def _schedule(t: int, T: int) -> float:
        if T == 0:
            return 0.0
        return float(alpha) if t / T > cutoff else 0.0

    return _schedule  # type: ignore[return-value]


def late_only_schedule(alpha: float, cutoff: float = 0.5) -> TimestepSchedule:
    """Apply *alpha* only during the late (low-noise) refinement portion."""

    def _schedule(t: int, T: int) -> float:
        if T == 0:
            return 0.0
        return float(alpha) if t / T <= cutoff else 0.0

    return _schedule  # type: ignore[return-value]


def linear_schedule(alpha_start: float, alpha_end: float = 0.0) -> TimestepSchedule:
    """Linearly decay from *alpha_start* (t=T) to *alpha_end* (t=0)."""

    def _schedule(t: int, T: int) -> float:
        if T == 0:
            return float(alpha_start)
        frac = t / T
        return float(alpha_end + (alpha_start - alpha_end) * frac)

    return _schedule  # type: ignore[return-value]


class LegacyTimestepAdaptiveSteerer:
    """Backward-compatible steerer that wraps a single :class:`SteeringVector`.

    This is the Phase 2.2 API preserved for code that uses the old
    ``TimestepAdaptiveSteerer(vector, schedule, layers)`` constructor.
    New code should use :class:`TimestepAdaptiveSteerer` instead.
    """

    def __init__(
        self,
        vector: SteeringVector,
        schedule: TimestepSchedule,
        layers: list[int] | None = None,
    ) -> None:
        self.vector = vector
        self.schedule = schedule
        self.layers: list[int] = layers if layers is not None else list(vector.layers)

    def schedule_values(self, num_inference_steps: int) -> list[float]:
        """Return the effective alpha at each step (for inspection/testing)."""
        T = num_inference_steps
        return [self.schedule(max(1, T - k), T) for k in range(T)]


# ---------------------------------------------------------------------------
# Schedule factory
# ---------------------------------------------------------------------------


def get_schedule(schedule_type: str) -> Callable[[int, int], float]:
    """Return a schedule function ``f(step, total_steps) -> float ∈ [0, 1]``.

    The returned callable is used to scale the steering alpha at each
    denoising step.  ``step`` is 0-indexed (0 = first step, total_steps = last).

    Available schedules:

    - ``"constant"``:   Always 1.0.
    - ``"linear"``:     Linearly decays 1.0 → 0.0 over steps.  Clamped to
                        [0, 1] when ``step`` is outside ``[0, total_steps]``.
    - ``"cosine"``:     ``(1 + cos(π · step / total_steps)) / 2``.  Equals
                        1.0 at step 0 and 0.0 at step ``total_steps``.
    - ``"early_only"``: 1.0 if ``step < total_steps * 0.4`` else 0.0.
    - ``"late_only"``:  0.0 if ``step < total_steps * 0.6`` else 1.0.

    Args:
        schedule_type: One of the five strings listed above.

    Returns:
        A callable ``(step: int, total_steps: int) -> float``.

    Raises:
        ValueError: If *schedule_type* is not recognised.
    """
    if schedule_type == "constant":

        def _constant(step: int, total_steps: int) -> float:
            return 1.0

        return _constant

    elif schedule_type == "linear":

        def _linear(step: int, total_steps: int) -> float:
            if total_steps <= 0:
                return 1.0
            # Clamp step so step >= total_steps → 0.0.
            t = min(step, total_steps) / total_steps
            return 1.0 - t

        return _linear

    elif schedule_type == "cosine":

        def _cosine(step: int, total_steps: int) -> float:
            if total_steps <= 0:
                return 1.0
            clamped = min(step, total_steps)
            return (1.0 + math.cos(math.pi * clamped / total_steps)) / 2.0

        return _cosine

    elif schedule_type == "early_only":

        def _early_only(step: int, total_steps: int) -> float:
            if total_steps <= 0:
                return 1.0
            return 1.0 if step < total_steps * 0.4 else 0.0

        return _early_only

    elif schedule_type == "late_only":

        def _late_only(step: int, total_steps: int) -> float:
            if total_steps <= 0:
                return 0.0
            return 0.0 if step < total_steps * 0.6 else 1.0

        return _late_only

    else:
        raise ValueError(
            f"Unknown schedule type '{schedule_type}'. "
            "Choose from: 'constant', 'linear', 'cosine', 'early_only', 'late_only'."
        )


# ---------------------------------------------------------------------------
# TimestepAdaptiveSteerer
# ---------------------------------------------------------------------------


class TimestepAdaptiveSteerer:
    """Wraps a :class:`~steer_audio.multi_steer.MultiConceptSteerer` with a
    per-step alpha schedule.

    At each forward pass the hook reads the internal step counter and scales
    the combined steering vector by ``schedule(step, total_steps)``.  The
    caller is responsible for calling :meth:`advance_step` after each
    denoising step and :meth:`reset` before a new generation run.

    Example::

        multi = MultiConceptSteerer(bank)
        multi.add_concept("tempo", alpha=50.0)

        steerer = TimestepAdaptiveSteerer(multi, schedule_type="cosine")
        handles = steerer.register_scheduled_hooks(model, [6, 7], total_steps=60)
        for _ in range(60):
            model(latent)          # hook applies cosine-scaled alpha
            steerer.advance_step()
        multi.remove_hooks(handles)
        steerer.reset()

    Args:
        multi_steerer:  Pre-configured :class:`MultiConceptSteerer` with at
                        least one concept added.
        schedule_type:  Schedule name passed to :func:`get_schedule`.
    """

    def __init__(
        self,
        multi_steerer: MultiConceptSteerer,
        schedule_type: str = "constant",
    ) -> None:
        self.multi_steerer: MultiConceptSteerer = multi_steerer
        self.schedule_fn: Callable[[int, int], float] = get_schedule(schedule_type)
        # Internal step state; use a dict so hook closures see live updates.
        self._state: Dict[str, int] = {"step": 0}

    # ------------------------------------------------------------------ #
    # Core schedule helper
    # ------------------------------------------------------------------ #

    def step_alpha(self, base_alpha: float, step: int, total_steps: int) -> float:
        """Return the effective alpha at *step*.

        Computes ``base_alpha * schedule(step, total_steps)``.

        Args:
            base_alpha:  The unscaled steering strength.
            step:        Current denoising step (0-indexed).
            total_steps: Total number of denoising steps.

        Returns:
            Scaled alpha value (same sign as *base_alpha*).
        """
        scale = self.schedule_fn(step, total_steps)
        return base_alpha * scale

    # ------------------------------------------------------------------ #
    # Step state management
    # ------------------------------------------------------------------ #

    def advance_step(self) -> None:
        """Increment the internal step counter by one.

        Must be called once after each diffusion denoising step so that the
        next forward pass uses the updated schedule value.
        """
        self._state["step"] += 1

    def reset(self) -> None:
        """Reset the internal step counter to zero.

        Call before starting a new generation run to ensure the schedule
        starts from step 0.
        """
        self._state["step"] = 0

    # ------------------------------------------------------------------ #
    # Hook registration
    # ------------------------------------------------------------------ #

    def register_scheduled_hooks(
        self,
        model: nn.Module,
        target_layers: List[int],
        total_steps: int,
    ) -> List[Any]:
        """Register time-varying steering hooks on *target_layers* of *model*.

        At each forward pass the hook:

        1. Reads ``self._state["step"]`` to get the current step.
        2. Computes ``scale = schedule(step, total_steps)``.
        3. Gets the combined steering vector from the wrapped
           :class:`MultiConceptSteerer`.
        4. Applies ``h' = ReNorm(h + scale * combined, h)``.

        Layer resolution follows the same logic as
        :meth:`~steer_audio.multi_steer.MultiConceptSteerer.register_hooks`:
        tries ``model.transformer_blocks[i]``, then
        ``list(model.children())[i]``, then *model* itself.

        Args:
            model:         PyTorch module.
            target_layers: List of layer indices.
            total_steps:   Total denoising steps (passed to the schedule).

        Returns:
            List of hook handles.  Pass to
            ``multi_steerer.remove_hooks(handles)`` to clean up.
        """
        children = list(model.children())
        handles: List[Any] = []

        for layer_idx in target_layers:
            # Resolve the target sub-module (same logic as MultiConceptSteerer).
            if hasattr(model, "transformer_blocks"):
                blocks = model.transformer_blocks  # type: ignore[attr-defined]
                if layer_idx < len(blocks):
                    target_mod: nn.Module = blocks[layer_idx]
                else:
                    log.warning(
                        "Layer %d out of range for transformer_blocks (len=%d); "
                        "registering hook on model root.",
                        layer_idx,
                        len(blocks),
                    )
                    target_mod = model
            elif layer_idx < len(children):
                target_mod = children[layer_idx]
            else:
                log.warning(
                    "Layer %d out of range for children (len=%d); "
                    "registering hook on model root.",
                    layer_idx,
                    len(children),
                )
                target_mod = model

            def _make_hook(l_idx: int) -> Any:
                # Capture l_idx; close over self / total_steps via outer scope.
                def hook(
                    module: nn.Module,
                    inputs: tuple,
                    output: Any,
                ) -> Any:
                    step = self._state["step"]
                    scale = self.schedule_fn(step, total_steps)

                    if isinstance(output, tuple):
                        h, *rest = output
                    else:
                        h = output
                        rest = None

                    h_orig = h.detach().clone()

                    # Get the combined (alpha-weighted) vector and scale it.
                    combined = self.multi_steerer.get_combined_vectors(l_idx)
                    v = (combined * scale).to(h.device)

                    h_out = _renorm(h.float() + v, h_orig.float())
                    h_out = h_out.to(h.dtype)

                    if rest is not None:
                        return (h_out, *rest)
                    return h_out

                return hook

            handle = target_mod.register_forward_hook(_make_hook(layer_idx))
            handles.append(handle)
            log.debug(
                "Registered scheduled hook on layer %d (%s).",
                layer_idx,
                type(target_mod).__name__,
            )

        return handles
