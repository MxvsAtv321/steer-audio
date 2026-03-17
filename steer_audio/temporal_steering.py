"""
Timestep-adaptive steering schedules for diffusion-based audio generation.

Phase 2, Prompt 2.2 — TADA roadmap (arXiv 2602.11910).

Scientific motivation
---------------------
Early diffusion timesteps (high noise, t ≈ T) determine global structure
(genre, mood, tempo).  Later timesteps (t ≈ 0) refine fine details (timbre,
production quality).  A cosine schedule that applies maximum alpha at the
start and tapers to zero at the end improves audio preservation (lower LPAPS)
while maintaining concept alignment (ΔAlignment CLAP).

Timestep convention
-------------------
``t`` is the *noise-level timestep*, decreasing from T (maximum noise, start
of diffusion) to 0 (clean signal, end of diffusion).  Given a forward-hook
call counter (0-indexed *step*) and *total_T* steps:

    t = total_T - step          (step 0 → t=T; step T-1 → t=1)

Schedule functions accept ``(t: int, T: int)`` and return a scalar alpha.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from typing import Any, Callable, Generator, Protocol, runtime_checkable

import numpy as np
import torch

from steer_audio.vector_bank import SteeringVector

log = logging.getLogger(__name__)

# Small epsilon to guard against zero-norm divisions.
_EPS: float = 1e-8


# ---------------------------------------------------------------------------
# TimestepSchedule protocol
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


# ---------------------------------------------------------------------------
# Built-in schedules
# ---------------------------------------------------------------------------


def constant_schedule(alpha: float) -> TimestepSchedule:
    """Return *alpha* regardless of the current timestep.

    Args:
        alpha: Fixed alpha value.

    Returns:
        A :class:`TimestepSchedule` that always returns *alpha*.

    Example::

        sched = constant_schedule(60.0)
        assert sched(15, 30) == 60.0
    """

    def _schedule(t: int, T: int) -> float:  # noqa: D401
        """Constant schedule: return ``alpha`` at every timestep."""
        return alpha

    return _schedule


def cosine_schedule(alpha_max: float, alpha_min: float = 0.0) -> TimestepSchedule:
    """Alpha follows a cosine curve from *alpha_max* (start) to *alpha_min* (end).

    The peak occurs at ``t/T = 1.0`` (maximum noise, start of diffusion) and
    the trough at ``t/T = 0.0`` (clean signal, end of diffusion).

    Formula::

        alpha(t) = alpha_min + (alpha_max - alpha_min)
                   * 0.5 * (1 + cos(π * (1 - t/T)))

    Args:
        alpha_max: Maximum alpha applied at the very first diffusion step.
        alpha_min: Minimum alpha applied at the very last diffusion step.
                   Defaults to ``0.0``.

    Returns:
        A :class:`TimestepSchedule` following a half-cosine decay.

    Example::

        sched = cosine_schedule(80.0)
        assert abs(sched(30, 30) - 80.0) < 1e-6  # peak at start
        assert abs(sched(0, 30) - 0.0) < 1e-6   # trough at end
    """

    def _schedule(t: int, T: int) -> float:
        """Cosine-decaying schedule from alpha_max (t=T) to alpha_min (t=0)."""
        if T == 0:
            return float(alpha_max)
        # t/T ∈ [0, 1]; t=T → factor=1 (alpha_max), t=0 → factor=0 (alpha_min)
        factor = 0.5 * (1.0 + math.cos(math.pi * (1.0 - t / T)))
        return float(alpha_min + (alpha_max - alpha_min) * factor)

    return _schedule


def early_only_schedule(alpha: float, cutoff: float = 0.5) -> TimestepSchedule:
    """Apply *alpha* only during the early (high-noise) portion of diffusion.

    Steering is active while ``t/T > cutoff`` (first half of the process when
    cutoff=0.5) and zero otherwise.  This focuses semantic edits on the stage
    where global structure is laid down.

    Args:
        alpha:  Alpha applied during early timesteps.
        cutoff: Fractional threshold (default ``0.5`` → first half).

    Returns:
        A :class:`TimestepSchedule` that is non-zero only for early steps.

    Example::

        sched = early_only_schedule(60.0, cutoff=0.5)
        assert sched(20, 30) == 60.0   # t/T ≈ 0.67 > 0.5 → active
        assert sched(10, 30) == 0.0    # t/T ≈ 0.33 ≤ 0.5 → inactive
    """

    def _schedule(t: int, T: int) -> float:
        """Early-only schedule: active for t/T > cutoff."""
        if T == 0:
            return 0.0
        return float(alpha) if t / T > cutoff else 0.0

    return _schedule


def late_only_schedule(alpha: float, cutoff: float = 0.5) -> TimestepSchedule:
    """Apply *alpha* only during the late (low-noise) refinement portion.

    Steering is active while ``t/T <= cutoff`` (second half when cutoff=0.5).
    This is useful for fine-grained edits that should not affect global
    structure (e.g., targeting specific timbral details).

    Args:
        alpha:  Alpha applied during late timesteps.
        cutoff: Fractional threshold (default ``0.5`` → second half).

    Returns:
        A :class:`TimestepSchedule` that is non-zero only for late steps.

    Example::

        sched = late_only_schedule(60.0, cutoff=0.5)
        assert sched(10, 30) == 60.0   # t/T ≈ 0.33 ≤ 0.5 → active
        assert sched(20, 30) == 0.0    # t/T ≈ 0.67 > 0.5 → inactive
    """

    def _schedule(t: int, T: int) -> float:
        """Late-only schedule: active for t/T <= cutoff."""
        if T == 0:
            return 0.0
        return float(alpha) if t / T <= cutoff else 0.0

    return _schedule


def linear_schedule(alpha_start: float, alpha_end: float = 0.0) -> TimestepSchedule:
    """Alpha decreases linearly from *alpha_start* (t=T) to *alpha_end* (t=0).

    Args:
        alpha_start: Alpha at the first diffusion step (t=T).
        alpha_end:   Alpha at the last diffusion step (t=0).  Default ``0.0``.

    Returns:
        A :class:`TimestepSchedule` with linear decay.

    Example::

        sched = linear_schedule(60.0)
        assert abs(sched(30, 30) - 60.0) < 1e-6
        assert abs(sched(0, 30) - 0.0) < 1e-6
    """

    def _schedule(t: int, T: int) -> float:
        """Linear schedule: decreases from alpha_start to alpha_end."""
        if T == 0:
            return float(alpha_start)
        frac = t / T  # 1 at start, 0 at end
        return float(alpha_end + (alpha_start - alpha_end) * frac)

    return _schedule


# ---------------------------------------------------------------------------
# Internal renorm helper (mirrors multi_steer._renorm)
# ---------------------------------------------------------------------------


def _renorm(h_steered: torch.Tensor, h_orig: torch.Tensor) -> torch.Tensor:
    """Renormalize *h_steered* to match the per-token L2 norm of *h_orig*.

    ReNorm(h', h) = h' / ||h'||₂ · ||h||₂  (broadcast over the last dim)

    Args:
        h_steered: Steered activation, shape ``(..., dim)``.
        h_orig:    Original activation (same shape).

    Returns:
        Activation with same shape and per-token L2 norm as *h_orig*.
    """
    orig_norm = h_orig.float().norm(dim=-1, keepdim=True)
    steered_norm = h_steered.float().norm(dim=-1, keepdim=True)
    return (h_steered.float() / (steered_norm + _EPS)) * orig_norm


# ---------------------------------------------------------------------------
# TimestepAdaptiveSteerer
# ---------------------------------------------------------------------------


class TimestepAdaptiveSteerer:
    """Apply a steering vector with a timestep-dependent alpha schedule.

    The hook reads the call-counter incremented per forward pass to infer the
    current diffusion timestep ``t``.  Given *num_inference_steps = T*, the
    hook on its ``k``-th call (0-indexed) maps ``t = T - k`` and queries
    ``schedule(t, T)`` for the effective alpha.

    This covers the complete TADA "per-step adaptive" steering interface:

    * ``constant_schedule``   — reproduce baseline fixed-alpha behaviour
    * ``cosine_schedule``     — taper alpha as denoising progresses
    * ``early_only_schedule`` — concentrate edits in the noisy/structural phase
    * ``late_only_schedule``  — concentrate edits in the refinement phase

    Args:
        vector:   A pre-computed :class:`~steer_audio.vector_bank.SteeringVector`.
        schedule: A :class:`TimestepSchedule` (callable ``(t, T) → float``).
        layers:   Transformer-block indices to steer.  Defaults to
                  ``vector.layers`` if ``None``.

    Example::

        from steer_audio import SteeringVector
        from steer_audio.temporal_steering import TimestepAdaptiveSteerer, cosine_schedule

        sv = SteeringVector(concept="tempo", method="caa",
                            model_name="ace-step", layers=[6, 7],
                            vector=torch.randn(3072))
        steerer = TimestepAdaptiveSteerer(sv, cosine_schedule(alpha_max=80))
        audio, sr = steerer.steer(model, "a jazz song", duration=30.0, seed=42)
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

    # ------------------------------------------------------------------ #
    # Hook factory
    # ------------------------------------------------------------------ #

    def _make_hook(
        self,
        schedule: TimestepSchedule,
        vector: SteeringVector,
        layer_state: dict[str, int],
        total_T: int,
    ) -> Callable:
        """Build a forward hook for a single transformer layer.

        The hook uses *layer_state* (a mutable dict shared only within this
        layer) to count calls and infer the current diffusion timestep.  Each
        layer maintains its own counter so that multi-layer configurations
        remain correct.

        Args:
            schedule:    The alpha schedule ``(t, T) → float``.
            vector:      The steering vector to apply.
            layer_state: Mutable dict with key ``"call_count"`` (starts at 0).
            total_T:     Total number of inference steps.

        Returns:
            A ``torch.nn.Module`` forward hook function.
        """
        sv = vector

        def hook(
            module: torch.nn.Module,
            inputs: tuple,
            output: torch.Tensor | tuple,
        ) -> torch.Tensor | tuple:
            """Forward hook: inject adaptive steering into cross-attn output."""
            step = layer_state["call_count"]
            # Map step counter to noise-level timestep t ∈ [T, 1]
            # step=0 → t=T (start, max noise); step=T-1 → t=1 (near clean)
            t: int = max(1, total_T - step)
            effective_alpha: float = schedule(t, total_T)
            layer_state["call_count"] += 1

            if abs(effective_alpha) < _EPS:
                # Schedule returned ~0 — skip hook to avoid unnecessary compute.
                return output

            # Unwrap tuple outputs (e.g. ``(hidden, attn_weights)``).
            if isinstance(output, tuple):
                h, *rest = output
            else:
                h = output
                rest = None

            h_orig = h.detach().clone()
            v = sv.vector.float().to(h.device)  # shape: (hidden_dim,)

            h_out = h.float()
            if sv.method == "caa":
                # CAA steering: add delta, then renorm to preserve activation magnitude.
                h_out = _renorm(h_out + effective_alpha * v, h_orig.float())
            else:
                # SAE steering: additive only, no renorm.
                h_out = h_out + effective_alpha * v

            h_out = h_out.to(h.dtype)

            log.debug(
                "Adaptive hook: layer step=%d t=%d T=%d α=%.2f method=%s",
                step,
                t,
                total_T,
                effective_alpha,
                sv.method,
            )

            if rest is not None:
                return (h_out, *rest)
            return h_out

        return hook

    # ------------------------------------------------------------------ #
    # Context manager: register / deregister hooks
    # ------------------------------------------------------------------ #

    @contextmanager
    def _hooked(
        self,
        transformer_blocks: torch.nn.ModuleList,
        total_T: int,
    ) -> Generator[None, None, None]:
        """Context manager: register per-layer adaptive hooks, yield, remove.

        Args:
            transformer_blocks: ``model.transformer_blocks`` ModuleList.
            total_T:            Total number of inference timesteps.
        """
        handles: list[Any] = []
        n_blocks = len(transformer_blocks)

        for layer_idx in self.layers:
            if layer_idx >= n_blocks:
                log.warning(
                    "Layer index %d is out of range for model with %d blocks; skipped.",
                    layer_idx,
                    n_blocks,
                )
                continue

            # Each layer gets its own independent call counter.
            layer_state: dict[str, int] = {"call_count": 0}
            hook_fn = self._make_hook(
                self.schedule, self.vector, layer_state, total_T
            )

            block = transformer_blocks[layer_idx]
            target = getattr(block, "cross_attn", block)
            handle = target.register_forward_hook(hook_fn)
            handles.append(handle)
            log.debug(
                "Registered adaptive hook on layer %d (%s).",
                layer_idx,
                type(target).__name__,
            )

        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            log.debug("Removed %d adaptive steering hook(s).", len(handles))

    # ------------------------------------------------------------------ #
    # Inference entry point
    # ------------------------------------------------------------------ #

    def steer(
        self,
        model: Any,
        prompt: str,
        duration: float = 30.0,
        seed: int = 42,
        num_inference_steps: int = 30,
    ) -> tuple[np.ndarray, int]:
        """Generate audio with timestep-adaptive concept steering.

        Registers per-layer forward hooks that apply the schedule function
        at each diffusion step, runs the diffusion pipeline, then removes
        all hooks.

        Args:
            model:               ACE-Step model (``PatchableACE`` or similar).
            prompt:              Text prompt for audio generation.
            duration:            Target audio duration in seconds.
            seed:                Random seed for reproducible generation.
            num_inference_steps: Total diffusion denoising steps (= T).

        Returns:
            ``(audio_array, sample_rate)`` where *audio_array* is a
            1-D (mono) or 2-D (stereo) NumPy float array.

        Raises:
            AttributeError: If the expected model attribute tree is not found.
        """
        transformer_blocks = self._get_transformer_blocks(model)

        with self._hooked(transformer_blocks, total_T=num_inference_steps):
            audio, sr = self._run_inference(
                model, prompt, duration, seed, num_inference_steps
            )

        return audio, sr

    # ------------------------------------------------------------------ #
    # Model interface helpers
    # ------------------------------------------------------------------ #

    def _get_transformer_blocks(self, model: Any) -> torch.nn.ModuleList:
        """Retrieve the transformer block ModuleList from *model*.

        Tries attribute paths in order:
        1. ``model.patchable_model.ace_step_transformer.transformer_blocks``
        2. ``model.ace_step_transformer.transformer_blocks``
        3. ``model.transformer_blocks``

        Args:
            model: Model object with transformer blocks.

        Returns:
            The ``transformer_blocks`` :class:`torch.nn.ModuleList`.

        Raises:
            AttributeError: If none of the expected paths are found.
        """
        candidates = [
            lambda m: m.patchable_model.ace_step_transformer.transformer_blocks,
            lambda m: m.ace_step_transformer.transformer_blocks,
            lambda m: m.transformer_blocks,
        ]
        for getter in candidates:
            try:
                blocks = getter(model)
                if blocks is not None:
                    return blocks
            except AttributeError:
                continue

        raise AttributeError(
            "Cannot find transformer_blocks on model.  Expected one of:\n"
            "  model.patchable_model.ace_step_transformer.transformer_blocks\n"
            "  model.ace_step_transformer.transformer_blocks\n"
            "  model.transformer_blocks\n"
            f"Got model type: {type(model).__name__}"
        )

    def _run_inference(
        self,
        model: Any,
        prompt: str,
        duration: float,
        seed: int,
        num_inference_steps: int,
    ) -> tuple[np.ndarray, int]:
        """Run one forward pass of the diffusion pipeline.

        Supports the ``SteeredACEStepPipeline`` / ``SimpleACEStepPipeline``
        interface.  Override for other models.

        Args:
            model:               Model with a ``.pipeline`` attribute.
            prompt:              Text prompt.
            duration:            Audio duration in seconds.
            seed:                Random seed.
            num_inference_steps: Total denoising steps.

        Returns:
            ``(audio_array, sample_rate)``.
        """
        pipeline = getattr(model, "pipeline", model)

        audio = pipeline(
            prompt=prompt,
            audio_duration=duration,
            manual_seed=seed,
            num_inference_steps=num_inference_steps,
            return_type="audio",
        )

        sr: int = getattr(pipeline, "sample_rate", 44100)

        if isinstance(audio, torch.Tensor):
            audio_np: np.ndarray = audio.squeeze().cpu().float().numpy()
        elif isinstance(audio, np.ndarray):
            audio_np = audio
        else:
            raise TypeError(
                f"Unexpected audio output type from pipeline: {type(audio).__name__}"
            )

        return audio_np, sr

    # ------------------------------------------------------------------ #
    # Utility: evaluate schedule over all steps
    # ------------------------------------------------------------------ #

    def schedule_values(self, num_inference_steps: int) -> list[float]:
        """Return the alpha value at every diffusion step for inspection.

        Useful for plotting the schedule before running inference.

        Args:
            num_inference_steps: Total number of denoising steps.

        Returns:
            List of length *num_inference_steps* where ``values[k]`` is the
            effective alpha at step k (k=0 = start, k=T-1 = end).
        """
        T = num_inference_steps
        return [self.schedule(max(1, T - k), T) for k in range(T)]
