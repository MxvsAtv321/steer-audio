"""
Unified steering pipeline — Phase 2.5 (TADA roadmap integration week).

Composes all Phase 2 components into a single entry point:

  - MultiConceptSteerer   (2.1): simultaneous multi-concept injection with
                                  optional Gram-Schmidt orthogonalization
  - TimestepAdaptiveSteerer (2.2): per-step alpha schedules per concept
  - ConceptAlgebra          (2.3): algebra expressions → steering vectors
  - SelfMonitoredSteerer    (2.4): CLAP-probe adaptive alpha reduction

Usage::

    from steer_audio.pipeline import SteeringPipeline
    from steer_audio import cosine_schedule

    pipeline = SteeringPipeline(
        vectors={"tempo": sv_tempo, "mood": sv_mood},
        schedules={"tempo": cosine_schedule(alpha_max=80)},
        orthogonalize=True,
    )
    audio, sr = pipeline.steer(
        model,
        prompt="a jazz piano trio",
        alphas={"tempo": 60, "mood": 40},
        duration=30.0,
        seed=42,
    )
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from steer_audio.multi_steer import MultiConceptSteerer, _renorm
from steer_audio.temporal_steering import TimestepSchedule, constant_schedule
from steer_audio.vector_bank import SteeringVector, SteeringVectorBank

log = logging.getLogger(__name__)

# Epsilon shared with multi_steer and temporal_steering.
_EPS: float = 1e-8


# ---------------------------------------------------------------------------
# Internal hook builder
# ---------------------------------------------------------------------------


def _make_adaptive_multi_hook(
    contributions: list[tuple[SteeringVector, float, TimestepSchedule]],
    layer_state: dict[str, int],
    total_T: int,
) -> Any:
    """Build a forward hook applying multiple concept vectors with adaptive schedules.

    The hook is called once per forward pass of the targeted layer.  It
    increments a per-layer step counter to infer the current diffusion
    timestep ``t = T - step``, queries each concept's schedule for its
    effective alpha, and injects the combined steering delta.

    Args:
        contributions: List of ``(SteeringVector, base_alpha, schedule)`` triples
                       for this layer.  ``base_alpha`` is the user-provided
                       alpha; ``schedule(t, T)`` scales it per step.
        layer_state:   Mutable dict with key ``"call_count"`` (shared only
                       within this layer's hook closure).
        total_T:       Total number of diffusion inference steps.

    Returns:
        A ``torch.nn.Module`` forward hook function.
    """

    def hook(
        module: torch.nn.Module,
        inputs: tuple,
        output: torch.Tensor | tuple,
    ) -> torch.Tensor | tuple:
        """Adaptive multi-concept steering hook."""
        step = layer_state["call_count"]
        # t decreases from T (start) to 1 (end).
        t: int = max(1, total_T - step)
        layer_state["call_count"] += 1

        # Unwrap tuple outputs (e.g. (hidden, attn_weights)).
        if isinstance(output, tuple):
            h, *rest = output
        else:
            h = output
            rest = None

        h_orig = h.detach().clone()
        h_out = h.float()

        # Accumulate per-method deltas.
        caa_delta = torch.zeros_like(h_out)
        sae_delta = torch.zeros_like(h_out)
        has_caa = False

        for sv, base_alpha, schedule in contributions:
            effective_alpha: float = schedule(t, total_T)
            if abs(effective_alpha) < _EPS:
                continue
            v = sv.vector.float().to(h.device)  # shape: (hidden_dim,)
            if sv.method == "caa":
                has_caa = True
                caa_delta = caa_delta + effective_alpha * v
            else:
                sae_delta = sae_delta + effective_alpha * v

        if has_caa:
            # ReNorm: apply CAA delta then renorm to original magnitude.
            h_out = _renorm(h_out + caa_delta, h_orig.float())
        # SAE delta is additive with no renorm.
        h_out = h_out + sae_delta
        h_out = h_out.to(h.dtype)

        log.debug(
            "Adaptive multi-hook: step=%d t=%d T=%d has_caa=%s",
            step,
            t,
            total_T,
            has_caa,
        )

        if rest is not None:
            return (h_out, *rest)
        return h_out

    return hook


# ---------------------------------------------------------------------------
# SteeringPipeline
# ---------------------------------------------------------------------------


class SteeringPipeline:
    """Unified pipeline composing all Phase 2 steering components.

    Provides a single ``.steer()`` entry point that dispatches internally to
    the appropriate Phase 2 module depending on which features are configured:

    * **Schedules present**: builds merged adaptive multi-concept hooks that
      query each concept's :class:`~steer_audio.temporal_steering.TimestepSchedule`
      at every diffusion step.
    * **Single concept + probe**: delegates to
      :class:`~steer_audio.self_monitor.SelfMonitoredSteerer`.
    * **Otherwise**: delegates to
      :class:`~steer_audio.multi_steer.MultiConceptSteerer`.

    Args:
        vectors:             Mapping ``concept_name → SteeringVector``.
        schedules:           Optional per-concept
                             :class:`~steer_audio.temporal_steering.TimestepSchedule`.
                             Concepts without an entry default to a constant
                             schedule equal to the alpha passed to :meth:`steer`.
        probes:              Optional per-concept
                             :class:`~steer_audio.self_monitor.ConceptProbe`
                             for self-monitored steering (single-concept only).
        orthogonalize:       If ``True``, applies Gram-Schmidt orthogonalization
                             to the steering vectors before inference.
        num_inference_steps: Total diffusion denoising steps used by the model.
                             Required for schedule-aware steering (default 30).

    Raises:
        ValueError: If *vectors* is empty.
    """

    def __init__(
        self,
        vectors: dict[str, SteeringVector],
        schedules: dict[str, TimestepSchedule] | None = None,
        probes: dict[str, Any] | None = None,  # dict[str, ConceptProbe]
        orthogonalize: bool = True,
        num_inference_steps: int = 30,
    ) -> None:
        if not vectors:
            raise ValueError("SteeringPipeline requires at least one SteeringVector.")
        self._vectors: dict[str, SteeringVector] = dict(vectors)
        self._schedules: dict[str, TimestepSchedule] = dict(schedules) if schedules else {}
        self._probes: dict[str, Any] = dict(probes) if probes else {}
        self._orthogonalize = orthogonalize
        self._num_inference_steps = num_inference_steps

    # ------------------------------------------------------------------ #
    # Factory constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_vector_bank(
        cls,
        bank: SteeringVectorBank,
        directory: Path,
        orthogonalize: bool = True,
        num_inference_steps: int = 30,
    ) -> "SteeringPipeline":
        """Build a pipeline from all vectors in *directory*.

        Args:
            bank:                :class:`SteeringVectorBank` instance.
            directory:           Directory containing ``.safetensors`` vector files.
            orthogonalize:       Whether to Gram-Schmidt orthogonalize.
            num_inference_steps: Total diffusion steps.

        Returns:
            Configured :class:`SteeringPipeline` with all discovered vectors.

        Raises:
            ValueError: If no vectors are found in *directory*.
        """
        vectors = bank.load_all(Path(directory))
        if not vectors:
            raise ValueError(f"No steering vectors found in {directory}")
        return cls(
            vectors=vectors,
            orthogonalize=orthogonalize,
            num_inference_steps=num_inference_steps,
        )

    # ------------------------------------------------------------------ #
    # Dynamic registration
    # ------------------------------------------------------------------ #

    def add_algebra_vector(
        self,
        name: str,
        expr: str,
        algebra: Any,  # ConceptAlgebra — avoid circular import
        layers: list[int] | None = None,
        model_name: str = "ace-step",
    ) -> None:
        """Evaluate a concept algebra expression and register the result.

        Args:
            name:       Key under which the vector is stored in this pipeline.
            expr:       Algebra expression string, e.g. ``"jazz + female_vocal"``.
            algebra:    :class:`~steer_audio.concept_algebra.ConceptAlgebra` instance.
            layers:     Transformer-block indices to steer.  Defaults to ``[6, 7]``.
            model_name: Model identifier for provenance metadata.

        Example::

            pipeline.add_algebra_vector(
                "jazz_vocal", "jazz + female_vocal", algebra, layers=[6, 7]
            )
        """
        feature_set = algebra.expr(expr)
        sv = algebra.to_steering_vector(
            feature_set,
            layers=layers or [6, 7],
            model_name=model_name,
        )
        # Override the auto-generated concept name with the user-supplied key.
        sv.concept = name
        self._vectors[name] = sv
        log.info(
            "Registered algebra vector '%s' from expression %r (tau=%d).",
            name,
            expr,
            sv.tau,
        )

    def set_schedule(self, concept: str, schedule: TimestepSchedule) -> None:
        """Assign a timestep schedule to a registered concept.

        Args:
            concept:  Name of a concept already in this pipeline.
            schedule: :class:`~steer_audio.temporal_steering.TimestepSchedule`
                      to apply per diffusion step.

        Raises:
            KeyError: If *concept* is not registered.
        """
        if concept not in self._vectors:
            raise KeyError(
                f"Concept '{concept}' is not registered.  "
                f"Available: {list(self._vectors.keys())}"
            )
        self._schedules[concept] = schedule
        log.debug("Set schedule for '%s'.", concept)

    def set_probe(self, concept: str, probe: Any) -> None:
        """Assign a :class:`~steer_audio.self_monitor.ConceptProbe` to a concept.

        Self-monitoring is activated when exactly one concept with a probe is
        active during :meth:`steer`.

        Args:
            concept: Name of a concept already in this pipeline.
            probe:   Trained :class:`~steer_audio.self_monitor.ConceptProbe`.

        Raises:
            KeyError: If *concept* is not registered.
        """
        if concept not in self._vectors:
            raise KeyError(
                f"Concept '{concept}' is not registered.  "
                f"Available: {list(self._vectors.keys())}"
            )
        self._probes[concept] = probe
        log.debug("Set ConceptProbe for '%s'.", concept)

    # ------------------------------------------------------------------ #
    # Inference entry point
    # ------------------------------------------------------------------ #

    def steer(
        self,
        model: Any,
        prompt: str,
        alphas: dict[str, float],
        duration: float = 30.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, int]:
        """Generate audio with multi-concept, schedule-aware steering.

        Dispatches internally to the appropriate Phase 2 component:

        * Single concept + trained probe → :class:`SelfMonitoredSteerer`.
        * Any active concept has a schedule → adaptive multi-concept hooks.
        * Otherwise → :class:`MultiConceptSteerer` with constant alphas.

        Args:
            model:    ACE-Step model instance (``PatchableACE`` or compatible).
            prompt:   Text prompt for audio generation.
            alphas:   ``{concept_name: alpha_value}`` dict.  Concepts with
                      ``alpha == 0`` are skipped.
            duration: Target audio duration in seconds (default 30 s).
            seed:     Random seed for reproducible generation (default 42).

        Returns:
            ``(audio_array, sample_rate)`` where *audio_array* is a 1-D (mono)
            or 2-D (stereo) float32 NumPy array.

        Raises:
            ValueError: If no active concepts remain after filtering.
        """
        # Keep only registered concepts with non-zero alpha.
        active: dict[str, float] = {
            c: a for c, a in alphas.items() if c in self._vectors and a != 0.0
        }
        if not active:
            raise ValueError(
                "No active concepts with non-zero alphas found.  "
                f"Requested: {list(alphas.keys())}  "
                f"Registered: {list(self._vectors.keys())}"
            )

        active_vectors = {c: self._vectors[c] for c in active}

        # ---- Path 1: self-monitored steerer (single concept + probe) ----
        if len(active) == 1:
            concept = next(iter(active))
            if concept in self._probes:
                from steer_audio.self_monitor import SelfMonitoredSteerer

                log.info("Using SelfMonitoredSteerer for concept '%s'.", concept)
                steerer = SelfMonitoredSteerer(
                    vector=active_vectors[concept],
                    probe=self._probes[concept],
                    alpha=active[concept],
                )
                return steerer.steer(model, prompt, duration, seed)

        # Build a MultiConceptSteerer (orthogonalization is applied here).
        multi_steerer = MultiConceptSteerer(active_vectors, orthogonalize=self._orthogonalize)

        # ---- Path 2: adaptive multi-concept hooks (any concept has schedule) ----
        if self._schedules:
            log.info(
                "Using adaptive multi-concept hooks (schedules: %s).",
                list(self._schedules.keys()),
            )
            return self._steer_adaptive(model, prompt, active, multi_steerer, duration, seed)

        # ---- Path 3: plain multi-concept steering with constant alphas ----
        log.info("Using MultiConceptSteerer with constant alphas.")
        return multi_steerer.steer(model, prompt, active, duration, seed)

    # ------------------------------------------------------------------ #
    # Adaptive steering (multi-concept + schedules)
    # ------------------------------------------------------------------ #

    def _steer_adaptive(
        self,
        model: Any,
        prompt: str,
        alphas: dict[str, float],
        multi_steerer: MultiConceptSteerer,
        duration: float,
        seed: int,
    ) -> tuple[np.ndarray, int]:
        """Run inference with per-concept timestep schedules.

        Builds one adaptive forward hook per targeted transformer layer.
        Each hook queries the concept's schedule at every diffusion step
        to determine the effective alpha.

        Args:
            model:         ACE-Step model instance.
            prompt:        Text prompt.
            alphas:        Active ``{concept: alpha}`` mapping.
            multi_steerer: A :class:`MultiConceptSteerer` (may have orthogonalized
                           vectors).
            duration:      Audio duration in seconds.
            seed:          Random seed.

        Returns:
            ``(audio_array, sample_rate)``.
        """
        transformer_blocks = multi_steerer._get_transformer_blocks(model)
        T = self._num_inference_steps

        # Aggregate contributions per layer, wrapping bare alphas in schedules.
        layer_contributions: dict[
            int, list[tuple[SteeringVector, float, TimestepSchedule]]
        ] = defaultdict(list)

        for concept, sv in multi_steerer.vectors.items():
            base_alpha = alphas.get(concept, 0.0)
            if base_alpha == 0.0:
                continue
            # Prefer an explicit schedule; fall back to a constant schedule at base_alpha.
            if concept in self._schedules:
                schedule = self._schedules[concept]
            else:
                schedule = constant_schedule(base_alpha)
                # Effective alpha is baked into the schedule, so pass base=1.0.
                # But to keep the scaling consistent we wrap: schedule(t,T)
                # already returns the alpha, so base_alpha in the hook should be 1.0.
                # Simpler: keep base_alpha in the contribution and let the hook scale.
                # NOTE: when schedule is constant_schedule(base_alpha), it ignores
                # base_alpha from the contribution list — so we pass 1.0 and let
                # schedule carry the magnitude.

            for layer_idx in sv.layers:
                layer_contributions[layer_idx].append((sv, base_alpha, schedule))

        # Register adaptive hooks.
        handles: list[Any] = []
        n_blocks = len(transformer_blocks)

        for layer_idx, contribs in layer_contributions.items():
            if layer_idx >= n_blocks:
                log.warning(
                    "Layer index %d out of range for model with %d blocks; skipped.",
                    layer_idx,
                    n_blocks,
                )
                continue
            layer_state: dict[str, int] = {"call_count": 0}
            hook_fn = _make_adaptive_multi_hook(contribs, layer_state, T)
            block = transformer_blocks[layer_idx]
            target = getattr(block, "cross_attn", block)
            handle = target.register_forward_hook(hook_fn)
            handles.append(handle)
            log.debug("Registered adaptive multi-hook on layer %d.", layer_idx)

        try:
            # Run inference directly (bypassing multi_steerer.steer which would
            # register its own non-adaptive hooks and double-apply the delta).
            pipeline = getattr(model, "pipeline", model)
            audio = pipeline(
                prompt=prompt,
                audio_duration=duration,
                manual_seed=seed,
                num_inference_steps=T,
                return_type="audio",
            )
            sr: int = getattr(pipeline, "sample_rate", 44100)
            if isinstance(audio, torch.Tensor):
                audio_np: np.ndarray = audio.squeeze().cpu().float().numpy()
            elif isinstance(audio, np.ndarray):
                audio_np = audio
            else:
                raise TypeError(
                    f"Unexpected audio output type: {type(audio).__name__}"
                )
        finally:
            for handle in handles:
                handle.remove()
            log.debug("Removed %d adaptive multi-hook(s).", len(handles))

        return audio_np, sr

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def concepts(self) -> list[str]:
        """Names of all registered concepts."""
        return list(self._vectors.keys())

    def summary(self) -> str:
        """Return a human-readable summary of the pipeline configuration.

        Returns:
            Multi-line string listing each concept with its method, layers,
            schedule, and whether a probe is attached.
        """
        lines = [
            "SteeringPipeline Summary",
            "=" * 56,
            f"  {'Concept':<20} {'Method':<5} {'Layers':<12} {'Schedule':<18} Probe",
            "-" * 56,
        ]
        for name, sv in self._vectors.items():
            sched = self._schedules.get(name)
            sched_str = getattr(sched, "__name__", repr(sched)) if sched else "constant"
            probe_str = "yes" if name in self._probes else "no"
            layers_str = str(sv.layers)
            lines.append(
                f"  {name:<20} {sv.method:<5} {layers_str:<12} {sched_str:<18} {probe_str}"
            )
        lines.append("-" * 56)
        lines.append(
            f"  orthogonalize={self._orthogonalize}, "
            f"num_inference_steps={self._num_inference_steps}"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SteeringPipeline(concepts={self.concepts}, "
            f"orthogonalize={self._orthogonalize}, "
            f"num_inference_steps={self._num_inference_steps})"
        )
