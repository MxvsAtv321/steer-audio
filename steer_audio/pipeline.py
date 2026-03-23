"""
Unified steering pipeline — Prompt 2.6 (TADA roadmap).

Composes all Phase 2 components into a single entry point:

  - SteeringVectorBank   (1.5): persistent vector storage
  - MultiConceptSteerer   (2.1): simultaneous multi-concept injection
  - TimestepAdaptiveSteerer (2.2): per-step alpha schedules
  - ConceptAlgebra          (2.3): algebra expressions → steering vectors
  - SelfMonitoredSteerer    (2.4): CLAP-probe adaptive alpha reduction

New API (Prompt 2.6)::

    from steer_audio.pipeline import SteeringPipeline
    from steer_audio.vector_bank import SteeringVectorBank

    bank = SteeringVectorBank()
    pipeline = (
        SteeringPipeline(bank, model=my_model, schedule_type="cosine")
        .add_concept("tempo", alpha=60)
        .add_concept("mood", alpha=40)
    )
    result = pipeline.generate("a jazz piano trio", dry_run=True)

Legacy API (backwards-compatible)::

    pipeline = SteeringPipeline(
        vectors={"tempo": sv_tempo, "mood": sv_mood},
        schedules={"tempo": cosine_schedule(alpha_max=80)},
        orthogonalize=True,
    )
    audio, sr = pipeline.steer(model, prompt="a jazz piano trio", alphas={"tempo": 60})
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from steer_audio.multi_steer import MultiConceptSteerer, _renorm
from steer_audio.temporal_steering import (
    TimestepSchedule,
    constant_schedule,
    get_schedule,
)
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

    Supports two construction styles:

    **New (Prompt 2.6) — fluent builder API**::

        pipeline = (
            SteeringPipeline(bank, model=model, schedule_type="cosine")
            .add_concept("tempo", alpha=60)
            .add_concept("mood", alpha=40)
        )
        result = pipeline.generate("a jazz piano trio", dry_run=True)

    **Legacy — dict-of-vectors API**::

        pipeline = SteeringPipeline(
            {"tempo": sv_tempo, "mood": sv_mood},
            schedules={"tempo": cosine_schedule(alpha_max=80)},
        )
        audio, sr = pipeline.steer(model, "a jazz piano", alphas={"tempo": 60})

    Args:
        vectors_or_bank:     Either a ``dict[str, SteeringVector]`` (legacy) or a
                             :class:`SteeringVectorBank` (new API).
        model:               Optional model instance; required for non-dry-run generation.
        target_layers:       Default transformer-block indices to use when
                             ``add_concept`` does not find layer info in the bank
                             (default ``[6, 7]``).
        orthogonalize:       Gram-Schmidt orthogonalize vectors before injection.
        schedule_type:       Global default schedule type string (``"constant"``,
                             ``"cosine"``, ``"linear"``, etc.) used by
                             :func:`~steer_audio.temporal_steering.get_schedule`.
        self_monitor:        Enable self-monitoring via
                             :class:`~steer_audio.self_monitor.SelfMonitoredSteerer`.
        monitor_every_n_steps: Step interval for self-monitoring probes.
        schedules:           Legacy: per-concept schedule callables.
        probes:              Legacy: per-concept ConceptProbe objects.
        num_inference_steps: Legacy: total diffusion steps (default 30).

    Raises:
        ValueError: If no vectors are available after construction.
    """

    def __init__(
        self,
        vectors_or_bank: "dict[str, SteeringVector] | SteeringVectorBank | None" = None,
        model: Any = None,
        target_layers: list[int] | None = None,
        orthogonalize: bool = True,
        schedule_type: str = "constant",
        self_monitor: bool = False,
        monitor_every_n_steps: int = 5,
        # Legacy keyword args for backwards compatibility
        vectors: "dict[str, SteeringVector] | None" = None,
        schedules: "dict[str, TimestepSchedule] | None" = None,
        probes: "dict[str, Any] | None" = None,
        num_inference_steps: int = 30,
    ) -> None:
        # Handle legacy positional dict or new bank input.
        if isinstance(vectors_or_bank, SteeringVectorBank):
            self._bank: SteeringVectorBank | None = vectors_or_bank
            # Populate _vectors from the bank if it already contains data,
            # so that the legacy dict-like API continues to work.
            self._vectors: dict[str, SteeringVector] = dict(vectors_or_bank)
        elif isinstance(vectors_or_bank, dict):
            # Legacy: dict[str, SteeringVector]
            if not vectors_or_bank and vectors is None:
                raise ValueError(
                    "SteeringPipeline requires at least one SteeringVector."
                )
            self._bank = None
            self._vectors = dict(vectors_or_bank)
        elif vectors_or_bank is None and vectors is not None:
            # Pure legacy keyword: vectors=...
            self._bank = None
            self._vectors = dict(vectors)
        elif vectors_or_bank is None:
            # Bank-first style with no initial vectors (add_concept later).
            self._bank = None
            self._vectors = {}
        else:
            raise TypeError(
                f"vectors_or_bank must be a dict or SteeringVectorBank, "
                f"got {type(vectors_or_bank).__name__}"
            )

        # Validate: if no vectors AND no bank source, raise.
        # A SteeringVectorBank with zero entries is still valid (add_concept fills it later).
        if not self._vectors and not isinstance(vectors_or_bank, SteeringVectorBank):
            raise ValueError("SteeringPipeline requires at least one SteeringVector.")

        self._model = model
        self._target_layers: list[int] = target_layers or [6, 7]
        self._orthogonalize = orthogonalize
        self._schedule_type = schedule_type
        self._self_monitor_enabled = self_monitor
        self._monitor_every_n_steps = monitor_every_n_steps
        self._num_inference_steps = num_inference_steps

        # Per-concept state (legacy + new API shared).
        self._schedules: dict[str, TimestepSchedule] = dict(schedules) if schedules else {}
        self._probes: dict[str, Any] = dict(probes) if probes else {}
        # New in Prompt 2.6: per-concept alphas and methods (set by add_concept).
        self._alphas: dict[str, float] = {}
        self._methods: dict[str, str] = {}

        # Active hook handles (used by context manager and generate()).
        self._handles: list[Any] = []

    # ------------------------------------------------------------------ #
    # Fluent builder methods (new API — Prompt 2.6)
    # ------------------------------------------------------------------ #

    def add_concept(
        self,
        concept: str,
        alpha: float,
        method: str = "caa",
    ) -> "SteeringPipeline":
        """Register a concept with its steering alpha.

        If the concept's :class:`SteeringVector` is already in ``self._vectors``
        or can be retrieved from the bank, it is linked immediately.
        Otherwise the concept is registered as a pending entry; its vector must
        be provided via the bank before :meth:`generate` is called.

        Args:
            concept: Concept name (e.g. ``"tempo"``, ``"mood"``).
            alpha:   Steering strength.
            method:  Vector method, ``"caa"`` (default) or ``"sae"``.

        Returns:
            ``self`` for fluent chaining.
        """
        self._alphas[concept] = alpha
        self._methods[concept] = method

        # Attempt to load from bank if not already present.
        if concept not in self._vectors and self._bank is not None:
            sv = self._bank.get(concept, method)
            if sv is not None:
                self._vectors[concept] = sv

        log.debug("add_concept: '%s' alpha=%.1f method=%s", concept, alpha, method)
        return self

    def remove_concept(self, concept: str) -> "SteeringPipeline":
        """Unregister a concept from the pipeline.

        Args:
            concept: Name of a concept to remove.

        Returns:
            ``self`` for fluent chaining.
        """
        self._alphas.pop(concept, None)
        self._methods.pop(concept, None)
        self._vectors.pop(concept, None)
        self._schedules.pop(concept, None)
        self._probes.pop(concept, None)
        log.debug("remove_concept: '%s'", concept)
        return self

    def enable_self_monitoring(
        self,
        probe: Any,  # ConceptProbe
        **kwargs: Any,
    ) -> "SteeringPipeline":
        """Enable self-monitoring with the given probe for all active concepts.

        Args:
            probe:   A :class:`~steer_audio.self_monitor.ConceptProbe`.
            **kwargs: Extra kwargs forwarded to
                      :class:`~steer_audio.self_monitor.SelfMonitoredSteerer`.

        Returns:
            ``self`` for fluent chaining.
        """
        self._self_monitor_enabled = True
        self._monitor_kwargs = kwargs
        # Register probe for all current concepts.
        for concept in list(self._alphas.keys()) + list(self._vectors.keys()):
            self._probes[concept] = probe
        log.debug("enable_self_monitoring: probe attached to %d concept(s).", len(self._probes))
        return self

    # ------------------------------------------------------------------ #
    # Schedule API (supports both new and legacy signatures)
    # ------------------------------------------------------------------ #

    def set_schedule(
        self,
        concept_or_type: str,
        schedule: TimestepSchedule | None = None,
    ) -> "SteeringPipeline":
        """Set a schedule for a concept or set the global default schedule type.

        **New API** — single string argument sets the global schedule type::

            pipeline.set_schedule("cosine")

        **Legacy API** — two arguments set a per-concept schedule::

            pipeline.set_schedule("tempo", cosine_schedule(alpha_max=80))

        Args:
            concept_or_type: Either a concept name (legacy) or a schedule type
                             string (new).
            schedule:        Schedule callable (legacy only; omit for new API).

        Returns:
            ``self`` for fluent chaining.

        Raises:
            KeyError: (legacy mode) If *concept_or_type* is not a registered concept.
        """
        if schedule is None:
            # New API: update global schedule type.
            self._schedule_type = concept_or_type
            log.debug("set_schedule: global type set to '%s'.", concept_or_type)
        else:
            # Legacy API: per-concept schedule callable.
            if concept_or_type not in self._vectors:
                raise KeyError(
                    f"Concept '{concept_or_type}' is not registered.  "
                    f"Available: {list(self._vectors.keys())}"
                )
            self._schedules[concept_or_type] = schedule
            log.debug("set_schedule: per-concept schedule for '%s'.", concept_or_type)
        return self

    def set_probe(self, concept: str, probe: Any) -> "SteeringPipeline":
        """Assign a :class:`~steer_audio.self_monitor.ConceptProbe` to a concept.

        Args:
            concept: Name of a concept already in this pipeline.
            probe:   Trained :class:`~steer_audio.self_monitor.ConceptProbe`.

        Returns:
            ``self`` for fluent chaining.

        Raises:
            KeyError: If *concept* is not registered.
        """
        if concept not in self._vectors:
            raise KeyError(
                f"Concept '{concept}' is not registered.  "
                f"Available: {list(self._vectors.keys())}"
            )
        self._probes[concept] = probe
        log.debug("set_probe: probe attached to '%s'.", concept)
        return self

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
            vectors,
            orthogonalize=orthogonalize,
            num_inference_steps=num_inference_steps,
        )

    # ------------------------------------------------------------------ #
    # Dynamic registration (legacy)
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
        """
        feature_set = algebra.expr(expr)
        sv = algebra.to_steering_vector(
            feature_set,
            layers=layers or [6, 7],
            model_name=model_name,
        )
        sv.concept = name
        self._vectors[name] = sv
        log.info(
            "Registered algebra vector '%s' from expression %r (tau=%d).",
            name,
            expr,
            sv.tau,
        )

    # ------------------------------------------------------------------ #
    # New generate() entry point — Prompt 2.6
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        seed: int = 42,
        num_inference_steps: int = 60,
        audio_length_seconds: float = 30.0,
        dry_run: bool = False,
    ) -> "dict[str, Any]":
        """Generate audio with the configured concepts.

        In **dry-run** mode (``dry_run=True``) no model inference is performed.
        The method validates configuration and returns a structured dict with
        ``audio=None``.  This is useful for unit tests and integration checks
        that do not require real model weights.

        Args:
            prompt:               Text prompt for audio generation.
            seed:                 Random seed for reproducible generation.
            num_inference_steps:  Total diffusion denoising steps.
            audio_length_seconds: Target audio duration in seconds.
            dry_run:              If ``True``, skip inference and return a stub result.

        Returns:
            Dict with keys:

            - ``"audio"``:       numpy array of audio samples, or ``None`` in dry-run.
            - ``"sample_rate"``: int sample rate, or ``44100`` in dry-run.
            - ``"prompt"``:      the prompt string.
            - ``"seed"``:        the seed used.
            - ``"concepts"``:    list of active concept names.
            - ``"alphas"``:      dict of ``{concept: alpha}``.
            - ``"dry_run"``:     bool.
            - ``"interference"`` (optional): interference report if > 1 active concept.

        Raises:
            ValueError: If no active concepts are available.
            RuntimeError: If ``dry_run=False`` and no model is attached.
        """
        # Determine active concepts and alphas (merge fluent + legacy).
        alphas = {**self._alphas}
        # Fall back to any vectors with alpha=1.0 if add_concept hasn't been called.
        if not alphas and self._vectors:
            alphas = {c: 1.0 for c in self._vectors}

        active: dict[str, float] = {
            c: a for c, a in alphas.items()
            if c in self._vectors and abs(a) > _EPS
        }

        if not active:
            raise ValueError(
                "No active concepts with non-zero alphas found.  "
                f"Requested: {list(alphas.keys())}  "
                f"Registered vectors: {list(self._vectors.keys())}"
            )

        # Build schedule callable for each concept.
        global_sched = get_schedule(self._schedule_type)

        def _get_concept_schedule(concept: str) -> TimestepSchedule:
            if concept in self._schedules:
                return self._schedules[concept]
            # Wrap global schedule so it scales by the concept's base alpha.
            base_alpha = active[concept]
            raw_sched = global_sched

            def _scaled(t: int, T: int) -> float:
                return base_alpha * raw_sched(t, T)

            return _scaled

        # Compute interference report if multiple concepts.
        interference: dict[str, Any] | None = None
        if len(active) > 1:
            interference = self.get_interference_report()

        if dry_run:
            return {
                "audio": None,
                "sample_rate": 44100,
                "prompt": prompt,
                "seed": seed,
                "concepts": list(active.keys()),
                "alphas": dict(active),
                "dry_run": True,
                **({"interference": interference} if interference is not None else {}),
            }

        # Non-dry-run: requires a model.
        if self._model is None:
            raise RuntimeError(
                "generate(dry_run=False) requires a model.  "
                "Pass model= to SteeringPipeline() or set pipeline._model."
            )

        active_vectors = {c: self._vectors[c] for c in active}
        multi_steerer = MultiConceptSteerer(
            active_vectors, orthogonalize=self._orthogonalize
        )

        # Build per-concept schedules (constant schedule wraps base alpha).
        schedules_for_run: dict[str, TimestepSchedule] = {
            c: _get_concept_schedule(c) for c in active
        }

        audio_np, sr = self._run_adaptive(
            model=self._model,
            prompt=prompt,
            active=active,
            multi_steerer=multi_steerer,
            schedules=schedules_for_run,
            duration=audio_length_seconds,
            seed=seed,
            T=num_inference_steps,
        )

        return {
            "audio": audio_np,
            "sample_rate": sr,
            "prompt": prompt,
            "seed": seed,
            "concepts": list(active.keys()),
            "alphas": dict(active),
            "dry_run": False,
            **({"interference": interference} if interference is not None else {}),
        }

    # ------------------------------------------------------------------ #
    # Interference report — Prompt 2.6
    # ------------------------------------------------------------------ #

    def get_interference_report(self) -> "dict[str, Any]":
        """Return an interference report for the currently active concepts.

        Delegates to :meth:`MultiConceptSteerer.interference_report`.

        Returns:
            Dict with keys ``"concepts"``, ``"cosine_matrix"``, ``"max_cosine"``,
            ``"warnings"``, and ``"clap_deltas"``.
        """
        # Build a MultiConceptSteerer over all registered vectors with alpha ≠ 0.
        alphas = {**self._alphas}
        if not alphas and self._vectors:
            alphas = {c: 1.0 for c in self._vectors}

        active_vectors = {
            c: self._vectors[c]
            for c, a in alphas.items()
            if c in self._vectors and abs(a) > _EPS
        }

        if not active_vectors:
            return {
                "concepts": [],
                "cosine_matrix": np.zeros((0, 0), dtype=np.float32),
                "max_cosine": 0.0,
                "warnings": [],
                "clap_deltas": {},
            }

        # Dict-based init auto-populates _active with alpha=1.0 for all concepts.
        steerer = MultiConceptSteerer(
            active_vectors, orthogonalize=False  # raw geometry, no orthogonalization
        )
        return steerer.interference_report()

    # ------------------------------------------------------------------ #
    # Legacy steer() entry point
    # ------------------------------------------------------------------ #

    def steer(
        self,
        model: Any,
        prompt: str,
        alphas: dict[str, float],
        duration: float = 30.0,
        seed: int = 42,
    ) -> "tuple[np.ndarray, int]":
        """Generate audio with multi-concept, schedule-aware steering.

        **Legacy method** — prefer :meth:`generate` for new code.

        Dispatches internally to the appropriate Phase 2 component:

        * Single concept + trained probe → :class:`SelfMonitoredSteerer`.
        * Any active concept has a schedule → adaptive multi-concept hooks.
        * Otherwise → :class:`MultiConceptSteerer` with constant alphas.

        Args:
            model:    ACE-Step model instance (``PatchableACE`` or compatible).
            prompt:   Text prompt for audio generation.
            alphas:   ``{concept_name: alpha_value}`` dict.
            duration: Target audio duration in seconds.
            seed:     Random seed.

        Returns:
            ``(audio_array, sample_rate)``.

        Raises:
            ValueError: If no active concepts remain after filtering.
        """
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
    # Adaptive steering (shared by steer() and generate())
    # ------------------------------------------------------------------ #

    def _steer_adaptive(
        self,
        model: Any,
        prompt: str,
        active: dict[str, float],
        multi_steerer: MultiConceptSteerer,
        duration: float,
        seed: int,
    ) -> "tuple[np.ndarray, int]":
        """Run inference with per-concept timestep schedules (legacy path).

        Args:
            model:         ACE-Step model instance.
            prompt:        Text prompt.
            active:        Active ``{concept: alpha}`` mapping.
            multi_steerer: A :class:`MultiConceptSteerer`.
            duration:      Audio duration in seconds.
            seed:          Random seed.

        Returns:
            ``(audio_array, sample_rate)``.
        """
        T = self._num_inference_steps
        schedules: dict[str, TimestepSchedule] = {
            c: self._schedules.get(c, constant_schedule(a))
            for c, a in active.items()
        }
        return self._run_adaptive(model, prompt, active, multi_steerer, schedules, duration, seed, T)

    def _run_adaptive(
        self,
        model: Any,
        prompt: str,
        active: dict[str, float],
        multi_steerer: MultiConceptSteerer,
        schedules: dict[str, TimestepSchedule],
        duration: float,
        seed: int,
        T: int,
    ) -> "tuple[np.ndarray, int]":
        """Core adaptive inference with hook registration / cleanup.

        Args:
            model:         ACE-Step model instance.
            prompt:        Text prompt.
            active:        Active ``{concept: alpha}`` mapping.
            multi_steerer: A :class:`MultiConceptSteerer`.
            schedules:     Per-concept schedule callables.
            duration:      Audio duration in seconds.
            seed:          Random seed.
            T:             Total number of inference steps.

        Returns:
            ``(audio_array, sample_rate)``.
        """
        transformer_blocks = multi_steerer._get_transformer_blocks(model)
        n_blocks = len(transformer_blocks)

        layer_contributions: dict[
            int, list[tuple[SteeringVector, float, TimestepSchedule]]
        ] = defaultdict(list)

        for concept, sv in multi_steerer.vectors.items():
            base_alpha = active.get(concept, 0.0)
            if abs(base_alpha) < _EPS:
                continue
            sched = schedules.get(concept, constant_schedule(base_alpha))
            for layer_idx in sv.layers:
                layer_contributions[layer_idx].append((sv, base_alpha, sched))

        handles: list[Any] = []
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

        # Store on self so __exit__ can also clean up.
        self._handles = handles
        try:
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
            self._handles = []
            log.debug("Removed %d adaptive multi-hook(s).", len(handles))

        return audio_np, sr

    # ------------------------------------------------------------------ #
    # Context manager — Prompt 2.6
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "SteeringPipeline":
        """Enter the pipeline context — returns self."""
        self._handles = []
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the pipeline context — remove any lingering hooks."""
        for handle in self._handles:
            handle.remove()
        if self._handles:
            log.debug("Context manager: removed %d lingering hook(s).", len(self._handles))
        self._handles = []
        return False  # do not suppress exceptions

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
            f"  {'Concept':<20} {'Method':<5} {'Layers':<12} {'Alpha':<8} {'Schedule':<14} Probe",
            "-" * 56,
        ]
        for name, sv in self._vectors.items():
            sched = self._schedules.get(name)
            sched_str = getattr(sched, "__name__", repr(sched)) if sched else self._schedule_type
            probe_str = "yes" if name in self._probes else "no"
            layers_str = str(sv.layers)
            alpha_str = f"{self._alphas.get(name, '-')}"
            lines.append(
                f"  {name:<20} {sv.method:<5} {layers_str:<12} {alpha_str:<8} {sched_str:<14} {probe_str}"
            )
        lines.append("-" * 56)
        lines.append(
            f"  orthogonalize={self._orthogonalize}, "
            f"schedule_type={self._schedule_type}, "
            f"num_inference_steps={self._num_inference_steps}"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SteeringPipeline(concepts={self.concepts}, "
            f"orthogonalize={self._orthogonalize}, "
            f"schedule_type={self._schedule_type})"
        )
