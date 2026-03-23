"""
Multi-concept steering with optional Gram-Schmidt orthogonalization.

Mathematical foundation (arXiv 2602.11910 §3.2):

  CAA multi-steer:  h'_l = ReNorm(h_l + Σ_c α_c · v_c,  h_l)
  SAE multi-steer:  h'_l = h_l  + Σ_c α_c · v_c^SAE   (no ReNorm)

When orthogonalize=True the steering vectors are Gram-Schmidt orthogonalized
before summation to minimize inter-concept interference.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from steer_audio.vector_bank import SteeringVector, SteeringVectorBank

log = logging.getLogger(__name__)

# Epsilon shared with caa_utils.renorm
_EPS: float = 1e-8
# Minimum L2 norm below which an orthogonalized vector is discarded.
_GRAM_SCHMIDT_EPS: float = 1e-8


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _renorm(h_steered: torch.Tensor, h_orig: torch.Tensor) -> torch.Tensor:
    """Renormalize *h_steered* to match the per-token magnitude of *h_orig*.

    ReNorm(h', h) = h' / ||h'||₂ · ||h||₂   (broadcast over last dim)

    Args:
        h_steered: Steered activation, shape ``(..., dim)``.
        h_orig:    Original activation (same shape).

    Returns:
        Activation with the same shape and per-token L2 norm as *h_orig*.
    """
    orig_norm = h_orig.float().norm(dim=-1, keepdim=True)
    steered_norm = h_steered.float().norm(dim=-1, keepdim=True)
    return (h_steered.float() / (steered_norm + _EPS)) * orig_norm


# ---------------------------------------------------------------------------
# MultiConceptSteerer
# ---------------------------------------------------------------------------


class MultiConceptSteerer:
    """Apply multiple steering vectors simultaneously during diffusion inference.

    Mathematical foundation:
        CAA multi-steer:  h'_l = ReNorm(h_l + Σ_c α_c · v_c, h_l)
        SAE multi-steer:  h'_l = h_l + Σ_c α_c · v_c^SAE  (no ReNorm)

    When ``orthogonalize=True``, Gram-Schmidt orthogonalization is applied to
    the steering vectors before summation.  Concepts are processed in order of
    decreasing ``clap_delta`` so the most reliable concept anchors the basis.

    Supports two usage modes:

    **Prompt 2.2 (bank-based) API** — pass a ``SteeringVectorBank``:
        Use :meth:`add_concept`, :meth:`remove_concept`, :meth:`set_alpha`,
        :meth:`get_combined_vectors`, :meth:`register_hooks`, :meth:`remove_hooks`,
        :meth:`interference_report`.

    **Legacy (dict-based) API** — pass a ``dict[str, SteeringVector]``:
        Use :meth:`get_hooks`, :meth:`interference_matrix`, :meth:`steer`.

    Args:
        vectors_or_bank: Either a ``dict[str, SteeringVector]`` (legacy) or a
                         ``SteeringVectorBank`` (Prompt 2.2 API).
        orthogonalize:   If ``True``, apply Gram-Schmidt orthogonalization.
                         For the dict API this is done in-place at construction;
                         for the bank API it is applied in :meth:`get_combined_vectors`.

    Raises:
        ValueError: If an empty dict is provided (legacy API).
        TypeError:  If an unsupported type is provided.
    """

    def __init__(
        self,
        vectors_or_bank: Union[Dict[str, SteeringVector], "SteeringVectorBank"],
        orthogonalize: bool = True,
    ) -> None:
        self.orthogonalize = orthogonalize

        if isinstance(vectors_or_bank, SteeringVectorBank):
            # --- Prompt 2.2 bank-based API ---
            self._bank: Optional[SteeringVectorBank] = vectors_or_bank
            self.vectors: dict[str, SteeringVector] = {}
            # _active tracks {concept: (method, alpha)} for active concepts.
            self._active: dict[str, tuple[str, float]] = {}
        elif isinstance(vectors_or_bank, dict):
            # --- Legacy dict-based API ---
            if not vectors_or_bank:
                raise ValueError("vectors must contain at least one SteeringVector.")
            self._bank = None
            # Work on a shallow copy so callers aren't surprised by in-place changes.
            self.vectors = dict(vectors_or_bank)
            # Populate _active from the dict (all concepts active with alpha=1.0).
            self._active = {k: (v.method, 1.0) for k, v in vectors_or_bank.items()}
            if orthogonalize:
                self._apply_gram_schmidt()
        else:
            raise TypeError(
                f"Expected SteeringVectorBank or dict[str, SteeringVector], "
                f"got {type(vectors_or_bank).__name__}"
            )

    # ------------------------------------------------------------------ #
    # Gram-Schmidt orthogonalization
    # ------------------------------------------------------------------ #

    def _apply_gram_schmidt(self) -> None:
        """In-place Gram-Schmidt orthogonalize all stored steering vectors.

        Processes concepts in order of decreasing ``clap_delta`` so that the
        most reliable concept direction is kept intact.

        After orthogonalization each vector is unit-normalized.  Vectors
        whose residual falls below ``_GRAM_SCHMIDT_EPS`` are left at their
        current (near-zero) values with a warning.
        """
        # Sort by descending clap_delta — most reliable concept anchors basis.
        sorted_keys = sorted(
            self.vectors.keys(),
            key=lambda k: -self.vectors[k].clap_delta,
        )

        basis: list[torch.Tensor] = []  # orthonormal vectors accumulated so far

        for key in sorted_keys:
            sv = self.vectors[key]
            v: torch.Tensor = sv.vector.float().clone()

            # Subtract projections onto all previously established basis vectors.
            for u in basis:
                v = v - v.dot(u) * u

            norm = v.norm()
            if norm < _GRAM_SCHMIDT_EPS:
                log.warning(
                    "Steering vector for '%s' collapsed to near-zero after "
                    "Gram-Schmidt (norm=%.2e); it is linearly dependent on "
                    "previously processed concepts and will have no effect.",
                    key,
                    norm.item(),
                )
                # Keep the (near-zero) vector but do NOT add to basis.
            else:
                v = v / norm
                basis.append(v)

            # Mutate stored vector in-place.
            sv.vector = v.to(sv.vector.dtype)

    # ------------------------------------------------------------------ #
    # Prompt 2.2: bank-based concept management
    # ------------------------------------------------------------------ #

    def add_concept(self, concept: str, alpha: float, method: str = "caa") -> None:
        """Add a concept to the active set with the given alpha.

        Args:
            concept: Concept name (must exist in the bank).
            alpha:   Steering strength.
            method:  Vector method (``"caa"`` or ``"sae"``).

        Raises:
            RuntimeError: If no ``SteeringVectorBank`` was provided.
            KeyError:     If the concept is not in the bank.
        """
        if self._bank is None:
            raise RuntimeError(
                "add_concept requires a SteeringVectorBank; "
                "this steerer was created with a plain dict."
            )
        sv = self._bank.get(concept, method)  # raises KeyError if missing
        self._active[concept] = (method, alpha)
        self.vectors[concept] = sv

    def remove_concept(self, concept: str) -> None:
        """Remove a concept from the active set.

        Args:
            concept: Concept name to deactivate.
        """
        self._active.pop(concept, None)
        self.vectors.pop(concept, None)

    def set_alpha(self, concept: str, alpha: float) -> None:
        """Update the alpha (steering strength) for an active concept.

        Args:
            concept: Must already be active (added via :meth:`add_concept`).
            alpha:   New steering strength.

        Raises:
            KeyError: If *concept* is not in the active set.
        """
        if concept not in self._active:
            raise KeyError(
                f"Concept '{concept}' is not active. "
                "Call add_concept() first."
            )
        method, _ = self._active[concept]
        self._active[concept] = (method, alpha)

    def get_combined_vectors(self, layer: int) -> torch.Tensor:
        """Return the summed steering delta for *layer*.

        Computes ``Σ_c α_c · v_c`` over all active concepts.
        If ``orthogonalize=True`` the vectors are Gram-Schmidt orthogonalized
        before weighting (parallel vectors collapse to near-zero with a warning).

        Args:
            layer: Layer index (used to look up per-concept vectors; concepts
                   whose ``SteeringVector.layers`` does not include *layer* are
                   still included — the vector field is shared across layers).

        Returns:
            Combined delta tensor of shape ``(d_model,)``.
            Returns a zero tensor if no concepts are active.
        """
        active_items = list(self._active.items())
        if not active_items:
            # Infer dim from vectors if any, else default to 1
            dim = next(
                (sv.vector.shape[0] for sv in self.vectors.values()), 1
            )
            return torch.zeros(dim)

        # Collect (vector, alpha) pairs for active concepts.
        pairs: list[tuple[torch.Tensor, float]] = []
        for concept, (method, alpha) in active_items:
            sv = self.vectors.get(concept)
            if sv is None and self._bank is not None:
                sv = self._bank.get(concept, method)
            if sv is None:
                continue
            pairs.append((sv.vector.float(), alpha))

        if not pairs:
            return torch.zeros(1)

        dim = pairs[0][0].shape[0]

        # alpha=0 → no contribution; short-circuit to zero when all are zero.
        if all(a == 0.0 for _, a in pairs):
            return torch.zeros(dim)

        # Threshold for near-zero residual after Gram-Schmidt.  Use 1e-5 rather
        # than machine-epsilon so that float32 "identical" vectors (which have
        # residuals ~2e-7 due to normalisation round-off) are correctly flagged.
        _GS_NEAR_ZERO: float = 1e-5

        if self.orthogonalize and len(pairs) > 1:
            # Gram-Schmidt: orthogonalize in insertion order.
            # Pre-normalize each vector so all inputs are unit-norm.
            basis: list[torch.Tensor] = []
            result = torch.zeros(dim)
            for v, a in pairs:
                # Ensure unit-norm before GS to avoid float32 precision drift.
                v = F.normalize(v.clone(), dim=0)
                for u in basis:
                    v = v - v.dot(u) * u
                norm = v.norm()
                if norm < _GS_NEAR_ZERO:
                    warnings.warn(
                        f"A concept vector collapsed to near-zero (norm={norm:.2e}) "
                        "after Gram-Schmidt; it is nearly parallel to a prior concept "
                        "and will have minimal effect. Consider removing it.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    # Still add a * v (near-zero contribution) but skip adding to basis.
                else:
                    v = v / norm
                    basis.append(v)
                result = result + a * v
            return result
        else:
            # No orthogonalization: sum alpha * unit(v).
            result = torch.zeros(dim)
            for v, a in pairs:
                result = result + a * F.normalize(v, dim=0)
            return result

    def register_hooks(
        self, model: nn.Module, target_layers: List[int]
    ) -> List[Any]:
        """Register forward hooks on *target_layers* of *model*.

        At each targeted layer the hook applies:
            h'_l = ReNorm(h_l + combined, h_l)

        where ``combined = get_combined_vectors(layer_idx)``.

        The hook captures the combined vector at registration time (current
        alpha values).  Call :meth:`remove_hooks` with the returned handles
        when done.

        Args:
            model:         PyTorch module.  Target sub-modules are found by
                           trying ``model.transformer_blocks[i]``, then
                           ``list(model.children())[i]``, then *model* itself.
            target_layers: List of layer indices to hook.

        Returns:
            List of hook handles (pass to :meth:`remove_hooks` to clean up).
        """
        children = list(model.children())
        handles: List[Any] = []

        for layer_idx in target_layers:
            combined = self.get_combined_vectors(layer_idx)

            # Find the target sub-module.
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

            def _make_hook(delta: torch.Tensor):
                def hook(
                    module: nn.Module,
                    inputs: tuple,
                    output: Any,
                ) -> Any:
                    if isinstance(output, tuple):
                        h, *rest = output
                    else:
                        h = output
                        rest = None
                    h_orig = h.detach().clone()
                    v = delta.to(h.device)
                    h_out = _renorm(h.float() + v, h_orig.float())
                    h_out = h_out.to(h.dtype)
                    if rest is not None:
                        return (h_out, *rest)
                    return h_out
                return hook

            handle = target_mod.register_forward_hook(_make_hook(combined))
            handles.append(handle)
            log.debug(
                "Registered steering hook on layer %d (%s).",
                layer_idx,
                type(target_mod).__name__,
            )

        return handles

    def remove_hooks(self, handles: List[Any]) -> None:
        """Remove all hook handles returned by :meth:`register_hooks`.

        Args:
            handles: List of hook handles to remove.
        """
        for h in handles:
            h.remove()
        log.debug("Removed %d hook(s).", len(handles))

    def interference_report(self) -> Dict[str, Any]:
        """Compute an interference report for the currently active concepts.

        Contains:

        - ``"concepts"``:      List of active concept names.
        - ``"cosine_matrix"``: N×N numpy array of pairwise cosine similarities.
        - ``"max_cosine"``:    Maximum off-diagonal cosine similarity.
        - ``"warnings"``:      List of warning strings for pairs with cosine > 0.5.
        - ``"clap_deltas"``:   Dict ``{concept: clap_delta}`` (placeholder −1.0 when
                               no real metric exists yet).

        Returns:
            Dict with the above keys.
        """
        active_concepts = list(self._active.keys())
        n = len(active_concepts)

        if n == 0:
            return {
                "concepts": [],
                "cosine_matrix": np.zeros((0, 0), dtype=np.float32),
                "max_cosine": 0.0,
                "warnings": [],
                "clap_deltas": {},
            }

        # Build unit-norm vectors for each active concept.
        unit_vecs: list[torch.Tensor] = []
        clap_deltas: dict[str, float] = {}
        for concept in active_concepts:
            sv = self.vectors.get(concept)
            if sv is None:
                method = self._active[concept][0]
                if self._bank is not None:
                    sv = self._bank.get(concept, method)
            if sv is None:
                unit_vecs.append(torch.zeros(1))
                clap_deltas[concept] = -1.0
            else:
                unit_vecs.append(F.normalize(sv.vector.float(), dim=0))
                clap_deltas[concept] = sv.clap_delta if sv.clap_delta != 0.0 else -1.0

        # Compute cosine matrix.
        try:
            stacked = torch.stack(unit_vecs)  # (N, D)
            cos_mat = (stacked @ stacked.T).cpu().numpy().astype(np.float32)
        except RuntimeError:
            # Vectors have different dims (degenerate case).
            cos_mat = np.eye(n, dtype=np.float32)

        # Find max off-diagonal cosine and build warnings.
        warn_msgs: list[str] = []
        max_cos = 0.0
        _INTERFERENCE_THRESHOLD = 0.5
        for i in range(n):
            for j in range(i + 1, n):
                c = float(cos_mat[i, j])
                if abs(c) > max_cos:
                    max_cos = abs(c)
                if abs(c) > _INTERFERENCE_THRESHOLD:
                    warn_msgs.append(
                        f"High interference between '{active_concepts[i]}' and "
                        f"'{active_concepts[j]}': cosine={c:.3f} > "
                        f"{_INTERFERENCE_THRESHOLD}. Consider removing one or "
                        "enabling orthogonalization."
                    )

        return {
            "concepts": active_concepts,
            "cosine_matrix": cos_mat,
            "max_cosine": max_cos,
            "warnings": warn_msgs,
            "clap_deltas": clap_deltas,
        }

    # ------------------------------------------------------------------ #
    # Interference matrix (legacy)
    # ------------------------------------------------------------------ #

    def interference_matrix(self) -> torch.Tensor:
        """Compute the pairwise cosine-similarity matrix of steering vectors.

        Values near ±1 indicate high interference; near 0 is orthogonal.

        Returns:
            Float tensor of shape ``(N_concepts, N_concepts)`` where
            ``matrix[i, j] = cos(v_i, v_j)``.
        """
        keys = list(self.vectors.keys())
        n = len(keys)
        # Stack as rows: shape (N, hidden_dim)
        stacked = torch.stack(
            [F.normalize(self.vectors[k].vector.float(), dim=0) for k in keys]
        )  # (N, D)
        # Cosine similarity matrix via matrix multiplication of unit-norm rows.
        matrix = stacked @ stacked.T  # (N, N)
        return matrix

    # ------------------------------------------------------------------ #
    # Hook factory
    # ------------------------------------------------------------------ #

    def get_hooks(
        self,
        alphas: dict[str, float],
    ) -> list[tuple[int, Any]]:
        """Build PyTorch forward hooks for multi-concept steering injection.

        The returned hooks should be registered on the cross-attention output
        of each targeted transformer block.  Hooks for the same layer are
        merged so that only a single forward call is made per block.

        Hook logic (CAA):
            h' = ReNorm(h + Σ_c α_c · v_c, h)   for layers in any vector
        Hook logic (SAE):
            h' = h + Σ_c α_c · v_c               (no ReNorm)

        Args:
            alphas: ``{concept_name: alpha_value}`` mapping.  Concepts absent
                    from this dict default to alpha=0 (no effect).

        Returns:
            List of ``(layer_index, hook_fn)`` pairs.
        """
        # Aggregate contributions per layer.
        layer_contributions: dict[
            int, list[tuple[torch.Tensor, float, str]]
        ] = defaultdict(list)

        for concept, sv in self.vectors.items():
            alpha = alphas.get(concept, 0.0)
            if alpha == 0.0:
                continue
            for layer_idx in sv.layers:
                # (steering_vector, alpha, method)
                layer_contributions[layer_idx].append(
                    (sv.vector, alpha, sv.method)
                )

        hooks: list[tuple[int, Any]] = []

        for layer_idx, contribs in layer_contributions.items():
            # Close over a local copy of contribs.
            def _make_hook(
                contributions: list[tuple[torch.Tensor, float, str]],
            ):
                def hook(
                    module: torch.nn.Module,
                    inputs: tuple,
                    output: torch.Tensor | tuple,
                ) -> torch.Tensor | tuple:
                    """Forward hook: inject concept steering into cross-attn output."""
                    # Unwrap tuple outputs (e.g. (hidden, weights)).
                    if isinstance(output, tuple):
                        h, *rest = output
                    else:
                        h = output
                        rest = None

                    h_orig = h.detach().clone()

                    # Compute per-method deltas separately.
                    # CAA delta (will be ReNorm-ed together with h).
                    caa_delta = torch.zeros_like(h.float())
                    sae_delta = torch.zeros_like(h.float())
                    has_caa = False

                    for v, a, method in contributions:
                        v_dev = v.float().to(h.device)  # shape: (D,)
                        if method == "caa":
                            has_caa = True
                            caa_delta = caa_delta + a * v_dev
                        else:  # sae
                            sae_delta = sae_delta + a * v_dev

                    # Apply CAA delta with ReNorm.
                    h_out = h.float()
                    if has_caa:
                        h_out = _renorm(h_out + caa_delta, h_orig.float())

                    # Apply SAE delta without ReNorm.
                    h_out = h_out + sae_delta

                    # Cast back to original dtype.
                    h_out = h_out.to(h.dtype)

                    if rest is not None:
                        return (h_out, *rest)
                    return h_out

                return hook

            hooks.append((layer_idx, _make_hook(contribs)))

        return hooks

    # ------------------------------------------------------------------ #
    # Context manager: register / deregister hooks
    # ------------------------------------------------------------------ #

    @contextmanager
    def _hooked(
        self,
        transformer_blocks: torch.nn.ModuleList,
        alphas: dict[str, float],
    ) -> Generator[None, None, None]:
        """Context manager: register steering hooks, yield, then remove them.

        The hooks are attached to the ``cross_attn`` sub-module of each
        targeted transformer block.

        Args:
            transformer_blocks: ``model.transformer_blocks`` module list.
            alphas:             Concept → alpha mapping.
        """
        hooks = self.get_hooks(alphas)
        handles: list[Any] = []

        for layer_idx, hook_fn in hooks:
            if layer_idx >= len(transformer_blocks):
                log.warning(
                    "Layer index %d is out of range for model with %d blocks; skipped.",
                    layer_idx,
                    len(transformer_blocks),
                )
                continue
            block = transformer_blocks[layer_idx]
            # Prefer the cross-attention sub-module; fall back to the block itself.
            target_module = getattr(block, "cross_attn", block)
            handle = target_module.register_forward_hook(hook_fn)
            handles.append(handle)
            log.debug("Registered steering hook on layer %d (%s).", layer_idx, type(target_module).__name__)

        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            log.debug("Removed %d steering hook(s).", len(handles))

    # ------------------------------------------------------------------ #
    # Main inference entry point
    # ------------------------------------------------------------------ #

    def steer(
        self,
        model: Any,
        prompt: str,
        alphas: dict[str, float],
        duration: float = 30.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, int]:
        """Generate audio steered by multiple concepts simultaneously.

        Registers forward hooks on the model's cross-attention layers,
        runs one forward pass of the diffusion pipeline, then removes hooks.

        This method targets the ``PatchableACE`` / ``SteeredACEStepPipeline``
        interface used by the rest of the TADA codebase:

            transformer_blocks = model.patchable_model          \\
                                      .ace_step_transformer     \\
                                      .transformer_blocks

        If *model* exposes a different attribute tree, subclass and override
        :meth:`_get_transformer_blocks`.

        Args:
            model:    ACE-Step model instance (``PatchableACE`` or similar).
            prompt:   Text prompt for audio generation.
            alphas:   ``{concept_name: alpha_value}`` dict.  Use 0 to skip.
            duration: Target audio duration in seconds.
            seed:     Random seed for reproducible latent sampling.

        Returns:
            ``(audio_array, sample_rate)`` where *audio_array* has shape
            ``(n_samples,)`` (mono) or ``(2, n_samples)`` (stereo).

        Raises:
            AttributeError: If the expected model attribute tree is not found.
        """
        transformer_blocks = self._get_transformer_blocks(model)

        with self._hooked(transformer_blocks, alphas):
            audio, sr = self._run_inference(model, prompt, duration, seed)

        return audio, sr

    # ------------------------------------------------------------------ #
    # Model interface helpers (override for non-ACE-Step models)
    # ------------------------------------------------------------------ #

    def _get_transformer_blocks(self, model: Any) -> torch.nn.ModuleList:
        """Retrieve the transformer block ModuleList from *model*.

        Tries the following attribute paths in order:
          1. ``model.patchable_model.ace_step_transformer.transformer_blocks``
          2. ``model.ace_step_transformer.transformer_blocks``
          3. ``model.transformer_blocks``

        Args:
            model: Any model object.

        Returns:
            The ``transformer_blocks`` ModuleList.

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
    ) -> tuple[np.ndarray, int]:
        """Run one forward pass of the diffusion pipeline.

        Supports the ``SteeredACEStepPipeline`` and ``SimpleACEStepPipeline``
        interfaces.  Extend or override for other models.

        Args:
            model:    Model with a ``.pipeline`` attribute or callable.
            prompt:   Text prompt.
            duration: Audio duration in seconds.
            seed:     Random seed.

        Returns:
            ``(audio_array, sample_rate)``.
        """
        # Prefer dedicated pipeline attribute.
        pipeline = getattr(model, "pipeline", model)

        audio = pipeline(
            prompt=prompt,
            audio_duration=duration,
            manual_seed=seed,
            return_type="audio",
        )

        sr: int = getattr(pipeline, "sample_rate", 44100)

        # Normalise output to numpy array.
        if isinstance(audio, torch.Tensor):
            audio_np: np.ndarray = audio.squeeze().cpu().float().numpy()
        elif isinstance(audio, np.ndarray):
            audio_np = audio
        else:
            raise TypeError(
                f"Unexpected audio output type from pipeline: {type(audio).__name__}"
            )

        return audio_np, sr
