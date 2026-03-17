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
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Generator

import numpy as np
import torch
import torch.nn.functional as F

from steer_audio.vector_bank import SteeringVector

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

    Args:
        vectors:       Mapping ``concept_name → SteeringVector``.
        orthogonalize: If ``True``, in-place Gram-Schmidt the stored vectors.

    Raises:
        ValueError: If *vectors* is empty.
    """

    def __init__(
        self,
        vectors: dict[str, SteeringVector],
        orthogonalize: bool = False,
    ) -> None:
        if not vectors:
            raise ValueError("vectors must contain at least one SteeringVector.")
        # Work on a shallow copy so callers aren't surprised by in-place changes.
        self.vectors: dict[str, SteeringVector] = dict(vectors)
        if orthogonalize:
            self._apply_gram_schmidt()

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
    # Interference matrix
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
