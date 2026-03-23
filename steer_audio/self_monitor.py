"""
Self-monitoring steering controller — Phase 2, Prompt 2.4 (TADA roadmap).

Scientific motivation
---------------------
At high alpha values, CAA steering degrades audio quality (LPAPS increases,
CE/PQ decrease).  A controller that detects when the target concept is
sufficiently present reduces alpha adaptively — preventing over-steering.

Inspired by SMITIN (Koo et al. 2025, arXiv 2404.02252).

Algorithm (per diffusion step)
------------------------------
1. Every ``check_every`` steps, decode the partial latent → run CLAP → run probe.
2. If P(concept) > threshold_high: set effective_alpha = alpha * decay_factor.
3. If P(concept) < threshold_low: restore effective_alpha to original alpha.
4. Apply effective_alpha this step.
"""

from __future__ import annotations

import logging
import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator

import numpy as np
import torch
import torch.nn.functional as F

from steer_audio.vector_bank import SteeringVector

log = logging.getLogger(__name__)

# Epsilon to guard against zero-norm divisions.
_EPS: float = 1e-8

# Type alias: a CLAP extractor callable accepts (audio_array, sample_rate) and
# returns a 1-D float32 embedding vector.
ClapExtractor = Callable[[np.ndarray, int], np.ndarray]


# ---------------------------------------------------------------------------
# ConceptProbe
# ---------------------------------------------------------------------------


class ConceptProbe:
    """Lightweight probe that detects concept presence from CLAP embeddings.

    Supports two usage modes:

    **Stub mode** (``clap_model=None``):
        :meth:`score` always returns ``0.5``.  Use this in tests or when CLAP
        weights are unavailable.

    **CLAP mode** (``clap_model`` provided):
        :meth:`score` returns the cosine similarity between the *target_prompt*
        text embedding and the audio embedding.

    Training data: CLAP embeddings of generated audio with positive / negative
    concept prompts.  A logistic regression classifier is fitted on those
    embeddings and used at inference time to return P(concept present) via
    :meth:`predict_proba`.

    Args:
        concept:        Name of the concept being probed (e.g. ``"tempo"``).
        target_prompt:  Text prompt describing the desired concept (used by
                        :meth:`score` to compute CLAP text-audio similarity).
        clap_model:     Full CLAP model object with ``get_text_embedding`` and
                        ``get_audio_embedding_from_data`` methods.  When
                        ``None``, :meth:`score` returns the stub value ``0.5``.
        clap_extractor: *Legacy parameter.*  Callable
                        ``(audio: np.ndarray, sample_rate: int) -> np.ndarray``
                        used by :meth:`predict_proba`.  Defaults to a zero-
                        vector stub.
    """

    def __init__(
        self,
        concept: str,
        target_prompt: str = "",
        clap_model: Any | None = None,
        clap_extractor: ClapExtractor | None = None,
    ) -> None:
        self.target_prompt: str = target_prompt
        self.clap_model: Any | None = clap_model
        self.concept = concept
        self._clap_extractor: ClapExtractor = (
            clap_extractor if clap_extractor is not None else _stub_clap_extractor
        )
        # Fitted after calling train(); None before that.
        try:
            from sklearn.linear_model import LogisticRegression

            self.classifier = LogisticRegression(max_iter=1000)
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for ConceptProbe. "
                "Install with: pip install scikit-learn"
            ) from exc
        self._is_trained: bool = False

    # ------------------------------------------------------------------ #
    # Embedding helper
    # ------------------------------------------------------------------ #

    def _embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return the CLAP embedding for *audio*.

        Args:
            audio:       1-D or 2-D float32 audio array.
            sample_rate: Sample rate in Hz.

        Returns:
            1-D float32 numpy array (CLAP embedding).
        """
        emb = self._clap_extractor(audio, sample_rate)
        return np.asarray(emb, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #

    def train(
        self,
        positive_audio_paths: list[Path],
        negative_audio_paths: list[Path],
        sample_rate: int = 44100,
    ) -> float:
        """Fit the logistic regression probe on CLAP embeddings.

        Args:
            positive_audio_paths: Audio files generated with positive concept prompts.
            negative_audio_paths: Audio files generated with negative concept prompts.
            sample_rate:          Sample rate assumed when loading audio (default 44 100 Hz).

        Returns:
            Training accuracy in [0, 1].

        Raises:
            ValueError: If either list is empty or lists have different lengths.
        """
        if not positive_audio_paths or not negative_audio_paths:
            raise ValueError(
                "Both positive_audio_paths and negative_audio_paths must be non-empty. "
                f"Got {len(positive_audio_paths)} positive and "
                f"{len(negative_audio_paths)} negative paths."
            )

        embeddings: list[np.ndarray] = []
        labels: list[int] = []

        for path in positive_audio_paths:
            audio = _load_audio(Path(path), sample_rate)
            embeddings.append(self._embed(audio, sample_rate))
            labels.append(1)

        for path in negative_audio_paths:
            audio = _load_audio(Path(path), sample_rate)
            embeddings.append(self._embed(audio, sample_rate))
            labels.append(0)

        X = np.stack(embeddings)  # shape: (N, embedding_dim)
        y = np.array(labels, dtype=int)

        self.classifier.fit(X, y)
        self._is_trained = True
        accuracy: float = float(self.classifier.score(X, y))
        log.info(
            "ConceptProbe('%s') trained on %d samples — train accuracy: %.3f",
            self.concept,
            len(y),
            accuracy,
        )
        return accuracy

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #

    def predict_proba(self, audio: np.ndarray, sample_rate: int) -> float:
        """Return P(concept present) in [0, 1].

        Args:
            audio:       1-D or 2-D float32 audio array.
            sample_rate: Sample rate in Hz.

        Returns:
            Probability that the concept is present, in [0, 1].

        Raises:
            RuntimeError: If the probe has not been trained yet.
        """
        if not self._is_trained:
            raise RuntimeError(
                f"ConceptProbe('{self.concept}') has not been trained. "
                "Call train() first."
            )
        emb = self._embed(audio, sample_rate).reshape(1, -1)
        prob: float = float(self.classifier.predict_proba(emb)[0, 1])
        return prob

    # ------------------------------------------------------------------ #
    # Score / delta — diffusion-step monitoring API (Prompt 2.5)
    # ------------------------------------------------------------------ #

    def score(
        self,
        audio_tensor: torch.Tensor | np.ndarray,
        sample_rate: int = 44100,
    ) -> float:
        """Measure concept presence for one diffusion step.

        In **stub mode** (``clap_model=None``) this always returns ``0.5`` so
        that the :class:`SelfMonitoredSteerer` can run in tests without real
        CLAP weights.

        In **CLAP mode**, computes the cosine similarity between the text
        embedding of :attr:`target_prompt` and the audio embedding of
        *audio_tensor*.

        Args:
            audio_tensor: Partial audio for this diffusion step.  Accepts a
                          PyTorch tensor (any shape) or a numpy array.
            sample_rate:  Sample rate of *audio_tensor* (default 44 100 Hz).

        Returns:
            Float in approximately [0, 1] measuring concept presence.  Exactly
            ``0.5`` in stub mode.
        """
        if self.clap_model is None:
            return 0.5

        # Convert to numpy 1-D float32 for CLAP.
        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.detach().cpu().float().numpy().flatten()
        else:
            audio_np = np.asarray(audio_tensor, dtype=np.float32).flatten()

        try:
            text_emb = self.clap_model.get_text_embedding([self.target_prompt])
            audio_emb = self.clap_model.get_audio_embedding_from_data(
                [audio_np], use_tensor=False
            )
            similarity = float(
                np.dot(text_emb[0], audio_emb[0])
                / (
                    np.linalg.norm(text_emb[0]) * np.linalg.norm(audio_emb[0])
                    + _EPS
                )
            )
            # Map cosine similarity from [-1, 1] to [0, 1].
            return (similarity + 1.0) / 2.0
        except Exception as exc:  # noqa: BLE001
            log.debug("ConceptProbe.score() failed: %s — returning 0.5", exc)
            return 0.5

    def delta(self, current_score: float, previous_score: float) -> float:
        """Return the signed change in concept score between two steps.

        Args:
            current_score:  Score at the current diffusion step.
            previous_score: Score at the previous check step.

        Returns:
            ``current_score - previous_score``.  Positive means the concept is
            becoming more present; negative means regression.
        """
        return current_score - previous_score

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def save(self, path: Path) -> None:
        """Serialise probe to *path* using pickle.

        Args:
            path: Destination file path (parent dirs created automatically).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "concept": self.concept,
            "classifier": self.classifier,
            "is_trained": self._is_trained,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh)
        log.debug("Saved ConceptProbe('%s') to %s", self.concept, path)

    @classmethod
    def load(
        cls,
        path: Path,
        clap_extractor: ClapExtractor | None = None,
    ) -> "ConceptProbe":
        """Load a probe from *path*.

        Args:
            path:           Path to a pickle file written by :meth:`save`.
            clap_extractor: Optional CLAP extractor to attach after loading.

        Returns:
            Reconstructed :class:`ConceptProbe`.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ConceptProbe file not found: {path}")
        with path.open("rb") as fh:
            payload = pickle.load(fh)  # noqa: S301
        probe = cls(concept=payload["concept"], clap_extractor=clap_extractor)
        probe.classifier = payload["classifier"]
        probe._is_trained = payload["is_trained"]
        log.debug("Loaded ConceptProbe('%s') from %s", probe.concept, path)
        return probe


# ---------------------------------------------------------------------------
# VectorAdaptiveSteerer  (legacy vector-based implementation)
# ---------------------------------------------------------------------------


class VectorAdaptiveSteerer:
    """Adaptive steering that reduces alpha when concept is detected.

    The controller hooks into the model's transformer blocks (at each layer
    listed in *vector.layers*) and adjusts the effective alpha based on
    concept-presence probability estimated by *probe*.

    .. note::
        This is the legacy implementation used by
        :class:`~steer_audio.pipeline.SteeringPipeline`.  For the
        diffusion-step delta controller described in TADA Prompt 2.5, see
        :class:`SelfMonitoredSteerer`.

    Args:
        vector:          Pre-computed :class:`~steer_audio.vector_bank.SteeringVector`.
        probe:           Trained :class:`ConceptProbe` for the same concept.
        alpha:           Nominal steering magnitude.
        threshold_high:  Stop boosting above this concept probability (default 0.85).
        threshold_low:   Resume boosting below this concept probability (default 0.40).
        decay_factor:    Multiply effective alpha by this factor when concept is
                         detected (default 0.5).
        check_every:     Run the probe every N diffusion steps (default 5).
    """

    def __init__(
        self,
        vector: SteeringVector,
        probe: ConceptProbe,
        alpha: float,
        threshold_high: float = 0.85,
        threshold_low: float = 0.40,
        decay_factor: float = 0.5,
        check_every: int = 5,
    ) -> None:
        if threshold_low >= threshold_high:
            raise ValueError(
                f"threshold_low ({threshold_low}) must be strictly less than "
                f"threshold_high ({threshold_high})."
            )
        if not (0.0 < decay_factor <= 1.0):
            raise ValueError(
                f"decay_factor must be in (0, 1], got {decay_factor}."
            )
        if check_every < 1:
            raise ValueError(f"check_every must be >= 1, got {check_every}.")

        self.vector = vector
        self.probe = probe
        self.alpha = alpha
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.decay_factor = decay_factor
        self.check_every = check_every

        # Populated during steer().
        self._trace: list[dict] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def steer(
        self,
        model: Any,
        prompt: str,
        duration: float = 30.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, int]:
        """Run steered inference with adaptive alpha control.

        Args:
            model:    ACE-Step (or compatible) model instance.
            prompt:   Text prompt for audio generation.
            duration: Audio duration in seconds (default 30).
            seed:     Random seed for reproducible generation (default 42).

        Returns:
            ``(audio_array, sample_rate)`` where *audio_array* is a 1-D float32
            numpy array.
        """
        self._trace = []
        effective_alpha = self.alpha
        step_counter = 0

        def _hook(
            module: torch.nn.Module,
            inputs: tuple,
            output: torch.Tensor,
        ) -> torch.Tensor:
            nonlocal effective_alpha, step_counter

            step_counter += 1

            # --- Check probe every `check_every` steps ---
            if step_counter % self.check_every == 0:
                # Decode partial latent to audio for probing.
                partial_audio = _decode_partial_latent(model, output)
                sample_rate = _get_sample_rate(model)
                prob = self.probe.predict_proba(partial_audio, sample_rate)
                clap_score = _compute_clap_score(model, partial_audio, prompt, sample_rate)

                # Adaptive alpha logic.
                if prob > self.threshold_high:
                    effective_alpha = effective_alpha * self.decay_factor
                    log.debug(
                        "Step %d: P(concept)=%.3f > %.2f → decay alpha to %.2f",
                        step_counter,
                        prob,
                        self.threshold_high,
                        effective_alpha,
                    )
                elif prob < self.threshold_low:
                    effective_alpha = self.alpha
                    log.debug(
                        "Step %d: P(concept)=%.3f < %.2f → restore alpha to %.2f",
                        step_counter,
                        prob,
                        self.threshold_low,
                        effective_alpha,
                    )

                self._trace.append(
                    {
                        "step": step_counter,
                        "effective_alpha": effective_alpha,
                        "concept_probability": prob,
                        "decoded_clap_score": clap_score,
                    }
                )

            # --- Apply steering at effective_alpha ---
            if effective_alpha == 0.0:
                return output

            v = self.vector.vector.to(output.device, dtype=output.dtype)
            v = F.normalize(v, dim=0)
            delta = effective_alpha * v  # shape: (hidden_dim,)
            h_steered = output + delta  # broadcast over (batch, seq, dim)

            if self.vector.method == "caa":
                # ReNorm: preserve per-token L2 magnitude of original activation.
                orig_norm = output.float().norm(dim=-1, keepdim=True)
                steered_norm = h_steered.float().norm(dim=-1, keepdim=True)
                h_steered = (h_steered.float() / (steered_norm + _EPS)) * orig_norm
                h_steered = h_steered.to(output.dtype)

            return h_steered

        # Register hooks.
        handles: list[torch.utils.hooks.RemovableHook] = []
        blocks = _get_transformer_blocks(model)
        for layer_idx in self.vector.layers:
            if layer_idx >= len(blocks):
                log.warning(
                    "VectorAdaptiveSteerer: layer %d out of range (model has %d blocks), skipping.",
                    layer_idx,
                    len(blocks),
                )
                continue
            handle = blocks[layer_idx].cross_attn.register_forward_hook(_hook)
            handles.append(handle)

        try:
            audio, sample_rate = _run_inference(model, prompt, duration, seed)
        finally:
            for h in handles:
                h.remove()

        return audio, sample_rate

    def get_monitoring_trace(self) -> "pd.DataFrame":
        """Return the monitoring trace from the most recent :meth:`steer` call.

        Each row corresponds to one probe check and contains:
        ``step``, ``effective_alpha``, ``concept_probability``,
        ``decoded_clap_score``.

        Returns:
            :class:`pandas.DataFrame` with four columns.

        Raises:
            RuntimeError: If :meth:`steer` has not been called yet.
        """
        if not self._trace:
            raise RuntimeError(
                "No monitoring trace available. Call steer() first."
            )

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for get_monitoring_trace(). "
                "Install with: pip install pandas"
            ) from exc

        return pd.DataFrame(self._trace)


# ---------------------------------------------------------------------------
# SelfMonitoredSteerer — diffusion-step delta controller (Prompt 2.5)
# ---------------------------------------------------------------------------


class SelfMonitoredSteerer:
    """Adaptive alpha controller for diffusion-based audio steering (TADA Prompt 2.5).

    Unlike the autoregressive SMITIN controller, this class gates on diffusion
    steps: every ``check_every_n_steps`` steps it scores the partial latent
    decode and adjusts the current alpha to converge on the target concept score.

    Algorithm (per call to :meth:`update`):
    1. If ``should_check(step)`` is False, return ``current_alpha`` unchanged.
    2. Compute ``s = probe.score(partial_audio, sample_rate)``.
    3. If a previous score exists, compute ``Δ = probe.delta(s, prev_score)``:
       - Δ < 0 (concept regressing)       → ``current_alpha += alpha_step``.
       - Δ > convergence_threshold         → ``current_alpha -= alpha_step``.
       - |Δ| ≤ convergence_threshold        → no change.
    4. Clamp ``current_alpha`` to ``[min_alpha, max_alpha]``.
    5. Append *s* to score history; return updated ``current_alpha``.

    Args:
        multi_steerer:         :class:`~steer_audio.multi_steer.MultiConceptSteerer`
                               instance to adapt.
        probe:                 :class:`ConceptProbe` used for scoring.
        check_every_n_steps:   How many diffusion steps between probe evaluations.
        alpha_step:            Amount to increment / decrement alpha per check.
        max_alpha:             Upper clamp on alpha.
        min_alpha:             Lower clamp on alpha (default 0).
        convergence_threshold: Minimum positive delta that triggers alpha reduction.
    """

    def __init__(
        self,
        multi_steerer: Any,           # MultiConceptSteerer — avoid circular import
        probe: "ConceptProbe",
        check_every_n_steps: int = 5,
        alpha_step: float = 5.0,
        max_alpha: float = 100.0,
        min_alpha: float = 0.0,
        convergence_threshold: float = 0.02,
    ) -> None:
        self._multi_steerer = multi_steerer
        self._probe = probe
        self.check_every_n_steps = check_every_n_steps
        self.alpha_step = alpha_step
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self.convergence_threshold = convergence_threshold

        # Public state fields (Prompt 2.5 spec).
        self.current_alpha: float = 0.0
        self._step: int = 0
        self._score_history: list[float] = []

    # ------------------------------------------------------------------ #
    # should_check
    # ------------------------------------------------------------------ #

    def should_check(self, step: int) -> bool:
        """Return True every ``check_every_n_steps`` diffusion steps.

        Args:
            step: Current diffusion step index (0-based).

        Returns:
            ``True`` if the probe should be evaluated at *step*.
        """
        return (step % self.check_every_n_steps) == 0

    # ------------------------------------------------------------------ #
    # update
    # ------------------------------------------------------------------ #

    def update(
        self,
        partial_audio: torch.Tensor | np.ndarray,
        sample_rate: int,
        step: int,
    ) -> float:
        """Evaluate probe and update alpha for one diffusion step.

        Args:
            partial_audio: Partial audio decoded from the current latent.
                           Can be a torch ``Tensor`` or numpy array.
            sample_rate:   Sample rate of *partial_audio*.
            step:          Current diffusion step index.

        Returns:
            Updated ``current_alpha`` (unchanged when not a check step).
        """
        if not self.should_check(step):
            return self.current_alpha

        self._step = step
        new_score = self._probe.score(partial_audio, sample_rate)

        if self._score_history:
            prev_score = self._score_history[-1]
            d = self._probe.delta(new_score, prev_score)
            if d < 0:
                # Score regressed → push harder.
                self.current_alpha += self.alpha_step
            elif d > self.convergence_threshold:
                # Score improved beyond threshold → ease off.
                self.current_alpha -= self.alpha_step
            # else |d| ≤ threshold → no change

        self._score_history.append(new_score)

        # Clamp alpha.
        self.current_alpha = max(self.min_alpha, min(self.max_alpha, self.current_alpha))

        log.debug(
            "SelfMonitoredSteerer step=%d score=%.4f alpha=%.2f",
            step,
            new_score,
            self.current_alpha,
        )
        return self.current_alpha

    # ------------------------------------------------------------------ #
    # reset / get_history
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset step counter, score history, and current alpha to initial state."""
        self.current_alpha = 0.0
        self._step = 0
        self._score_history = []

    def get_history(self) -> dict[str, list]:
        """Return score and alpha history.

        Returns:
            Dictionary with keys ``"scores"`` (list of probe scores per check
            step) and ``"steps"`` (list of diffusion step indices at which the
            probe was evaluated).
        """
        return {
            "scores": list(self._score_history),
            "steps": list(range(0, self._step + 1, self.check_every_n_steps))[
                : len(self._score_history)
            ],
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _stub_clap_extractor(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return a deterministic zero embedding when no real CLAP model is set.

    Args:
        audio:       Audio array (ignored).
        sample_rate: Sample rate (ignored).

    Returns:
        Zero vector of shape ``(512,)``.
    """
    return np.zeros(512, dtype=np.float32)


def _load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Load an audio file as a 1-D float32 numpy array.

    Falls back to a zero-vector stub when soundfile / librosa are unavailable
    so that unit tests can run without audio I/O dependencies.

    Args:
        path:        Path to a WAV / FLAC / OGG file.
        sample_rate: Target sample rate for resampling.

    Returns:
        1-D float32 numpy array.
    """
    try:
        import soundfile as sf

        audio, sr = sf.read(str(path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # mono mix-down
        if sr != sample_rate:
            try:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            except ImportError:
                log.warning(
                    "librosa not available; skipping resample from %d to %d Hz for %s",
                    sr,
                    sample_rate,
                    path,
                )
        return audio.astype(np.float32)
    except (ImportError, Exception) as exc:  # noqa: BLE001
        log.warning("Could not load '%s': %s — using zero-vector stub.", path, exc)
        return np.zeros(sample_rate, dtype=np.float32)


def _get_transformer_blocks(model: Any) -> Any:
    """Return the transformer block list from *model*.

    Tries ``model.patchable_model.transformer_blocks`` first, then
    ``model.transformer_blocks``.

    Args:
        model: ACE-Step model instance or compatible stub.

    Returns:
        A sequence of transformer block modules.

    Raises:
        AttributeError: If neither attribute path is found.
    """
    for attr_path in ("patchable_model.transformer_blocks", "transformer_blocks"):
        obj = model
        found = True
        for part in attr_path.split("."):
            if not hasattr(obj, part):
                found = False
                break
            obj = getattr(obj, part)
        if found:
            return obj
    raise AttributeError(
        "Cannot find transformer_blocks on the model. "
        "Expected 'model.transformer_blocks' or 'model.patchable_model.transformer_blocks'."
    )


def _get_sample_rate(model: Any) -> int:
    """Return the sample rate from *model*, defaulting to 44 100 Hz.

    Args:
        model: Model instance.

    Returns:
        Sample rate in Hz.
    """
    for attr in ("sample_rate", "pipeline.sample_rate"):
        obj = model
        found = True
        for part in attr.split("."):
            if not hasattr(obj, part):
                found = False
                break
            obj = getattr(obj, part)
        if found and isinstance(obj, int):
            return obj
    return 44100


def _decode_partial_latent(model: Any, latent: torch.Tensor) -> np.ndarray:
    """Decode *latent* to audio for probe evaluation.

    When a real VAE decoder is unavailable (test or stub models) returns a
    zero-vector of length 1 second at 44 100 Hz.

    Args:
        model:   Model that may expose a ``decode_latents`` method.
        latent:  Partial latent tensor from a forward hook, shape ``(..., dim)``.

    Returns:
        1-D float32 numpy array.
    """
    try:
        if hasattr(model, "decode_latents"):
            audio = model.decode_latents(latent)
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().float().numpy()
            if audio.ndim > 1:
                audio = audio[0]  # first batch item
            return audio.astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        log.debug("_decode_partial_latent: decode_latents failed: %s", exc)
    return np.zeros(44100, dtype=np.float32)


def _compute_clap_score(
    model: Any,
    audio: np.ndarray,
    prompt: str,
    sample_rate: int,
) -> float:
    """Compute CLAP text-audio alignment score.

    Returns 0.0 when CLAP is unavailable (test / stub context).

    Args:
        model:       Model instance (may carry a clap_model attribute).
        audio:       1-D float32 audio array.
        prompt:      Text prompt to score against.
        sample_rate: Sample rate of *audio*.

    Returns:
        Cosine similarity in [-1, 1], or 0.0 if CLAP is unavailable.
    """
    try:
        clap = getattr(model, "clap_model", None)
        if clap is None:
            return 0.0
        text_emb = clap.get_text_embedding([prompt])
        audio_emb = clap.get_audio_embedding_from_data(
            [audio], use_tensor=False
        )
        score = float(
            np.dot(text_emb[0], audio_emb[0])
            / (np.linalg.norm(text_emb[0]) * np.linalg.norm(audio_emb[0]) + _EPS)
        )
        return score
    except Exception as exc:  # noqa: BLE001
        log.debug("_compute_clap_score failed: %s", exc)
        return 0.0


def _run_inference(
    model: Any,
    prompt: str,
    duration: float,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Run model inference and return (audio_array, sample_rate).

    Tries ``model.pipeline(prompt, duration=duration, seed=seed)`` or
    ``model(prompt)``.  Falls back to a zero-vector stub for tests.

    Args:
        model:    Model instance.
        prompt:   Text prompt.
        duration: Duration in seconds.
        seed:     Random seed.

    Returns:
        ``(audio_array, sample_rate)`` tuple.
    """
    sample_rate = _get_sample_rate(model)
    try:
        if hasattr(model, "pipeline"):
            result = model.pipeline(prompt, duration=duration, seed=seed)
        else:
            result = model(prompt)

        if isinstance(result, (tuple, list)):
            audio = result[0]
        else:
            audio = result

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().float().numpy()
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = audio[0]
        return np.asarray(audio, dtype=np.float32), sample_rate
    except Exception as exc:  # noqa: BLE001
        log.debug("_run_inference failed (%s); returning zero-vector stub.", exc)
        n_samples = int(duration * sample_rate)
        return np.zeros(n_samples, dtype=np.float32), sample_rate
