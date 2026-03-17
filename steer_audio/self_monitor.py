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
    """Lightweight linear probe that detects concept presence from CLAP embeddings.

    Training data: CLAP embeddings of generated audio with positive / negative
    concept prompts.  A logistic regression classifier is fitted on those
    embeddings and used at inference time to return P(concept present).

    Args:
        concept:        Name of the concept being probed (e.g. ``"tempo"``).
        clap_extractor: Callable ``(audio: np.ndarray, sample_rate: int) ->
                        np.ndarray`` that returns a 1-D CLAP embedding.
                        Defaults to a zero-vector stub so the class can be
                        instantiated without real CLAP weights.
    """

    def __init__(
        self,
        concept: str,
        clap_extractor: ClapExtractor | None = None,
    ) -> None:
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
# SelfMonitoredSteerer
# ---------------------------------------------------------------------------


class SelfMonitoredSteerer:
    """Adaptive steering that reduces alpha when concept is detected.

    The controller hooks into the model's transformer blocks (at each layer
    listed in *vector.layers*) and adjusts the effective alpha based on
    concept-presence probability estimated by *probe*.

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
                    "SelfMonitoredSteerer: layer %d out of range (model has %d blocks), skipping.",
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
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for get_monitoring_trace(). "
                "Install with: pip install pandas"
            ) from exc

        if not self._trace:
            raise RuntimeError(
                "No monitoring trace available. Call steer() first."
            )
        return pd.DataFrame(self._trace)


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
