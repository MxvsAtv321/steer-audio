"""Unified evaluation metrics for steering vector quality — Phase 3.4-pre.

Provides a single, testable entry point for all steering evaluation metrics
(CLAP alignment, FAD, LPAPS) used across experiment scripts, the Gradio app,
and the scaling law experiment.

Design
------
- Each metric is wrapped in a :class:`MetricBackend` that is loaded *lazily*
  so the module imports cleanly even without GPU or model weights.
- Backends advertise availability via :meth:`MetricBackend.is_available`.
  When unavailable they return ``float("nan")`` and log a warning rather than
  crashing, keeping experiment loops intact.
- :class:`EvalSuite` composes any subset of backends and exposes a single
  :meth:`EvalSuite.evaluate_dir` call that returns a :class:`MetricResult`.
- :func:`compute_alpha_sweep` iterates over the ``alpha_{value}/`` directory
  structure written by ``eval_steering_vectors.py`` and returns a
  ``pandas.DataFrame`` of per-alpha metric values.

Metric backends
---------------
``clap``  — CLAP cosine similarity (text prompt ↔ audio); requires
            ``laion_clap`` and model weights.  Stubs to NaN otherwise.
``fad``   — Fréchet Audio Distance vs a reference directory; requires
            ``audioldm_eval``.  Stubs to NaN otherwise.
``lpaps`` — LPAPS audio preservation (steered vs unsteered baseline).
            Requires ACE-Step / ``editing.eval_medley``.  Stubs to NaN.

Usage
-----
  # Fully self-contained dry-run (no models needed):
  suite = EvalSuite(backends=["clap", "fad"], stub=True)
  result = suite.evaluate_dir(audio_dir=Path("outputs/alpha_50"), prompt="piano")

  # Real evaluation (GPU + model weights):
  suite = EvalSuite(backends=["clap", "fad", "lpaps"])
  df = compute_alpha_sweep(steered_dir=Path("outputs/tempo/tf7"), suite=suite,
                           prompt="fast tempo music", reference_dir=Path("baseline/"))
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MetricResult
# ---------------------------------------------------------------------------

_SENTINEL = float("nan")  # returned by unavailable backends


@dataclass
class MetricResult:
    """Scalar metric values for a single audio directory evaluation.

    Attributes:
        clap_score:  Mean cosine similarity between audio embeddings and the
                     text prompt embedding (range ≈ 0–1; higher = more aligned).
        fad_score:   Fréchet Audio Distance vs a reference set (lower = better).
        lpaps_score: LPAPS preservation metric vs unsteered baseline
                     (lower = better preservation).
        extra:       Any additional scalar metrics keyed by name.
    """

    clap_score: float = _SENTINEL
    fad_score: float = _SENTINEL
    lpaps_score: float = _SENTINEL
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Return all metric values as a flat dictionary."""
        d: Dict[str, float] = {
            "clap": self.clap_score,
            "fad": self.fad_score,
            "lpaps": self.lpaps_score,
        }
        d.update(self.extra)
        return d

    def is_complete(self) -> bool:
        """Return True only if no metric value is NaN."""
        return all(not np.isnan(v) for v in self.to_dict().values())


# ---------------------------------------------------------------------------
# MetricBackend protocol
# ---------------------------------------------------------------------------


class MetricBackend:
    """Base class for lazily-loaded metric implementations.

    Subclasses must implement :meth:`is_available` and :meth:`compute`.
    Instantiation must never import optional heavyweight dependencies.
    """

    name: str = "base"

    def is_available(self) -> bool:
        """Return True if all runtime dependencies are importable."""
        return False

    def compute(
        self,
        audio_dir: Path,
        *,
        prompt: Optional[str] = None,
        reference_dir: Optional[Path] = None,
        **kwargs,
    ) -> float:
        """Compute the metric over *audio_dir* and return a scalar.

        Args:
            audio_dir:     Directory of ``.wav`` files to evaluate.
            prompt:        Text description used by text-audio metrics.
            reference_dir: Unsteered baseline directory (used by FAD, LPAPS).
            **kwargs:      Additional backend-specific parameters.

        Returns:
            Scalar metric value, or ``float("nan")`` if unavailable.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# CLAP backend
# ---------------------------------------------------------------------------


class ClapBackend(MetricBackend):
    """CLAP cosine similarity between audio files and a text prompt.

    Wraps ``laion_clap`` lazily.  Falls back to NaN when unavailable.

    Args:
        use_music_checkpoint: If True, use the CLAP music checkpoint instead
                              of the general-purpose one.
        device:               PyTorch device string (default: ``"cpu"``).
        batch_size:           Audio batch size for embedding extraction.
    """

    name = "clap"

    def __init__(
        self,
        use_music_checkpoint: bool = False,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.use_music_checkpoint = use_music_checkpoint
        self.device = device
        self.batch_size = batch_size
        self._model = None  # loaded on first compute() call

    def is_available(self) -> bool:
        """Return True if ``laion_clap`` is importable."""
        try:
            import laion_clap  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_model(self):
        import laion_clap
        import torch

        if self.use_music_checkpoint:
            model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
            model.load_ckpt(
                "res/clap/pretrained/music_audioset_epoch_15_esc_90.14.pt",
                verbose=False,
            )
        else:
            model = laion_clap.CLAP_Module(enable_fusion=True)
            model.load_ckpt(verbose=False)
        model = model.to(torch.device(self.device)).eval()
        return model

    def compute(
        self,
        audio_dir: Path,
        *,
        prompt: Optional[str] = None,
        reference_dir: Optional[Path] = None,
        **kwargs,
    ) -> float:
        """Compute mean CLAP similarity for WAV files in *audio_dir*.

        Args:
            audio_dir: Directory containing ``.wav`` files.
            prompt:    Text description to compare against.

        Returns:
            Mean cosine similarity in [0, 1], or ``nan`` if unavailable.
        """
        if not self.is_available():
            log.warning("laion_clap not installed — CLAP score is NaN")
            return _SENTINEL

        if prompt is None:
            log.warning("No prompt given for CLAP evaluation — returning NaN")
            return _SENTINEL

        wav_files = sorted(audio_dir.glob("*.wav"))
        if not wav_files:
            log.warning("No WAV files found in %s — CLAP score is NaN", audio_dir)
            return _SENTINEL

        import torch

        if self._model is None:
            self._model = self._load_model()

        template = kwargs.get("prompt_template", "This is a music of {p}")
        text = [template.format(p=prompt)]

        with torch.no_grad():
            text_emb = torch.tensor(
                self._model.get_text_embedding(text)
            ).cpu()  # (1, emb_dim)
            audio_embs = []
            for i in range(0, len(wav_files), self.batch_size):
                batch = [str(p) for p in wav_files[i : i + self.batch_size]]
                emb = self._model.get_audio_embedding_from_filelist(x=batch)
                audio_embs.append(torch.tensor(emb).cpu())
            audio_emb = torch.cat(audio_embs, dim=0)  # (N, emb_dim)

        import torch.nn.functional as F

        audio_norm = F.normalize(audio_emb.float(), dim=1)
        text_norm = F.normalize(text_emb.float(), dim=1)
        sims = (audio_norm @ text_norm.T).squeeze(1)  # (N,)
        return float(sims.mean())


# ---------------------------------------------------------------------------
# FAD backend
# ---------------------------------------------------------------------------


class FadBackend(MetricBackend):
    """Fréchet Audio Distance between generated and reference audio.

    Wraps ``audioldm_eval`` lazily.

    Args:
        device: PyTorch device string (default: ``"cpu"``).
    """

    name = "fad"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def is_available(self) -> bool:
        """Return True if ``audioldm_eval`` is importable."""
        try:
            from audioldm_eval.metrics.fad import FrechetAudioDistance  # noqa: F401
            return True
        except ImportError:
            return False

    def compute(
        self,
        audio_dir: Path,
        *,
        prompt: Optional[str] = None,
        reference_dir: Optional[Path] = None,
        **kwargs,
    ) -> float:
        """Compute FAD between *audio_dir* and *reference_dir*.

        Args:
            audio_dir:     Directory of generated ``.wav`` files.
            reference_dir: Directory of unsteered baseline ``.wav`` files.

        Returns:
            FAD score (lower = better), or ``nan`` if unavailable.
        """
        if not self.is_available():
            log.warning("audioldm_eval not installed — FAD score is NaN")
            return _SENTINEL

        if reference_dir is None or not reference_dir.exists():
            log.warning("No reference_dir for FAD — returning NaN")
            return _SENTINEL

        from audioldm_eval.metrics.fad import FrechetAudioDistance

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fad = FrechetAudioDistance()
            result = fad.score(str(audio_dir), str(reference_dir))
        return float(result.get("frechet_audio_distance", _SENTINEL))


# ---------------------------------------------------------------------------
# LPAPS backend
# ---------------------------------------------------------------------------


class LpapsBackend(MetricBackend):
    """LPAPS audio preservation metric (steered vs unsteered baseline).

    Requires ``editing.eval_medley`` which is part of ACE-Step.
    Stubs to NaN on Python 3.13 / without ACE-Step.
    """

    name = "lpaps"

    def is_available(self) -> bool:
        """Return True if ``editing.eval_medley`` is importable."""
        try:
            from editing.eval_medley import get_lpaps  # noqa: F401
            return True
        except ImportError:
            return False

    def compute(
        self,
        audio_dir: Path,
        *,
        prompt: Optional[str] = None,
        reference_dir: Optional[Path] = None,
        **kwargs,
    ) -> float:
        """Compute LPAPS between steered audio and baseline.

        Args:
            audio_dir:     Directory of steered ``.wav`` files.
            reference_dir: Directory of unsteered baseline ``.wav`` files.

        Returns:
            Mean LPAPS score (lower = better preservation), or ``nan``.
        """
        if not self.is_available():
            log.warning(
                "editing.eval_medley (ACE-Step) not installed — LPAPS is NaN. "
                "Requires Python 3.10-3.12. See docs/scaling_real_runs.md."
            )
            return _SENTINEL

        if reference_dir is None or not reference_dir.exists():
            log.warning("No reference_dir for LPAPS — returning NaN")
            return _SENTINEL

        from editing.eval_medley import get_lpaps

        return float(get_lpaps(str(audio_dir), str(reference_dir)))


# ---------------------------------------------------------------------------
# Stub backend (for tests / dry-run)
# ---------------------------------------------------------------------------


class StubBackend(MetricBackend):
    """Returns a fixed value — for tests and dry-run mode.

    Args:
        name:  Metric name this stub mimics.
        value: Fixed return value (default: 0.5).
    """

    def __init__(self, name: str, value: float = 0.5) -> None:
        self.name = name
        self._value = value

    def is_available(self) -> bool:
        return True

    def compute(self, audio_dir: Path, **kwargs) -> float:  # type: ignore[override]
        """Return the fixed stub value, ignoring all inputs."""
        return self._value


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKEND_CLASSES: Dict[str, type] = {
    "clap": ClapBackend,
    "fad": FadBackend,
    "lpaps": LpapsBackend,
}

# Default stub values used when stub=True (realistic-looking numbers)
_STUB_VALUES: Dict[str, float] = {
    "clap": 0.32,
    "fad": 12.4,
    "lpaps": 0.18,
}


def _make_backend(name: str, stub: bool, **kwargs) -> MetricBackend:
    """Instantiate a named backend, optionally as a stub.

    Args:
        name:   One of ``"clap"``, ``"fad"``, ``"lpaps"``.
        stub:   If True, return a :class:`StubBackend` regardless of
                real backend availability.
        **kwargs: Forwarded to the real backend constructor.

    Returns:
        A :class:`MetricBackend` instance.

    Raises:
        ValueError: If *name* is not a registered backend.
    """
    if name not in _BACKEND_CLASSES:
        raise ValueError(
            f"Unknown backend {name!r}. Choose from: {sorted(_BACKEND_CLASSES)}"
        )
    if stub:
        return StubBackend(name=name, value=_STUB_VALUES.get(name, 0.5))
    return _BACKEND_CLASSES[name](**kwargs)


# ---------------------------------------------------------------------------
# EvalSuite
# ---------------------------------------------------------------------------


class EvalSuite:
    """Compose and run a set of metric backends over an audio directory.

    Args:
        backends: Names of backends to include (any of ``"clap"``, ``"fad"``,
                  ``"lpaps"``).  Defaults to all three.
        stub:     If True, every backend is replaced with a :class:`StubBackend`
                  that returns a fixed value.  Useful for tests and dry-run.
        device:   Device for backends that support it.

    Example::

        suite = EvalSuite(backends=["clap", "fad"], stub=True)
        result = suite.evaluate_dir(Path("outputs/alpha_50"), prompt="piano music")
        print(result.clap_score)  # 0.32 (stub value)
    """

    def __init__(
        self,
        backends: Optional[Sequence[str]] = None,
        stub: bool = False,
        device: str = "cpu",
    ) -> None:
        if backends is None:
            backends = list(_BACKEND_CLASSES.keys())
        self._backends: Dict[str, MetricBackend] = {}
        for name in backends:
            self._backends[name] = _make_backend(name, stub, device=device)

    def availability(self) -> Dict[str, bool]:
        """Return a dict of ``{backend_name: is_available}``."""
        return {name: b.is_available() for name, b in self._backends.items()}

    def evaluate_dir(
        self,
        audio_dir: Path,
        *,
        prompt: Optional[str] = None,
        reference_dir: Optional[Path] = None,
    ) -> MetricResult:
        """Run all registered backends on *audio_dir*.

        Args:
            audio_dir:     Directory containing ``.wav`` files.
            prompt:        Text description (used by CLAP).
            reference_dir: Unsteered baseline directory (used by FAD, LPAPS).

        Returns:
            :class:`MetricResult` with one value per backend.
        """
        result = MetricResult()
        for name, backend in self._backends.items():
            try:
                value = backend.compute(
                    audio_dir,
                    prompt=prompt,
                    reference_dir=reference_dir,
                )
            except Exception as exc:
                log.error("Backend %s raised: %s", name, exc)
                value = _SENTINEL
            setattr(result, f"{name}_score", value)
            log.info("  %-8s = %.4f", name, value)
        return result


# ---------------------------------------------------------------------------
# Alpha-sweep evaluation
# ---------------------------------------------------------------------------


def _parse_alpha_from_dirname(dirname: str) -> Optional[float]:
    """Extract alpha from a directory name like ``alpha_50`` or ``alpha_-10.5``.

    Args:
        dirname: Bare directory name (not a full path).

    Returns:
        Parsed float, or None if the name does not match the pattern.
    """
    match = re.fullmatch(r"alpha_(-?[\d.]+)", dirname)
    if match:
        return float(match.group(1))
    return None


def compute_alpha_sweep(
    steered_dir: Path,
    suite: EvalSuite,
    *,
    prompt: Optional[str] = None,
    reference_dir: Optional[Path] = None,
    alpha_filter: Optional[List[float]] = None,
) -> "pd.DataFrame":
    """Evaluate all ``alpha_*/`` subdirectories in *steered_dir*.

    Expected layout (as written by ``eval_steering_vectors.py``)::

        steered_dir/
          alpha_-100/  *.wav
          alpha_-90/   *.wav
          …
          alpha_100/   *.wav

    Args:
        steered_dir:   Root directory containing ``alpha_*/`` subdirectories.
        suite:         :class:`EvalSuite` instance to use for evaluation.
        prompt:        Forwarded to :meth:`EvalSuite.evaluate_dir`.
        reference_dir: Forwarded to :meth:`EvalSuite.evaluate_dir`.
        alpha_filter:  If given, only evaluate these alpha values.

    Returns:
        ``pd.DataFrame`` with columns ``alpha``, ``clap``, ``fad``, ``lpaps``
        (and any ``extra`` metrics), sorted by alpha.

    Raises:
        FileNotFoundError: If *steered_dir* does not exist.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for compute_alpha_sweep.") from exc

    if not steered_dir.exists():
        raise FileNotFoundError(f"steered_dir not found: {steered_dir}")

    rows = []
    alpha_dirs = [
        (alpha, d)
        for d in sorted(steered_dir.iterdir())
        if d.is_dir() and (alpha := _parse_alpha_from_dirname(d.name)) is not None
    ]
    if not alpha_dirs:
        log.warning("No alpha_*/ subdirectories found in %s", steered_dir)
        return pd.DataFrame(columns=["alpha", "clap", "fad", "lpaps"])

    for alpha, alpha_dir in sorted(alpha_dirs, key=lambda x: x[0]):
        if alpha_filter is not None and alpha not in alpha_filter:
            continue
        log.info("Evaluating alpha=%.1f  (%s)", alpha, alpha_dir.name)
        result = suite.evaluate_dir(
            alpha_dir, prompt=prompt, reference_dir=reference_dir
        )
        row: Dict = {"alpha": alpha}
        row.update(result.to_dict())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def plot_alpha_sweep(
    df: "pd.DataFrame",
    out_dir: Path,
    concept: str = "concept",
) -> List[Path]:
    """Save alignment and preservation plots from an alpha-sweep DataFrame.

    Args:
        df:      Output of :func:`compute_alpha_sweep`.
        out_dir: Directory for output PNGs.
        concept: Concept name used in plot titles.

    Returns:
        List of paths to written PNG files.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[Path] = []

    alpha = df["alpha"].to_numpy()

    # — CLAP alignment vs alpha —
    if "clap" in df.columns and not df["clap"].isna().all():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(alpha, df["clap"].to_numpy(), "b-o", markersize=4)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel("Steering alpha (α)")
        ax.set_ylabel("CLAP alignment score")
        ax.set_title(f"CLAP Alignment vs Alpha — {concept}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / "clap_vs_alpha.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out_paths.append(p)

    # — FAD and LPAPS vs alpha —
    for metric, ylabel, color in [
        ("fad", "FAD (lower = better)", "orange"),
        ("lpaps", "LPAPS (lower = better preservation)", "green"),
    ]:
        if metric in df.columns and not df[metric].isna().all():
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(alpha, df[metric].to_numpy(), f"-o", color=color, markersize=4)
            ax.axvline(0, color="gray", linestyle="--", alpha=0.4)
            ax.set_xlabel("Steering alpha (α)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{metric.upper()} vs Alpha — {concept}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = out_dir / f"{metric}_vs_alpha.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            out_paths.append(p)

    log.info("Saved %d plots to %s", len(out_paths), out_dir)
    return out_paths
