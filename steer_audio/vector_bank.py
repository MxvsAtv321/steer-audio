"""
Steering vector data type and serialization registry.

Implements SteeringVector (provenance-rich container) and SteeringVectorBank
(stateful registry with save/load/compose/summarise).

Reference: TADA roadmap Prompt 2.1 — arXiv 2602.11910.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GRAM_SCHMIDT_EPS: float = 1e-8
_DEFAULT_ALPHA_RANGE: List[float] = [float(a) for a in range(-100, 101, 10)]


# ---------------------------------------------------------------------------
# SteeringVector
# ---------------------------------------------------------------------------


@dataclass
class SteeringVector:
    """A computed steering vector with full provenance metadata.

    Attributes:
        concept:        Semantic concept name, e.g. "tempo".
        method:         Computation method: "caa" or "sae".
        model_name:     Model identifier, e.g. "ace-step".
        layers:         Transformer-block indices where this vector is applied.
        vector:         Steering direction; shape ``(hidden_dim,)``.
                        Unit-norm for CAA; unnormalized for SAE.
        alpha_range:    List of alpha values to sweep; defaults to −100…100 step 10.
        metadata:       Arbitrary provenance metadata dict.
        n_prompt_pairs: Number of prompt pairs used in CAA computation.
        tau:            Top-τ SAE features (``None`` for CAA).
        created_at:     ISO-8601 creation timestamp.
        clap_delta:     ΔAlignment at alpha=50 — higher means stronger concept signal.
        lpaps_at_50:    LPAPS at alpha=50 — lower means better audio preservation.
    """

    concept: str
    method: str
    model_name: str
    layers: list[int]
    vector: torch.Tensor
    alpha_range: list[float] = field(default_factory=lambda: list(_DEFAULT_ALPHA_RANGE))
    metadata: dict[str, Any] = field(default_factory=dict)
    n_prompt_pairs: int = 0
    tau: int | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    clap_delta: float = 0.0
    lpaps_at_50: float = 0.0

    # ------------------------------------------------------------------
    # Prompt 2.1: model property alias
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """Alias for ``model_name`` (Prompt 2.1 field name)."""
        return self.model_name

    # ------------------------------------------------------------------
    # Prompt 2.1: norm / cosine_similarity
    # ------------------------------------------------------------------

    def norm(self, layer: int | None = None) -> float:
        """L2 norm of the steering vector.

        Args:
            layer: Ignored (vector is shared across all layers).

        Returns:
            Scalar L2 norm.
        """
        return self.vector.float().norm().item()

    def cosine_similarity(self, other: "SteeringVector", layer: int | None = None) -> float:
        """Cosine similarity between this vector and *other*.

        Args:
            other: Another :class:`SteeringVector`.
            layer: Ignored (vector is shared across all layers).

        Returns:
            Scalar in ``[-1, 1]``.
        """
        v1 = F.normalize(self.vector.float(), dim=0)
        v2 = F.normalize(other.vector.float(), dim=0)
        return v1.dot(v2).item()

    # ------------------------------------------------------------------
    # Prompt 2.1: __repr__
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SteeringVector(concept={self.concept!r}, method={self.method!r}, "
            f"model={self.model_name!r}, layers={self.layers}, "
            f"dim={self.vector.shape[0]}, norm={self.norm():.4f}, "
            f"clap_delta={self.clap_delta:.3f})"
        )

    # ------------------------------------------------------------------
    # Prompt 2.1: save / load (instance methods on SteeringVector)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize to ``.safetensors`` + sidecar ``.json``.

        Args:
            path: Destination path without extension (or with ``.safetensors``).
                  A ``.json`` sidecar is written alongside the tensor file.

        Raises:
            ImportError: If ``safetensors`` is not installed.
        """
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required. Install with: pip install safetensors"
            ) from exc

        p = Path(path)
        if p.suffix != ".safetensors":
            p = p.with_suffix(".safetensors")
        p.parent.mkdir(parents=True, exist_ok=True)

        # Build metadata dict for sidecar.
        meta: dict[str, Any] = {
            "concept": self.concept,
            "method": self.method,
            "model_name": self.model_name,
            "layers": self.layers,
            "alpha_range": self.alpha_range,
            "metadata": self.metadata,
            "n_prompt_pairs": self.n_prompt_pairs,
            "tau": self.tau,
            "created_at": self.created_at,
            "clap_delta": self.clap_delta,
            "lpaps_at_50": self.lpaps_at_50,
        }

        # safetensors requires all metadata values to be strings.
        meta_str: dict[str, str] = {k: json.dumps(v) for k, v in meta.items()}
        save_file({"vector": self.vector.float().contiguous()}, str(p), metadata=meta_str)

        # Sidecar JSON for human readability.
        sidecar = p.with_suffix(".json")
        sidecar.write_text(json.dumps(meta, indent=2))

        log.debug("Saved SteeringVector '%s' to %s", self.concept, p)

    @classmethod
    def load(cls, path: str) -> "SteeringVector":
        """Load a :class:`SteeringVector` from a ``.safetensors`` file.

        Args:
            path: Path to a ``.safetensors`` file written by :meth:`save`.

        Returns:
            Reconstructed :class:`SteeringVector`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ImportError: If ``safetensors`` is not installed.
        """
        try:
            from safetensors.torch import load_file, safe_open
        except ImportError as exc:
            raise ImportError(
                "safetensors is required. Install with: pip install safetensors"
            ) from exc

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"SteeringVector file not found: {p}")

        tensors = load_file(str(p))
        vector = tensors["vector"]

        with safe_open(str(p), framework="pt") as f:
            raw_meta = f.metadata()

        meta: dict[str, Any] = {k: json.loads(v) for k, v in raw_meta.items()}
        meta["layers"] = [int(x) for x in meta["layers"]]
        # alpha_range stored as list; ensure list[float]
        meta["alpha_range"] = [float(x) for x in meta.get("alpha_range", _DEFAULT_ALPHA_RANGE)]
        # metadata dict
        meta.setdefault("metadata", {})

        return cls(vector=vector, **meta)


# ---------------------------------------------------------------------------
# SteeringVectorBank
# ---------------------------------------------------------------------------


class SteeringVectorBank:
    """Stateful registry and I/O helper for :class:`SteeringVector` objects.

    Vectors are keyed internally as ``"{concept}_{method}"`` (e.g. ``"tempo_caa"``).

    Implements a dict-like interface so that code expecting a plain mapping
    (e.g. ``SteeringPipeline(bank.load_all(dir))``) continues to work.

    Prompt 2.1 API
    --------------
    ``add`` / ``get`` / ``list`` / ``save_all`` / ``load_all`` (classmethod) /
    ``compose`` / ``interference_matrix``

    Legacy helpers (kept for backward compatibility)
    ------------------------------------------------
    ``save(sv, path)`` / ``load(path)`` / ``summary_table(loaded)``
    """

    def __init__(self) -> None:
        self._registry: dict[str, SteeringVector] = {}

    # ------------------------------------------------------------------
    # Prompt 2.1: add / get / list
    # ------------------------------------------------------------------

    def add(self, vector: SteeringVector) -> None:
        """Register *vector* in the bank.

        Args:
            vector: :class:`SteeringVector` to add.
        """
        key = f"{vector.concept}_{vector.method}"
        self._registry[key] = vector
        log.debug("Added SteeringVector '%s' to bank.", key)

    def get(self, concept: str, method: str = "caa") -> SteeringVector:
        """Retrieve a vector by concept and method.

        Args:
            concept: Concept name, e.g. ``"tempo"``.
            method:  Computation method (``"caa"`` or ``"sae"``).

        Returns:
            The stored :class:`SteeringVector`.

        Raises:
            KeyError: If no matching vector is found.
        """
        key = f"{concept}_{method}"
        if key not in self._registry:
            raise KeyError(f"No SteeringVector for concept='{concept}' method='{method}'.")
        return self._registry[key]

    def list(self) -> List[str]:
        """Return a list of registered ``"concept/method"`` identifiers.

        Returns:
            Sorted list of strings, e.g. ``["mood/caa", "tempo/caa"]``.
        """
        # Convert internal underscore keys to slash-separated for Prompt 2.1 API.
        result = []
        for key in self._registry:
            # key is "{concept}_{method}"; method has no underscores (caa/sae).
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                result.append(f"{parts[0]}/{parts[1]}")
            else:
                result.append(key)
        return sorted(result)

    # ------------------------------------------------------------------
    # Prompt 2.1: save_all / load_all (classmethod)
    # ------------------------------------------------------------------

    def save_all(self, directory: str) -> None:
        """Save all registered vectors to *directory*.

        Args:
            directory: Target directory (created if absent).
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for key, sv in self._registry.items():
            sv.save(str(d / f"{key}.safetensors"))
        log.debug("Saved %d vectors to %s.", len(self._registry), d)

    @classmethod
    def load_all(cls, directory: "str | Path") -> "SteeringVectorBank":
        """Load every ``.safetensors`` file in *directory* into a new bank.

        Args:
            directory: Directory to scan for ``.safetensors`` files.

        Returns:
            A new :class:`SteeringVectorBank` populated with the loaded vectors.

        Note:
            When called on an *instance* this still returns a **new** bank (it
            behaves as a classmethod).  Existing code that treats the return
            value as a dict continues to work because :class:`SteeringVectorBank`
            implements a dict-like interface.
        """
        bank = cls()
        for p in sorted(Path(directory).glob("*.safetensors")):
            try:
                sv = SteeringVector.load(str(p))
                key = f"{sv.concept}_{sv.method}"
                bank._registry[key] = sv
                log.debug("Loaded '%s' from %s", key, p)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to load %s: %s", p, exc)
        return bank

    # ------------------------------------------------------------------
    # Prompt 2.1: compose
    # ------------------------------------------------------------------

    def compose(
        self,
        concepts_or_vectors,
        method: str = "caa",
        orthogonalize: bool = True,
    ) -> dict[int, torch.Tensor]:
        """Compose steering vectors into a per-layer delta dict.

        Supports two call signatures:

        **New (Prompt 2.1):**
            ``compose(["tempo", "mood"], method="caa", orthogonalize=True)``
            Fetches vectors from the internal registry and applies Gram-Schmidt
            if ``orthogonalize=True``.  Returns ``{layer: combined_tensor}``.

        **Legacy:**
            ``compose([(sv1, alpha1), (sv2, alpha2)])``
            Computes ``Σ α_c · v_c`` per layer with optional Gram-Schmidt.

        Args:
            concepts_or_vectors:
                Either a list of concept name strings (new API) or a list of
                ``(SteeringVector, alpha)`` tuples (legacy API).
            method:         Method to use when fetching by name (new API only).
            orthogonalize:  If ``True``, apply Gram-Schmidt before summing.

        Returns:
            ``{layer_idx: delta_tensor}`` for all affected layers.
        """
        if not concepts_or_vectors:
            return {}

        # Detect API variant.
        first = concepts_or_vectors[0]
        if isinstance(first, str):
            # New API: look up by concept name.
            pairs: list[tuple[SteeringVector, float]] = []
            for concept in concepts_or_vectors:
                sv = self.get(concept, method)
                pairs.append((sv, 1.0))  # unit alpha; caller scales externally
        else:
            # Legacy API: list of (SteeringVector, alpha) tuples.
            pairs = list(concepts_or_vectors)

        # Collect per-layer contributions.
        layer_vecs: dict[int, list[tuple[torch.Tensor, float, str]]] = {}
        for sv, alpha in pairs:
            for layer in sv.layers:
                layer_vecs.setdefault(layer, []).append(
                    (sv.vector.float(), alpha, sv.method)
                )

        result: dict[int, torch.Tensor] = {}
        for layer, contributions in layer_vecs.items():
            if len(contributions) == 1:
                v, a, _m = contributions[0]
                result[layer] = a * F.normalize(v, dim=0)
            else:
                # Sort by descending |alpha| to anchor the most important vector.
                contributions.sort(key=lambda x: -abs(x[1]))
                if orthogonalize:
                    ortho: list[torch.Tensor] = []
                    delta = torch.zeros_like(contributions[0][0])
                    for v, a, _m in contributions:
                        v = v.clone()
                        for u in ortho:
                            v = v - v.dot(u) * u
                        norm = v.norm()
                        if norm < _GRAM_SCHMIDT_EPS:
                            warnings.warn(
                                f"A vector has near-zero residual norm {norm:.4f} "
                                "after Gram-Schmidt orthogonalization — it is nearly "
                                "parallel to a prior concept and will have no effect. "
                                "Consider removing it or choosing a different concept.",
                                stacklevel=3,
                            )
                        else:
                            v = v / norm
                        ortho.append(v)
                        delta = delta + a * v
                    result[layer] = delta
                else:
                    delta = torch.zeros_like(contributions[0][0])
                    for v, a, _m in contributions:
                        delta = delta + a * F.normalize(v, dim=0)
                    result[layer] = delta

        return result

    # ------------------------------------------------------------------
    # Prompt 2.1: interference_matrix
    # ------------------------------------------------------------------

    def interference_matrix(self, layer: int | None = None) -> np.ndarray:
        """Compute the N×N pairwise cosine-similarity matrix.

        Args:
            layer: Ignored (each :class:`SteeringVector` stores a single
                   direction shared across all layers).

        Returns:
            ``np.ndarray`` of shape ``(N, N)`` where ``matrix[i, j]``
            is the cosine similarity between concepts *i* and *j*.
        """
        if not self._registry:
            return np.zeros((0, 0), dtype=np.float32)

        keys = list(self._registry.keys())
        vecs = torch.stack(
            [F.normalize(self._registry[k].vector.float(), dim=0) for k in keys]
        )  # (N, D)
        mat = (vecs @ vecs.T).cpu().numpy().astype(np.float32)
        return mat

    # ------------------------------------------------------------------
    # Dict-like interface (for backward compatibility)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._registry)

    def __bool__(self) -> bool:
        return bool(self._registry)

    def __contains__(self, item: object) -> bool:
        return item in self._registry

    def __getitem__(self, key: str) -> SteeringVector:
        return self._registry[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._registry)

    def keys(self):
        return self._registry.keys()

    def values(self):
        return self._registry.values()

    def items(self):
        return self._registry.items()

    # ------------------------------------------------------------------
    # Legacy instance helpers (kept for backward compatibility)
    # ------------------------------------------------------------------

    def save(self, sv: SteeringVector, path: "Path | str") -> None:
        """Serialize *sv* to a ``.safetensors`` file at *path*.

        Legacy helper — new code should call ``sv.save(path)`` directly.
        """
        sv.save(str(path))
        # Also register in the bank so the instance stays in sync.
        self.add(sv)

    def load(self, path: "Path | str") -> SteeringVector:
        """Load a :class:`SteeringVector` from a ``.safetensors`` file.

        Legacy helper — new code should call ``SteeringVector.load(path)`` directly.
        """
        sv = SteeringVector.load(str(path))
        self.add(sv)
        return sv

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def summary_table(self, loaded: "dict[str, SteeringVector] | SteeringVectorBank | None" = None) -> str:
        """Return a rich-formatted table of loaded vectors.

        Args:
            loaded: Mapping of vectors to display.  Defaults to this bank's
                    internal registry when ``None``.

        Returns:
            Multi-line string (rich markup) suitable for console output.
        """
        if loaded is None:
            loaded = self._registry

        try:
            from rich.table import Table
            from rich.console import Console
            import io

            table = Table(title="Steering Vector Bank")
            table.add_column("Key", style="cyan")
            table.add_column("Concept")
            table.add_column("Method")
            table.add_column("Model")
            table.add_column("Layers")
            table.add_column("CLAP Δ", justify="right")
            table.add_column("LPAPS@50", justify="right")
            table.add_column("Created")

            for key, sv in sorted(loaded.items()):
                table.add_row(
                    key,
                    sv.concept,
                    sv.method,
                    sv.model_name,
                    str(sv.layers),
                    f"{sv.clap_delta:.3f}",
                    f"{sv.lpaps_at_50:.3f}",
                    sv.created_at[:10],
                )

            buf = io.StringIO()
            console = Console(file=buf, no_color=False)
            console.print(table)
            return buf.getvalue()
        except ImportError:
            lines = ["Steering Vector Bank", "-" * 60]
            for key, sv in sorted(loaded.items()):
                lines.append(
                    f"{key:30s}  layers={sv.layers}  clap_delta={sv.clap_delta:.3f}"
                )
            return "\n".join(lines)


# ---------------------------------------------------------------------------
# Download stub
# ---------------------------------------------------------------------------


def download_pretrained(concept: str, model: str = "ace-step") -> SteeringVector:
    """Download a pre-trained steering vector from HuggingFace Hub.

    Args:
        concept: Concept name, e.g. ``"tempo"``.
        model:   Model identifier (default ``"ace-step"``).

    Raises:
        NotImplementedError: Always — vectors are not yet hosted on the Hub.
    """
    raise NotImplementedError(
        f"Pre-trained vectors for '{concept}' on '{model}' are not yet available "
        "on HuggingFace Hub.\n"
        "To upload your own, run:\n"
        "  from huggingface_hub import upload_file\n"
        f"  upload_file(path_or_fileobj='vectors/{concept}.safetensors',\n"
        f"              path_in_repo='{concept}.safetensors',\n"
        f"              repo_id='tada-steering-vectors/{model}')"
    )
