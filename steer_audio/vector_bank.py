"""
Steering vector data type and serialisation registry.

Implements SteeringVector (provenance-rich container) and SteeringVectorBank
(save / load / compose / summarise).

Reference: TADA roadmap Prompt 1.5 — arXiv 2602.11910.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SteeringVector dataclass
# ---------------------------------------------------------------------------

_GRAM_SCHMIDT_EPS: float = 1e-8


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
        alpha_range:    ``(min_alpha, max_alpha)`` empirically validated range.
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
    alpha_range: tuple[float, float] = (-100.0, 100.0)
    n_prompt_pairs: int = 0
    tau: int | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    clap_delta: float = 0.0
    lpaps_at_50: float = 0.0


# ---------------------------------------------------------------------------
# SteeringVectorBank
# ---------------------------------------------------------------------------


class SteeringVectorBank:
    """Registry and I/O for :class:`SteeringVector` objects.

    Vectors are serialised as ``.safetensors`` files with all metadata
    packed into the ``metadata`` field as JSON.
    """

    # Keys excluded from the metadata JSON (stored as tensor data instead).
    _TENSOR_KEYS: frozenset[str] = frozenset({"vector"})

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #

    def save(self, sv: SteeringVector, path: Path) -> None:
        """Serialise *sv* to a ``.safetensors`` file at *path*.

        Args:
            sv:   Steering vector to save.
            path: Destination file path (parent dirs created automatically).

        Raises:
            ImportError: If ``safetensors`` is not installed.
        """
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required for SteeringVectorBank.save(). "
                "Install with: pip install safetensors"
            ) from exc

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialise non-tensor fields as JSON metadata.
        meta: dict[str, Any] = asdict(sv)
        meta.pop("vector")  # stored as tensor, not metadata
        # safetensors metadata values must be strings.
        metadata_str: dict[str, str] = {k: json.dumps(v) for k, v in meta.items()}

        tensors = {"vector": sv.vector.float().contiguous()}
        save_file(tensors, str(path), metadata=metadata_str)
        log.debug("Saved SteeringVector '%s' to %s", sv.concept, path)

    def load(self, path: Path) -> SteeringVector:
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
                "safetensors is required for SteeringVectorBank.load(). "
                "Install with: pip install safetensors"
            ) from exc

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SteeringVector file not found: {path}")

        tensors = load_file(str(path))
        vector = tensors["vector"]

        # Read raw metadata strings.
        with safe_open(str(path), framework="pt") as f:
            raw_meta = f.metadata()

        meta: dict[str, Any] = {k: json.loads(v) for k, v in raw_meta.items()}
        # layers is stored as a JSON list; convert to list[int]
        meta["layers"] = [int(x) for x in meta["layers"]]
        # alpha_range stored as list; convert to tuple
        meta["alpha_range"] = tuple(meta["alpha_range"])

        return SteeringVector(vector=vector, **meta)

    def load_all(self, directory: Path) -> dict[str, SteeringVector]:
        """Load every ``.safetensors`` file in *directory*.

        Args:
            directory: Directory to scan for ``.safetensors`` files.

        Returns:
            Mapping ``"{concept}_{method}" → SteeringVector``.
        """
        directory = Path(directory)
        result: dict[str, SteeringVector] = {}
        for p in sorted(directory.glob("*.safetensors")):
            try:
                sv = self.load(p)
                key = f"{sv.concept}_{sv.method}"
                result[key] = sv
                log.debug("Loaded '%s' from %s", key, p)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to load %s: %s", p, exc)
        return result

    # ------------------------------------------------------------------ #
    # Compose
    # ------------------------------------------------------------------ #

    def compose(
        self,
        vectors: list[tuple[SteeringVector, float]],
    ) -> dict[int, torch.Tensor]:
        """Compose multiple (vector, alpha) pairs into a per-layer delta dict.

        Applies Gram-Schmidt orthogonalization when two vectors share a layer,
        ordering by decreasing ``clap_delta`` (most reliable concept first).

        Args:
            vectors: List of ``(SteeringVector, alpha)`` pairs.

        Returns:
            ``{layer_idx: Σ_c α_c · v_c}`` for all affected layers.
        """
        # Collect per-layer contributions before orthogonalization.
        layer_vecs: dict[int, list[tuple[torch.Tensor, float]]] = {}
        for sv, alpha in vectors:
            for layer in sv.layers:
                layer_vecs.setdefault(layer, []).append((sv.vector.float(), alpha))

        result: dict[int, torch.Tensor] = {}
        for layer, contributions in layer_vecs.items():
            if len(contributions) == 1:
                v, a = contributions[0]
                result[layer] = a * F.normalize(v, dim=0)
            else:
                # Gram-Schmidt: sort by descending |alpha| as a proxy for importance.
                contributions.sort(key=lambda x: -abs(x[1]))
                ortho: list[torch.Tensor] = []
                delta = torch.zeros_like(contributions[0][0])
                for v, a in contributions:
                    v = v.clone()
                    for u in ortho:
                        v = v - v.dot(u) * u
                    norm = v.norm()
                    if norm > _GRAM_SCHMIDT_EPS:
                        v = v / norm
                    ortho.append(v)
                    delta = delta + a * v
                result[layer] = delta
        return result

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def summary_table(self, loaded: dict[str, SteeringVector]) -> str:
        """Return a rich-formatted table of *loaded* vectors.

        Args:
            loaded: Mapping returned by :meth:`load_all`.

        Returns:
            Multi-line string (rich markup) suitable for console output.
        """
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
            # Fallback plain-text table if rich is not installed.
            lines = ["Steering Vector Bank", "-" * 60]
            for key, sv in sorted(loaded.items()):
                lines.append(
                    f"{key:30s}  layers={sv.layers}  clap_delta={sv.clap_delta:.3f}"
                )
            return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Download stub
    # ------------------------------------------------------------------ #


def download_pretrained(concept: str, model: str = "ace-step") -> SteeringVector:
    """Download a pre-trained steering vector from HuggingFace Hub.

    Args:
        concept: Concept name, e.g. ``"tempo"``.
        model:   Model identifier (default ``"ace-step"``).

    Raises:
        NotImplementedError: Always — vectors are not yet hosted on the Hub.
            Upload vectors first via ``huggingface_hub.upload_file`` to the
            ``tada-steering-vectors/{model}`` repository.
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
