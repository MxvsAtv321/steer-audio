"""
SAE Concept Algebra System — Phase 2.3.

Implements concept arithmetic over SAE feature sets: add (union), subtract
(set difference), intersect, and weighted blend of musical concepts.

Mathematical foundation (arXiv 2602.11910 §3.3):
  Concept c ↔ top-τ feature set F_c ⊆ {1 … num_features}
  Steering vector: v_c = Σ_{j ∈ F_c} W_dec[:, j]

  Addition  (A + B): F_A ∪ F_B → scores combined additively
  Subtraction (A - B): F_A \\ F_B → use F_A scores for kept indices
  Intersection (A & B): F_A ∩ F_B → use min of scores
  Weighted (0.7*A): scale all TF-IDF scores by 0.7

Reference: TADA roadmap Prompt 2.3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from steer_audio.vector_bank import SteeringVector

matplotlib.use("Agg")  # headless-safe backend; override before pyplot import

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ConceptFeatureSet
# ---------------------------------------------------------------------------


@dataclass
class ConceptFeatureSet:
    """Sparse representation of a concept as a set of SAE feature indices + scores.

    Attributes:
        concept:        Human-readable concept name or algebra expression string.
        feature_indices: Top-τ feature indices, shape ``(τ,)``.
        tfidf_scores:   TF-IDF score for each feature, shape ``(τ,)``.
                        Used as per-feature weights in :meth:`to_steering_vector`.
        decoder_matrix: W_dec from the trained SAE, shape ``(hidden_dim, num_features)``.
                        Column j is the decoder direction for feature j.
    """

    concept: str
    feature_indices: np.ndarray   # shape (τ,), dtype int64
    tfidf_scores: np.ndarray      # shape (τ,), dtype float32
    decoder_matrix: torch.Tensor  # shape (hidden_dim, num_features)

    # ------------------------------------------------------------------ #
    # Steering vector construction
    # ------------------------------------------------------------------ #

    def to_steering_vector(self, weight: float = 1.0) -> torch.Tensor:
        """Sum decoder columns weighted by TF-IDF scores.

        Computes:
            weight * Σ_{j ∈ feature_indices} tfidf_score[j] * W_dec[:, j]

        Args:
            weight: Overall scale factor applied after the feature sum.

        Returns:
            Float tensor of shape ``(hidden_dim,)``.  Zero vector when
            ``feature_indices`` is empty.
        """
        hidden_dim = self.decoder_matrix.shape[0]

        if len(self.feature_indices) == 0:
            log.warning(
                "ConceptFeatureSet '%s' has no features; returning zero vector.",
                self.concept,
            )
            return torch.zeros(hidden_dim, dtype=torch.float32)

        # Vectorised column sum: W_dec[:, F_c] @ scores  → shape (hidden_dim,)
        indices = torch.from_numpy(self.feature_indices.astype(np.int64))
        scores = torch.from_numpy(self.tfidf_scores.astype(np.float32))

        selected_cols = self.decoder_matrix.float()[:, indices]  # (hidden_dim, τ)
        vector = selected_cols @ scores  # (hidden_dim,)

        return weight * vector

    # ------------------------------------------------------------------ #
    # Jaccard overlap
    # ------------------------------------------------------------------ #

    def overlap(self, other: ConceptFeatureSet) -> float:
        """Jaccard similarity between two concept feature sets.

        Computes |F_a ∩ F_b| / |F_a ∪ F_b|.

        Args:
            other: The other concept feature set.

        Returns:
            Float in ``[0, 1]``.  Returns ``0.0`` if both sets are empty.
        """
        set_a = set(self.feature_indices.tolist())
        set_b = set(other.feature_indices.tolist())
        union_size = len(set_a | set_b)
        if union_size == 0:
            return 0.0
        return len(set_a & set_b) / union_size

    # ------------------------------------------------------------------ #
    # Arithmetic operators
    # ------------------------------------------------------------------ #

    def __add__(self, other: ConceptFeatureSet) -> ConceptFeatureSet:
        """Union of feature sets (F_A ∪ F_B) with additive score combination.

        For features present in both sets, scores are summed to reinforce
        shared discriminative features.

        Args:
            other: Another :class:`ConceptFeatureSet` from the *same* SAE.

        Returns:
            New :class:`ConceptFeatureSet` with unioned indices.
        """
        scores_a = dict(
            zip(self.feature_indices.tolist(), self.tfidf_scores.tolist())
        )
        scores_b = dict(
            zip(other.feature_indices.tolist(), other.tfidf_scores.tolist())
        )
        all_indices = sorted(set(scores_a.keys()) | set(scores_b.keys()))
        result_indices = np.array(all_indices, dtype=np.int64)
        result_scores = np.array(
            [scores_a.get(j, 0.0) + scores_b.get(j, 0.0) for j in all_indices],
            dtype=np.float32,
        )
        return ConceptFeatureSet(
            concept=f"({self.concept}+{other.concept})",
            feature_indices=result_indices,
            tfidf_scores=result_scores,
            decoder_matrix=self.decoder_matrix,
        )

    def __sub__(self, other: ConceptFeatureSet) -> ConceptFeatureSet:
        """Set difference F_A \\ F_B — keep features in A not in B.

        Useful for removing a sub-concept (e.g. fast_tempo - drums).

        Args:
            other: Concept whose features are to be excluded.

        Returns:
            New :class:`ConceptFeatureSet` containing only F_A \\ F_B features.
        """
        set_b = set(other.feature_indices.tolist())
        mask = np.array(
            [j not in set_b for j in self.feature_indices.tolist()], dtype=bool
        )
        return ConceptFeatureSet(
            concept=f"({self.concept}-{other.concept})",
            feature_indices=self.feature_indices[mask],
            tfidf_scores=self.tfidf_scores[mask],
            decoder_matrix=self.decoder_matrix,
        )

    def __and__(self, other: ConceptFeatureSet) -> ConceptFeatureSet:
        """Intersection F_A ∩ F_B — keep only features shared by both.

        Shared feature scores are taken as the minimum of the two scores
        (conservative: only features *both* concepts strongly agree on).

        Args:
            other: Concept to intersect with.

        Returns:
            New :class:`ConceptFeatureSet` containing F_A ∩ F_B features.
        """
        scores_b = dict(
            zip(other.feature_indices.tolist(), other.tfidf_scores.tolist())
        )
        set_b = set(scores_b.keys())

        indices_a = self.feature_indices.tolist()
        scores_a_list = self.tfidf_scores.tolist()

        result_indices: list[int] = []
        result_scores: list[float] = []
        for idx, score_a in zip(indices_a, scores_a_list):
            if idx in set_b:
                result_indices.append(idx)
                result_scores.append(min(score_a, scores_b[idx]))

        return ConceptFeatureSet(
            concept=f"({self.concept}&{other.concept})",
            feature_indices=np.array(result_indices, dtype=np.int64),
            tfidf_scores=np.array(result_scores, dtype=np.float32),
            decoder_matrix=self.decoder_matrix,
        )

    def __mul__(self, scalar: float) -> ConceptFeatureSet:
        """Scale TF-IDF weights by *scalar* (proportional weighting).

        Used for weighted blends, e.g. ``0.7 * jazz``.

        Args:
            scalar: Non-negative scale factor.

        Returns:
            New :class:`ConceptFeatureSet` with scores scaled by *scalar*.
        """
        if scalar < 0:
            log.warning(
                "Negative scalar (%.3f) passed to ConceptFeatureSet.__mul__; "
                "this will invert feature contributions.",
                scalar,
            )
        return ConceptFeatureSet(
            concept=f"({scalar}*{self.concept})",
            feature_indices=self.feature_indices.copy(),
            tfidf_scores=self.tfidf_scores * float(scalar),
            decoder_matrix=self.decoder_matrix,
        )

    def __rmul__(self, scalar: float) -> ConceptFeatureSet:
        """Support ``scalar * ConceptFeatureSet`` syntax."""
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        return (
            f"ConceptFeatureSet(concept={self.concept!r}, "
            f"n_features={len(self.feature_indices)}, "
            f"hidden_dim={self.decoder_matrix.shape[0]})"
        )

    # ------------------------------------------------------------------ #
    # Factory helper: build from a trained Sae model
    # ------------------------------------------------------------------ #

    @classmethod
    def from_sae(
        cls,
        sae_model: Any,
        concept: str,
        feature_indices: np.ndarray,
        tfidf_scores: np.ndarray,
    ) -> ConceptFeatureSet:
        """Construct a :class:`ConceptFeatureSet` from a trained ``Sae`` model.

        Extracts and transposes ``sae_model.W_dec`` from its native shape
        ``(num_features, hidden_dim)`` to the expected ``(hidden_dim, num_features)``.

        Args:
            sae_model:      Trained ``Sae`` instance (from ``sae_src.sae.sae``).
            concept:        Human-readable concept name.
            feature_indices: Top-τ feature indices, shape ``(τ,)``.
            tfidf_scores:   TF-IDF scores corresponding to *feature_indices*.

        Returns:
            :class:`ConceptFeatureSet` with the decoder matrix embedded.

        Raises:
            AttributeError: If *sae_model* does not expose ``W_dec``.
        """
        # Sae.W_dec is a Parameter of shape (num_features, hidden_dim)
        w_dec_raw: torch.Tensor = sae_model.W_dec.detach().float()
        # Transpose → (hidden_dim, num_features)
        decoder_matrix = w_dec_raw.T.contiguous()
        return cls(
            concept=concept,
            feature_indices=feature_indices,
            tfidf_scores=tfidf_scores,
            decoder_matrix=decoder_matrix,
        )


# ---------------------------------------------------------------------------
# Expression parser (recursive descent)
# ---------------------------------------------------------------------------


class _Token:
    """Single lexer token."""

    __slots__ = ("type", "value")

    def __init__(self, type_: str, value: Any) -> None:
        self.type = type_
        self.value = value

    def __repr__(self) -> str:
        return f"_Token({self.type!r}, {self.value!r})"


def _tokenize(text: str) -> list[_Token]:
    """Lex *text* into a list of tokens followed by an EOF token.

    Supported tokens: NUMBER, IDENT, PLUS, MINUS, STAR, AMP, LPAREN, RPAREN, EOF.

    Args:
        text: Raw expression string.

    Returns:
        List of :class:`_Token` objects (last element is always EOF).

    Raises:
        ValueError: On an unrecognised character.
    """
    _SINGLE = {"+": "PLUS", "-": "MINUS", "*": "STAR", "&": "AMP", "(": "LPAREN", ")": "RPAREN"}
    tokens: list[_Token] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
        elif ch in _SINGLE:
            tokens.append(_Token(_SINGLE[ch], ch))
            i += 1
        elif ch.isdigit() or ch == ".":
            j = i
            while j < len(text) and (text[j].isdigit() or text[j] == "."):
                j += 1
            tokens.append(_Token("NUMBER", float(text[i:j])))
            i = j
        elif ch.isalpha() or ch == "_":
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(_Token("IDENT", text[i:j]))
            i = j
        else:
            raise ValueError(
                f"Unexpected character {ch!r} at position {i} in expression {text!r}"
            )
    tokens.append(_Token("EOF", None))
    return tokens


class _ExpressionParser:
    """Recursive descent parser for concept algebra expressions.

    Grammar::

        expr     := term (('+' | '-') term)*
        term     := factor ('&' factor)*
        factor   := NUMBER '*' atom | atom ('*' NUMBER)?
        atom     := IDENT | '(' expr ')'

    Args:
        text:     Raw expression string.
        features: Mapping from concept name to :class:`ConceptFeatureSet`.
    """

    def __init__(self, text: str, features: dict[str, ConceptFeatureSet]) -> None:
        self.features = features
        self.tokens = _tokenize(text)
        self.pos = 0

    # -- token primitives ---------------------------------------------------

    def _peek(self) -> _Token:
        return self.tokens[self.pos]

    def _consume(self, expected_type: str | None = None) -> _Token:
        tok = self.tokens[self.pos]
        if expected_type is not None and tok.type != expected_type:
            raise ValueError(
                f"Expected token {expected_type!r} but got {tok.type!r} "
                f"(value={tok.value!r}) at position {self.pos}."
            )
        self.pos += 1
        return tok

    # -- grammar rules ------------------------------------------------------

    def parse(self) -> ConceptFeatureSet:
        """Parse the full expression and return its :class:`ConceptFeatureSet`.

        Raises:
            ValueError: On syntax errors or trailing tokens.
            KeyError:   On unknown concept identifiers.
        """
        result = self._parse_expr()
        if self._peek().type != "EOF":
            raise ValueError(
                f"Unexpected trailing token {self._peek().value!r} in expression."
            )
        return result

    def _parse_expr(self) -> ConceptFeatureSet:
        """Parse: term (('+' | '-') term)*"""
        left = self._parse_term()
        while self._peek().type in ("PLUS", "MINUS"):
            op = self._consume().type
            right = self._parse_term()
            if op == "PLUS":
                left = left + right
            else:
                left = left - right
        return left

    def _parse_term(self) -> ConceptFeatureSet:
        """Parse: factor ('&' factor)*"""
        left = self._parse_factor()
        while self._peek().type == "AMP":
            self._consume("AMP")
            right = self._parse_factor()
            left = left & right
        return left

    def _parse_factor(self) -> ConceptFeatureSet:
        """Parse: NUMBER '*' atom | atom ('*' NUMBER)?

        Supports both ``0.7 * jazz`` and ``jazz * 0.7`` syntaxes.
        """
        if self._peek().type == "NUMBER":
            scalar = self._consume("NUMBER").value
            self._consume("STAR")
            atom = self._parse_atom()
            return scalar * atom
        else:
            atom = self._parse_atom()
            if self._peek().type == "STAR":
                self._consume("STAR")
                scalar = self._consume("NUMBER").value
                return atom * scalar
            return atom

    def _parse_atom(self) -> ConceptFeatureSet:
        """Parse: IDENT | '(' expr ')'"""
        if self._peek().type == "LPAREN":
            self._consume("LPAREN")
            result = self._parse_expr()
            self._consume("RPAREN")
            return result
        elif self._peek().type == "IDENT":
            name = self._consume("IDENT").value
            if name not in self.features:
                available = sorted(self.features.keys())
                raise KeyError(
                    f"Concept {name!r} not found in algebra. "
                    f"Available concepts: {available}"
                )
            # Return a copy so mutations don't affect the registry.
            cfs = self.features[name]
            return ConceptFeatureSet(
                concept=cfs.concept,
                feature_indices=cfs.feature_indices.copy(),
                tfidf_scores=cfs.tfidf_scores.copy(),
                decoder_matrix=cfs.decoder_matrix,
            )
        else:
            raise ValueError(
                f"Expected concept name or '(' but got {self._peek().type!r} "
                f"(value={self._peek().value!r})."
            )


# ---------------------------------------------------------------------------
# ConceptAlgebra
# ---------------------------------------------------------------------------


class ConceptAlgebra:
    """Perform concept arithmetic and produce steering vectors from SAE feature algebra.

    Provides an expression evaluator (via :meth:`expr`) and a heatmap visualiser
    (via :meth:`feature_overlap_heatmap`).

    Args:
        sae_model:        Trained ``Sae`` model instance.  May be ``None`` when
                          ``ConceptFeatureSet`` objects are built externally (the SAE
                          is only used by :meth:`to_steering_vector`).
        concept_features: Mapping ``concept_name → ConceptFeatureSet``.

    Example::

        algebra = ConceptAlgebra(sae, features)
        result = algebra.expr("0.7 * jazz + 0.3 * reggae")
        sv = algebra.to_steering_vector(result)
    """

    def __init__(
        self,
        sae_model: Any,
        concept_features: dict[str, ConceptFeatureSet],
    ) -> None:
        self.sae = sae_model
        self.features: dict[str, ConceptFeatureSet] = dict(concept_features)

    # ------------------------------------------------------------------ #
    # Expression evaluator
    # ------------------------------------------------------------------ #

    def expr(self, expression: str) -> ConceptFeatureSet:
        """Parse and evaluate a concept algebra expression.

        Supported syntax examples::

            algebra.expr("jazz + female_vocal - piano")
            algebra.expr("0.7 * jazz + 0.3 * techno")
            algebra.expr("fast_tempo & energetic_mood")

        Args:
            expression: Algebra expression string.

        Returns:
            :class:`ConceptFeatureSet` representing the result.

        Raises:
            ValueError: On parse / syntax errors.
            KeyError:   On unknown concept names in the expression.
        """
        parser = _ExpressionParser(expression, self.features)
        result = parser.parse()
        log.debug("Evaluated expression %r → %r", expression, result.concept)
        return result

    # ------------------------------------------------------------------ #
    # Convert algebra result to SteeringVector
    # ------------------------------------------------------------------ #

    def to_steering_vector(
        self,
        expr_result: ConceptFeatureSet,
        layers: list[int] | None = None,
        model_name: str = "ace-step",
    ) -> SteeringVector:
        """Convert an algebra result to a :class:`SteeringVector` for inference.

        Args:
            expr_result: Output of :meth:`expr`.
            layers:      Transformer-block indices to apply the vector at.
                         Defaults to ``[6, 7]`` (ACE-Step functional layers).
            model_name:  Model identifier stored in the provenance metadata.

        Returns:
            :class:`SteeringVector` with ``method="sae"`` and the summed decoder
            direction as the vector.
        """
        if layers is None:
            layers = [6, 7]  # ACE-Step functional layers (arXiv 2602.11910 Table 2)

        vector = expr_result.to_steering_vector(weight=1.0)

        return SteeringVector(
            concept=expr_result.concept,
            method="sae",
            model_name=model_name,
            layers=list(layers),
            vector=vector,
            tau=int(len(expr_result.feature_indices)),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------ #
    # Feature overlap heatmap
    # ------------------------------------------------------------------ #

    def feature_overlap_heatmap(self) -> plt.Figure:
        """Render a heatmap of pairwise Jaccard overlaps between all loaded concepts.

        Returns:
            :class:`matplotlib.figure.Figure` ready for saving or display.
        """
        concepts = list(self.features.keys())
        n = len(concepts)

        # Build pairwise Jaccard matrix: shape (n, n).
        matrix = np.zeros((n, n), dtype=np.float32)
        for i, c1 in enumerate(concepts):
            for j, c2 in enumerate(concepts):
                matrix[i, j] = self.features[c1].overlap(self.features[c2])

        fig_size = max(6, n * 0.8)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

        im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label="Jaccard Overlap")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(concepts, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(concepts, fontsize=9)
        ax.set_title("SAE Feature Overlap Heatmap (Jaccard Similarity)")

        # Annotate cells.
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                color = "white" if val > 0.6 else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=7, color=color,
                )

        fig.tight_layout()
        return fig
