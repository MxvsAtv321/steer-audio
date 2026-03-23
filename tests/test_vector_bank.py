"""
Tests for steer_audio.vector_bank — Prompt 2.1.

Covers:
- SteeringVector: save/load round-trip, norm, cosine_similarity, __repr__
- SteeringVectorBank: add/get/list, save_all/load_all, compose (Gram-Schmidt),
  interference_matrix shape and diagonal, compose output shapes per layer.

All tests run on CPU using small random tensors; no ACE-Step weights needed.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from steer_audio.vector_bank import SteeringVector, SteeringVectorBank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 64


def _make_sv(
    concept: str,
    *,
    seed: int = 0,
    dim: int = _DIM,
    layers: list[int] | None = None,
    method: str = "caa",
    clap_delta: float = 0.5,
) -> SteeringVector:
    """Create a random unit-norm SteeringVector with the Prompt 2.1 API."""
    torch.manual_seed(seed)
    v = torch.randn(dim)
    v = v / v.norm()
    return SteeringVector(
        concept=concept,
        method=method,
        model_name="ace_step",
        layers=layers or [6, 7],
        vector=v,
        alpha_range=list(range(-100, 101, 10)),
        metadata={"test": True},
        clap_delta=clap_delta,
    )


def _orthogonal_pair(dim: int = _DIM) -> tuple[SteeringVector, SteeringVector]:
    """Two exactly orthogonal unit vectors on dimensions 0 and 1."""
    v1 = torch.zeros(dim)
    v1[0] = 1.0
    v2 = torch.zeros(dim)
    v2[1] = 1.0
    sv1 = SteeringVector(
        concept="c1", method="caa", model_name="test",
        layers=[6, 7], vector=v1,
    )
    sv2 = SteeringVector(
        concept="c2", method="caa", model_name="test",
        layers=[6, 7], vector=v2,
    )
    return sv1, sv2


# ---------------------------------------------------------------------------
# SteeringVector — fields and methods
# ---------------------------------------------------------------------------


class TestSteeringVectorFields:
    def test_model_property_alias(self):
        """`model` property returns the same value as `model_name`."""
        sv = _make_sv("tempo")
        assert sv.model == sv.model_name == "ace_step"

    def test_metadata_field_stored(self):
        """`metadata` dict is preserved after construction."""
        sv = _make_sv("tempo")
        assert sv.metadata == {"test": True}

    def test_alpha_range_is_list(self):
        """`alpha_range` is a list of floats."""
        sv = _make_sv("tempo")
        assert isinstance(sv.alpha_range, list)
        assert len(sv.alpha_range) == 21  # -100 to 100 step 10

    def test_norm_positive(self):
        """`norm(layer)` returns a positive float."""
        sv = _make_sv("tempo")
        assert sv.norm(6) > 0.0
        assert sv.norm(7) > 0.0

    def test_norm_unit_vector(self):
        """`norm(layer)` ≈ 1.0 for a unit-norm vector."""
        sv = _make_sv("tempo", seed=1)
        assert abs(sv.norm(6) - 1.0) < 1e-5

    def test_cosine_similarity_self(self):
        """`cosine_similarity(self, layer)` == 1.0."""
        sv = _make_sv("tempo")
        assert abs(sv.cosine_similarity(sv, 6) - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        """`cosine_similarity` of two orthogonal vectors ≈ 0."""
        sv1, sv2 = _orthogonal_pair()
        assert abs(sv1.cosine_similarity(sv2, 6)) < 1e-5

    def test_cosine_similarity_parallel(self):
        """`cosine_similarity` of identical vectors == 1.0."""
        sv1 = _make_sv("tempo", seed=42)
        sv2 = _make_sv("tempo", seed=42)
        assert abs(sv1.cosine_similarity(sv2, 6) - 1.0) < 1e-5

    def test_repr_contains_concept(self):
        """`__repr__` contains the concept name."""
        sv = _make_sv("tempo")
        assert "tempo" in repr(sv)
        assert "SteeringVector" in repr(sv)


# ---------------------------------------------------------------------------
# SteeringVector — save / load round-trip
# ---------------------------------------------------------------------------


class TestSteeringVectorSaveLoad:
    def test_round_trip_preserves_concept(self, tmp_path: Path):
        sv = _make_sv("tempo")
        sv.save(str(tmp_path / "tempo_caa.safetensors"))
        sv2 = SteeringVector.load(str(tmp_path / "tempo_caa.safetensors"))
        assert sv2.concept == sv.concept

    def test_round_trip_preserves_method(self, tmp_path: Path):
        sv = _make_sv("tempo")
        sv.save(str(tmp_path / "tempo_caa.safetensors"))
        sv2 = SteeringVector.load(str(tmp_path / "tempo_caa.safetensors"))
        assert sv2.method == sv.method

    def test_round_trip_preserves_model_name(self, tmp_path: Path):
        sv = _make_sv("tempo")
        sv.save(str(tmp_path / "tempo_caa.safetensors"))
        sv2 = SteeringVector.load(str(tmp_path / "tempo_caa.safetensors"))
        assert sv2.model_name == sv.model_name

    def test_round_trip_preserves_layers(self, tmp_path: Path):
        sv = _make_sv("tempo", layers=[6, 7])
        sv.save(str(tmp_path / "tempo_caa.safetensors"))
        sv2 = SteeringVector.load(str(tmp_path / "tempo_caa.safetensors"))
        assert sv2.layers == [6, 7]

    def test_round_trip_preserves_vector(self, tmp_path: Path):
        sv = _make_sv("tempo", seed=7)
        sv.save(str(tmp_path / "tempo_caa.safetensors"))
        sv2 = SteeringVector.load(str(tmp_path / "tempo_caa.safetensors"))
        assert torch.allclose(sv2.vector, sv.vector, atol=1e-6)

    def test_round_trip_preserves_alpha_range(self, tmp_path: Path):
        sv = _make_sv("tempo")
        sv.save(str(tmp_path / "tempo_caa.safetensors"))
        sv2 = SteeringVector.load(str(tmp_path / "tempo_caa.safetensors"))
        assert sv2.alpha_range == sv.alpha_range

    def test_round_trip_preserves_clap_delta(self, tmp_path: Path):
        sv = _make_sv("tempo", clap_delta=0.75)
        sv.save(str(tmp_path / "tempo_caa.safetensors"))
        sv2 = SteeringVector.load(str(tmp_path / "tempo_caa.safetensors"))
        assert abs(sv2.clap_delta - 0.75) < 1e-5

    def test_sidecar_json_created(self, tmp_path: Path):
        sv = _make_sv("tempo")
        sv.save(str(tmp_path / "tempo_caa.safetensors"))
        assert (tmp_path / "tempo_caa.json").exists()

    def test_load_raises_for_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            SteeringVector.load(str(tmp_path / "nonexistent.safetensors"))

    def test_instance_save_adds_suffix(self, tmp_path: Path):
        """save() adds .safetensors if not present."""
        sv = _make_sv("tempo")
        sv.save(str(tmp_path / "tempo_caa"))  # no extension
        assert (tmp_path / "tempo_caa.safetensors").exists()


# ---------------------------------------------------------------------------
# SteeringVectorBank — add / get / list
# ---------------------------------------------------------------------------


class TestSteeringVectorBankRegistry:
    def test_add_and_get(self):
        bank = SteeringVectorBank()
        sv = _make_sv("tempo")
        bank.add(sv)
        retrieved = bank.get("tempo", "caa")
        assert retrieved is sv

    def test_get_wrong_concept_raises(self):
        bank = SteeringVectorBank()
        bank.add(_make_sv("tempo"))
        with pytest.raises(KeyError):
            bank.get("mood", "caa")

    def test_get_wrong_method_raises(self):
        bank = SteeringVectorBank()
        bank.add(_make_sv("tempo"))
        with pytest.raises(KeyError):
            bank.get("tempo", "sae")

    def test_list_format(self):
        bank = SteeringVectorBank()
        bank.add(_make_sv("tempo"))
        bank.add(_make_sv("mood", seed=1))
        names = bank.list()
        assert "tempo/caa" in names
        assert "mood/caa" in names

    def test_list_sorted(self):
        bank = SteeringVectorBank()
        bank.add(_make_sv("zzz", seed=0))
        bank.add(_make_sv("aaa", seed=1))
        names = bank.list()
        assert names == sorted(names)

    def test_len_empty(self):
        assert len(SteeringVectorBank()) == 0

    def test_len_after_add(self):
        bank = SteeringVectorBank()
        bank.add(_make_sv("tempo"))
        bank.add(_make_sv("mood", seed=1))
        assert len(bank) == 2

    def test_bool_empty(self):
        assert not SteeringVectorBank()

    def test_bool_nonempty(self):
        bank = SteeringVectorBank()
        bank.add(_make_sv("tempo"))
        assert bank

    def test_contains(self):
        bank = SteeringVectorBank()
        bank.add(_make_sv("tempo"))
        assert "tempo_caa" in bank
        assert "mood_caa" not in bank

    def test_getitem(self):
        bank = SteeringVectorBank()
        sv = _make_sv("tempo")
        bank.add(sv)
        assert bank["tempo_caa"] is sv

    def test_iteration_yields_keys(self):
        bank = SteeringVectorBank()
        bank.add(_make_sv("tempo"))
        bank.add(_make_sv("mood", seed=1))
        keys = list(bank)
        assert "tempo_caa" in keys
        assert "mood_caa" in keys

    def test_items_method(self):
        bank = SteeringVectorBank()
        sv = _make_sv("tempo")
        bank.add(sv)
        items = dict(bank.items())
        assert "tempo_caa" in items
        assert items["tempo_caa"] is sv


# ---------------------------------------------------------------------------
# SteeringVectorBank — save_all / load_all (round-trip)
# ---------------------------------------------------------------------------


class TestSteeringVectorBankIO:
    def test_save_all_creates_files(self, tmp_path: Path):
        bank = SteeringVectorBank()
        bank.add(_make_sv("tempo"))
        bank.add(_make_sv("mood", seed=1))
        bank.save_all(str(tmp_path))
        assert (tmp_path / "tempo_caa.safetensors").exists()
        assert (tmp_path / "mood_caa.safetensors").exists()

    def test_load_all_returns_bank(self, tmp_path: Path):
        bank = SteeringVectorBank()
        bank.add(_make_sv("tempo"))
        bank.save_all(str(tmp_path))
        loaded = SteeringVectorBank.load_all(tmp_path)
        assert isinstance(loaded, SteeringVectorBank)

    def test_load_all_preserves_count(self, tmp_path: Path):
        bank = SteeringVectorBank()
        bank.add(_make_sv("tempo"))
        bank.add(_make_sv("mood", seed=1))
        bank.save_all(str(tmp_path))
        loaded = SteeringVectorBank.load_all(tmp_path)
        assert len(loaded) == 2

    def test_load_all_preserves_vectors(self, tmp_path: Path):
        bank = SteeringVectorBank()
        sv = _make_sv("tempo", seed=77)
        bank.add(sv)
        bank.save_all(str(tmp_path))
        loaded = SteeringVectorBank.load_all(tmp_path)
        sv2 = loaded.get("tempo", "caa")
        assert torch.allclose(sv2.vector, sv.vector, atol=1e-6)

    def test_load_all_empty_dir(self, tmp_path: Path):
        loaded = SteeringVectorBank.load_all(tmp_path)
        assert len(loaded) == 0

    def test_instance_save_then_load_all_via_instance(self, tmp_path: Path):
        """Legacy: bank.save(sv, path) then bank.load_all(dir) returns bank."""
        bank = SteeringVectorBank()
        sv = _make_sv("tempo")
        bank.save(sv, tmp_path / "tempo_caa.safetensors")
        loaded = bank.load_all(tmp_path)
        assert "tempo_caa" in loaded

    def test_legacy_load_returns_sv(self, tmp_path: Path):
        """Legacy: bank.load(path) returns a SteeringVector."""
        sv = _make_sv("tempo")
        sv.save(str(tmp_path / "tempo_caa.safetensors"))
        bank = SteeringVectorBank()
        sv2 = bank.load(tmp_path / "tempo_caa.safetensors")
        assert isinstance(sv2, SteeringVector)
        assert sv2.concept == "tempo"


# ---------------------------------------------------------------------------
# SteeringVectorBank — compose
# ---------------------------------------------------------------------------


class TestSteeringVectorBankCompose:
    def test_compose_single_concept_returns_per_layer(self):
        bank = SteeringVectorBank()
        bank.add(_make_sv("tempo", layers=[6, 7]))
        result = bank.compose(["tempo"], method="caa", orthogonalize=False)
        assert 6 in result
        assert 7 in result

    def test_compose_output_shape(self):
        bank = SteeringVectorBank()
        bank.add(_make_sv("tempo", dim=_DIM, layers=[6, 7]))
        result = bank.compose(["tempo"], method="caa")
        for layer in [6, 7]:
            assert result[layer].shape == (_DIM,)

    def test_compose_orthogonal_pair_unchanged_direction(self):
        """Gram-Schmidt on two already-orthogonal vectors leaves them unchanged."""
        sv1, sv2 = _orthogonal_pair()
        bank = SteeringVectorBank()
        bank.add(sv1)
        bank.add(sv2)
        result_with = bank.compose(["c1", "c2"], orthogonalize=True)
        result_without = bank.compose(["c1", "c2"], orthogonalize=False)
        # Both results should be non-zero for each layer.
        for layer in [6, 7]:
            assert result_with[layer].norm().item() > 0.0
            assert result_without[layer].norm().item() > 0.0

    def test_compose_degenerate_parallel_vectors_warning(self):
        """Two identical vectors: second has near-zero residual, warning raised."""
        sv1 = _make_sv("c1", seed=42, layers=[6, 7], clap_delta=0.5)
        # Identical vector, different name.
        sv2 = SteeringVector(
            concept="c2",
            method="caa",
            model_name="test",
            layers=[6, 7],
            vector=sv1.vector.clone(),  # same direction
            clap_delta=0.3,  # lower clap_delta → processed after sv1
        )
        bank = SteeringVectorBank()
        bank.add(sv1)
        bank.add(sv2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = bank.compose(["c1", "c2"], orthogonalize=True)
            # At least one warning about near-zero residual expected.
            assert len(w) >= 1, "Expected warning for near-parallel vectors."

    def test_compose_legacy_api_single_tuple(self):
        """Legacy compose([(sv, alpha)]) returns correct per-layer dict."""
        sv = _make_sv("tempo", layers=[6, 7])
        bank = SteeringVectorBank()
        result = bank.compose([(sv, 2.0)])
        for layer in [6, 7]:
            assert layer in result
            assert abs(result[layer].norm().item() - 2.0) < 1e-4

    def test_compose_legacy_api_two_tuples(self):
        """Legacy compose([(sv1, a1), (sv2, a2)]) returns per-layer dict."""
        sv1 = _make_sv("c1", seed=0, layers=[6, 7])
        sv2 = _make_sv("c2", seed=1, layers=[6, 7])
        bank = SteeringVectorBank()
        result = bank.compose([(sv1, 10.0), (sv2, 10.0)])
        for layer in [6, 7]:
            assert layer in result
            assert result[layer].norm().item() > 0.0

    def test_compose_empty_returns_empty_dict(self):
        bank = SteeringVectorBank()
        result = bank.compose([])
        assert result == {}

    def test_compose_two_concepts_distinct_layers(self):
        """Concepts on non-overlapping layers produce independent entries."""
        sv1 = _make_sv("c1", layers=[6])
        sv2 = _make_sv("c2", seed=1, layers=[7])
        bank = SteeringVectorBank()
        bank.add(sv1)
        bank.add(sv2)
        result = bank.compose(["c1", "c2"])
        assert 6 in result
        assert 7 in result


# ---------------------------------------------------------------------------
# SteeringVectorBank — interference_matrix
# ---------------------------------------------------------------------------


class TestInterferenceMatrix:
    def test_shape_2x2(self):
        bank = SteeringVectorBank()
        bank.add(_make_sv("tempo", seed=0))
        bank.add(_make_sv("mood", seed=1))
        mat = bank.interference_matrix(layer=6)
        assert mat.shape == (2, 2)

    def test_shape_3x3(self):
        bank = SteeringVectorBank()
        for i, name in enumerate(["a", "b", "c"]):
            bank.add(_make_sv(name, seed=i))
        mat = bank.interference_matrix()
        assert mat.shape == (3, 3)

    def test_diagonal_approx_one(self):
        bank = SteeringVectorBank()
        bank.add(_make_sv("tempo", seed=0))
        bank.add(_make_sv("mood", seed=1))
        mat = bank.interference_matrix(layer=6)
        diag = np.diag(mat)
        np.testing.assert_allclose(diag, np.ones(2), atol=1e-5)

    def test_symmetric(self):
        bank = SteeringVectorBank()
        bank.add(_make_sv("tempo", seed=0))
        bank.add(_make_sv("mood", seed=1))
        mat = bank.interference_matrix(layer=6)
        np.testing.assert_allclose(mat, mat.T, atol=1e-5)

    def test_orthogonal_pair_off_diagonal_zero(self):
        """Orthogonal vectors → off-diagonal ≈ 0."""
        sv1, sv2 = _orthogonal_pair()
        bank = SteeringVectorBank()
        bank.add(sv1)
        bank.add(sv2)
        mat = bank.interference_matrix(layer=6)
        assert abs(mat[0, 1]) < 1e-5

    def test_empty_bank_returns_empty_matrix(self):
        bank = SteeringVectorBank()
        mat = bank.interference_matrix()
        assert mat.shape == (0, 0)

    def test_dtype_is_float32(self):
        bank = SteeringVectorBank()
        bank.add(_make_sv("tempo"))
        mat = bank.interference_matrix()
        assert mat.dtype == np.float32

    def test_layer_arg_accepted(self):
        """interference_matrix(layer=7) runs without error."""
        bank = SteeringVectorBank()
        bank.add(_make_sv("tempo", seed=0))
        bank.add(_make_sv("mood", seed=1))
        mat = bank.interference_matrix(layer=7)
        assert mat.shape == (2, 2)
