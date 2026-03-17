"""
Tests for steer_audio.concept_algebra — ConceptFeatureSet and ConceptAlgebra.

All tests run on CPU without SAE model weights; decoder_matrix is constructed
synthetically so no GPU or checkpoint files are needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from steer_audio.concept_algebra import (
    ConceptAlgebra,
    ConceptFeatureSet,
    AlgebraPreset,
    AlgebraPresetBank,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_HIDDEN_DIM = 32
_NUM_FEATURES = 64
_TAU = 8  # feature set size used in tests


def _make_decoder(seed: int = 0) -> torch.Tensor:
    """Return a random (hidden_dim, num_features) decoder matrix."""
    torch.manual_seed(seed)
    return torch.randn(_HIDDEN_DIM, _NUM_FEATURES)


def _make_cfs(
    concept: str,
    feature_indices: list[int] | None = None,
    tfidf_scores: list[float] | None = None,
    decoder: torch.Tensor | None = None,
    seed: int = 0,
) -> ConceptFeatureSet:
    """Build a ConceptFeatureSet for tests."""
    if decoder is None:
        decoder = _make_decoder(seed)
    if feature_indices is None:
        rng = np.random.default_rng(seed)
        feature_indices = sorted(
            rng.choice(_NUM_FEATURES, size=_TAU, replace=False).tolist()
        )
    if tfidf_scores is None:
        rng = np.random.default_rng(seed + 1)
        tfidf_scores = rng.uniform(0.1, 1.0, size=len(feature_indices)).tolist()

    return ConceptFeatureSet(
        concept=concept,
        feature_indices=np.array(feature_indices, dtype=np.int64),
        tfidf_scores=np.array(tfidf_scores, dtype=np.float32),
        decoder_matrix=decoder,
    )


@pytest.fixture
def decoder() -> torch.Tensor:
    """Shared (hidden_dim=32, num_features=64) decoder for all tests."""
    return _make_decoder(seed=42)


@pytest.fixture
def cfs_jazz(decoder) -> ConceptFeatureSet:
    return _make_cfs("jazz", feature_indices=list(range(0, 8)), decoder=decoder, seed=0)


@pytest.fixture
def cfs_techno(decoder) -> ConceptFeatureSet:
    # Indices 4-11 → overlap {4,5,6,7} with jazz
    return _make_cfs("techno", feature_indices=list(range(4, 12)), decoder=decoder, seed=1)


@pytest.fixture
def cfs_drums(decoder) -> ConceptFeatureSet:
    # Disjoint from jazz and techno
    return _make_cfs("drums", feature_indices=list(range(20, 28)), decoder=decoder, seed=2)


@pytest.fixture
def algebra(decoder, cfs_jazz, cfs_techno, cfs_drums) -> ConceptAlgebra:
    return ConceptAlgebra(
        sae_model=None,  # not needed for feature-algebra tests
        concept_features={
            "jazz": cfs_jazz,
            "techno": cfs_techno,
            "drums": cfs_drums,
        },
    )


# ---------------------------------------------------------------------------
# 1. to_steering_vector — shape and content
# ---------------------------------------------------------------------------


def test_to_steering_vector_shape(cfs_jazz):
    """to_steering_vector returns a 1-D tensor of length hidden_dim."""
    v = cfs_jazz.to_steering_vector()
    assert v.shape == (
        _HIDDEN_DIM,
    ), f"Expected shape ({_HIDDEN_DIM},), got {v.shape}"


def test_to_steering_vector_nonzero(cfs_jazz):
    """to_steering_vector returns a non-zero vector for non-empty feature sets."""
    v = cfs_jazz.to_steering_vector()
    assert v.norm().item() > 0.0, "Steering vector should be non-zero for non-empty CFS."


def test_to_steering_vector_empty_features(decoder):
    """to_steering_vector returns zeros for an empty feature set (no crash)."""
    empty = ConceptFeatureSet(
        concept="empty",
        feature_indices=np.array([], dtype=np.int64),
        tfidf_scores=np.array([], dtype=np.float32),
        decoder_matrix=decoder,
    )
    v = empty.to_steering_vector()
    assert v.shape == (_HIDDEN_DIM,)
    assert v.norm().item() == 0.0, "Empty CFS should produce zero vector."


def test_to_steering_vector_weight_scales(cfs_jazz):
    """Passing weight=2.0 doubles the output vs weight=1.0."""
    v1 = cfs_jazz.to_steering_vector(weight=1.0)
    v2 = cfs_jazz.to_steering_vector(weight=2.0)
    assert torch.allclose(2.0 * v1, v2, atol=1e-5), (
        "weight=2.0 should produce exactly 2× the weight=1.0 vector."
    )


def test_to_steering_vector_manual(decoder):
    """Manually verify the weighted sum for a single-feature CFS."""
    feat_idx = 5
    score = 2.0
    cfs = ConceptFeatureSet(
        concept="single",
        feature_indices=np.array([feat_idx], dtype=np.int64),
        tfidf_scores=np.array([score], dtype=np.float32),
        decoder_matrix=decoder,
    )
    v = cfs.to_steering_vector(weight=1.0)
    expected = score * decoder[:, feat_idx].float()
    assert torch.allclose(v, expected, atol=1e-5), (
        f"Single-feature CFS should equal score * W_dec[:, {feat_idx}]."
    )


# ---------------------------------------------------------------------------
# 2. overlap (Jaccard similarity)
# ---------------------------------------------------------------------------


def test_overlap_self(cfs_jazz):
    """A concept has Jaccard overlap 1.0 with itself."""
    assert abs(cfs_jazz.overlap(cfs_jazz) - 1.0) < 1e-6


def test_overlap_disjoint(cfs_jazz, cfs_drums):
    """Disjoint feature sets have Jaccard overlap 0.0."""
    assert cfs_jazz.overlap(cfs_drums) == 0.0


def test_overlap_partial(cfs_jazz, cfs_techno):
    """Partial overlap is in (0, 1) and equals |intersection| / |union|."""
    set_j = set(cfs_jazz.feature_indices.tolist())
    set_t = set(cfs_techno.feature_indices.tolist())
    expected = len(set_j & set_t) / len(set_j | set_t)
    got = cfs_jazz.overlap(cfs_techno)
    assert abs(got - expected) < 1e-6, f"Expected {expected:.4f}, got {got:.4f}"


# ---------------------------------------------------------------------------
# 3. __add__ (union)
# ---------------------------------------------------------------------------


def test_add_union_indices(cfs_jazz, cfs_techno):
    """Union includes all indices from both sets."""
    result = cfs_jazz + cfs_techno
    expected = set(cfs_jazz.feature_indices.tolist()) | set(cfs_techno.feature_indices.tolist())
    assert set(result.feature_indices.tolist()) == expected


def test_add_shared_indices_have_summed_scores(cfs_jazz, cfs_techno):
    """Shared feature indices get additive score combination."""
    result = cfs_jazz + cfs_techno
    scores_j = dict(zip(cfs_jazz.feature_indices.tolist(), cfs_jazz.tfidf_scores.tolist()))
    scores_t = dict(zip(cfs_techno.feature_indices.tolist(), cfs_techno.tfidf_scores.tolist()))
    scores_r = dict(zip(result.feature_indices.tolist(), result.tfidf_scores.tolist()))

    for idx in set(scores_j.keys()) & set(scores_t.keys()):
        expected = scores_j[idx] + scores_t[idx]
        assert abs(scores_r[idx] - expected) < 1e-5, (
            f"Shared index {idx}: expected score {expected:.4f}, got {scores_r[idx]:.4f}"
        )


def test_add_disjoint_preserves_sizes(cfs_jazz, cfs_drums):
    """Union of disjoint sets has size = |jazz| + |drums|."""
    result = cfs_jazz + cfs_drums
    expected_size = len(cfs_jazz.feature_indices) + len(cfs_drums.feature_indices)
    assert len(result.feature_indices) == expected_size


# ---------------------------------------------------------------------------
# 4. __sub__ (set difference)
# ---------------------------------------------------------------------------


def test_sub_removes_b_indices(cfs_jazz, cfs_techno):
    """Set difference removes all B features from A."""
    result = cfs_jazz - cfs_techno
    set_b = set(cfs_techno.feature_indices.tolist())
    for idx in result.feature_indices.tolist():
        assert idx not in set_b, f"Index {idx} from B should not appear in A - B."


def test_sub_keeps_exclusive_a(cfs_jazz, cfs_techno):
    """Set difference keeps features exclusive to A."""
    result = cfs_jazz - cfs_techno
    set_a = set(cfs_jazz.feature_indices.tolist())
    set_b = set(cfs_techno.feature_indices.tolist())
    exclusive_a = set_a - set_b
    assert set(result.feature_indices.tolist()) == exclusive_a


def test_sub_self_is_empty(cfs_jazz):
    """Subtracting a concept from itself yields an empty feature set."""
    result = cfs_jazz - cfs_jazz
    assert len(result.feature_indices) == 0


def test_sub_disjoint_preserves_a(cfs_jazz, cfs_drums):
    """Subtracting a disjoint concept leaves A unchanged."""
    result = cfs_jazz - cfs_drums
    assert set(result.feature_indices.tolist()) == set(cfs_jazz.feature_indices.tolist())


# ---------------------------------------------------------------------------
# 5. __and__ (intersection)
# ---------------------------------------------------------------------------


def test_and_indices_are_intersection(cfs_jazz, cfs_techno):
    """Intersection keeps only shared indices."""
    result = cfs_jazz & cfs_techno
    set_j = set(cfs_jazz.feature_indices.tolist())
    set_t = set(cfs_techno.feature_indices.tolist())
    assert set(result.feature_indices.tolist()) == set_j & set_t


def test_and_disjoint_is_empty(cfs_jazz, cfs_drums):
    """Intersection of disjoint sets is empty."""
    result = cfs_jazz & cfs_drums
    assert len(result.feature_indices) == 0


def test_and_scores_are_min(cfs_jazz, cfs_techno):
    """Intersection scores are the minimum of each concept's score for shared features."""
    result = cfs_jazz & cfs_techno
    scores_j = dict(zip(cfs_jazz.feature_indices.tolist(), cfs_jazz.tfidf_scores.tolist()))
    scores_t = dict(zip(cfs_techno.feature_indices.tolist(), cfs_techno.tfidf_scores.tolist()))
    for idx, score in zip(result.feature_indices.tolist(), result.tfidf_scores.tolist()):
        expected = min(scores_j[idx], scores_t[idx])
        assert abs(score - expected) < 1e-5, (
            f"Intersection score at index {idx}: expected {expected:.4f}, got {score:.4f}"
        )


# ---------------------------------------------------------------------------
# 6. __mul__ (scalar scaling)
# ---------------------------------------------------------------------------


def test_mul_scales_scores(cfs_jazz):
    """Multiplying by scalar scales all tfidf_scores proportionally."""
    scaled = 0.5 * cfs_jazz
    assert np.allclose(scaled.tfidf_scores, cfs_jazz.tfidf_scores * 0.5, atol=1e-6)


def test_mul_preserves_indices(cfs_jazz):
    """Multiplying does not change the set of feature indices."""
    scaled = 3.0 * cfs_jazz
    assert list(scaled.feature_indices) == list(cfs_jazz.feature_indices)


def test_mul_rmul_equivalent(cfs_jazz):
    """0.7 * cfs and cfs * 0.7 produce identical results."""
    a = 0.7 * cfs_jazz
    b = cfs_jazz * 0.7
    assert np.allclose(a.tfidf_scores, b.tfidf_scores, atol=1e-6)


def test_weighted_blend_steering_vector(cfs_jazz, cfs_techno, decoder):
    """0.7*jazz + 0.3*techno produces a different vector than 1.0*jazz + 1.0*techno."""
    blend = 0.7 * cfs_jazz + 0.3 * cfs_techno
    equal_weight = cfs_jazz + cfs_techno
    v_blend = blend.to_steering_vector()
    v_equal = equal_weight.to_steering_vector()
    # They should generally differ (unless pathological random seed collision).
    assert not torch.allclose(v_blend, v_equal, atol=1e-3), (
        "Weighted blend should differ from equal-weight union."
    )


# ---------------------------------------------------------------------------
# 7. ConceptAlgebra.expr — parsing
# ---------------------------------------------------------------------------


def test_expr_single_concept(algebra, cfs_jazz):
    """Parsing a bare concept name returns a copy of that ConceptFeatureSet."""
    result = algebra.expr("jazz")
    assert set(result.feature_indices.tolist()) == set(cfs_jazz.feature_indices.tolist())


def test_expr_addition(algebra, cfs_jazz, cfs_drums):
    """Parsing 'jazz + drums' returns union of jazz and drums."""
    result = algebra.expr("jazz + drums")
    expected = set(cfs_jazz.feature_indices.tolist()) | set(cfs_drums.feature_indices.tolist())
    assert set(result.feature_indices.tolist()) == expected


def test_expr_subtraction(algebra, cfs_jazz, cfs_techno):
    """Parsing 'jazz - techno' returns set difference."""
    result = algebra.expr("jazz - techno")
    set_j = set(cfs_jazz.feature_indices.tolist())
    set_t = set(cfs_techno.feature_indices.tolist())
    assert set(result.feature_indices.tolist()) == set_j - set_t


def test_expr_intersection(algebra, cfs_jazz, cfs_techno):
    """Parsing 'jazz & techno' returns intersection."""
    result = algebra.expr("jazz & techno")
    set_j = set(cfs_jazz.feature_indices.tolist())
    set_t = set(cfs_techno.feature_indices.tolist())
    assert set(result.feature_indices.tolist()) == set_j & set_t


def test_expr_weighted_blend(algebra):
    """Parsing '0.7 * jazz + 0.3 * techno' runs without error."""
    result = algebra.expr("0.7 * jazz + 0.3 * techno")
    assert len(result.feature_indices) > 0


def test_expr_chained(algebra):
    """Parsing 'jazz + drums - techno' chains three concepts."""
    result = algebra.expr("jazz + drums - techno")
    # Should not crash and should have at most |jazz|+|drums| features.
    max_size = len(algebra.features["jazz"].feature_indices) + len(
        algebra.features["drums"].feature_indices
    )
    assert len(result.feature_indices) <= max_size


def test_expr_parenthesised(algebra, cfs_jazz, cfs_drums, cfs_techno):
    """Parsing '(jazz + drums) & techno' respects parentheses."""
    result = algebra.expr("(jazz + drums) & techno")
    union_jd = set(cfs_jazz.feature_indices.tolist()) | set(cfs_drums.feature_indices.tolist())
    set_t = set(cfs_techno.feature_indices.tolist())
    assert set(result.feature_indices.tolist()) == union_jd & set_t


def test_expr_unknown_concept_raises(algebra):
    """Using an unknown concept name raises KeyError."""
    with pytest.raises(KeyError, match="violin"):
        algebra.expr("jazz + violin")


def test_expr_syntax_error_raises(algebra):
    """A syntax error in the expression raises ValueError."""
    with pytest.raises((ValueError, KeyError)):
        algebra.expr("jazz ++ drums")


# ---------------------------------------------------------------------------
# 8. ConceptAlgebra.to_steering_vector
# ---------------------------------------------------------------------------


def test_to_sv_returns_steering_vector(algebra):
    """to_steering_vector wraps an algebra result in a SteeringVector."""
    from steer_audio.vector_bank import SteeringVector

    result = algebra.expr("jazz + drums")
    sv = algebra.to_steering_vector(result)
    assert isinstance(sv, SteeringVector)
    assert sv.method == "sae"
    assert sv.vector.shape == (_HIDDEN_DIM,)


def test_to_sv_custom_layers(algebra):
    """to_steering_vector respects custom layer list."""
    result = algebra.expr("jazz")
    sv = algebra.to_steering_vector(result, layers=[3, 5])
    assert sv.layers == [3, 5]


def test_to_sv_tau_matches_feature_count(algebra):
    """SteeringVector.tau matches the number of features in the algebra result."""
    result = algebra.expr("jazz & techno")
    sv = algebra.to_steering_vector(result)
    assert sv.tau == len(result.feature_indices)


# ---------------------------------------------------------------------------
# 9. feature_overlap_heatmap
# ---------------------------------------------------------------------------


def test_overlap_heatmap_returns_figure(algebra):
    """feature_overlap_heatmap returns a matplotlib Figure without error."""
    import matplotlib.pyplot as plt

    fig = algebra.feature_overlap_heatmap()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_overlap_heatmap_diagonal_is_one(algebra):
    """Diagonal of Jaccard matrix should be 1.0 (self-similarity)."""
    concepts = list(algebra.features.keys())
    for c in concepts:
        j = algebra.features[c].overlap(algebra.features[c])
        assert abs(j - 1.0) < 1e-6, f"Self-overlap for {c!r} should be 1.0, got {j:.4f}"


# ---------------------------------------------------------------------------
# 10. ConceptFeatureSet.from_sae class method
# ---------------------------------------------------------------------------


def test_from_sae_transposes_decoder():
    """from_sae correctly transposes W_dec from (num_features, hidden_dim) to (hidden_dim, num_features)."""
    import types

    # Mock Sae with W_dec parameter shape (num_features=64, hidden_dim=32)
    torch.manual_seed(99)
    w_dec_raw = torch.randn(_NUM_FEATURES, _HIDDEN_DIM)

    mock_sae = types.SimpleNamespace()
    mock_sae.W_dec = torch.nn.Parameter(w_dec_raw)

    feature_indices = np.array([0, 1, 2], dtype=np.int64)
    tfidf_scores = np.array([1.0, 0.8, 0.6], dtype=np.float32)

    cfs = ConceptFeatureSet.from_sae(mock_sae, "test", feature_indices, tfidf_scores)

    assert cfs.decoder_matrix.shape == (
        _HIDDEN_DIM,
        _NUM_FEATURES,
    ), f"Expected ({_HIDDEN_DIM}, {_NUM_FEATURES}), got {cfs.decoder_matrix.shape}"


# ---------------------------------------------------------------------------
# 11. AlgebraPreset — construction and metadata
# ---------------------------------------------------------------------------


def test_preset_defaults():
    """AlgebraPreset sets created_at automatically and defaults to empty tags."""
    preset = AlgebraPreset(name="test", expression="jazz + drums")
    assert preset.created_at  # non-empty ISO string
    assert preset.tags == []
    assert preset.description == ""
    assert preset.author == ""


def test_preset_evaluate(algebra, cfs_jazz, cfs_drums):
    """AlgebraPreset.evaluate delegates to ConceptAlgebra.expr."""
    preset = AlgebraPreset(
        name="jazz_drums",
        expression="jazz + drums",
        description="Jazz with drums",
    )
    result = preset.evaluate(algebra)
    expected = set(cfs_jazz.feature_indices.tolist()) | set(cfs_drums.feature_indices.tolist())
    assert set(result.feature_indices.tolist()) == expected


def test_preset_evaluate_unknown_concept_raises(algebra):
    """AlgebraPreset.evaluate raises KeyError for unknown concept names."""
    preset = AlgebraPreset(name="bad", expression="jazz + unknown_concept")
    with pytest.raises(KeyError):
        preset.evaluate(algebra)


# ---------------------------------------------------------------------------
# 12. AlgebraPresetBank — save / load / list_all / delete
# ---------------------------------------------------------------------------


@pytest.fixture
def preset_dir(tmp_path) -> Path:
    """Temporary directory for preset files."""
    return tmp_path / "presets"


@pytest.fixture
def sample_preset() -> AlgebraPreset:
    return AlgebraPreset(
        name="jazz_reggae_blend",
        expression="0.5 * jazz + 0.5 * reggae",
        description="Equal-weight jazz-reggae hybrid",
        tags=["genre", "blend"],
        author="test_suite",
    )


def test_preset_bank_save_creates_file(preset_dir, sample_preset):
    """save() writes a JSON file named after the preset."""
    bank = AlgebraPresetBank()
    out = bank.save(sample_preset, preset_dir)
    assert out.exists(), "save() should create the JSON file."
    assert out.suffix == ".json"
    assert out.stem == sample_preset.name


def test_preset_bank_save_round_trip(preset_dir, sample_preset):
    """load() restores all fields saved by save()."""
    bank = AlgebraPresetBank()
    bank.save(sample_preset, preset_dir)
    loaded = bank.load(preset_dir / f"{sample_preset.name}.json")
    assert loaded.name == sample_preset.name
    assert loaded.expression == sample_preset.expression
    assert loaded.description == sample_preset.description
    assert loaded.tags == sample_preset.tags
    assert loaded.author == sample_preset.author
    assert loaded.created_at == sample_preset.created_at


def test_preset_bank_load_missing_file(preset_dir):
    """load() raises FileNotFoundError for a missing file."""
    bank = AlgebraPresetBank()
    with pytest.raises(FileNotFoundError):
        bank.load(preset_dir / "nonexistent.json")


def test_preset_bank_load_invalid_json(tmp_path):
    """load() raises ValueError when the file is not valid JSON."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not valid json", encoding="utf-8")
    bank = AlgebraPresetBank()
    with pytest.raises(ValueError, match="Could not parse"):
        bank.load(bad_file)


def test_preset_bank_load_missing_required_field(tmp_path):
    """load() raises ValueError when required field 'expression' is absent."""
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"name": "oops"}), encoding="utf-8")
    bank = AlgebraPresetBank()
    with pytest.raises(ValueError, match="missing required field"):
        bank.load(incomplete)


def test_preset_bank_list_all_empty_dir(preset_dir):
    """list_all() returns empty dict for a directory with no JSON files."""
    preset_dir.mkdir(parents=True, exist_ok=True)
    bank = AlgebraPresetBank()
    result = bank.list_all(preset_dir)
    assert result == {}


def test_preset_bank_list_all_multiple(preset_dir):
    """list_all() returns all saved presets keyed by name."""
    bank = AlgebraPresetBank()
    p1 = AlgebraPreset(name="alpha", expression="jazz")
    p2 = AlgebraPreset(name="beta", expression="drums")
    bank.save(p1, preset_dir)
    bank.save(p2, preset_dir)

    result = bank.list_all(preset_dir)
    assert set(result.keys()) == {"alpha", "beta"}
    assert result["alpha"].expression == "jazz"
    assert result["beta"].expression == "drums"


def test_preset_bank_delete_existing(preset_dir, sample_preset):
    """delete() removes the file and returns True."""
    bank = AlgebraPresetBank()
    bank.save(sample_preset, preset_dir)
    deleted = bank.delete(sample_preset.name, preset_dir)
    assert deleted is True
    assert not (preset_dir / f"{sample_preset.name}.json").exists()


def test_preset_bank_delete_nonexistent(preset_dir):
    """delete() returns False (not an error) when the preset does not exist."""
    preset_dir.mkdir(parents=True, exist_ok=True)
    bank = AlgebraPresetBank()
    result = bank.delete("no_such_preset", preset_dir)
    assert result is False


def test_preset_bank_name_with_path_separator_raises(preset_dir):
    """save() raises ValueError if preset.name contains path separators."""
    bank = AlgebraPresetBank()
    bad = AlgebraPreset(name="a/b", expression="jazz")
    with pytest.raises(ValueError, match="path separators"):
        bank.save(bad, preset_dir)


def test_preset_bank_summary_table_returns_string(preset_dir, sample_preset):
    """summary_table() returns a non-empty string."""
    bank = AlgebraPresetBank()
    bank.save(sample_preset, preset_dir)
    loaded = bank.list_all(preset_dir)
    table = bank.summary_table(loaded)
    assert isinstance(table, str) and len(table) > 0


def test_preset_bank_list_skips_corrupt_file(preset_dir, sample_preset):
    """list_all() logs a warning and skips unreadable JSON files."""
    bank = AlgebraPresetBank()
    bank.save(sample_preset, preset_dir)
    # Plant a corrupt file alongside the valid one.
    (preset_dir / "corrupt.json").write_text("{bad json", encoding="utf-8")
    result = bank.list_all(preset_dir)
    # Only the valid preset should appear.
    assert sample_preset.name in result
    assert "corrupt" not in result


def test_preset_evaluate_and_to_sv_pipeline(preset_dir, algebra):
    """End-to-end: save preset → load → evaluate → to_steering_vector."""
    bank = AlgebraPresetBank()
    preset = AlgebraPreset(
        name="jazz_minus_techno",
        expression="jazz - techno",
        description="Jazz without techno features",
    )
    bank.save(preset, preset_dir)
    loaded = bank.load(preset_dir / "jazz_minus_techno.json")

    cfs_result = loaded.evaluate(algebra)
    sv = algebra.to_steering_vector(cfs_result)

    from steer_audio.vector_bank import SteeringVector
    assert isinstance(sv, SteeringVector)
    assert sv.method == "sae"
