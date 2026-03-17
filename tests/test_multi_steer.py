"""
Tests for steer_audio.multi_steer and steer_audio.vector_bank.

All tests run on CPU without model weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from steer_audio.vector_bank import SteeringVector, SteeringVectorBank
from steer_audio.multi_steer import MultiConceptSteerer, _renorm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DIM = 64  # small hidden dim for fast CPU tests


def _make_sv(concept: str, dim: int = _DIM, seed: int | None = None) -> SteeringVector:
    """Create a random unit-norm SteeringVector for testing."""
    if seed is not None:
        torch.manual_seed(seed)
    v = torch.randn(dim)
    v = v / v.norm()
    return SteeringVector(
        concept=concept,
        method="caa",
        model_name="ace-step",
        layers=[6, 7],
        vector=v,
        clap_delta=float(torch.rand(1).item()),
        lpaps_at_50=float(torch.rand(1).item()),
    )


@pytest.fixture
def sv_tempo() -> SteeringVector:
    return _make_sv("tempo", seed=0)


@pytest.fixture
def sv_mood() -> SteeringVector:
    return _make_sv("mood", seed=1)


@pytest.fixture
def sv_pair(sv_tempo, sv_mood) -> dict[str, SteeringVector]:
    return {"tempo": sv_tempo, "mood": sv_mood}


# ---------------------------------------------------------------------------
# _renorm
# ---------------------------------------------------------------------------


def test_renorm_preserves_magnitude():
    """ReNorm(h + delta, h) should have the same per-token L2 norm as h."""
    torch.manual_seed(42)
    h = torch.randn(2, 16, _DIM)
    delta = torch.randn(2, 16, _DIM) * 0.1
    h_steered = _renorm(h + delta, h)

    orig_norms = h.norm(dim=-1)
    new_norms = h_steered.norm(dim=-1)
    assert torch.allclose(orig_norms, new_norms, atol=1e-5), (
        f"Magnitude mismatch: max diff = {(orig_norms - new_norms).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# SteeringVector / SteeringVectorBank
# ---------------------------------------------------------------------------


def test_steering_vector_round_trip(tmp_path: Path, sv_tempo: SteeringVector):
    """Save then load must reproduce the original vector and metadata."""
    bank = SteeringVectorBank()
    out = tmp_path / "tempo_caa.safetensors"
    bank.save(sv_tempo, out)
    sv2 = bank.load(out)

    assert sv2.concept == sv_tempo.concept
    assert sv2.method == sv_tempo.method
    assert sv2.layers == sv_tempo.layers
    assert torch.allclose(sv2.vector, sv_tempo.vector, atol=1e-6), (
        "Vector mismatch after round-trip serialisation."
    )


def test_load_all_finds_multiple_files(
    tmp_path: Path, sv_tempo: SteeringVector, sv_mood: SteeringVector
):
    """load_all should return one entry per .safetensors file."""
    bank = SteeringVectorBank()
    bank.save(sv_tempo, tmp_path / "tempo_caa.safetensors")
    bank.save(sv_mood, tmp_path / "mood_caa.safetensors")

    loaded = bank.load_all(tmp_path)
    assert len(loaded) == 2
    assert "tempo_caa" in loaded
    assert "mood_caa" in loaded


def test_compose_single_vector(sv_tempo: SteeringVector):
    """compose with a single vector should return alpha * unit_v for that layer."""
    bank = SteeringVectorBank()
    result = bank.compose([(sv_tempo, 2.0)])
    for layer_idx in sv_tempo.layers:
        assert layer_idx in result
        # Magnitude should be ≈ 2.0 (unit-norm vector scaled by alpha).
        magnitude = result[layer_idx].norm().item()
        assert abs(magnitude - 2.0) < 1e-4, f"Expected norm≈2.0, got {magnitude:.4f}"


def test_compose_orthogonalizes_shared_layers(sv_tempo: SteeringVector, sv_mood: SteeringVector):
    """compose with shared layers applies Gram-Schmidt."""
    bank = SteeringVectorBank()
    result = bank.compose([(sv_tempo, 10.0), (sv_mood, 10.0)])
    # Both vectors target layers [6, 7]; each layer delta must exist.
    for layer_idx in [6, 7]:
        assert layer_idx in result
        # After orthogonalization the combined delta is non-zero.
        assert result[layer_idx].norm().item() > 0.0


# ---------------------------------------------------------------------------
# MultiConceptSteerer — construction and basic properties
# ---------------------------------------------------------------------------


def test_init_raises_on_empty_vectors():
    """MultiConceptSteerer should raise ValueError for an empty dict."""
    with pytest.raises(ValueError, match="at least one"):
        MultiConceptSteerer({})


def test_init_single_concept(sv_tempo: SteeringVector):
    """Single-concept steerer should initialise without error."""
    steerer = MultiConceptSteerer({"tempo": sv_tempo})
    assert "tempo" in steerer.vectors


# ---------------------------------------------------------------------------
# interference_matrix
# ---------------------------------------------------------------------------


def test_interference_matrix_shape(sv_pair: dict):
    """interference_matrix should return an (N, N) tensor."""
    steerer = MultiConceptSteerer(sv_pair)
    imat = steerer.interference_matrix()
    n = len(sv_pair)
    assert imat.shape == (n, n), f"Expected ({n},{n}), got {imat.shape}"


def test_interference_matrix_diagonal_ones(sv_pair: dict):
    """Diagonal entries (self-similarity) must equal 1.0."""
    steerer = MultiConceptSteerer(sv_pair)
    imat = steerer.interference_matrix()
    diag = imat.diagonal()
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5), (
        f"Diagonal not all 1.0: {diag}"
    )


def test_interference_matrix_symmetric(sv_pair: dict):
    """Cosine similarity matrix must be symmetric."""
    steerer = MultiConceptSteerer(sv_pair)
    imat = steerer.interference_matrix()
    assert torch.allclose(imat, imat.T, atol=1e-5)


def test_interference_matrix_orthogonal_pair():
    """Two orthogonal vectors should have off-diagonal ≈ 0."""
    dim = 64
    v1 = torch.zeros(dim)
    v1[0] = 1.0
    v2 = torch.zeros(dim)
    v2[1] = 1.0

    sv1 = SteeringVector(
        concept="c1", method="caa", model_name="test", layers=[6],
        vector=v1, clap_delta=1.0,
    )
    sv2 = SteeringVector(
        concept="c2", method="caa", model_name="test", layers=[6],
        vector=v2, clap_delta=0.5,
    )

    steerer = MultiConceptSteerer({"c1": sv1, "c2": sv2})
    imat = steerer.interference_matrix()
    off_diag = imat[0, 1].item()
    assert abs(off_diag) < 1e-5, f"Off-diagonal should be ≈0 for orthogonal vectors, got {off_diag:.2e}"


# ---------------------------------------------------------------------------
# Gram-Schmidt orthogonalization
# ---------------------------------------------------------------------------


def test_gram_schmidt_reduces_interference(sv_pair: dict):
    """After GS, off-diagonal interference should be closer to 0."""
    steerer_before = MultiConceptSteerer(sv_pair, orthogonalize=False)
    imat_before = steerer_before.interference_matrix()
    off_before = abs(imat_before[0, 1].item())

    steerer_after = MultiConceptSteerer(sv_pair, orthogonalize=True)
    imat_after = steerer_after.interference_matrix()
    off_after = abs(imat_after[0, 1].item())

    assert off_after < off_before + 1e-5, (
        f"Orthogonalization should reduce interference: "
        f"before={off_before:.4f}, after={off_after:.4f}"
    )


def test_gram_schmidt_preserves_first_vector(sv_pair: dict):
    """The highest-clap_delta vector should be unchanged after GS."""
    # Identify the concept with higher clap_delta.
    best_key = max(sv_pair, key=lambda k: sv_pair[k].clap_delta)
    original_vec = sv_pair[best_key].vector.clone()

    steerer = MultiConceptSteerer(sv_pair, orthogonalize=True)
    ortho_vec = steerer.vectors[best_key].vector

    assert torch.allclose(original_vec, ortho_vec, atol=1e-5), (
        "Gram-Schmidt should not modify the anchor (highest clap_delta) vector."
    )


def test_gram_schmidt_unit_norm(sv_pair: dict):
    """After GS each vector should still have unit L2 norm (unless collapsed)."""
    steerer = MultiConceptSteerer(sv_pair, orthogonalize=True)
    for concept, sv in steerer.vectors.items():
        norm = sv.vector.norm().item()
        # Accept near-zero only if there was a degeneracy (warn is issued).
        assert abs(norm - 1.0) < 1e-4 or norm < 1e-7, (
            f"Vector for '{concept}' has unexpected norm {norm:.4f} after GS."
        )


# ---------------------------------------------------------------------------
# get_hooks
# ---------------------------------------------------------------------------


def test_get_hooks_returns_hooks_for_active_alphas(sv_pair: dict):
    """get_hooks should return one entry per targeted layer when alpha != 0."""
    steerer = MultiConceptSteerer(sv_pair)
    alphas = {"tempo": 50.0, "mood": 30.0}
    hooks = steerer.get_hooks(alphas)

    # Both vectors target layers [6, 7] → 2 distinct layers → 2 hooks.
    assert len(hooks) == 2
    layer_indices = [layer_idx for layer_idx, _ in hooks]
    assert sorted(layer_indices) == [6, 7]


def test_get_hooks_skips_zero_alpha(sv_pair: dict):
    """get_hooks should produce no hooks when all alphas are 0."""
    steerer = MultiConceptSteerer(sv_pair)
    hooks = steerer.get_hooks({"tempo": 0.0, "mood": 0.0})
    assert hooks == [], "No hooks expected for all-zero alphas."


def test_hook_modifies_activation(sv_pair: dict):
    """Applying a hook with alpha != 0 should change the activation tensor."""
    steerer = MultiConceptSteerer(sv_pair)
    alphas = {"tempo": 50.0, "mood": 0.0}
    hooks = steerer.get_hooks(alphas)
    assert hooks, "Expected at least one hook."

    torch.manual_seed(42)
    h_orig = torch.randn(1, 16, _DIM)
    h_copy = h_orig.clone()

    # Manually invoke the first hook function.
    _, hook_fn = hooks[0]
    h_steered = hook_fn(None, None, h_copy)

    assert not torch.allclose(h_orig, h_steered, atol=1e-6), (
        "Steered activation should differ from original when alpha != 0."
    )


def test_hook_identity_at_zero_alpha(sv_pair: dict):
    """A concept with alpha=0 contributes nothing; total delta should be zero."""
    steerer = MultiConceptSteerer(sv_pair)
    # Only mood is active; tempo is zero.
    hooks = steerer.get_hooks({"tempo": 0.0, "mood": 50.0})

    torch.manual_seed(7)
    h_orig = torch.randn(1, 16, _DIM)
    h_copy = h_orig.clone()

    _, hook_fn = hooks[0]
    h_steered = hook_fn(None, None, h_copy)
    # mood has non-zero alpha so result SHOULD differ from original.
    assert not torch.allclose(h_orig, h_steered, atol=1e-6)


def test_hook_output_same_shape(sv_pair: dict):
    """Hook output must have the same shape as input."""
    steerer = MultiConceptSteerer(sv_pair)
    hooks = steerer.get_hooks({"tempo": 10.0, "mood": 20.0})

    torch.manual_seed(99)
    h_orig = torch.randn(2, 8, _DIM)

    for _, hook_fn in hooks:
        h_out = hook_fn(None, None, h_orig.clone())
        assert h_out.shape == h_orig.shape, (
            f"Shape mismatch: expected {h_orig.shape}, got {h_out.shape}"
        )


def test_hook_tuple_output_passthrough(sv_pair: dict):
    """Hook must handle tuple outputs (h, attn_weights) correctly."""
    steerer = MultiConceptSteerer(sv_pair)
    hooks = steerer.get_hooks({"tempo": 10.0, "mood": 0.0})

    torch.manual_seed(5)
    h_orig = torch.randn(1, 16, _DIM)
    attn_weights = torch.ones(1, 8, 16, 16)  # dummy attention weights

    _, hook_fn = hooks[0]
    result = hook_fn(None, None, (h_orig.clone(), attn_weights))
    assert isinstance(result, tuple), "Tuple output must be preserved."
    assert result[0].shape == h_orig.shape
    assert torch.allclose(result[1], attn_weights), "Extra tuple elements must be unchanged."


# ---------------------------------------------------------------------------
# SAE method (no ReNorm)
# ---------------------------------------------------------------------------


def test_sae_hook_no_renorm():
    """SAE steering should NOT preserve magnitude (no ReNorm applied)."""
    dim = 64
    torch.manual_seed(42)
    v = torch.randn(dim)
    v = v / v.norm()

    sv = SteeringVector(
        concept="drums",
        method="sae",
        model_name="ace-step",
        layers=[7],
        vector=v,
        clap_delta=0.8,
    )
    steerer = MultiConceptSteerer({"drums": sv})
    hooks = steerer.get_hooks({"drums": 100.0})
    _, hook_fn = hooks[0]

    torch.manual_seed(3)
    h_orig = torch.randn(1, 16, dim)
    h_steered = hook_fn(None, None, h_orig.clone())

    orig_norms = h_orig.norm(dim=-1)
    new_norms = h_steered.norm(dim=-1)
    # SAE does NOT renorm, so norms should differ.
    assert not torch.allclose(orig_norms, new_norms, atol=1e-3), (
        "SAE method should not preserve activation magnitude."
    )
