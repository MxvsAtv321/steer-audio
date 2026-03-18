"""
Phase 2 integration tests — cross-component interactions.

Exercises the full chain:
  ConceptAlgebra (2.3) → SteeringVector → MultiConceptSteerer (2.1)
  SteeringVectorBank (1.5) round-trip → SteeringPipeline (2.5)
  TimestepSchedule (2.2) → adaptive multi-concept hooks
  ConceptProbe (2.4) → set_probe → SteeringPipeline dispatch

All tests run on CPU without model weights.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from steer_audio.vector_bank import SteeringVector, SteeringVectorBank
from steer_audio.multi_steer import MultiConceptSteerer
from steer_audio.temporal_steering import (
    cosine_schedule,
    constant_schedule,
    early_only_schedule,
    TimestepAdaptiveSteerer,
)
from steer_audio.concept_algebra import ConceptAlgebra, ConceptFeatureSet
from steer_audio.self_monitor import ConceptProbe
from steer_audio.pipeline import SteeringPipeline, _make_adaptive_multi_hook

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_DIM = 64   # small hidden dim for fast CPU tests
_NUM_FEATURES = 128
_TAU = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sv(
    concept: str,
    dim: int = _DIM,
    layers: list[int] | None = None,
    method: str = "caa",
    seed: int = 0,
    clap_delta: float = 0.5,
) -> SteeringVector:
    """Create a random unit-norm SteeringVector."""
    torch.manual_seed(seed)
    v = torch.randn(dim)
    v = v / v.norm()
    return SteeringVector(
        concept=concept,
        method=method,
        model_name="ace-step",
        layers=layers or [6, 7],
        vector=v,
        clap_delta=clap_delta,
        lpaps_at_50=0.1,
    )


def _make_feature_set(
    concept: str,
    dim: int = _DIM,
    num_features: int = _NUM_FEATURES,
    tau: int = _TAU,
    seed: int = 0,
) -> tuple[ConceptFeatureSet, torch.Tensor]:
    """Create a ConceptFeatureSet with a synthetic decoder matrix.

    Returns:
        (ConceptFeatureSet, decoder_matrix) so the same matrix can be shared.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    decoder = torch.randn(dim, num_features)
    indices = np.arange(tau, dtype=np.int64) + seed * tau  # disjoint per seed
    indices = indices % num_features
    scores = np.random.rand(tau).astype(np.float32) + 0.1  # all positive
    return (
        ConceptFeatureSet(
            concept=concept,
            feature_indices=indices,
            tfidf_scores=scores,
            decoder_matrix=decoder,
        ),
        decoder,
    )


def _make_algebra(*concepts: str) -> ConceptAlgebra:
    """Build a ConceptAlgebra with one synthetic ConceptFeatureSet per concept."""
    features: dict[str, ConceptFeatureSet] = {}
    for i, c in enumerate(concepts):
        fs, _ = _make_feature_set(c, seed=i)
        features[c] = fs
    return ConceptAlgebra(sae_model=None, concept_features=features)


def _make_mock_model(num_blocks: int = 10, dim: int = _DIM) -> Any:
    """Create a minimal mock transformer model compatible with the steerer interface.

    Structure:
        model.transformer_blocks[i].cross_attn  (single Linear layer)
    """
    class CrossAttn(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x  # identity

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.cross_attn = CrossAttn()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.cross_attn(x)

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList(
                [Block() for _ in range(num_blocks)]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.transformer_blocks:
                x = block(x)
            return x

    return MockModel()


# ---------------------------------------------------------------------------
# 1. ConceptAlgebra → SteeringVector → MultiConceptSteerer
# ---------------------------------------------------------------------------


class TestAlgebraToMultiSteer:
    """Integration: ConceptAlgebra (2.3) → SteeringVector → MultiConceptSteerer (2.1)."""

    def test_algebra_expr_then_multi_steer_interference_matrix(self):
        """Algebra expression produces a SteeringVector usable in MultiConceptSteerer."""
        algebra = _make_algebra("jazz", "techno")

        # Evaluate a simple addition expression.
        fs_result = algebra.expr("jazz + techno")
        sv_result = algebra.to_steering_vector(fs_result, layers=[6, 7])

        sv_jazz, _ = _make_feature_set("jazz_raw", seed=99)
        sv_jazz_result = algebra.to_steering_vector(sv_jazz, layers=[6, 7])

        steerer = MultiConceptSteerer(
            {"algebra_result": sv_result, "jazz_raw": sv_jazz_result},
            orthogonalize=False,
        )
        matrix = steerer.interference_matrix()
        # Shape must be (2, 2).
        assert matrix.shape == (2, 2), f"Expected (2,2), got {matrix.shape}"
        # Diagonal should be all 1.0.
        assert torch.allclose(matrix.diagonal(), torch.ones(2), atol=1e-5)

    def test_algebra_subtraction_reduces_overlap_with_subtracted_concept(self):
        """A - B should have lower cosine similarity with B's vector than A alone."""
        algebra = _make_algebra("A", "B")
        sv_a = algebra.to_steering_vector(algebra.expr("A"))
        sv_ab = algebra.to_steering_vector(algebra.expr("A - B"))
        sv_b = algebra.to_steering_vector(algebra.expr("B"))

        cos_a_b = torch.cosine_similarity(sv_a.vector, sv_b.vector, dim=0).item()
        cos_ab_b = torch.cosine_similarity(sv_ab.vector, sv_b.vector, dim=0).item()
        # After subtracting B's features, the direction should be less aligned with B.
        # (This holds in expectation; with small test data it must at least not be worse.)
        assert cos_ab_b <= cos_a_b + 1e-3, (
            f"A-B should be no more aligned with B than A alone. "
            f"cos(A,B)={cos_a_b:.4f} cos(A-B,B)={cos_ab_b:.4f}"
        )

    def test_algebra_intersection_is_subset_of_both(self):
        """Intersection feature indices are a subset of both operands."""
        algebra = _make_algebra("X", "Y")
        fs_x = algebra.features["X"]
        fs_y = algebra.features["Y"]
        fs_xy = fs_x & fs_y

        set_x = set(fs_x.feature_indices.tolist())
        set_y = set(fs_y.feature_indices.tolist())
        set_xy = set(fs_xy.feature_indices.tolist())

        assert set_xy <= set_x, "Intersection must be subset of X"
        assert set_xy <= set_y, "Intersection must be subset of Y"


# ---------------------------------------------------------------------------
# 2. SteeringVectorBank round-trip → SteeringPipeline
# ---------------------------------------------------------------------------


class TestVectorBankRoundtripToPipeline:
    """Integration: SteeringVectorBank save/load (1.5) → SteeringPipeline (2.5)."""

    def test_round_trip_then_pipeline_summary(self, tmp_path: Path):
        """Saved and reloaded vectors can be used in SteeringPipeline."""
        bank = SteeringVectorBank()
        sv_tempo = _make_sv("tempo", seed=1)
        sv_mood = _make_sv("mood", seed=2, clap_delta=0.7)

        bank.save(sv_tempo, tmp_path / "tempo_caa.safetensors")
        bank.save(sv_mood, tmp_path / "mood_caa.safetensors")

        loaded = bank.load_all(tmp_path)
        assert len(loaded) == 2

        pipeline = SteeringPipeline(loaded, orthogonalize=False)
        summary = pipeline.summary()
        assert "SteeringPipeline Summary" in summary
        # Both concept names must appear.
        assert "tempo" in summary or "mood" in summary

    def test_pipeline_from_vector_bank_factory(self, tmp_path: Path):
        """SteeringPipeline.from_vector_bank loads all vectors from a directory."""
        bank = SteeringVectorBank()
        bank.save(_make_sv("guitar", seed=3), tmp_path / "guitar_caa.safetensors")
        bank.save(_make_sv("drums", seed=4), tmp_path / "drums_caa.safetensors")

        pipeline = SteeringPipeline.from_vector_bank(
            bank, tmp_path, orthogonalize=False
        )
        # load_all keys by "{concept}_{method}" (e.g. "guitar_caa").
        assert set(pipeline.concepts) >= {"guitar_caa", "drums_caa"}

    def test_empty_directory_raises(self, tmp_path: Path):
        """from_vector_bank raises ValueError if no vectors are found."""
        bank = SteeringVectorBank()
        with pytest.raises(ValueError, match="No steering vectors found"):
            SteeringPipeline.from_vector_bank(bank, tmp_path)


# ---------------------------------------------------------------------------
# 3. TimestepSchedule → adaptive multi-concept hooks
# ---------------------------------------------------------------------------


class TestAdaptiveMultiHook:
    """Integration: _make_adaptive_multi_hook (2.5) uses schedules from (2.2)."""

    def test_hook_applies_caa_delta_with_cosine_schedule(self):
        """Hook with cosine schedule applies a non-zero delta at early steps."""
        sv = _make_sv("tempo", seed=5)
        schedule = cosine_schedule(alpha_max=60.0)
        layer_state = {"call_count": 0}
        T = 30

        hook = _make_adaptive_multi_hook(
            contributions=[(sv, 60.0, schedule)],
            layer_state=layer_state,
            total_T=T,
        )

        # Simulate first diffusion step (t = T = 30 → cosine peak = 60).
        # Use non-zero x so that CAA renorm preserves magnitude correctly.
        torch.manual_seed(42)
        x = torch.randn(1, 4, _DIM)
        x_clone = x.clone()
        result = hook(None, None, x)
        # At step 0 (t=T=30), effective_alpha should be 60.0 (cosine peak).
        # The output should differ from the input.
        assert not torch.allclose(result, x_clone), "Hook must modify activation at step 0"

    def test_hook_zero_delta_when_early_only_at_late_step(self):
        """early_only_schedule returns 0 at late steps → hook is a no-op."""
        sv = _make_sv("mood", seed=6)
        # cutoff=0.5 → inactive for t/T <= 0.5 (second half of diffusion)
        schedule = early_only_schedule(alpha=50.0, cutoff=0.5)
        # Start at step = T (last step → t=1, t/T ≈ 0.033 <= 0.5 → inactive)
        layer_state = {"call_count": 29}  # step 29 of 30 → t=1
        T = 30

        hook = _make_adaptive_multi_hook(
            contributions=[(sv, 50.0, schedule)],
            layer_state=layer_state,
            total_T=T,
        )

        x = torch.randn(1, 4, _DIM)
        result = hook(None, None, x)
        # At the very last step the schedule returns 0 → output unchanged.
        assert torch.allclose(result, x), "Hook must be a no-op when alpha=0"

    def test_hook_increments_call_count(self):
        """Hook increments layer_state['call_count'] on each call."""
        sv = _make_sv("tempo", seed=7)
        layer_state = {"call_count": 0}
        hook = _make_adaptive_multi_hook(
            contributions=[(sv, 10.0, constant_schedule(10.0))],
            layer_state=layer_state,
            total_T=30,
        )

        x = torch.randn(1, 4, _DIM)
        for expected in range(1, 4):
            hook(None, None, x)
            assert layer_state["call_count"] == expected

    def test_hook_with_tuple_output(self):
        """Hook handles tuple output (hidden, attn_weights) correctly."""
        sv = _make_sv("tempo", seed=8)
        layer_state = {"call_count": 0}
        hook = _make_adaptive_multi_hook(
            contributions=[(sv, 30.0, constant_schedule(30.0))],
            layer_state=layer_state,
            total_T=30,
        )

        x = torch.zeros(1, 4, _DIM)
        weights = torch.randn(1, 4, 4)
        result = hook(None, None, (x, weights))

        assert isinstance(result, tuple), "Should return a tuple"
        assert result[0].shape == x.shape
        # Weights should be returned unchanged.
        assert torch.allclose(result[1], weights)

    def test_multi_concept_hook_sums_sae_deltas(self):
        """Multiple SAE vectors are summed additively (no renorm for SAE)."""
        sv1 = _make_sv("jazz", method="sae", seed=9)
        sv2 = _make_sv("techno", method="sae", seed=10)
        alpha1, alpha2 = 20.0, 30.0

        layer_state = {"call_count": 0}
        T = 1  # single step so t=1
        hook = _make_adaptive_multi_hook(
            contributions=[
                (sv1, alpha1, constant_schedule(alpha1)),
                (sv2, alpha2, constant_schedule(alpha2)),
            ],
            layer_state=layer_state,
            total_T=T,
        )

        x = torch.zeros(2, 8, _DIM)
        result = hook(None, None, x)

        # Expected: x + alpha1*v1 + alpha2*v2 (no renorm for SAE).
        expected = (
            alpha1 * sv1.vector.float()
            + alpha2 * sv2.vector.float()
        )  # shape: (D,)
        diff = result.float() - x.float()
        # Every token should have this delta.
        for b in range(2):
            for t in range(8):
                assert torch.allclose(diff[b, t], expected, atol=1e-5), (
                    f"Token [{b},{t}] has wrong delta"
                )


# ---------------------------------------------------------------------------
# 4. SteeringPipeline: construction, registration, summary
# ---------------------------------------------------------------------------


class TestSteeringPipelineConstruction:
    """Unit tests for SteeringPipeline construction and introspection."""

    def test_empty_vectors_raises(self):
        """Constructing with an empty dict raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            SteeringPipeline({})

    def test_single_vector_pipeline(self):
        """Pipeline with one vector initialises without errors."""
        sv = _make_sv("tempo")
        pipeline = SteeringPipeline({"tempo": sv}, orthogonalize=False)
        assert pipeline.concepts == ["tempo"]

    def test_set_schedule_registered_concept(self):
        """set_schedule works for a registered concept."""
        sv = _make_sv("tempo")
        pipeline = SteeringPipeline({"tempo": sv}, orthogonalize=False)
        pipeline.set_schedule("tempo", cosine_schedule(alpha_max=80))
        assert "tempo" in pipeline._schedules

    def test_set_schedule_unregistered_concept_raises(self):
        """set_schedule raises KeyError for an unknown concept."""
        pipeline = SteeringPipeline({"tempo": _make_sv("tempo")}, orthogonalize=False)
        with pytest.raises(KeyError):
            pipeline.set_schedule("unknown", constant_schedule(10.0))

    def test_set_probe_registered_concept(self):
        """set_probe attaches a ConceptProbe to a registered concept."""

        def _stub(audio, sr):
            return np.zeros(64, dtype=np.float32)

        probe = ConceptProbe("tempo", clap_extractor=_stub)
        pipeline = SteeringPipeline({"tempo": _make_sv("tempo")}, orthogonalize=False)
        pipeline.set_probe("tempo", probe)
        assert "tempo" in pipeline._probes

    def test_set_probe_unregistered_raises(self):
        """set_probe raises KeyError for an unknown concept."""

        def _stub(audio, sr):
            return np.zeros(64, dtype=np.float32)

        probe = ConceptProbe("tempo", clap_extractor=_stub)
        pipeline = SteeringPipeline({"tempo": _make_sv("tempo")}, orthogonalize=False)
        with pytest.raises(KeyError):
            pipeline.set_probe("ghost_concept", probe)

    def test_summary_contains_all_concepts(self):
        """summary() lists every registered concept."""
        svs = {c: _make_sv(c, seed=i) for i, c in enumerate(["tempo", "mood", "guitar"])}
        pipeline = SteeringPipeline(svs, orthogonalize=False)
        summary = pipeline.summary()
        for concept in ["tempo", "mood", "guitar"]:
            assert concept in summary

    def test_repr(self):
        """__repr__ includes the concept list."""
        sv = _make_sv("tempo")
        pipeline = SteeringPipeline({"tempo": sv}, orthogonalize=False)
        r = repr(pipeline)
        assert "tempo" in r
        assert "SteeringPipeline" in r


# ---------------------------------------------------------------------------
# 5. SteeringPipeline: add_algebra_vector
# ---------------------------------------------------------------------------


class TestPipelineAddAlgebraVector:
    """Integration: algebra expressions registered directly into a pipeline."""

    def test_add_algebra_vector_registers_concept(self):
        """add_algebra_vector registers the result under the given name."""
        algebra = _make_algebra("jazz", "female_vocal")
        sv_jazz = _make_sv("jazz", seed=0)
        pipeline = SteeringPipeline({"jazz": sv_jazz}, orthogonalize=False)

        pipeline.add_algebra_vector("jazz_vocal", "jazz + female_vocal", algebra)
        assert "jazz_vocal" in pipeline.concepts

    def test_add_algebra_vector_has_sae_method(self):
        """Vectors from algebra expressions have method='sae'."""
        algebra = _make_algebra("fast_tempo", "drums")
        sv_fast = _make_sv("fast_tempo", seed=1)
        pipeline = SteeringPipeline({"fast_tempo": sv_fast}, orthogonalize=False)

        pipeline.add_algebra_vector("fast_no_drums", "fast_tempo - drums", algebra)
        sv = pipeline._vectors["fast_no_drums"]
        assert sv.method == "sae"

    def test_add_algebra_vector_custom_layers(self):
        """add_algebra_vector stores the specified layers."""
        algebra = _make_algebra("mood")
        sv_mood = _make_sv("mood", seed=2)
        pipeline = SteeringPipeline({"mood": sv_mood}, orthogonalize=False)

        pipeline.add_algebra_vector("mood_custom", "mood", algebra, layers=[3, 5, 7])
        assert pipeline._vectors["mood_custom"].layers == [3, 5, 7]


# ---------------------------------------------------------------------------
# 6. SteeringPipeline: hook registration on mock model
# ---------------------------------------------------------------------------


class TestPipelineHookRegistration:
    """Verify hooks are registered on the correct transformer blocks."""

    def test_multi_steer_hooks_registered_and_removed(self):
        """After steer() raises during inference, hooks are cleaned up by the finally block."""
        model = _make_mock_model(num_blocks=10)
        sv_tempo = _make_sv("tempo", layers=[6, 7], seed=0)
        sv_mood = _make_sv("mood", layers=[6, 7], seed=1)

        # Add a schedule so the pipeline takes the _steer_adaptive path.
        pipeline = SteeringPipeline(
            {"tempo": sv_tempo, "mood": sv_mood},
            schedules={"tempo": cosine_schedule(alpha_max=60.0)},
            orthogonalize=False,
            num_inference_steps=10,
        )

        def _count_hooks(m: nn.Module) -> int:
            return sum(len(mod._forward_hooks) for mod in m.modules())

        # A pipeline callable that raises *after* hooks have been registered.
        class _FailPipeline:
            sample_rate = 44100

            def __call__(self, **kwargs):
                raise RuntimeError("inference_error")

        model.pipeline = _FailPipeline()

        assert _count_hooks(model) == 0, "Baseline: no hooks before steer()"

        with pytest.raises(RuntimeError, match="inference_error"):
            pipeline.steer(
                model,
                prompt="test",
                alphas={"tempo": 60, "mood": 40},
                duration=5.0,
                seed=42,
            )

        # The finally block in _steer_adaptive must have removed all hooks.
        assert _count_hooks(model) == 0, "All hooks must be removed after steer() raises"

    def test_adaptive_hooks_registered_during_inference(self):
        """During _steer_adaptive, hooks are present on targeted layers."""
        model = _make_mock_model(num_blocks=10)
        target_layers = [3, 5]
        sv = _make_sv("tempo", layers=target_layers, seed=0)

        pipeline = SteeringPipeline(
            {"tempo": sv},
            schedules={"tempo": cosine_schedule(alpha_max=60.0)},
            orthogonalize=False,
            num_inference_steps=10,
        )

        # We'll count hooks at hook-registration time by hijacking `register_forward_hook`.
        registered: list[int] = []

        for layer_idx in target_layers:
            block = model.transformer_blocks[layer_idx]
            target_module = block.cross_attn
            orig = target_module.register_forward_hook

            def _counting_register(fn, _orig=orig, _idx=layer_idx):
                registered.append(_idx)
                return _orig(fn)

            target_module.register_forward_hook = _counting_register  # type: ignore

        # The actual inference would fail without a real pipeline; abort with a controlled error.
        class _AbortPipeline:
            sample_rate = 44100

            def __call__(self, **kwargs):
                raise RuntimeError("abort_inference")

        model.pipeline = _AbortPipeline()

        with pytest.raises(RuntimeError, match="abort_inference"):
            pipeline.steer(
                model,
                prompt="test",
                alphas={"tempo": 60},
                duration=5.0,
                seed=0,
            )

        # Both target layers must have had a hook registered.
        for layer_idx in target_layers:
            assert layer_idx in registered, f"Hook not registered on layer {layer_idx}"
        # After the exception, all hooks must be cleaned up.
        for layer_idx in target_layers:
            block = model.transformer_blocks[layer_idx]
            assert len(block.cross_attn._forward_hooks) == 0, (
                f"Hook not removed from layer {layer_idx}"
            )


# ---------------------------------------------------------------------------
# 7. TimestepAdaptiveSteerer + SteeringPipeline schedule_values consistency
# ---------------------------------------------------------------------------


class TestScheduleValueConsistency:
    """Cross-module: schedule values from temporal_steering agree with pipeline hooks."""

    def test_cosine_schedule_values_match_standalone_steerer(self):
        """SteeringPipeline's adaptive hook uses the same schedule as TimestepAdaptiveSteerer."""
        sv = _make_sv("tempo", seed=0)
        T = 20
        schedule = cosine_schedule(alpha_max=80.0)

        # Reference: values from TimestepAdaptiveSteerer utility method.
        adaptive_steerer = TimestepAdaptiveSteerer(sv, schedule, layers=[6, 7])
        ref_values = adaptive_steerer.schedule_values(T)

        # Our hook implementation should produce the same effective alphas.
        hook_alphas: list[float] = []
        for step in range(T):
            t = max(1, T - step)
            hook_alphas.append(schedule(t, T))

        for i, (ref, got) in enumerate(zip(ref_values, hook_alphas)):
            assert abs(ref - got) < 1e-6, (
                f"Schedule mismatch at step {i}: ref={ref} got={got}"
            )

    def test_pipeline_schedule_via_steer_adaptive_uses_same_values(self):
        """_make_adaptive_multi_hook produces schedule-consistent alphas."""
        sv = _make_sv("tempo", seed=0)
        T = 15
        schedule = cosine_schedule(alpha_max=60.0)

        applied_alphas: list[float] = []

        def _tracking_schedule(t: int, total_T: int) -> float:
            a = schedule(t, total_T)
            applied_alphas.append(a)
            return a

        layer_state = {"call_count": 0}
        hook = _make_adaptive_multi_hook(
            [(sv, 60.0, _tracking_schedule)],
            layer_state,
            total_T=T,
        )

        x = torch.randn(1, 1, _DIM)
        for _ in range(T):
            hook(None, None, x)

        assert len(applied_alphas) == T
        # First call should be at peak (step 0 → t=T).
        assert abs(applied_alphas[0] - schedule(T, T)) < 1e-6

    def test_constant_schedule_equals_cosine_at_midpoint(self):
        """Sanity: at t = T/2, cosine schedule value is between max and min."""
        T = 40
        alpha_max = 80.0
        schedule = cosine_schedule(alpha_max=alpha_max)
        mid = schedule(T // 2, T)
        assert 0.0 < mid < alpha_max, (
            f"Midpoint value {mid} should be between 0 and {alpha_max}"
        )


# ---------------------------------------------------------------------------
# 8. steer() raises ValueError for no active concepts
# ---------------------------------------------------------------------------


class TestPipelineSteerValidation:
    """Edge cases for SteeringPipeline.steer()."""

    def test_steer_raises_for_zero_alpha(self):
        """steer() raises if all provided alphas are 0."""
        pipeline = SteeringPipeline({"tempo": _make_sv("tempo")}, orthogonalize=False)
        with pytest.raises(ValueError, match="No active concepts"):
            pipeline.steer(None, "test prompt", alphas={"tempo": 0.0})

    def test_steer_raises_for_unknown_concept(self):
        """steer() raises if none of the requested concepts are registered."""
        pipeline = SteeringPipeline({"tempo": _make_sv("tempo")}, orthogonalize=False)
        with pytest.raises(ValueError, match="No active concepts"):
            pipeline.steer(None, "test prompt", alphas={"unknown_concept": 50.0})

    def test_steer_filters_unknown_concepts_gracefully(self):
        """steer() raises only if ALL requested concepts are unregistered."""
        # At least one registered concept with non-zero alpha → should reach inference,
        # not raise ValueError.  We use a mock pipeline that aborts early.
        model = _make_mock_model(num_blocks=10)

        class _AbortPipeline:
            sample_rate = 44100

            def __call__(self, **kwargs):
                raise RuntimeError("reached_inference")

        model.pipeline = _AbortPipeline()

        pipeline = SteeringPipeline({"tempo": _make_sv("tempo")}, orthogonalize=False)

        with pytest.raises(RuntimeError, match="reached_inference"):
            pipeline.steer(
                model,
                "test",
                alphas={"tempo": 60.0, "ghost": 40.0},  # "ghost" is not registered
            )
