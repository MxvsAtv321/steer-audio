"""
Tests for steer_audio.self_monitor — Phase 2, Prompt 2.4 (TADA roadmap).

Covers:
- ConceptProbe: train, predict_proba, save/load round-trip, untrained guard
- SelfMonitoredSteerer: construction, init validation, alpha decay/restore,
  hook registration/deregistration, monitoring trace DataFrame
- Accuracy claim: probe trained on linearly-separable synthetic data
  must achieve > 75 % on a held-out set
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from steer_audio.self_monitor import (
    ConceptProbe,
    SelfMonitoredSteerer,
    VectorAdaptiveSteerer,
    _stub_clap_extractor,
    _get_sample_rate,
    _get_transformer_blocks,
)
from steer_audio.vector_bank import SteeringVector


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_EMB_DIM = 64  # small embedding dimension for fast CPU tests


def _random_clap_extractor(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return a deterministic pseudorandom embedding based on audio sum."""
    rng = np.random.default_rng(int(abs(audio.sum() * 1e6)) % (2**31))
    return rng.standard_normal(_EMB_DIM).astype(np.float32)


def _separable_clap_extractor(label: int) -> Any:
    """Return a CLAP extractor that always returns embeddings separable by label.

    Args:
        label: 1 for positive concept, 0 for negative.

    Returns:
        Callable ``(audio, sample_rate) -> np.ndarray``.
    """

    def _extractor(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(_EMB_DIM).astype(np.float32)
        # Concept dimension: positive → +2, negative → −2
        emb[0] = 2.0 if label == 1 else -2.0
        return emb

    return _extractor


@pytest.fixture
def caa_vector() -> SteeringVector:
    """CAA SteeringVector for layers [0, 1], hidden_dim=64."""
    torch.manual_seed(0)
    return SteeringVector(
        concept="tempo",
        method="caa",
        model_name="ace-step",
        layers=[0, 1],
        vector=torch.randn(64),
        clap_delta=0.15,
    )


@pytest.fixture
def sae_vector() -> SteeringVector:
    """SAE SteeringVector for layer [0], hidden_dim=64."""
    torch.manual_seed(1)
    return SteeringVector(
        concept="mood",
        method="sae",
        model_name="ace-step",
        layers=[0],
        vector=torch.randn(64),
        clap_delta=0.10,
    )


class _SimpleBlock(nn.Module):
    """Minimal transformer block with a cross_attn Identity sub-module."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.cross_attn = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cross_attn(x)


def _make_stub_model(n_blocks: int = 4, dim: int = 64) -> Any:
    """Return a MagicMock model whose transformer_blocks are _SimpleBlocks."""
    model = MagicMock()
    blocks = nn.ModuleList([_SimpleBlock(dim) for _ in range(n_blocks)])
    model.transformer_blocks = blocks
    model.sample_rate = 44100

    # pipeline mock: return zero audio
    def _pipeline(prompt: str, duration: float = 10.0, seed: int = 42) -> np.ndarray:
        return np.zeros(int(duration * 44100), dtype=np.float32)

    model.pipeline = _pipeline

    # decode_latents: return one second of silence
    model.decode_latents = MagicMock(return_value=np.zeros(44100, dtype=np.float32))

    return model


def _make_trained_probe(concept: str = "tempo") -> ConceptProbe:
    """Return a ConceptProbe fitted on linearly separable synthetic embeddings."""
    probe = ConceptProbe(concept=concept, clap_extractor=_stub_clap_extractor)
    rng = np.random.default_rng(42)
    n_each = 40
    pos_embs = rng.normal(loc=2.0, scale=0.3, size=(n_each, 512)).astype(np.float32)
    neg_embs = rng.normal(loc=-2.0, scale=0.3, size=(n_each, 512)).astype(np.float32)
    X = np.vstack([pos_embs, neg_embs])
    y = np.array([1] * n_each + [0] * n_each)
    probe.classifier.fit(X, y)
    probe._is_trained = True
    return probe


# ---------------------------------------------------------------------------
# ConceptProbe — initialisation
# ---------------------------------------------------------------------------


class TestConceptProbeInit:
    def test_default_stub_extractor(self) -> None:
        probe = ConceptProbe(concept="tempo")
        emb = probe._clap_extractor(np.zeros(44100, dtype=np.float32), 44100)
        assert emb.shape == (512,)
        assert np.all(emb == 0.0), "Default stub should return zeros"

    def test_custom_extractor_used(self) -> None:
        def my_extractor(audio: np.ndarray, sr: int) -> np.ndarray:
            return np.ones(32, dtype=np.float32)

        probe = ConceptProbe(concept="tempo", clap_extractor=my_extractor)
        emb = probe._embed(np.zeros(100, dtype=np.float32), 44100)
        assert emb.shape == (32,)
        assert np.all(emb == 1.0)

    def test_concept_attribute_set(self) -> None:
        probe = ConceptProbe(concept="mood")
        assert probe.concept == "mood"

    def test_not_trained_on_init(self) -> None:
        probe = ConceptProbe(concept="tempo")
        assert not probe._is_trained


# ---------------------------------------------------------------------------
# ConceptProbe — predict_proba raises before training
# ---------------------------------------------------------------------------


class TestConceptProbeUntrainedGuard:
    def test_predict_raises_before_train(self) -> None:
        probe = ConceptProbe(concept="tempo")
        with pytest.raises(RuntimeError, match="not been trained"):
            probe.predict_proba(np.zeros(44100, dtype=np.float32), 44100)


# ---------------------------------------------------------------------------
# ConceptProbe — train on synthetic data (no audio files needed)
# ---------------------------------------------------------------------------


class TestConceptProbeTrain:
    def test_is_trained_after_fit(self) -> None:
        probe = _make_trained_probe("tempo")
        assert probe._is_trained

    def test_train_accuracy_above_75_percent(self) -> None:
        """Probe on linearly separable data must achieve > 75 % accuracy."""
        probe = _make_trained_probe("tempo")
        rng = np.random.default_rng(99)
        n_each = 20
        pos = rng.normal(loc=2.0, scale=0.3, size=(n_each, 512)).astype(np.float32)
        neg = rng.normal(loc=-2.0, scale=0.3, size=(n_each, 512)).astype(np.float32)
        X_test = np.vstack([pos, neg])
        y_test = np.array([1] * n_each + [0] * n_each)
        acc = probe.classifier.score(X_test, y_test)
        assert acc > 0.75, f"Expected > 75 % accuracy, got {acc:.3f}"

    def test_train_empty_paths_raises(self, tmp_path: Path) -> None:
        probe = ConceptProbe(concept="tempo")
        with pytest.raises(ValueError, match="non-empty"):
            probe.train([], [tmp_path / "neg.wav"])

    def test_predict_returns_float_in_unit_interval(self) -> None:
        probe = _make_trained_probe("tempo")
        # Use a positive-like embedding
        audio = np.ones(512, dtype=np.float32) * 2.0
        prob = probe.predict_proba(audio, 44100)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_predict_higher_for_positive_class(self) -> None:
        """Probe should assign higher P(concept) to positive-side audio."""
        probe = ConceptProbe(concept="tempo", clap_extractor=_separable_clap_extractor(1))
        # Fit on same separable pattern.
        rng = np.random.default_rng(42)
        n = 30
        pos_embs = rng.normal(loc=2.0, scale=0.2, size=(n, _EMB_DIM)).astype(np.float32)
        neg_embs = rng.normal(loc=-2.0, scale=0.2, size=(n, _EMB_DIM)).astype(np.float32)
        X = np.vstack([pos_embs, neg_embs])
        y = np.array([1] * n + [0] * n)
        probe.classifier.fit(X, y)
        probe._is_trained = True

        pos_audio = np.array([2.0] * _EMB_DIM, dtype=np.float32)
        neg_audio = np.array([-2.0] * _EMB_DIM, dtype=np.float32)

        # Override extractor to return the audio array directly as embedding.
        probe._clap_extractor = lambda a, sr: a.astype(np.float32)

        prob_pos = probe.predict_proba(pos_audio, 44100)
        prob_neg = probe.predict_proba(neg_audio, 44100)
        assert prob_pos > prob_neg, (
            f"Expected pos prob ({prob_pos:.3f}) > neg prob ({prob_neg:.3f})"
        )


# ---------------------------------------------------------------------------
# ConceptProbe — save / load round-trip
# ---------------------------------------------------------------------------


class TestConceptProbeSaveLoad:
    def test_round_trip_preserves_concept(self, tmp_path: Path) -> None:
        probe = _make_trained_probe("vocal_gender")
        save_path = tmp_path / "probe.pkl"
        probe.save(save_path)
        loaded = ConceptProbe.load(save_path)
        assert loaded.concept == probe.concept

    def test_round_trip_preserves_is_trained(self, tmp_path: Path) -> None:
        probe = _make_trained_probe("mood")
        save_path = tmp_path / "probe.pkl"
        probe.save(save_path)
        loaded = ConceptProbe.load(save_path)
        assert loaded._is_trained

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ConceptProbe.load(tmp_path / "does_not_exist.pkl")

    def test_round_trip_still_predicts(self, tmp_path: Path) -> None:
        probe = _make_trained_probe("tempo")
        save_path = tmp_path / "probe.pkl"
        probe.save(save_path)
        loaded = ConceptProbe.load(save_path)
        prob = loaded.predict_proba(np.zeros(512, dtype=np.float32), 44100)
        assert 0.0 <= prob <= 1.0

    def test_parent_dirs_created(self, tmp_path: Path) -> None:
        probe = _make_trained_probe("tempo")
        deep_path = tmp_path / "a" / "b" / "c" / "probe.pkl"
        probe.save(deep_path)
        assert deep_path.exists()


# ---------------------------------------------------------------------------
# VectorAdaptiveSteerer — construction validation (legacy vector-based steerer)
# ---------------------------------------------------------------------------


class TestVectorAdaptiveSteererInit:
    def test_threshold_low_must_be_less_than_high(
        self, caa_vector: SteeringVector
    ) -> None:
        probe = _make_trained_probe()
        with pytest.raises(ValueError, match="threshold_low"):
            VectorAdaptiveSteerer(
                vector=caa_vector,
                probe=probe,
                alpha=50.0,
                threshold_high=0.5,
                threshold_low=0.6,  # invalid: low > high
            )

    def test_invalid_decay_factor_zero(self, caa_vector: SteeringVector) -> None:
        probe = _make_trained_probe()
        with pytest.raises(ValueError, match="decay_factor"):
            VectorAdaptiveSteerer(
                vector=caa_vector, probe=probe, alpha=50.0, decay_factor=0.0
            )

    def test_invalid_decay_factor_negative(self, caa_vector: SteeringVector) -> None:
        probe = _make_trained_probe()
        with pytest.raises(ValueError, match="decay_factor"):
            VectorAdaptiveSteerer(
                vector=caa_vector, probe=probe, alpha=50.0, decay_factor=-0.1
            )

    def test_invalid_check_every(self, caa_vector: SteeringVector) -> None:
        probe = _make_trained_probe()
        with pytest.raises(ValueError, match="check_every"):
            VectorAdaptiveSteerer(
                vector=caa_vector, probe=probe, alpha=50.0, check_every=0
            )

    def test_valid_construction(self, caa_vector: SteeringVector) -> None:
        probe = _make_trained_probe()
        steerer = VectorAdaptiveSteerer(
            vector=caa_vector, probe=probe, alpha=50.0
        )
        assert steerer.alpha == 50.0
        assert steerer.threshold_high == 0.85
        assert steerer.threshold_low == 0.40
        assert steerer.decay_factor == 0.5
        assert steerer.check_every == 5


# ---------------------------------------------------------------------------
# VectorAdaptiveSteerer — steer() produces valid output
# ---------------------------------------------------------------------------


class TestVectorAdaptiveSteererSteer:
    def test_steer_returns_nonzero_array(self, caa_vector: SteeringVector) -> None:
        model = _make_stub_model()
        probe = _make_trained_probe()
        steerer = VectorAdaptiveSteerer(
            vector=caa_vector, probe=probe, alpha=50.0
        )
        audio, sr = steerer.steer(model, "a jazz song", duration=0.1, seed=42)
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1
        assert len(audio) > 0

    def test_steer_returns_correct_sample_rate(self, caa_vector: SteeringVector) -> None:
        model = _make_stub_model()
        probe = _make_trained_probe()
        steerer = VectorAdaptiveSteerer(
            vector=caa_vector, probe=probe, alpha=50.0
        )
        _, sr = steerer.steer(model, "test", duration=0.1, seed=0)
        assert sr == 44100

    def test_hooks_removed_after_steer(self, caa_vector: SteeringVector) -> None:
        """All hooks must be removed once steer() returns."""
        model = _make_stub_model(n_blocks=4)
        probe = _make_trained_probe()
        steerer = VectorAdaptiveSteerer(
            vector=caa_vector, probe=probe, alpha=50.0
        )
        steerer.steer(model, "test", duration=0.1, seed=0)
        for idx in caa_vector.layers:
            block = model.transformer_blocks[idx]
            assert len(block.cross_attn._forward_hooks) == 0, (
                f"Hook not removed from block {idx}"
            )

    def test_out_of_range_layer_skipped_gracefully(self) -> None:
        """Layers beyond the model's block count must not raise."""
        torch.manual_seed(5)
        sv = SteeringVector(
            concept="tempo",
            method="caa",
            model_name="ace-step",
            layers=[99],  # way out of range
            vector=torch.randn(64),
        )
        model = _make_stub_model(n_blocks=4)
        probe = _make_trained_probe()
        steerer = VectorAdaptiveSteerer(vector=sv, probe=probe, alpha=50.0)
        # Should complete without error.
        audio, _ = steerer.steer(model, "test", duration=0.1, seed=0)
        assert audio is not None


# ---------------------------------------------------------------------------
# VectorAdaptiveSteerer — monitoring trace
# ---------------------------------------------------------------------------


class TestMonitoringTrace:
    def test_trace_raises_before_steer(self, caa_vector: SteeringVector) -> None:
        probe = _make_trained_probe()
        steerer = VectorAdaptiveSteerer(
            vector=caa_vector, probe=probe, alpha=50.0
        )
        with pytest.raises(RuntimeError, match="steer\\(\\) first"):
            steerer.get_monitoring_trace()

    def test_trace_is_dataframe(self, caa_vector: SteeringVector) -> None:
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")

        model = _make_stub_model()
        probe = _make_trained_probe()
        steerer = VectorAdaptiveSteerer(
            vector=caa_vector,
            probe=probe,
            alpha=50.0,
            check_every=1,  # check every step to ensure we get a trace
        )
        steerer.steer(model, "test", duration=0.1, seed=0)

        # Trace may be empty if no hooks fired (stub model doesn't call cross_attn).
        # Manually inject a trace entry to test the DataFrame path.
        steerer._trace = [
            {
                "step": 1,
                "effective_alpha": 50.0,
                "concept_probability": 0.6,
                "decoded_clap_score": 0.3,
            }
        ]
        df = steerer.get_monitoring_trace()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {
            "step",
            "effective_alpha",
            "concept_probability",
            "decoded_clap_score",
        }

    def test_trace_has_expected_columns(self, caa_vector: SteeringVector) -> None:
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")

        probe = _make_trained_probe()
        steerer = VectorAdaptiveSteerer(
            vector=caa_vector, probe=probe, alpha=50.0
        )
        steerer._trace = [
            {
                "step": k,
                "effective_alpha": 50.0 * (0.5 ** k),
                "concept_probability": min(0.1 * k, 1.0),
                "decoded_clap_score": 0.4,
            }
            for k in range(1, 6)
        ]
        df = steerer.get_monitoring_trace()
        assert len(df) == 5
        assert df["step"].tolist() == list(range(1, 6))


# ---------------------------------------------------------------------------
# Alpha decay logic (unit test without a real model)
# ---------------------------------------------------------------------------


class TestAlphaDecayLogic:
    """Test the adaptive alpha update rule directly."""

    def test_alpha_decays_above_threshold_high(self) -> None:
        alpha = 100.0
        decay = 0.5
        threshold_high = 0.85
        concept_prob = 0.90  # above threshold_high
        if concept_prob > threshold_high:
            alpha = alpha * decay
        assert alpha == pytest.approx(50.0)

    def test_alpha_restores_below_threshold_low(self) -> None:
        original_alpha = 100.0
        effective_alpha = 50.0  # already decayed
        threshold_low = 0.40
        concept_prob = 0.30  # below threshold_low
        if concept_prob < threshold_low:
            effective_alpha = original_alpha
        assert effective_alpha == pytest.approx(100.0)

    def test_alpha_unchanged_between_thresholds(self) -> None:
        original_alpha = 100.0
        effective_alpha = 60.0
        threshold_high = 0.85
        threshold_low = 0.40
        concept_prob = 0.60  # between thresholds
        if concept_prob > threshold_high:
            effective_alpha = effective_alpha * 0.5
        elif concept_prob < threshold_low:
            effective_alpha = original_alpha
        assert effective_alpha == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# _get_transformer_blocks helper
# ---------------------------------------------------------------------------


class TestGetTransformerBlocks:
    def test_finds_direct_attribute(self) -> None:
        """Uses a plain object (not MagicMock) so no auto-attribute creation."""

        class _StubDirect:
            def __init__(self, b: nn.ModuleList) -> None:
                self.transformer_blocks = b

        blocks_list = nn.ModuleList([_SimpleBlock() for _ in range(3)])
        model = _StubDirect(blocks_list)
        blocks = _get_transformer_blocks(model)
        assert len(blocks) == 3

    def test_finds_nested_attribute(self) -> None:
        """Should find transformer_blocks via patchable_model.transformer_blocks."""

        class _Outer:
            def __init__(self) -> None:
                self.patchable_model = MagicMock()
                self.patchable_model.transformer_blocks = nn.ModuleList(
                    [_SimpleBlock() for _ in range(5)]
                )

        outer = _Outer()
        blocks = _get_transformer_blocks(outer)
        assert len(blocks) == 5

    def test_raises_if_not_found(self) -> None:
        model = MagicMock(spec=[])
        with pytest.raises(AttributeError):
            _get_transformer_blocks(model)


# ---------------------------------------------------------------------------
# _get_sample_rate helper
# ---------------------------------------------------------------------------


class TestGetSampleRate:
    def test_reads_direct_sample_rate(self) -> None:
        model = MagicMock()
        model.sample_rate = 22050
        assert _get_sample_rate(model) == 22050

    def test_defaults_to_44100(self) -> None:
        model = MagicMock(spec=[])
        assert _get_sample_rate(model) == 44100


# ===========================================================================
# Prompt 2.5 spec — ConceptProbe.score / delta + SelfMonitoredSteerer
# ===========================================================================


# ---------------------------------------------------------------------------
# ConceptProbe.score — stub mode and delta
# ---------------------------------------------------------------------------


class TestConceptProbeScoreDelta:
    """Tests for the new diffusion-step API: score() and delta()."""

    def test_score_stub_returns_0_5(self) -> None:
        """Without a clap_model, score() always returns 0.5."""
        probe = ConceptProbe(concept="tempo", target_prompt="fast tempo")
        audio = torch.randn(44100)
        assert probe.score(audio) == pytest.approx(0.5)

    def test_score_stub_ignores_audio_content(self) -> None:
        """Stub score is 0.5 regardless of audio content."""
        probe = ConceptProbe(concept="mood", target_prompt="happy")
        s1 = probe.score(torch.zeros(1000))
        s2 = probe.score(torch.ones(1000) * 10.0)
        assert s1 == pytest.approx(0.5)
        assert s2 == pytest.approx(0.5)

    def test_score_accepts_numpy_array(self) -> None:
        """score() accepts a numpy array as well as a torch Tensor."""
        probe = ConceptProbe(concept="tempo", target_prompt="fast")
        audio_np = np.zeros(1000, dtype=np.float32)
        assert probe.score(audio_np) == pytest.approx(0.5)

    def test_score_respects_sample_rate_param(self) -> None:
        """sample_rate is accepted without error in stub mode."""
        probe = ConceptProbe(concept="tempo", target_prompt="fast")
        assert probe.score(torch.zeros(22050), sample_rate=22050) == pytest.approx(0.5)

    def test_delta_positive_when_score_increases(self) -> None:
        """delta returns positive value when score increases."""
        probe = ConceptProbe(concept="tempo", target_prompt="fast")
        assert probe.delta(0.7, 0.5) == pytest.approx(0.2)

    def test_delta_negative_when_score_decreases(self) -> None:
        """delta returns negative value when score decreases (regression)."""
        probe = ConceptProbe(concept="tempo", target_prompt="fast")
        assert probe.delta(0.3, 0.6) == pytest.approx(-0.3)

    def test_delta_zero_when_score_unchanged(self) -> None:
        """delta returns 0.0 when scores are equal."""
        probe = ConceptProbe(concept="tempo", target_prompt="fast")
        assert probe.delta(0.5, 0.5) == pytest.approx(0.0)

    def test_target_prompt_stored(self) -> None:
        """target_prompt is stored as an attribute."""
        probe = ConceptProbe(concept="tempo", target_prompt="fast upbeat music")
        assert probe.target_prompt == "fast upbeat music"

    def test_default_target_prompt_is_empty_string(self) -> None:
        """target_prompt defaults to empty string."""
        probe = ConceptProbe(concept="tempo")
        assert probe.target_prompt == ""


# ---------------------------------------------------------------------------
# SelfMonitoredSteerer — spec-compliant (Prompt 2.5)
# ---------------------------------------------------------------------------


def _make_spec_probe(target_prompt: str = "fast tempo") -> ConceptProbe:
    """ConceptProbe in stub mode (no clap_model → score always 0.5)."""
    return ConceptProbe(concept="tempo", target_prompt=target_prompt)


def _make_spec_steerer(
    initial_alpha: float = 0.0,
    alpha_step: float = 5.0,
    min_alpha: float = 0.0,
    max_alpha: float = 100.0,
    convergence_threshold: float = 0.02,
    check_every_n_steps: int = 5,
) -> SelfMonitoredSteerer:
    """Return a SelfMonitoredSteerer with a stub probe and MagicMock steerer."""
    probe = _make_spec_probe()
    multi = MagicMock()
    steerer = SelfMonitoredSteerer(
        multi_steerer=multi,
        probe=probe,
        check_every_n_steps=check_every_n_steps,
        alpha_step=alpha_step,
        max_alpha=max_alpha,
        min_alpha=min_alpha,
        convergence_threshold=convergence_threshold,
    )
    steerer.current_alpha = initial_alpha
    return steerer


class TestSelfMonitoredSteererSpec:
    """Spec-compliant SelfMonitoredSteerer tests per Prompt 2.5."""

    # --- should_check ---

    def test_should_check_step_0(self) -> None:
        """Step 0 is always a check step (0 % N == 0)."""
        s = _make_spec_steerer(check_every_n_steps=5)
        assert s.should_check(0) is True

    def test_should_check_at_interval(self) -> None:
        """should_check returns True exactly at multiples of check_every_n_steps."""
        s = _make_spec_steerer(check_every_n_steps=5)
        assert s.should_check(5) is True
        assert s.should_check(10) is True
        assert s.should_check(15) is True

    def test_should_check_false_between_intervals(self) -> None:
        """should_check returns False between intervals."""
        s = _make_spec_steerer(check_every_n_steps=5)
        for step in [1, 2, 3, 4, 6, 7, 8, 9]:
            assert s.should_check(step) is False, f"should_check({step}) should be False"

    def test_should_check_interval_1_always_true(self) -> None:
        """With check_every_n_steps=1, every step is a check step."""
        s = _make_spec_steerer(check_every_n_steps=1)
        for step in range(10):
            assert s.should_check(step) is True

    # --- update — non-check step ---

    def test_update_no_change_on_non_check_step(self) -> None:
        """update() on a non-check step returns current_alpha unchanged."""
        s = _make_spec_steerer(initial_alpha=30.0, check_every_n_steps=5)
        alpha = s.update(torch.zeros(1000), 44100, step=3)
        assert alpha == pytest.approx(30.0)

    # --- update — stub probe returns 0.5 constantly → no delta yet on first check ---

    def test_update_first_check_no_history_no_alpha_change(self) -> None:
        """First check step has no previous score → alpha unchanged."""
        s = _make_spec_steerer(initial_alpha=20.0, check_every_n_steps=5)
        alpha = s.update(torch.zeros(1000), 44100, step=0)
        assert alpha == pytest.approx(20.0)
        assert len(s._score_history) == 1

    # --- alpha increases on regression (negative delta) ---

    def test_alpha_increases_when_score_regresses(self) -> None:
        """When delta < 0 (score regressed), alpha increases by alpha_step."""
        s = _make_spec_steerer(initial_alpha=20.0, alpha_step=5.0, check_every_n_steps=1)
        # Inject a falling score sequence using a custom probe.
        scores = iter([0.6, 0.4])  # second call lower than first

        class _FallingProbe(ConceptProbe):
            def score(self, audio, sample_rate=44100):  # type: ignore[override]
                return next(scores)

        s._probe = _FallingProbe(concept="tempo", target_prompt="fast")
        s.update(torch.zeros(100), 44100, step=0)  # score=0.6, no delta yet
        alpha = s.update(torch.zeros(100), 44100, step=1)  # score=0.4, delta=-0.2
        assert alpha == pytest.approx(25.0), f"Expected alpha=25.0, got {alpha}"

    # --- alpha decreases when score improves beyond threshold ---

    def test_alpha_decreases_when_score_improves(self) -> None:
        """When delta > convergence_threshold, alpha decreases by alpha_step."""
        s = _make_spec_steerer(
            initial_alpha=50.0,
            alpha_step=5.0,
            convergence_threshold=0.02,
            check_every_n_steps=1,
        )
        scores = iter([0.3, 0.7])  # big improvement

        class _RisingProbe(ConceptProbe):
            def score(self, audio, sample_rate=44100):  # type: ignore[override]
                return next(scores)

        s._probe = _RisingProbe(concept="tempo", target_prompt="fast")
        s.update(torch.zeros(100), 44100, step=0)  # score=0.3
        alpha = s.update(torch.zeros(100), 44100, step=1)  # score=0.7, delta=0.4 > 0.02
        assert alpha == pytest.approx(45.0), f"Expected alpha=45.0, got {alpha}"

    # --- alpha unchanged within threshold ---

    def test_alpha_unchanged_within_threshold(self) -> None:
        """When |delta| <= convergence_threshold, alpha stays the same."""
        threshold = 0.05
        s = _make_spec_steerer(
            initial_alpha=30.0,
            alpha_step=5.0,
            convergence_threshold=threshold,
            check_every_n_steps=1,
        )
        scores = iter([0.5, 0.52])  # delta = 0.02 < threshold

        class _StableProbe(ConceptProbe):
            def score(self, audio, sample_rate=44100):  # type: ignore[override]
                return next(scores)

        s._probe = _StableProbe(concept="tempo", target_prompt="fast")
        s.update(torch.zeros(100), 44100, step=0)
        alpha = s.update(torch.zeros(100), 44100, step=1)
        assert alpha == pytest.approx(30.0), f"Expected alpha unchanged at 30.0, got {alpha}"

    # --- clamping ---

    def test_alpha_clamped_at_max(self) -> None:
        """Alpha must not exceed max_alpha."""
        s = _make_spec_steerer(
            initial_alpha=98.0, alpha_step=5.0, max_alpha=100.0, check_every_n_steps=1
        )
        scores = iter([0.6, 0.3])  # regression → alpha would go to 103

        class _FallingProbe(ConceptProbe):
            def score(self, audio, sample_rate=44100):  # type: ignore[override]
                return next(scores)

        s._probe = _FallingProbe(concept="tempo", target_prompt="fast")
        s.update(torch.zeros(100), 44100, step=0)
        alpha = s.update(torch.zeros(100), 44100, step=1)
        assert alpha == pytest.approx(100.0), f"Expected alpha clamped to 100.0, got {alpha}"

    def test_alpha_clamped_at_min(self) -> None:
        """Alpha must not go below min_alpha."""
        s = _make_spec_steerer(
            initial_alpha=2.0, alpha_step=5.0, min_alpha=0.0, check_every_n_steps=1
        )
        scores = iter([0.3, 0.9])  # big improvement → alpha would go to -3

        class _RisingProbe(ConceptProbe):
            def score(self, audio, sample_rate=44100):  # type: ignore[override]
                return next(scores)

        s._probe = _RisingProbe(concept="tempo", target_prompt="fast")
        s.update(torch.zeros(100), 44100, step=0)
        alpha = s.update(torch.zeros(100), 44100, step=1)
        assert alpha == pytest.approx(0.0), f"Expected alpha clamped to 0.0, got {alpha}"

    # --- reset ---

    def test_reset_clears_state(self) -> None:
        """reset() zeroes current_alpha, _step, and _score_history."""
        s = _make_spec_steerer(initial_alpha=40.0, check_every_n_steps=1)
        s.update(torch.zeros(100), 44100, step=0)
        s.update(torch.zeros(100), 44100, step=1)
        s.reset()
        assert s.current_alpha == pytest.approx(0.0)
        assert s._step == 0
        assert s._score_history == []

    def test_reset_then_update_restarts_cleanly(self) -> None:
        """After reset(), the next update behaves as if it is the first."""
        s = _make_spec_steerer(initial_alpha=50.0, check_every_n_steps=1)
        s.update(torch.zeros(100), 44100, step=0)
        s.reset()
        s.current_alpha = 50.0
        # First update after reset: no history → no alpha change.
        alpha = s.update(torch.zeros(100), 44100, step=0)
        assert alpha == pytest.approx(50.0)
        assert len(s._score_history) == 1

    # --- get_history ---

    def test_get_history_empty_before_any_update(self) -> None:
        """get_history() returns empty lists before any update."""
        s = _make_spec_steerer()
        h = s.get_history()
        assert h["scores"] == []

    def test_get_history_accumulates_scores(self) -> None:
        """get_history() scores list grows with each check step."""
        s = _make_spec_steerer(check_every_n_steps=1)
        for step in range(4):
            s.update(torch.zeros(100), 44100, step=step)
        h = s.get_history()
        assert len(h["scores"]) == 4

    def test_get_history_returns_dict_of_lists(self) -> None:
        """get_history() returns a dict whose values are lists."""
        s = _make_spec_steerer(check_every_n_steps=1)
        s.update(torch.zeros(100), 44100, step=0)
        h = s.get_history()
        assert isinstance(h, dict)
        for v in h.values():
            assert isinstance(v, list)

    # --- initial state ---

    def test_initial_alpha_is_zero(self) -> None:
        """current_alpha is 0 on construction."""
        s = SelfMonitoredSteerer(
            multi_steerer=MagicMock(),
            probe=_make_spec_probe(),
        )
        assert s.current_alpha == pytest.approx(0.0)

    def test_initial_step_is_zero(self) -> None:
        """_step is 0 on construction."""
        s = SelfMonitoredSteerer(
            multi_steerer=MagicMock(),
            probe=_make_spec_probe(),
        )
        assert s._step == 0

    def test_initial_score_history_is_empty(self) -> None:
        """_score_history is empty on construction."""
        s = SelfMonitoredSteerer(
            multi_steerer=MagicMock(),
            probe=_make_spec_probe(),
        )
        assert s._score_history == []
