"""
Phase 2 End-to-End Integration Smoke Test.

Exercises all four Phase 2 components together without requiring model weights:

  Component       Module
  ─────────────── ─────────────────────────────────────────────
  2.1             steer_audio.multi_steer.MultiConceptSteerer
  2.2             steer_audio.temporal_steering.*
  2.3             steer_audio.concept_algebra.ConceptAlgebra
  2.4             steer_audio.self_monitor.ConceptProbe
  2.5 (this)      steer_audio.pipeline.SteeringPipeline

Run with:
    python experiments/phase2_integration.py

Exits with code 0 on success, 1 on failure.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

# Add repo root to sys.path when running as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from steer_audio.vector_bank import SteeringVector, SteeringVectorBank
from steer_audio.multi_steer import MultiConceptSteerer
from steer_audio.temporal_steering import (
    cosine_schedule,
    constant_schedule,
    early_only_schedule,
    late_only_schedule,
    TimestepAdaptiveSteerer,
)
from steer_audio.concept_algebra import ConceptAlgebra, ConceptFeatureSet
from steer_audio.self_monitor import ConceptProbe
from steer_audio.pipeline import SteeringPipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DIM = 256          # hidden dim (smaller than real ACE-Step 3072, but representative)
_NUM_FEATURES = 512
_TAU = 16
_PASS = "  PASS"
_FAIL = "  FAIL"


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
) -> ConceptFeatureSet:
    torch.manual_seed(seed)
    np.random.seed(seed)
    decoder = torch.randn(dim, num_features)
    indices = (np.arange(tau, dtype=np.int64) + seed * tau) % num_features
    scores = np.random.rand(tau).astype(np.float32) + 0.1
    return ConceptFeatureSet(
        concept=concept,
        feature_indices=indices,
        tfidf_scores=scores,
        decoder_matrix=decoder,
    )


_CLAP_DIM = 64  # stub CLAP embedding dimension (real CLAP is 512)


def _stub_clap(audio: np.ndarray, sr: int) -> np.ndarray:
    """Stub CLAP extractor returning a deterministic embedding."""
    rng = np.random.RandomState(int(np.abs(audio).sum() * 1000) % 2**31)
    return rng.rand(_CLAP_DIM).astype(np.float32)


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = _PASS if condition else _FAIL
    detail_str = f"  ({detail})" if detail else ""
    print(f"{status}  {name}{detail_str}")
    return condition


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_01_multi_concept_steerer() -> bool:
    """2.1: MultiConceptSteerer — interference matrix and Gram-Schmidt."""
    section("2.1 Multi-Concept Steerer")
    ok = True

    sv_tempo = _make_sv("tempo", seed=0, clap_delta=0.8)
    sv_mood = _make_sv("mood", seed=1, clap_delta=0.6)
    sv_guitar = _make_sv("guitar", seed=2, clap_delta=0.4)

    steerer = MultiConceptSteerer(
        {"tempo": sv_tempo, "mood": sv_mood, "guitar": sv_guitar},
        orthogonalize=False,
    )

    matrix = steerer.interference_matrix()
    ok &= check(
        "interference_matrix shape (3,3)",
        matrix.shape == (3, 3),
        f"got {matrix.shape}",
    )
    ok &= check(
        "interference_matrix diagonal == 1",
        torch.allclose(matrix.diagonal(), torch.ones(3), atol=1e-5),
    )
    ok &= check(
        "interference_matrix symmetric",
        torch.allclose(matrix, matrix.T, atol=1e-5),
    )

    # With orthogonalization enabled, vectors should be more orthogonal.
    steerer_orth = MultiConceptSteerer(
        {"tempo": _make_sv("tempo", seed=0), "mood": _make_sv("mood", seed=1)},
        orthogonalize=True,
    )
    mat_orth = steerer_orth.interference_matrix()
    off_diag_orth = mat_orth[0, 1].abs().item()
    ok &= check(
        "orthogonalized off-diagonal near 0",
        off_diag_orth < 1e-5,
        f"|cos(v0,v1)|={off_diag_orth:.6f}",
    )

    return ok


def test_02_timestep_schedules() -> bool:
    """2.2: Timestep schedules — values, monotonicity, complementarity."""
    section("2.2 Timestep-Adaptive Schedules")
    ok = True
    T = 30

    cosine = cosine_schedule(alpha_max=80.0)
    values = [cosine(max(1, T - k), T) for k in range(T)]
    ok &= check("cosine_schedule peak at step 0", abs(values[0] - 80.0) < 1e-5, f"got {values[0]}")
    ok &= check("cosine_schedule trough near step T-1", values[-1] < 5.0, f"got {values[-1]:.4f}")
    ok &= check("cosine_schedule monotonically decreasing", all(a >= b for a, b in zip(values, values[1:])))

    early = early_only_schedule(alpha=60.0, cutoff=0.5)
    late = late_only_schedule(alpha=60.0, cutoff=0.5)
    for t_step in range(T):
        t = max(1, T - t_step)
        e = early(t, T)
        l = late(t, T)
        # At every step, exactly one of them is non-zero (complementary).
        if not ((e == 0.0) != (l == 0.0)):
            ok &= check(
                f"early+late complementary at t={t}",
                False,
                f"early={e} late={l}",
            )
            break
    else:
        ok &= check("early+late schedules are complementary across all steps", True)

    # TimestepAdaptiveSteerer.schedule_values utility.
    sv = _make_sv("tempo", seed=3)
    adaptive = TimestepAdaptiveSteerer(sv, cosine_schedule(alpha_max=50.0), layers=[6, 7])
    svals = adaptive.schedule_values(T)
    ok &= check("schedule_values length == T", len(svals) == T, f"got {len(svals)}")
    ok &= check(
        "schedule_values[0] == cosine peak",
        abs(svals[0] - 50.0) < 1e-5,
        f"got {svals[0]}",
    )

    return ok


def test_03_concept_algebra() -> bool:
    """2.3: ConceptAlgebra — expr, operators, to_steering_vector."""
    section("2.3 SAE Concept Algebra")
    ok = True

    concepts = ["jazz", "techno", "female_vocal", "fast_tempo"]
    features = {c: _make_feature_set(c, seed=i) for i, c in enumerate(concepts)}
    algebra = ConceptAlgebra(sae_model=None, concept_features=features)

    # Addition.
    result_add = algebra.expr("jazz + techno")
    set_j = set(features["jazz"].feature_indices.tolist())
    set_t = set(features["techno"].feature_indices.tolist())
    union = set_j | set_t
    result_set = set(result_add.feature_indices.tolist())
    ok &= check("addition ⊆ union", result_set <= union, f"result={result_set}")

    # Subtraction.
    result_sub = algebra.expr("jazz - techno")
    result_sub_set = set(result_sub.feature_indices.tolist())
    ok &= check("subtraction has no techno features", result_sub_set.isdisjoint(set_t))

    # Weighted blend.
    result_blend = algebra.expr("0.5 * jazz + 0.5 * techno")
    ok &= check("weighted blend returns ConceptFeatureSet", result_blend is not None)

    # Intersection.
    result_and = algebra.expr("jazz & techno")
    result_and_set = set(result_and.feature_indices.tolist())
    ok &= check("intersection ⊆ jazz", result_and_set <= set_j)
    ok &= check("intersection ⊆ techno", result_and_set <= set_t)

    # Convert to SteeringVector.
    sv = algebra.to_steering_vector(result_add, layers=[6, 7], model_name="ace-step")
    ok &= check("to_steering_vector returns SteeringVector", isinstance(sv, SteeringVector))
    ok &= check("vector has correct dim", sv.vector.shape == (_DIM,), f"got {sv.vector.shape}")
    ok &= check("method is 'sae'", sv.method == "sae")
    ok &= check("layers are [6,7]", sv.layers == [6, 7])

    # Feature overlap heatmap.
    fig = algebra.feature_overlap_heatmap()
    ok &= check("feature_overlap_heatmap returns Figure", fig is not None)

    return ok


def test_04_concept_probe() -> bool:
    """2.4: ConceptProbe — train with synthetic embeddings, predict."""
    section("2.4 Self-Monitoring / ConceptProbe")
    ok = True

    rng = np.random.RandomState(42)
    # Positive class: cluster around +1 in first dimension.
    pos_emb = rng.randn(20, _CLAP_DIM).astype(np.float32)
    pos_emb[:, 0] += 3.0
    # Negative class: cluster around -1 in first dimension.
    neg_emb = rng.randn(20, _CLAP_DIM).astype(np.float32)
    neg_emb[:, 0] -= 3.0

    call_count = 0

    def _make_extractor(embeddings: np.ndarray, idx_counter: list[int]):
        """Return embeddings sequentially, cycling through the array."""
        def extractor(audio: np.ndarray, sr: int) -> np.ndarray:
            i = idx_counter[0] % len(embeddings)
            idx_counter[0] += 1
            return embeddings[i]
        return extractor

    pos_counter: list[int] = [0]
    neg_counter: list[int] = [0]

    probe = ConceptProbe("tempo", clap_extractor=_stub_clap)
    # Override the extractor to serve pre-built embeddings via train_on_embeddings path.
    # Since ConceptProbe.train() loads audio from disk, we exercise the classifier directly.
    from sklearn.linear_model import LogisticRegression

    probe.classifier = LogisticRegression(max_iter=1000)
    X = np.vstack([pos_emb, neg_emb])
    y = np.array([1] * 20 + [0] * 20, dtype=int)
    probe.classifier.fit(X, y)
    probe._is_trained = True

    train_acc = float(probe.classifier.score(X, y))
    ok &= check(
        "logistic probe train accuracy > 0.75",
        train_acc > 0.75,
        f"accuracy={train_acc:.3f}",
    )

    # predict_proba via CLAP should return float in [0,1].
    audio = np.zeros(44100, dtype=np.float32)
    prob = probe.predict_proba(audio, sample_rate=44100)
    ok &= check("predict_proba returns float in [0,1]", 0.0 <= prob <= 1.0, f"got {prob}")

    # Save and load round-trip.
    with tempfile.TemporaryDirectory() as tmp:
        probe_path = Path(tmp) / "tempo_probe.pkl"
        probe.save(probe_path)
        probe2 = ConceptProbe.load(probe_path, clap_extractor=_stub_clap)
        ok &= check("probe concept preserved after save/load", probe2.concept == "tempo")
        ok &= check("probe is_trained after load", probe2._is_trained)

    return ok


def test_05_steering_pipeline_construction() -> bool:
    """2.5: SteeringPipeline — construction, add_algebra_vector, summary."""
    section("2.5 SteeringPipeline — Construction")
    ok = True

    sv_tempo = _make_sv("tempo", seed=0, clap_delta=0.8)
    sv_mood = _make_sv("mood", seed=1, clap_delta=0.6)

    pipeline = SteeringPipeline(
        {"tempo": sv_tempo, "mood": sv_mood},
        schedules={"tempo": cosine_schedule(alpha_max=80.0)},
        orthogonalize=True,
        num_inference_steps=30,
    )

    ok &= check("pipeline has 2 concepts", len(pipeline.concepts) == 2)
    ok &= check("tempo in pipeline.concepts", "tempo" in pipeline.concepts)

    # set_schedule.
    pipeline.set_schedule("mood", early_only_schedule(60.0))
    ok &= check("mood schedule registered", "mood" in pipeline._schedules)

    # set_probe.
    probe = ConceptProbe("tempo", clap_extractor=_stub_clap)
    pipeline.set_probe("tempo", probe)
    ok &= check("probe registered for tempo", "tempo" in pipeline._probes)

    # summary.
    summary = pipeline.summary()
    ok &= check("summary contains 'SteeringPipeline Summary'", "SteeringPipeline Summary" in summary)
    ok &= check("summary contains concept names", "tempo" in summary and "mood" in summary)

    # repr.
    r = repr(pipeline)
    ok &= check("repr contains 'SteeringPipeline'", "SteeringPipeline" in r)

    # add_algebra_vector.
    features = {c: _make_feature_set(c, seed=i) for i, c in enumerate(["jazz", "techno"])}
    algebra = ConceptAlgebra(sae_model=None, concept_features=features)
    pipeline.add_algebra_vector("jazz_techno", "jazz + techno", algebra)
    ok &= check("algebra vector registered", "jazz_techno" in pipeline.concepts)
    ok &= check("algebra vector method is sae", pipeline._vectors["jazz_techno"].method == "sae")

    return ok


def test_06_pipeline_vector_bank_roundtrip() -> bool:
    """2.5: SteeringPipeline.from_vector_bank loads saved vectors."""
    section("2.5 SteeringPipeline.from_vector_bank Round-Trip")
    ok = True

    bank = SteeringVectorBank()
    svs = {c: _make_sv(c, seed=i) for i, c in enumerate(["tempo", "mood", "guitar"])}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for name, sv in svs.items():
            bank.save(sv, tmp_path / f"{name}_caa.safetensors")

        pipeline = SteeringPipeline.from_vector_bank(bank, tmp_path, orthogonalize=False)

    ok &= check("pipeline loaded 3 concepts", len(pipeline.concepts) == 3)
    # load_all keys by "{concept}_{method}", e.g. "tempo_caa".
    for c in ["tempo_caa", "mood_caa", "guitar_caa"]:
        ok &= check(f"'{c}' in pipeline", c in pipeline.concepts)

    return ok


def test_07_steer_validation() -> bool:
    """2.5: steer() raises correct exceptions on bad inputs."""
    section("2.5 SteeringPipeline.steer() — Validation")
    ok = True

    pipeline = SteeringPipeline({"tempo": _make_sv("tempo")}, orthogonalize=False)

    # Zero alpha → no active concepts.
    try:
        pipeline.steer(None, "test", alphas={"tempo": 0.0})
        ok &= check("steer() raises for zero alpha", False)
    except ValueError as exc:
        ok &= check("steer() raises ValueError for zero alpha", "No active concepts" in str(exc))

    # Unknown concept (not registered).
    try:
        pipeline.steer(None, "test", alphas={"unknown": 50.0})
        ok &= check("steer() raises for unknown concept", False)
    except ValueError as exc:
        ok &= check("steer() raises ValueError for unknown concept", "No active concepts" in str(exc))

    return ok


def test_08_adaptive_hook_correctness() -> bool:
    """2.5: Adaptive multi-concept hook produces correct deltas."""
    section("2.5 Adaptive Multi-Hook — Delta Correctness")
    ok = True

    from steer_audio.pipeline import _make_adaptive_multi_hook
    from steer_audio.multi_steer import _renorm

    sv1 = _make_sv("tempo", method="caa", seed=10)
    sv2 = _make_sv("mood", method="sae", seed=11)
    T = 20
    alpha1, alpha2 = 50.0, 30.0

    layer_state = {"call_count": 0}
    hook = _make_adaptive_multi_hook(
        [(sv1, alpha1, constant_schedule(alpha1)),
         (sv2, alpha2, constant_schedule(alpha2))],
        layer_state,
        total_T=T,
    )

    batch, seq = 2, 8
    x = torch.randn(batch, seq, _DIM)
    result = hook(None, None, x)

    # Expected:
    # CAA delta = alpha1 * v1
    # After renorm: h_out_caa = renorm(x + caa_delta, x)
    # Then SAE delta: h_out = h_out_caa + alpha2 * v2
    caa_delta = alpha1 * sv1.vector.float()  # (D,)
    h_renormed = _renorm(x.float() + caa_delta, x.float())
    expected = h_renormed + alpha2 * sv2.vector.float()

    ok &= check(
        "adaptive hook output shape correct",
        result.shape == x.shape,
        f"got {result.shape}",
    )
    ok &= check(
        "adaptive hook produces correct combined delta",
        torch.allclose(result.float(), expected.float(), atol=1e-5),
        f"max_err={( result.float() - expected.float()).abs().max().item():.6f}",
    )

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 60)
    print("  TADA Phase 2 Integration Smoke Test")
    print("=" * 60)

    tests = [
        ("2.1 MultiConceptSteerer", test_01_multi_concept_steerer),
        ("2.2 Timestep Schedules", test_02_timestep_schedules),
        ("2.3 ConceptAlgebra", test_03_concept_algebra),
        ("2.4 ConceptProbe", test_04_concept_probe),
        ("2.5 SteeringPipeline construction", test_05_steering_pipeline_construction),
        ("2.5 SteeringPipeline.from_vector_bank", test_06_pipeline_vector_bank_roundtrip),
        ("2.5 steer() validation", test_07_steer_validation),
        ("2.5 adaptive hook correctness", test_08_adaptive_hook_correctness),
    ]

    results: dict[str, bool] = {}
    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception:
            print(f"\n  EXCEPTION in '{name}':")
            traceback.print_exc()
            results[name] = False

    # Summary table.
    section("Summary")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {name}")
        all_passed = all_passed and passed

    print(f"\n  {'All tests passed!' if all_passed else 'Some tests FAILED.'}")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
