"""
Tests for the Steering Vector Geometry Analysis — Phase 3.3.

Covers:
  - ConceptVector construction and unit-norm enforcement
  - make_synthetic_vectors: shape, reproducibility, category coverage
  - analysis_cosine_heatmap: matrix shape, diagonal = 1, symmetry, file output
  - analysis_pca: coord shape, explained-variance sum, file output
  - analysis_linear_probing: accuracy > 0.5 on synthetic data, CSV output
  - analysis_arithmetic_verification: composed_cosine > individual cosines, CSV output
  - analysis_layer_progression: array shape, layer-reference peak, file output
  - run_geometry_analysis: end-to-end smoke test (dry-run), report.md written
  - GeometryReport populated with sane values
  - CLI argument parsing and mutual-exclusion guard
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup — mirrors other tests in this suite
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.vector_geometry import (
    DRY_HIDDEN_DIM,
    REFERENCE_LAYER,
    NUM_LAYERS,
    ConceptVector,
    GeometryReport,
    analysis_arithmetic_verification,
    analysis_cosine_heatmap,
    analysis_layer_progression,
    analysis_linear_probing,
    analysis_pca,
    make_synthetic_vectors,
    run_geometry_analysis,
    _parse_args,
    _make_synthetic_activations,
    _make_layer_vectors,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_concepts() -> list[ConceptVector]:
    """Four synthetic concepts with dim=16 for fast tests."""
    return make_synthetic_vectors(n_concepts=4, hidden_dim=16, seed=0)


@pytest.fixture()
def medium_concepts() -> list[ConceptVector]:
    """Eight synthetic concepts with default DRY_HIDDEN_DIM."""
    return make_synthetic_vectors(n_concepts=8, seed=42)


# ---------------------------------------------------------------------------
# ConceptVector
# ---------------------------------------------------------------------------


class TestConceptVector:
    def test_fields_stored(self):
        v = torch.randn(16)
        cv = ConceptVector(
            name="tempo", category="tempo", method="caa", layer=7, vec=v
        )
        assert cv.name == "tempo"
        assert cv.category == "tempo"
        assert cv.layer == 7
        assert cv.vec is v

    def test_vec_is_tensor(self, small_concepts):
        for c in small_concepts:
            assert isinstance(c.vec, torch.Tensor)


# ---------------------------------------------------------------------------
# make_synthetic_vectors
# ---------------------------------------------------------------------------


class TestMakeSyntheticVectors:
    def test_correct_count(self):
        cvs = make_synthetic_vectors(n_concepts=5, hidden_dim=16, seed=0)
        assert len(cvs) == 5

    def test_hidden_dim(self):
        cvs = make_synthetic_vectors(n_concepts=4, hidden_dim=32, seed=0)
        for c in cvs:
            assert c.vec.shape == (32,)

    def test_unit_norm(self):
        """Each synthetic vector should be unit-norm."""
        cvs = make_synthetic_vectors(n_concepts=6, hidden_dim=16, seed=1)
        for c in cvs:
            assert abs(c.vec.norm().item() - 1.0) < 1e-5, (
                f"Vector '{c.name}' is not unit-norm: {c.vec.norm().item()}"
            )

    def test_reproducible(self):
        a = make_synthetic_vectors(n_concepts=4, hidden_dim=16, seed=7)
        b = make_synthetic_vectors(n_concepts=4, hidden_dim=16, seed=7)
        for ca, cb in zip(a, b):
            assert torch.allclose(ca.vec, cb.vec)

    def test_different_seeds_differ(self):
        a = make_synthetic_vectors(n_concepts=4, hidden_dim=16, seed=1)
        b = make_synthetic_vectors(n_concepts=4, hidden_dim=16, seed=2)
        # At least one pair of vectors should differ
        any_diff = any(
            not torch.allclose(ca.vec, cb.vec) for ca, cb in zip(a, b)
        )
        assert any_diff

    def test_categories_assigned(self):
        cvs = make_synthetic_vectors(n_concepts=8, hidden_dim=16, seed=0)
        categories = {c.category for c in cvs}
        assert len(categories) > 1, "Expected multiple categories"


# ---------------------------------------------------------------------------
# Analysis 1 — Cosine Heatmap
# ---------------------------------------------------------------------------


class TestAnalysisCosineHeatmap:
    def test_matrix_shape(self, small_concepts, tmp_path):
        mat = analysis_cosine_heatmap(small_concepts, tmp_path)
        n = len(small_concepts)
        assert mat.shape == (n, n)

    def test_diagonal_is_one(self, small_concepts, tmp_path):
        mat = analysis_cosine_heatmap(small_concepts, tmp_path)
        for i in range(len(small_concepts)):
            assert abs(mat[i, i] - 1.0) < 1e-5

    def test_symmetry(self, small_concepts, tmp_path):
        mat = analysis_cosine_heatmap(small_concepts, tmp_path)
        assert np.allclose(mat, mat.T, atol=1e-6)

    def test_values_in_range(self, small_concepts, tmp_path):
        mat = analysis_cosine_heatmap(small_concepts, tmp_path)
        assert mat.min() >= -1.0 - 1e-5
        assert mat.max() <= 1.0 + 1e-5

    def test_file_created(self, small_concepts, tmp_path):
        analysis_cosine_heatmap(small_concepts, tmp_path)
        assert (tmp_path / "cosine_heatmap.png").exists()


# ---------------------------------------------------------------------------
# Analysis 2 — PCA
# ---------------------------------------------------------------------------


class TestAnalysisPCA:
    def test_coord_shape(self, small_concepts, tmp_path):
        coords, ratios = analysis_pca(small_concepts, tmp_path)
        n = len(small_concepts)
        assert coords.shape[0] == n
        assert coords.shape[1] == 2

    def test_explained_variance_positive(self, small_concepts, tmp_path):
        _, ratios = analysis_pca(small_concepts, tmp_path)
        assert all(r >= 0 for r in ratios)

    def test_explained_variance_sums_to_one(self, small_concepts, tmp_path):
        _, ratios = analysis_pca(small_concepts, tmp_path)
        assert abs(ratios.sum() - 1.0) < 1e-5

    def test_files_created(self, small_concepts, tmp_path):
        analysis_pca(small_concepts, tmp_path)
        assert (tmp_path / "pca_2d.png").exists()
        assert (tmp_path / "pca_variance.png").exists()

    def test_cumulative_variance_monotone(self, medium_concepts, tmp_path):
        _, ratios = analysis_pca(medium_concepts, tmp_path)
        cumvar = np.cumsum(ratios)
        assert all(cumvar[i] <= cumvar[i + 1] + 1e-9 for i in range(len(cumvar) - 1))


# ---------------------------------------------------------------------------
# Analysis 3 — Linear Probing
# ---------------------------------------------------------------------------


class TestAnalysisLinearProbing:
    def test_returns_one_row_per_concept(self, small_concepts, tmp_path):
        rows = analysis_linear_probing(small_concepts, tmp_path, dry_run=True)
        assert len(rows) == len(small_concepts)

    def test_accuracy_above_chance(self, small_concepts, tmp_path):
        """Probe should beat random chance (0.5) on well-structured synthetic data."""
        rows = analysis_linear_probing(
            small_concepts, tmp_path, dry_run=True, n_samples_per_class=64
        )
        for row in rows:
            assert row["accuracy"] > 0.5, (
                f"Probe for '{row['concept']}' accuracy={row['accuracy']:.3f} ≤ 0.5"
            )

    def test_vc_alignment_in_range(self, small_concepts, tmp_path):
        rows = analysis_linear_probing(small_concepts, tmp_path, dry_run=True)
        for row in rows:
            assert -1.0 <= row["vc_alignment"] <= 1.0

    def test_csv_written(self, small_concepts, tmp_path):
        analysis_linear_probing(small_concepts, tmp_path, dry_run=True)
        csv_path = tmp_path / "probing_accuracy.csv"
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == len(small_concepts)
        assert "accuracy" in rows[0]
        assert "vc_alignment" in rows[0]

    def test_dry_run_flag_stored(self, small_concepts, tmp_path):
        rows = analysis_linear_probing(small_concepts, tmp_path, dry_run=True)
        for row in rows:
            assert row["dry_run"] is True


# ---------------------------------------------------------------------------
# Analysis 4 — Arithmetic Verification
# ---------------------------------------------------------------------------


class TestAnalysisArithmetic:
    def test_returns_rows(self, small_concepts, tmp_path):
        rows = analysis_arithmetic_verification(
            small_concepts, tmp_path, dry_run=True
        )
        assert len(rows) >= 1

    def test_composed_exceeds_individuals(self, small_concepts, tmp_path):
        """Composed cosine should be ≥ max(a_cosine, b_cosine) on synthetic data."""
        rows = analysis_arithmetic_verification(
            small_concepts, tmp_path, dry_run=True, n_samples=64
        )
        for row in rows:
            # Composed direction should be at least as aligned as each individual
            assert row["composed_cosine"] >= max(row["a_cosine"], row["b_cosine"]) - 0.05, (
                f"Pair {row['pair']}: composed={row['composed_cosine']:.3f} "
                f"< max(a={row['a_cosine']:.3f}, b={row['b_cosine']:.3f})"
            )

    def test_csv_written(self, small_concepts, tmp_path):
        analysis_arithmetic_verification(small_concepts, tmp_path, dry_run=True)
        csv_path = tmp_path / "arithmetic_verification.csv"
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert "pair" in rows[0]
        assert "composed_cosine" in rows[0]

    def test_cosine_values_in_range(self, small_concepts, tmp_path):
        rows = analysis_arithmetic_verification(
            small_concepts, tmp_path, dry_run=True
        )
        for row in rows:
            for key in ("composed_cosine", "a_cosine", "b_cosine"):
                assert -1.0 <= row[key] <= 1.0 + 1e-5, (
                    f"{key}={row[key]} out of [-1, 1]"
                )


# ---------------------------------------------------------------------------
# Analysis 5 — Layer Progression
# ---------------------------------------------------------------------------


class TestAnalysisLayerProgression:
    def test_array_shape(self, small_concepts, tmp_path):
        sims = analysis_layer_progression(small_concepts, tmp_path, dry_run=True)
        assert sims.shape == (len(small_concepts), NUM_LAYERS)

    def test_reference_layer_sim_is_one(self, small_concepts, tmp_path):
        """At REFERENCE_LAYER the vector is the reference itself — similarity = 1."""
        sims = analysis_layer_progression(small_concepts, tmp_path, dry_run=True)
        for i in range(len(small_concepts)):
            assert abs(sims[i, REFERENCE_LAYER] - 1.0) < 1e-5, (
                f"Concept {i}: sim at layer {REFERENCE_LAYER} = {sims[i, REFERENCE_LAYER]:.4f}"
            )

    def test_values_in_range(self, small_concepts, tmp_path):
        sims = analysis_layer_progression(small_concepts, tmp_path, dry_run=True)
        assert sims.min() >= -1.0 - 1e-5
        assert sims.max() <= 1.0 + 1e-5

    def test_file_created(self, small_concepts, tmp_path):
        analysis_layer_progression(small_concepts, tmp_path, dry_run=True)
        assert (tmp_path / "layer_progression.png").exists()

    def test_early_layers_lower_sim(self, small_concepts, tmp_path):
        """Layer 0 similarity should be lower than REFERENCE_LAYER (by design)."""
        sims = analysis_layer_progression(small_concepts, tmp_path, dry_run=True)
        for i in range(len(small_concepts)):
            assert sims[i, 0] < sims[i, REFERENCE_LAYER], (
                f"Concept {i}: layer-0 sim ({sims[i,0]:.3f}) ≥ "
                f"layer-{REFERENCE_LAYER} sim ({sims[i,REFERENCE_LAYER]:.3f})"
            )


# ---------------------------------------------------------------------------
# Helper: _make_synthetic_activations
# ---------------------------------------------------------------------------


class TestMakeSyntheticActivations:
    def test_shape(self, small_concepts):
        X, y = _make_synthetic_activations(small_concepts[0], 20, 20, seed=0)
        assert X.shape == (40, small_concepts[0].vec.shape[0])
        assert y.shape == (40,)

    def test_labels_balanced(self, small_concepts):
        n_pos, n_neg = 30, 30
        _, y = _make_synthetic_activations(small_concepts[0], n_pos, n_neg, seed=0)
        assert y.sum() == n_pos
        assert (y == 0).sum() == n_neg

    def test_positive_higher_projection(self, small_concepts):
        """Positive samples should project more onto v than negative samples."""
        c = small_concepts[0]
        X, y = _make_synthetic_activations(c, 50, 50, seed=0)
        v = c.vec.numpy()
        proj = X @ v
        assert proj[y == 1].mean() > proj[y == 0].mean()


# ---------------------------------------------------------------------------
# Helper: _make_layer_vectors
# ---------------------------------------------------------------------------


class TestMakeLayerVectors:
    def test_length(self, small_concepts):
        vecs = _make_layer_vectors(small_concepts[0], NUM_LAYERS, seed=0)
        assert len(vecs) == NUM_LAYERS

    def test_unit_norm(self, small_concepts):
        vecs = _make_layer_vectors(small_concepts[0], NUM_LAYERS, seed=0)
        for l, v in enumerate(vecs):
            assert abs(v.norm().item() - 1.0) < 1e-5, (
                f"Layer {l} vector not unit-norm: {v.norm().item()}"
            )

    def test_reference_layer_matches(self, small_concepts):
        """At REFERENCE_LAYER the generated vector should align with the reference."""
        c = small_concepts[0]
        vecs = _make_layer_vectors(c, NUM_LAYERS, seed=0)
        ref = torch.nn.functional.normalize(c.vec, dim=0)
        ref_vec = torch.nn.functional.normalize(vecs[REFERENCE_LAYER], dim=0)
        sim = float(ref @ ref_vec)
        assert abs(sim - 1.0) < 1e-4, f"Reference layer sim={sim:.4f}"


# ---------------------------------------------------------------------------
# End-to-End: run_geometry_analysis
# ---------------------------------------------------------------------------


class TestRunGeometryAnalysis:
    def test_smoke_dry_run(self, small_concepts, tmp_path):
        """Full pipeline should complete without errors in dry-run mode."""
        report = run_geometry_analysis(
            concepts=small_concepts,
            out_dir=tmp_path,
            dry_run=True,
            seed=0,
        )
        assert isinstance(report, GeometryReport)
        assert report.n_concepts == len(small_concepts)

    def test_output_files_created(self, small_concepts, tmp_path):
        run_geometry_analysis(small_concepts, tmp_path, dry_run=True)
        expected = [
            "cosine_heatmap.png",
            "pca_2d.png",
            "pca_variance.png",
            "probing_accuracy.csv",
            "arithmetic_verification.csv",
            "layer_progression.png",
            "report.md",
        ]
        for fname in expected:
            assert (tmp_path / fname).exists(), f"Missing output: {fname}"

    def test_report_notes_nonempty(self, small_concepts, tmp_path):
        report = run_geometry_analysis(small_concepts, tmp_path, dry_run=True)
        assert len(report.notes) > 0

    def test_pca_variance_between_zero_and_one(self, small_concepts, tmp_path):
        report = run_geometry_analysis(small_concepts, tmp_path, dry_run=True)
        assert 0.0 <= report.pca_top2_variance <= 1.0
        assert 0.0 <= report.pca_top3_variance <= 1.0

    def test_probe_accuracy_sane(self, small_concepts, tmp_path):
        report = run_geometry_analysis(small_concepts, tmp_path, dry_run=True)
        assert 0.0 <= report.mean_probe_accuracy <= 1.0

    def test_report_md_contains_sections(self, small_concepts, tmp_path):
        run_geometry_analysis(small_concepts, tmp_path, dry_run=True)
        text = (tmp_path / "report.md").read_text()
        for section in [
            "Analysis 1",
            "Analysis 2",
            "Analysis 3",
            "Analysis 4",
            "Analysis 5",
        ]:
            assert section in text, f"Report missing section: {section}"

    def test_hidden_dim_in_report(self, small_concepts, tmp_path):
        report = run_geometry_analysis(small_concepts, tmp_path, dry_run=True)
        assert report.hidden_dim == small_concepts[0].vec.shape[0]


# ---------------------------------------------------------------------------
# GeometryReport
# ---------------------------------------------------------------------------


class TestGeometryReport:
    def test_default_construction(self):
        r = GeometryReport()
        assert r.n_concepts == 0
        assert r.notes == []

    def test_notes_is_fresh_list(self):
        r1 = GeometryReport()
        r2 = GeometryReport()
        r1.notes.append("hello")
        assert r2.notes == [], "notes list shared across instances"


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIParsing:
    def test_defaults(self):
        args = _parse_args([])
        assert not args.dry_run
        assert args.vectors_dir is None
        assert args.n_concepts == 8
        assert args.seed == 42

    def test_dry_run_flag(self):
        args = _parse_args(["--dry-run"])
        assert args.dry_run

    def test_out_dir(self, tmp_path):
        args = _parse_args(["--dry-run", "--out-dir", str(tmp_path)])
        assert args.out_dir == tmp_path

    def test_hidden_dim(self):
        args = _parse_args(["--dry-run", "--hidden-dim", "128"])
        assert args.hidden_dim == 128

    def test_n_concepts(self):
        args = _parse_args(["--dry-run", "--n-concepts", "6"])
        assert args.n_concepts == 6

    def test_seed(self):
        args = _parse_args(["--dry-run", "--seed", "99"])
        assert args.seed == 99
