"""
Tests for steer_audio.eval_metrics and experiments.eval_sweep — Phase 3.4-pre.

Covers:
  - MetricResult: construction, to_dict, is_complete
  - StubBackend: always available, returns fixed value
  - ClapBackend: is_available matches laion_clap importability
  - FadBackend: is_available matches audioldm_eval importability
  - LpapsBackend: is_available matches editing.eval_medley importability
  - _make_backend: raises on unknown name, returns StubBackend when stub=True
  - EvalSuite: construction, availability(), evaluate_dir() with stubs
  - _parse_alpha_from_dirname: correct parsing and rejection
  - compute_alpha_sweep: returns DataFrame with expected columns and rows
  - plot_alpha_sweep: creates PNG files
  - eval_sweep.make_dry_run_dirs: creates correct directory structure
  - eval_sweep.run_eval_sweep: end-to-end dry-run with stub backends
  - eval_sweep._parse_args: CLI flag parsing
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from steer_audio.eval_metrics import (
    MetricResult,
    StubBackend,
    ClapBackend,
    FadBackend,
    LpapsBackend,
    EvalSuite,
    _make_backend,
    _parse_alpha_from_dirname,
    compute_alpha_sweep,
    plot_alpha_sweep,
)
from experiments.eval_sweep import (
    make_dry_run_dirs,
    run_eval_sweep,
    _parse_args,
    _DRY_ALPHAS,
    _DRY_N_WAVS,
)


# ---------------------------------------------------------------------------
# MetricResult
# ---------------------------------------------------------------------------


class TestMetricResult:
    def test_default_all_nan(self):
        r = MetricResult()
        assert np.isnan(r.clap_score)
        assert np.isnan(r.fad_score)
        assert np.isnan(r.lpaps_score)

    def test_to_dict_keys(self):
        r = MetricResult(clap_score=0.3, fad_score=10.0, lpaps_score=0.1)
        d = r.to_dict()
        assert set(d.keys()) >= {"clap", "fad", "lpaps"}

    def test_to_dict_values(self):
        r = MetricResult(clap_score=0.3, fad_score=10.0, lpaps_score=0.1)
        assert r.to_dict()["clap"] == pytest.approx(0.3)

    def test_is_complete_all_set(self):
        r = MetricResult(clap_score=0.3, fad_score=5.0, lpaps_score=0.2)
        assert r.is_complete()

    def test_is_complete_false_when_nan(self):
        r = MetricResult(clap_score=0.3)  # fad and lpaps are NaN
        assert not r.is_complete()

    def test_extra_included_in_to_dict(self):
        r = MetricResult(extra={"muqt": 0.42})
        d = r.to_dict()
        assert "muqt" in d
        assert d["muqt"] == pytest.approx(0.42)

    def test_extra_default_empty(self):
        r1 = MetricResult()
        r2 = MetricResult()
        r1.extra["x"] = 1.0
        assert r2.extra == {}, "extra dict shared across instances"


# ---------------------------------------------------------------------------
# StubBackend
# ---------------------------------------------------------------------------


class TestStubBackend:
    def test_is_available(self):
        assert StubBackend("clap").is_available()

    def test_returns_fixed_value(self, tmp_path):
        stub = StubBackend("clap", value=0.77)
        result = stub.compute(tmp_path)
        assert result == pytest.approx(0.77)

    def test_ignores_all_kwargs(self, tmp_path):
        stub = StubBackend("fad", value=3.14)
        assert stub.compute(tmp_path, prompt="ignored", reference_dir=tmp_path) == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# Real backend availability (importability only — no models loaded)
# ---------------------------------------------------------------------------


class TestBackendAvailability:
    def test_clap_backend_availability_type(self):
        b = ClapBackend()
        assert isinstance(b.is_available(), bool)

    def test_fad_backend_availability_type(self):
        b = FadBackend()
        assert isinstance(b.is_available(), bool)

    def test_lpaps_backend_availability_type(self):
        b = LpapsBackend()
        assert isinstance(b.is_available(), bool)

    def test_clap_backend_name(self):
        assert ClapBackend.name == "clap"

    def test_fad_backend_name(self):
        assert FadBackend.name == "fad"

    def test_lpaps_backend_name(self):
        assert LpapsBackend.name == "lpaps"


# ---------------------------------------------------------------------------
# _make_backend
# ---------------------------------------------------------------------------


class TestMakeBackend:
    def test_returns_stub_when_stub_true(self):
        b = _make_backend("clap", stub=True)
        assert isinstance(b, StubBackend)

    def test_stub_value_realistic(self):
        b = _make_backend("clap", stub=True)
        val = b.compute(Path("."))
        assert 0.0 <= val <= 1.0

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            _make_backend("unknown_metric", stub=False)

    def test_all_known_names(self):
        for name in ("clap", "fad", "lpaps"):
            b = _make_backend(name, stub=True)
            assert b.name == name


# ---------------------------------------------------------------------------
# EvalSuite
# ---------------------------------------------------------------------------


class TestEvalSuite:
    def test_construction_defaults(self):
        suite = EvalSuite(stub=True)
        avail = suite.availability()
        assert set(avail.keys()) == {"clap", "fad", "lpaps"}

    def test_subset_of_backends(self):
        suite = EvalSuite(backends=["clap"], stub=True)
        assert "clap" in suite.availability()
        assert "fad" not in suite.availability()

    def test_stub_always_available(self):
        suite = EvalSuite(stub=True)
        for ok in suite.availability().values():
            assert ok

    def test_evaluate_dir_returns_metric_result(self, tmp_path):
        suite = EvalSuite(backends=["clap"], stub=True)
        result = suite.evaluate_dir(tmp_path, prompt="piano")
        assert isinstance(result, MetricResult)

    def test_stub_values_in_expected_range(self, tmp_path):
        suite = EvalSuite(backends=["clap", "fad", "lpaps"], stub=True)
        result = suite.evaluate_dir(tmp_path, prompt="drums")
        assert 0.0 <= result.clap_score <= 1.0
        assert result.fad_score > 0.0
        assert result.lpaps_score >= 0.0

    def test_evaluate_dir_sets_extra_to_empty(self, tmp_path):
        suite = EvalSuite(backends=["clap"], stub=True)
        result = suite.evaluate_dir(tmp_path, prompt="test")
        assert result.extra == {}


# ---------------------------------------------------------------------------
# _parse_alpha_from_dirname
# ---------------------------------------------------------------------------


class TestParseAlpha:
    @pytest.mark.parametrize("name, expected", [
        ("alpha_50", 50.0),
        ("alpha_-10", -10.0),
        ("alpha_0", 0.0),
        ("alpha_100", 100.0),
        ("alpha_-100", -100.0),
        ("alpha_3.5", 3.5),
    ])
    def test_valid_names(self, name, expected):
        assert _parse_alpha_from_dirname(name) == pytest.approx(expected)

    @pytest.mark.parametrize("name", [
        "beta_50", "alpha", "50", "alpha_abc", "", "alpha_50_extra",
    ])
    def test_invalid_names_return_none(self, name):
        assert _parse_alpha_from_dirname(name) is None


# ---------------------------------------------------------------------------
# compute_alpha_sweep
# ---------------------------------------------------------------------------


@pytest.mark.optional  # requires pandas
class TestComputeAlphaSweep:
    def _make_alpha_dirs(self, base: Path, alphas=(0, 50, -50)):
        """Create minimal alpha_*/ directories with one dummy WAV each."""
        for alpha in alphas:
            d = base / f"alpha_{alpha}"
            d.mkdir()
            # Write a minimal valid WAV (44 bytes header + 2 bytes data)
            import wave, struct
            wav_path = d / "sample.wav"
            with wave.open(str(wav_path), "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(struct.pack("<h", 0))

    def test_returns_dataframe(self, tmp_path):
        import pandas as pd
        self._make_alpha_dirs(tmp_path)
        suite = EvalSuite(backends=["clap"], stub=True)
        df = compute_alpha_sweep(tmp_path, suite, prompt="test")
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_dirs(self, tmp_path):
        self._make_alpha_dirs(tmp_path, alphas=(-50, 0, 50))
        suite = EvalSuite(backends=["clap"], stub=True)
        df = compute_alpha_sweep(tmp_path, suite, prompt="test")
        assert len(df) == 3

    def test_sorted_by_alpha(self, tmp_path):
        self._make_alpha_dirs(tmp_path, alphas=(100, -100, 0))
        suite = EvalSuite(backends=["clap"], stub=True)
        df = compute_alpha_sweep(tmp_path, suite, prompt="test")
        assert list(df["alpha"]) == sorted(df["alpha"].tolist())

    def test_clap_column_present(self, tmp_path):
        self._make_alpha_dirs(tmp_path)
        suite = EvalSuite(backends=["clap"], stub=True)
        df = compute_alpha_sweep(tmp_path, suite, prompt="test")
        assert "clap" in df.columns

    def test_missing_dir_raises(self, tmp_path):
        suite = EvalSuite(stub=True)
        with pytest.raises(FileNotFoundError):
            compute_alpha_sweep(tmp_path / "nonexistent", suite)

    def test_empty_dir_returns_empty_df(self, tmp_path):
        suite = EvalSuite(stub=True)
        df = compute_alpha_sweep(tmp_path, suite)
        assert len(df) == 0

    def test_alpha_filter(self, tmp_path):
        self._make_alpha_dirs(tmp_path, alphas=(-100, -50, 0, 50, 100))
        suite = EvalSuite(backends=["clap"], stub=True)
        df = compute_alpha_sweep(tmp_path, suite, alpha_filter=[0, 50])
        assert set(df["alpha"].tolist()) == {0.0, 50.0}


# ---------------------------------------------------------------------------
# plot_alpha_sweep
# ---------------------------------------------------------------------------


@pytest.mark.optional  # requires pandas
class TestPlotAlphaSweep:
    def _make_df(self):
        import pandas as pd
        return pd.DataFrame({
            "alpha": [-50.0, 0.0, 50.0],
            "clap": [0.25, 0.30, 0.45],
            "fad": [15.0, 12.0, 14.0],
            "lpaps": [0.1, 0.05, 0.2],
        })

    def test_clap_plot_created(self, tmp_path):
        df = self._make_df()
        plot_alpha_sweep(df, tmp_path, concept="tempo")
        assert (tmp_path / "clap_vs_alpha.png").exists()

    def test_fad_plot_created(self, tmp_path):
        df = self._make_df()
        plot_alpha_sweep(df, tmp_path, concept="tempo")
        assert (tmp_path / "fad_vs_alpha.png").exists()

    def test_lpaps_plot_created(self, tmp_path):
        df = self._make_df()
        plot_alpha_sweep(df, tmp_path, concept="tempo")
        assert (tmp_path / "lpaps_vs_alpha.png").exists()

    def test_returns_list_of_paths(self, tmp_path):
        df = self._make_df()
        paths = plot_alpha_sweep(df, tmp_path, concept="tempo")
        assert len(paths) == 3
        for p in paths:
            assert p.exists()

    def test_all_nan_column_skipped(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({
            "alpha": [-50.0, 0.0, 50.0],
            "clap": [float("nan")] * 3,
            "fad": [15.0, 12.0, 14.0],
            "lpaps": [float("nan")] * 3,
        })
        paths = plot_alpha_sweep(df, tmp_path, concept="tempo")
        names = [p.name for p in paths]
        assert "clap_vs_alpha.png" not in names
        assert "fad_vs_alpha.png" in names


# ---------------------------------------------------------------------------
# make_dry_run_dirs (from eval_sweep)
# ---------------------------------------------------------------------------


class TestMakeDryRunDirs:
    def test_creates_alpha_dirs(self, tmp_path):
        make_dry_run_dirs(tmp_path)
        alpha_dirs = [d for d in tmp_path.iterdir()
                      if d.is_dir() and d.name.startswith("alpha_")]
        assert len(alpha_dirs) == len(_DRY_ALPHAS)

    def test_creates_reference_dir(self, tmp_path):
        make_dry_run_dirs(tmp_path)
        assert (tmp_path / "reference").is_dir()

    def test_wav_files_in_each_dir(self, tmp_path):
        make_dry_run_dirs(tmp_path)
        for alpha in _DRY_ALPHAS:
            alpha_dir = tmp_path / f"alpha_{alpha:.0f}"
            wavs = list(alpha_dir.glob("*.wav"))
            assert len(wavs) == _DRY_N_WAVS, (
                f"Expected {_DRY_N_WAVS} WAVs in {alpha_dir.name}, got {len(wavs)}"
            )

    def test_wav_files_are_non_empty(self, tmp_path):
        make_dry_run_dirs(tmp_path)
        for alpha in _DRY_ALPHAS:
            for wav in (tmp_path / f"alpha_{alpha:.0f}").glob("*.wav"):
                assert wav.stat().st_size > 44, f"{wav.name} appears empty"

    def test_custom_alphas(self, tmp_path):
        make_dry_run_dirs(tmp_path, alphas=[0.0, 25.0])
        assert (tmp_path / "alpha_0").is_dir()
        assert (tmp_path / "alpha_25").is_dir()


# ---------------------------------------------------------------------------
# run_eval_sweep end-to-end (dry-run with stubs)
# ---------------------------------------------------------------------------


@pytest.mark.optional  # requires pandas
class TestRunEvalSweep:
    def test_produces_csv(self, tmp_path):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            steered_dir = Path(td) / "steered"
            make_dry_run_dirs(steered_dir)
            out = tmp_path / "out"
            run_eval_sweep(
                steered_dir=steered_dir,
                out_dir=out,
                concept="tempo",
                prompt="fast tempo music",
                reference_dir=steered_dir / "reference",
                stub=True,
                backends=["clap"],
            )
            assert (out / "metrics.csv").exists()

    def test_csv_has_correct_columns(self, tmp_path):
        import tempfile, csv

        with tempfile.TemporaryDirectory() as td:
            steered_dir = Path(td) / "steered"
            make_dry_run_dirs(steered_dir, alphas=[-50, 0, 50])
            out = tmp_path / "out"
            run_eval_sweep(
                steered_dir=steered_dir,
                out_dir=out,
                concept="tempo",
                prompt="test",
                reference_dir=None,
                stub=True,
                backends=["clap"],
            )
            with open(out / "metrics.csv") as f:
                cols = csv.DictReader(f).fieldnames or []
            assert "alpha" in cols
            assert "clap" in cols

    def test_csv_row_count(self, tmp_path):
        import tempfile, pandas as pd

        with tempfile.TemporaryDirectory() as td:
            steered_dir = Path(td) / "steered"
            alphas = [-50, 0, 50]
            make_dry_run_dirs(steered_dir, alphas=alphas)
            out = tmp_path / "out"
            run_eval_sweep(
                steered_dir=steered_dir,
                out_dir=out,
                concept="tempo",
                prompt="test",
                reference_dir=None,
                stub=True,
                backends=["clap"],
            )
            df = pd.read_csv(out / "metrics.csv")
            assert len(df) == len(alphas)

    def test_produces_plot(self, tmp_path):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            steered_dir = Path(td) / "steered"
            make_dry_run_dirs(steered_dir)
            out = tmp_path / "out"
            run_eval_sweep(
                steered_dir=steered_dir,
                out_dir=out,
                concept="tempo",
                prompt="test",
                reference_dir=None,
                stub=True,
                backends=["clap"],
            )
            assert (out / "clap_vs_alpha.png").exists()


# ---------------------------------------------------------------------------
# CLI argument parsing (eval_sweep)
# ---------------------------------------------------------------------------


class TestEvalSweepCLI:
    def test_dry_run_flag(self):
        args = _parse_args(["--dry-run"])
        assert args.dry_run

    def test_default_backends(self):
        args = _parse_args(["--dry-run"])
        assert set(args.backends) == {"clap", "fad", "lpaps"}

    def test_subset_backends(self):
        args = _parse_args(["--dry-run", "--backends", "clap", "fad"])
        assert set(args.backends) == {"clap", "fad"}

    def test_concept_default(self):
        args = _parse_args(["--dry-run"])
        assert args.concept == "concept"

    def test_custom_concept(self):
        args = _parse_args(["--dry-run", "--concept", "tempo"])
        assert args.concept == "tempo"

    def test_out_dir(self, tmp_path):
        args = _parse_args(["--dry-run", "--out-dir", str(tmp_path)])
        assert args.out_dir == tmp_path
