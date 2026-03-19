"""
Tests for the SAE Scaling Laws experiment — Phase 3.2.

Covers:
  - ScalingConfig and ScalingResult dataclass correctness
  - Synthetic activation generation (shape, dtype, reproducibility)
  - Single-config training pipeline (smallest config: m=2, k=4, n=50)
  - FVU, dead feature fraction, and mean sparsity computation
  - Power law fitting (known data and degenerate cases)
  - Pareto frontier identification
  - CSV and Markdown I/O (file existence and schema)
  - Plot file generation (files created, no display)
  - End-to-end smoke test (m=2, k=8, n=100 in dry-run)
  - resolve_grid(): priority rules for CLI flags, presets, and modes
  - _PRESET_REAL_SMALL: grid size and specific values
  - CLI parsing of --m-values / --k-values / --data-sizes / --seeds /
    --preset-real-small
"""

from __future__ import annotations

import csv
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup: mirrors conftest.py so tests run with `pytest tests/`
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SAE_ROOT = _REPO_ROOT / "sae"

for _p in [str(_REPO_ROOT), str(_SAE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse

from experiments.sae_scaling import (  # noqa: E402
    ScalingConfig,
    ScalingResult,
    _PRESET_REAL_SMALL,
    _build_parser,
    compute_dead_features,
    compute_fvu,
    compute_mean_sparsity,
    find_pareto_frontier,
    fit_power_law,
    generate_synthetic_activations,
    plot_alignment_vs_k,
    plot_fvu_vs_expansion,
    plot_pareto_frontier,
    resolve_grid,
    run_single_config,
    save_results_csv,
    save_summary_table,
    train_sae_quick,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_cfg() -> ScalingConfig:
    """Minimal ScalingConfig for fast CPU tests (d_in=32, 10 steps)."""
    return ScalingConfig(
        expansion_factors=[2],
        k_values=[4],
        data_sizes=[50],
        seeds=[42],
        hidden_dim=32,
        num_train_steps=10,
        dry_run_hidden_dim=32,
        dry_run_steps=10,
        batch_size=16,
        lr=1e-3,
        val_fraction=0.2,
    )


@pytest.fixture
def small_train_val() -> tuple[torch.Tensor, torch.Tensor]:
    """50-sample train and 10-sample val set with d_in=32."""
    all_data = generate_synthetic_activations(n=60, hidden_dim=32, seed=42)
    return all_data[:50], all_data[50:]


@pytest.fixture
def tiny_sae(small_train_val: tuple[torch.Tensor, torch.Tensor]):
    """Small trained SAE (m=2, k=4, d_in=32) on CPU (10 steps)."""
    train_data, val_data = small_train_val
    sae, _ = train_sae_quick(
        m=2,
        k=4,
        train_data=train_data,
        val_data=val_data,
        device=torch.device("cpu"),
        seed=42,
        num_steps=10,
        batch_size=16,
        lr=1e-3,
    )
    return sae


@pytest.fixture
def sample_results() -> list[ScalingResult]:
    """12 synthetic ScalingResults spanning 2×2 grid × 3 seeds.

    FVU follows a rough power law: fvu ≈ 0.8 / (m^0.5 * k^0.3).
    Dead % decreases as m increases (more capacity).
    """
    results = []
    for m in [2, 4]:
        for k in [8, 32]:
            for seed in [42, 123, 456]:
                rng = np.random.default_rng(seed)
                fvu = max(
                    0.01,
                    0.8 / (m**0.5 * k**0.3) + rng.uniform(-0.02, 0.02),
                )
                dead = max(0.0, 0.1 / m + rng.uniform(-0.02, 0.02))
                results.append(
                    ScalingResult(
                        expansion_factor=m,
                        k=k,
                        data_size=100,
                        seed=seed,
                        num_latents=m * 64,
                        fvu=fvu,
                        dead_feature_pct=dead,
                        mean_sparsity=float(k),
                    )
                )
    return results


# ---------------------------------------------------------------------------
# 1. Config and result dataclasses
# ---------------------------------------------------------------------------


def test_scaling_config_paper_defaults() -> None:
    """Default grid contains the paper config values m=4 and k=64."""
    cfg = ScalingConfig()
    assert 4 in cfg.expansion_factors, "Paper m=4 should be in expansion_factors"
    assert 64 in cfg.k_values, "Paper k=64 should be in k_values"
    assert cfg.hidden_dim == 3072, "ACE-Step hidden_dim must be 3072"


def test_scaling_result_to_dict() -> None:
    """ScalingResult converts to a dict with all expected keys."""
    r = ScalingResult(
        expansion_factor=4,
        k=64,
        data_size=100,
        seed=42,
        num_latents=256,
        fvu=0.3,
        dead_feature_pct=0.05,
        mean_sparsity=64.0,
    )
    d = asdict(r)
    assert isinstance(d, dict)
    assert d["expansion_factor"] == 4
    assert d["clap_delta"] == -1.0, "Sentinel value for uncomputed metrics must be -1.0"
    assert d["lpaps"] == -1.0
    assert d["interpretability_score"] == -1.0


# ---------------------------------------------------------------------------
# 2. Synthetic data generation
# ---------------------------------------------------------------------------


def test_generate_shape() -> None:
    """Output shape is exactly (n, hidden_dim) with float32 dtype."""
    acts = generate_synthetic_activations(n=100, hidden_dim=32, seed=42)
    assert acts.shape == (100, 32)
    assert acts.dtype == torch.float32


def test_generate_reproducible() -> None:
    """Same seed → identical tensors; different seeds → different tensors."""
    a1 = generate_synthetic_activations(n=50, hidden_dim=16, seed=42)
    a2 = generate_synthetic_activations(n=50, hidden_dim=16, seed=42)
    a3 = generate_synthetic_activations(n=50, hidden_dim=16, seed=99)

    assert torch.allclose(a1, a2), "Identical seeds should give identical activations"
    assert not torch.allclose(a1, a3), "Different seeds should differ"


def test_generate_nontrivial_variance() -> None:
    """Activations have std > 0.1 (not all zeros or constant)."""
    acts = generate_synthetic_activations(n=50, hidden_dim=32, seed=42)
    assert acts.std().item() > 0.1


# ---------------------------------------------------------------------------
# 3. Training pipeline
# ---------------------------------------------------------------------------


def test_train_sae_quick_returns_sae_and_metrics(
    small_train_val: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """train_sae_quick returns (Sae, dict) with required metric keys."""
    train_data, val_data = small_train_val
    sae, metrics = train_sae_quick(
        m=2,
        k=4,
        train_data=train_data,
        val_data=val_data,
        device=torch.device("cpu"),
        seed=42,
        num_steps=10,
        batch_size=16,
        lr=1e-3,
    )
    assert "fvu" in metrics
    assert "dead_feature_pct" in metrics
    assert "mean_sparsity" in metrics


def test_train_sae_quick_num_latents(
    small_train_val: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Trained SAE has num_latents = m × d_in."""
    train_data, val_data = small_train_val
    m, d_in = 2, 32
    sae, _ = train_sae_quick(
        m=m,
        k=4,
        train_data=train_data,
        val_data=val_data,
        device=torch.device("cpu"),
        seed=42,
        num_steps=5,
        batch_size=16,
        lr=1e-3,
    )
    assert sae.num_latents == m * d_in, (
        f"Expected num_latents={m * d_in}, got {sae.num_latents}"
    )


def test_train_sae_quick_loss_decreases(
    small_train_val: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """L2 reconstruction loss decreases over 5 gradient steps."""
    from sae_src.sae.config import SaeConfig
    from sae_src.sae.sae import Sae

    torch.manual_seed(42)
    cfg = SaeConfig(expansion_factor=2, k=4)
    sae = Sae(d_in=32, cfg=cfg, device="cpu")
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

    train_data, _ = small_train_val
    batch = train_data[:16].unsqueeze(1)  # (16, 1, 32)

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        out = sae.forward(batch, dead_mask=None)
        out.l2_loss.backward()
        optimizer.step()
        losses.append(out.l2_loss.item())

    assert losses[-1] < losses[0], (
        f"L2 loss should decrease over training; got {losses}"
    )


def test_train_sae_quick_k_clamped_to_num_latents() -> None:
    """k is clamped to num_latents when k > m × d_in (guard against bad configs)."""
    data = generate_synthetic_activations(n=40, hidden_dim=8, seed=0)
    train_data, val_data = data[:32], data[32:]
    # m=1, d_in=8 → num_latents=8; requesting k=100 > 8
    sae, metrics = train_sae_quick(
        m=1,
        k=100,
        train_data=train_data,
        val_data=val_data,
        device=torch.device("cpu"),
        seed=0,
        num_steps=5,
        batch_size=16,
        lr=1e-3,
    )
    assert sae.cfg.k <= sae.num_latents, "k must not exceed num_latents"


# ---------------------------------------------------------------------------
# 4. Metric computation
# ---------------------------------------------------------------------------


def test_fvu_in_valid_range(
    tiny_sae, small_train_val: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """FVU is non-negative (can exceed 1.0 for poorly trained SAEs)."""
    _, val_data = small_train_val
    fvu = compute_fvu(tiny_sae, val_data)
    assert fvu >= 0.0, f"FVU must be non-negative, got {fvu}"


def test_dead_features_in_range(
    tiny_sae, small_train_val: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Dead feature fraction is in [0, 1]."""
    _, val_data = small_train_val
    dead_pct = compute_dead_features(tiny_sae, val_data)
    assert 0.0 <= dead_pct <= 1.0, f"Dead % out of [0, 1]: {dead_pct}"


def test_mean_sparsity_nonnegative(
    tiny_sae, small_train_val: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Mean sparsity (pre-TopK L0) is non-negative."""
    _, val_data = small_train_val
    sparsity = compute_mean_sparsity(tiny_sae, val_data)
    assert sparsity >= 0.0, f"Mean sparsity must be non-negative, got {sparsity}"


def test_dead_features_all_data_zero_k() -> None:
    """With a very low k relative to num_latents, many features are dead."""
    from sae_src.sae.config import SaeConfig
    from sae_src.sae.sae import Sae

    torch.manual_seed(0)
    # m=8, k=1: only 1 feature selected per token → most of 128 are dead
    cfg = SaeConfig(expansion_factor=8, k=1)
    sae = Sae(d_in=16, cfg=cfg, device="cpu")

    val_data = generate_synthetic_activations(n=50, hidden_dim=16, seed=0)
    dead_pct = compute_dead_features(sae, val_data, batch_size=50)

    # With 50 tokens each selecting 1 feature, at most 50 of 128 features
    # are ever active → at least (128 - 50) / 128 ≈ 0.60 are dead.
    expected_min_dead = (128 - 50) / 128
    assert dead_pct >= expected_min_dead * 0.9, (
        f"Expected dead% >= {expected_min_dead:.2f}, got {dead_pct:.2f}"
    )


def test_perfect_reconstruction_gives_low_fvu() -> None:
    """After many training steps on a tiny repeated batch, FVU should decrease."""
    from sae_src.sae.config import SaeConfig
    from sae_src.sae.sae import Sae

    torch.manual_seed(42)
    d_in = 16
    cfg = SaeConfig(expansion_factor=4, k=8)
    sae = Sae(d_in=d_in, cfg=cfg, device="cpu")
    optimizer = torch.optim.Adam(sae.parameters(), lr=5e-3)

    # Overfit on a single batch of 8 vectors
    data = generate_synthetic_activations(n=8, hidden_dim=d_in, seed=42)
    batch = data.unsqueeze(1)  # (8, 1, d_in)

    initial_fvu = compute_fvu(sae, data)

    for _ in range(50):
        optimizer.zero_grad()
        out = sae.forward(batch, dead_mask=None)
        out.l2_loss.backward()
        if sae.W_dec is not None and sae.W_dec.grad is not None:
            sae.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()
        if _ % 10 == 9:
            sae.set_decoder_norm_to_unit_norm()

    final_fvu = compute_fvu(sae, data)
    assert final_fvu < initial_fvu, (
        f"FVU should decrease after training; "
        f"initial={initial_fvu:.4f}, final={final_fvu:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. Power law fitting
# ---------------------------------------------------------------------------


def test_power_law_fit_known_exponents() -> None:
    """Power law fit recovers approximate exponents on clean synthetic data."""
    rng = np.random.default_rng(42)
    # True model: FVU = 2.0 × m^(-0.5) × k^(-0.3)
    results = []
    for m in [2, 4, 8, 16]:
        for k in [8, 16, 32, 64]:
            fvu = 2.0 * m**(-0.5) * k**(-0.3) + rng.uniform(-1e-4, 1e-4)
            results.append(
                ScalingResult(
                    expansion_factor=m,
                    k=k,
                    data_size=100,
                    seed=42,
                    num_latents=m * 64,
                    fvu=max(1e-4, fvu),
                    dead_feature_pct=0.0,
                    mean_sparsity=float(k),
                )
            )
    fit = fit_power_law(results)
    assert fit["r2"] > 0.90, f"R² should be high for clean power law: {fit['r2']:.4f}"
    assert 0.2 < fit["a"] < 0.8, f"Exponent a ≈ 0.5 expected, got {fit['a']:.3f}"
    assert 0.1 < fit["b"] < 0.5, f"Exponent b ≈ 0.3 expected, got {fit['b']:.3f}"
    assert fit["C"] > 0.0, "Coefficient C must be positive"


def test_power_law_fit_too_few_points() -> None:
    """fit_power_law returns zero-valued dict when < 3 valid data points."""
    results = [
        ScalingResult(
            expansion_factor=4,
            k=64,
            data_size=100,
            seed=42,
            num_latents=256,
            fvu=0.3,
            dead_feature_pct=0.0,
            mean_sparsity=64.0,
        ),
        ScalingResult(
            expansion_factor=4,
            k=64,
            data_size=100,
            seed=123,
            num_latents=256,
            fvu=0.0,  # filtered out (log(0) undefined)
            dead_feature_pct=0.0,
            mean_sparsity=64.0,
        ),
    ]
    fit = fit_power_law(results)
    assert fit["r2"] == 0.0
    assert fit["a"] == 0.0
    assert fit["C"] == 0.0


def test_power_law_fit_empty() -> None:
    """fit_power_law handles empty results list without raising."""
    fit = fit_power_law([])
    assert fit["r2"] == 0.0


# ---------------------------------------------------------------------------
# 6. Pareto frontier
# ---------------------------------------------------------------------------


def test_pareto_frontier_nonempty(sample_results: list[ScalingResult]) -> None:
    """Pareto frontier is always non-empty."""
    pareto = find_pareto_frontier(sample_results)
    assert len(pareto) >= 1


def test_pareto_frontier_subset_of_all(
    sample_results: list[ScalingResult],
) -> None:
    """Pareto keys are a subset of all config keys."""
    pareto = find_pareto_frontier(sample_results)
    all_keys = {(r.expansion_factor, r.k) for r in sample_results}
    pareto_keys = {(r.expansion_factor, r.k) for r in pareto}
    assert pareto_keys.issubset(all_keys)


def test_pareto_frontier_single_config() -> None:
    """With one config, the Pareto frontier contains that config."""
    results = [
        ScalingResult(
            expansion_factor=4,
            k=64,
            data_size=100,
            seed=42,
            num_latents=256,
            fvu=0.3,
            dead_feature_pct=0.05,
            mean_sparsity=64.0,
        )
    ]
    pareto = find_pareto_frontier(results)
    assert len(pareto) == 1
    assert pareto[0].expansion_factor == 4


def test_pareto_frontier_all_nondominated() -> None:
    """When each config is strictly best on one metric, all should be Pareto-optimal."""
    results = [
        # A: best FVU, worst dead%
        ScalingResult(
            expansion_factor=2,
            k=8,
            data_size=100,
            seed=42,
            num_latents=128,
            fvu=0.05,
            dead_feature_pct=0.80,
            mean_sparsity=8.0,
        ),
        # B: worst FVU, best dead%
        ScalingResult(
            expansion_factor=4,
            k=64,
            data_size=100,
            seed=42,
            num_latents=256,
            fvu=0.60,
            dead_feature_pct=0.01,
            mean_sparsity=64.0,
        ),
    ]
    pareto = find_pareto_frontier(results)
    pareto_keys = {(r.expansion_factor, r.k) for r in pareto}
    assert (2, 8) in pareto_keys, "Config with best FVU must be Pareto-optimal"
    assert (4, 64) in pareto_keys, "Config with best dead% must be Pareto-optimal"


def test_pareto_frontier_dominated_excluded() -> None:
    """A strictly dominated config should not appear in the Pareto frontier."""
    results = [
        # A: good on both
        ScalingResult(
            expansion_factor=4,
            k=64,
            data_size=100,
            seed=42,
            num_latents=256,
            fvu=0.1,
            dead_feature_pct=0.05,
            mean_sparsity=64.0,
        ),
        # B: strictly dominated by A on both objectives
        ScalingResult(
            expansion_factor=2,
            k=8,
            data_size=100,
            seed=42,
            num_latents=128,
            fvu=0.9,
            dead_feature_pct=0.50,
            mean_sparsity=8.0,
        ),
    ]
    pareto = find_pareto_frontier(results)
    pareto_keys = {(r.expansion_factor, r.k) for r in pareto}
    assert (4, 64) in pareto_keys, "Dominating config must be Pareto-optimal"
    assert (2, 8) not in pareto_keys, "Dominated config must not be Pareto-optimal"


# ---------------------------------------------------------------------------
# 7. I/O helpers
# ---------------------------------------------------------------------------


def test_save_results_csv_schema(
    sample_results: list[ScalingResult], tmp_path: Path
) -> None:
    """CSV has correct number of rows and contains required column headers."""
    out_dir = tmp_path / "scaling"
    csv_path = save_results_csv(sample_results, out_dir)

    assert csv_path.exists()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == len(sample_results)
    assert reader.fieldnames is not None
    for col in ("fvu", "dead_feature_pct", "expansion_factor", "k", "seed"):
        assert col in reader.fieldnames, f"Column '{col}' missing from CSV"


def test_save_results_csv_values(
    sample_results: list[ScalingResult], tmp_path: Path
) -> None:
    """CSV values round-trip correctly for the first result."""
    out_dir = tmp_path / "scaling"
    save_results_csv(sample_results, out_dir)

    with open(out_dir / "all_results.csv", newline="") as f:
        reader = csv.DictReader(f)
        first = next(reader)

    assert int(first["expansion_factor"]) == sample_results[0].expansion_factor
    assert int(first["k"]) == sample_results[0].k


def test_save_summary_table_sections(
    sample_results: list[ScalingResult], tmp_path: Path
) -> None:
    """Summary table Markdown contains the key headings and paper config annotation."""
    out_dir = tmp_path / "scaling"
    pareto = find_pareto_frontier(sample_results)
    power_law = fit_power_law(sample_results)
    md_path = save_summary_table(sample_results, pareto, power_law, out_dir)

    assert md_path.exists()
    content = md_path.read_text()
    assert "Power Law Fit" in content
    assert "Pareto-Optimal" in content
    assert "m=4, k=64" in content
    assert "FVU" in content


# ---------------------------------------------------------------------------
# 8. Plot generation
# ---------------------------------------------------------------------------


def test_plot_fvu_vs_expansion_creates_file(
    sample_results: list[ScalingResult], tmp_path: Path
) -> None:
    """fvu_vs_expansion.png is written to the output directory."""
    out_dir = tmp_path / "scaling"
    plot_fvu_vs_expansion(sample_results, out_dir)
    assert (out_dir / "fvu_vs_expansion.png").exists()


def test_plot_alignment_vs_k_creates_file(
    sample_results: list[ScalingResult], tmp_path: Path
) -> None:
    """alignment_vs_k.png is created (placeholder when clap_delta == -1.0)."""
    out_dir = tmp_path / "scaling"
    plot_alignment_vs_k(sample_results, out_dir)
    assert (out_dir / "alignment_vs_k.png").exists()


def test_plot_pareto_frontier_creates_file(
    sample_results: list[ScalingResult], tmp_path: Path
) -> None:
    """pareto_frontier.png is written to the output directory."""
    out_dir = tmp_path / "scaling"
    plot_pareto_frontier(sample_results, out_dir)
    assert (out_dir / "pareto_frontier.png").exists()


def test_plots_handle_empty_results(tmp_path: Path) -> None:
    """Plot functions create files even when given an empty results list."""
    out_dir = tmp_path / "scaling_empty"
    empty: list[ScalingResult] = []
    plot_fvu_vs_expansion(empty, out_dir)
    plot_alignment_vs_k(empty, out_dir)
    plot_pareto_frontier(empty, out_dir)
    # All three files should exist (even if blank/minimal)
    assert (out_dir / "fvu_vs_expansion.png").exists()
    assert (out_dir / "alignment_vs_k.png").exists()
    assert (out_dir / "pareto_frontier.png").exists()


# ---------------------------------------------------------------------------
# 9. End-to-end smoke test
# ---------------------------------------------------------------------------


def test_run_single_config_smoke_test() -> None:
    """Smallest valid config trains and returns correct ScalingResult.

    This is the primary 'DONE WHEN' criterion from the Phase 3.2 roadmap:
    (m=2, k=8, n=100) trains and evaluates successfully.
    """
    cfg = ScalingConfig(
        dry_run_hidden_dim=32,
        dry_run_steps=10,
        batch_size=16,
        val_fraction=0.2,
    )
    result = run_single_config(
        m=2,
        k=8,
        n=100,
        seed=42,
        cfg=cfg,
        dry_run=True,
    )

    assert isinstance(result, ScalingResult)

    # Grid coordinates
    assert result.expansion_factor == 2
    assert result.k == 8
    assert result.data_size == 100
    assert result.seed == 42

    # num_latents = m × hidden_dim
    assert result.num_latents == 2 * 32

    # Metric validity
    assert result.fvu >= 0.0, f"FVU must be non-negative, got {result.fvu}"
    assert 0.0 <= result.dead_feature_pct <= 1.0
    assert result.mean_sparsity >= 0.0
    assert result.train_time_s >= 0.0

    # Sentinel values for uncomputed metrics
    assert result.clap_delta == -1.0
    assert result.lpaps == -1.0
    assert result.interpretability_score == -1.0


# ---------------------------------------------------------------------------
# 10. resolve_grid() — priority rules
# ---------------------------------------------------------------------------

def _ns(**kwargs) -> argparse.Namespace:
    """Build an argparse.Namespace with sensible defaults for resolve_grid."""
    defaults = dict(
        smoke_test=False,
        dry_run=False,
        full_grid=False,
        preset_real_small=False,
        m_values=None,
        k_values=None,
        data_sizes=None,
        seeds=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_resolve_grid_smoke_test_wins() -> None:
    """--smoke-test always returns the single fixed config, ignoring everything else."""
    cfg = ScalingConfig()
    # Even with explicit overrides, smoke-test wins.
    args = _ns(smoke_test=True, m_values=[8, 16], k_values=[128], preset_real_small=True)
    m, k, n, s = resolve_grid(args, cfg)
    assert m == [2]
    assert k == [8]
    assert n == [100]
    assert s == [42]


def test_resolve_grid_dry_run_default() -> None:
    """--dry-run without --full-grid gives the compact 2×2 default grid."""
    cfg = ScalingConfig()
    args = _ns(dry_run=True)
    m, k, n, s = resolve_grid(args, cfg)
    assert set(m) == {2, 4}
    assert set(k) == {8, 32}
    assert n == [100]
    assert s == [42]


def test_resolve_grid_full_grid_uses_cfg() -> None:
    """--full-grid returns the full cfg defaults, not the compact dry-run grid."""
    cfg = ScalingConfig(expansion_factors=[2, 4, 8])
    args = _ns(dry_run=True, full_grid=True)
    m, k, n, s = resolve_grid(args, cfg)
    assert m == [2, 4, 8]


def test_resolve_grid_real_mode_uses_cfg() -> None:
    """No --dry-run and no --full-grid still uses cfg (real mode)."""
    cfg = ScalingConfig(expansion_factors=[4, 8], k_values=[64])
    args = _ns()
    m, k, n, s = resolve_grid(args, cfg)
    assert m == [4, 8]
    assert k == [64]


def test_resolve_grid_preset_real_small_values() -> None:
    """--preset-real-small sets exactly the values in _PRESET_REAL_SMALL."""
    cfg = ScalingConfig()
    args = _ns(preset_real_small=True)
    m, k, n, s = resolve_grid(args, cfg)
    assert m == _PRESET_REAL_SMALL["expansion_factors"]
    assert k == _PRESET_REAL_SMALL["k_values"]
    assert n == _PRESET_REAL_SMALL["data_sizes"]
    assert s == _PRESET_REAL_SMALL["seeds"]


def test_resolve_grid_preset_overrides_dry_run() -> None:
    """--preset-real-small beats the compact --dry-run grid."""
    cfg = ScalingConfig()
    args = _ns(dry_run=True, preset_real_small=True)
    m, k, n, s = resolve_grid(args, cfg)
    # Preset values (not the dry-run 2×2 defaults)
    assert m == _PRESET_REAL_SMALL["expansion_factors"]
    assert k == _PRESET_REAL_SMALL["k_values"]


def test_resolve_grid_explicit_m_values() -> None:
    """--m-values overrides the mode default."""
    cfg = ScalingConfig()
    args = _ns(dry_run=True, m_values=[4, 16])
    m, k, n, s = resolve_grid(args, cfg)
    assert m == [4, 16]
    # k still comes from dry-run default
    assert set(k) == {8, 32}


def test_resolve_grid_explicit_k_values() -> None:
    """--k-values overrides the mode default."""
    cfg = ScalingConfig()
    args = _ns(dry_run=True, k_values=[64, 128])
    m, k, n, s = resolve_grid(args, cfg)
    assert k == [64, 128]


def test_resolve_grid_explicit_data_sizes() -> None:
    """--data-sizes overrides the mode default."""
    cfg = ScalingConfig()
    args = _ns(dry_run=True, data_sizes=[200, 1000])
    m, k, n, s = resolve_grid(args, cfg)
    assert n == [200, 1000]


def test_resolve_grid_explicit_seeds() -> None:
    """--seeds overrides the mode default."""
    cfg = ScalingConfig()
    args = _ns(dry_run=True, seeds=[7, 99])
    m, k, n, s = resolve_grid(args, cfg)
    assert s == [7, 99]


def test_resolve_grid_explicit_beats_preset() -> None:
    """Explicit --m-values overrides --preset-real-small for that dimension."""
    cfg = ScalingConfig()
    args = _ns(preset_real_small=True, m_values=[16, 32])
    m, k, n, s = resolve_grid(args, cfg)
    assert m == [16, 32]
    # Other dimensions still come from preset
    assert k == _PRESET_REAL_SMALL["k_values"]
    assert n == _PRESET_REAL_SMALL["data_sizes"]


def test_resolve_grid_full_grid_plus_explicit_override() -> None:
    """Explicit --k-values overrides --full-grid for that dimension."""
    cfg = ScalingConfig()
    args = _ns(full_grid=True, k_values=[32])
    m, k, n, s = resolve_grid(args, cfg)
    # m comes from full cfg, k is overridden
    assert m == cfg.expansion_factors
    assert k == [32]


def test_resolve_grid_all_explicit() -> None:
    """All four explicit flags together produce exactly those values."""
    cfg = ScalingConfig()  # defaults irrelevant
    args = _ns(m_values=[8], k_values=[128], data_sizes=[999], seeds=[7])
    m, k, n, s = resolve_grid(args, cfg)
    assert m == [8]
    assert k == [128]
    assert n == [999]
    assert s == [7]


# ---------------------------------------------------------------------------
# 11. _PRESET_REAL_SMALL constant
# ---------------------------------------------------------------------------


def test_preset_real_small_grid_size() -> None:
    """_PRESET_REAL_SMALL defines a 3×3×2×2 = 36-config grid."""
    total = (
        len(_PRESET_REAL_SMALL["expansion_factors"])
        * len(_PRESET_REAL_SMALL["k_values"])
        * len(_PRESET_REAL_SMALL["data_sizes"])
        * len(_PRESET_REAL_SMALL["seeds"])
    )
    assert total == 36, f"Expected 36 configs, got {total}"


def test_preset_real_small_m_values() -> None:
    """_PRESET_REAL_SMALL expansion_factors = [2, 4, 8]."""
    assert _PRESET_REAL_SMALL["expansion_factors"] == [2, 4, 8]


def test_preset_real_small_k_values() -> None:
    """_PRESET_REAL_SMALL k_values = [32, 64, 128]."""
    assert _PRESET_REAL_SMALL["k_values"] == [32, 64, 128]


def test_preset_real_small_includes_paper_config() -> None:
    """_PRESET_REAL_SMALL contains the paper's m=4, k=64 configuration."""
    assert 4 in _PRESET_REAL_SMALL["expansion_factors"]
    assert 64 in _PRESET_REAL_SMALL["k_values"]


# ---------------------------------------------------------------------------
# 12. CLI parsing of new flags via _build_parser
# ---------------------------------------------------------------------------


def test_cli_parse_m_values() -> None:
    """--m-values parses into a list of ints on args.m_values."""
    parser = _build_parser()
    args = parser.parse_args(["--dry-run", "--m-values", "2", "8", "16"])
    assert args.m_values == [2, 8, 16]


def test_cli_parse_k_values() -> None:
    """--k-values parses into a list of ints on args.k_values."""
    parser = _build_parser()
    args = parser.parse_args(["--dry-run", "--k-values", "32", "64"])
    assert args.k_values == [32, 64]


def test_cli_parse_data_sizes() -> None:
    """--data-sizes parses into a list of ints on args.data_sizes."""
    parser = _build_parser()
    args = parser.parse_args(["--dry-run", "--data-sizes", "500", "2000"])
    assert args.data_sizes == [500, 2000]


def test_cli_parse_seeds() -> None:
    """--seeds parses into a list of ints on args.seeds."""
    parser = _build_parser()
    args = parser.parse_args(["--dry-run", "--seeds", "42", "123", "456"])
    assert args.seeds == [42, 123, 456]


def test_cli_parse_preset_real_small() -> None:
    """--preset-real-small sets args.preset_real_small = True."""
    parser = _build_parser()
    args = parser.parse_args(["--preset-real-small"])
    assert args.preset_real_small is True


def test_cli_defaults_are_none() -> None:
    """All new override flags default to None / False when not supplied."""
    parser = _build_parser()
    args = parser.parse_args(["--dry-run"])
    assert args.m_values is None
    assert args.k_values is None
    assert args.data_sizes is None
    assert args.seeds is None
    assert args.preset_real_small is False


def test_cli_resolve_grid_integration() -> None:
    """Full parse → resolve_grid pipeline produces expected grid values."""
    parser = _build_parser()
    cfg = ScalingConfig()
    args = parser.parse_args([
        "--dry-run",
        "--m-values", "4", "8",
        "--k-values", "64",
        "--data-sizes", "200",
        "--seeds", "42",
    ])
    m, k, n, s = resolve_grid(args, cfg)
    assert m == [4, 8]
    assert k == [64]
    assert n == [200]
    assert s == [42]


def test_cli_smoke_test_ignores_explicit_flags() -> None:
    """--smoke-test overrides --m-values and --preset-real-small."""
    parser = _build_parser()
    cfg = ScalingConfig()
    args = parser.parse_args([
        "--smoke-test",
        "--m-values", "16", "32",
        "--preset-real-small",
    ])
    m, k, n, s = resolve_grid(args, cfg)
    assert m == [2]
    assert k == [8]
    assert n == [100]
    assert s == [42]
