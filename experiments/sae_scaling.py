#!/usr/bin/env python3
"""SAE Scaling Laws Experiment — Phase 3.2

Sweeps over SAE expansion factors (m), sparsity values (k), and training
data sizes (n) to characterise scaling behaviour for audio diffusion SAEs.

Scientific question:
    How does SAE quality (FVU, dead features, interpretability, steering
    effectiveness) scale with m, k, and n for ACE-Step activations?

Key outputs
-----------
  experiments/results/scaling/all_results.csv
  experiments/results/scaling/fvu_vs_expansion.png
  experiments/results/scaling/alignment_vs_k.png
  experiments/results/scaling/pareto_frontier.png
  experiments/results/scaling/summary_table.md

Usage
-----
  # Smoke test — single smallest config, synthetic data, CPU:
  python experiments/sae_scaling.py --smoke-test --out-dir results/scaling

  # Dry run — default 2×2 grid, synthetic activations:
  python experiments/sae_scaling.py --dry-run --out-dir results/scaling

  # Full dry-run grid (5×5×4×3 = 300 configs, synthetic activations):
  python experiments/sae_scaling.py --dry-run --full-grid --out-dir results/scaling

  # Real run — M4 Air safe preset (3×3×2×2 = 36 configs, ~12 min on MPS):
  python experiments/sae_scaling.py \\
      --preset-real-small \\
      --activation-cache $TADA_WORKDIR/cache/layer7 \\
      --out-dir results/scaling

  # Real run — larger preset (4×4×3×2 = 96 configs, ~40 min on MPS):
  python experiments/sae_scaling.py \\
      --m-values 2 4 8 16 --k-values 16 32 64 128 --data-sizes 500 2000 5000 --seeds 42 123 \\
      --activation-cache $TADA_WORKDIR/cache/layer7 \\
      --out-dir results/scaling

  # Explicit override (beats all presets and mode flags):
  python experiments/sae_scaling.py \\
      --dry-run --m-values 4 8 --k-values 64 --data-sizes 200 \\
      --out-dir results/scaling
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup: make sae_src importable from any working directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SAE_ROOT = _REPO_ROOT / "sae"

for _p in [str(_REPO_ROOT), str(_SAE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sae_src.sae.config import SaeConfig  # noqa: E402
from sae_src.sae.sae import Sae  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named grid presets
# ---------------------------------------------------------------------------

#: RAM-safe preset for 16 GB M1/M2/M4 MacBook Air.
#: 3×3×2×2 = 36 total configs.
#: Estimated time: ~12 min on M4 (MPS) or ~2 h on CPU at 200 steps/config.
_PRESET_REAL_SMALL: dict[str, list[int]] = {
    "expansion_factors": [2, 4, 8],
    "k_values": [32, 64, 128],
    "data_sizes": [500, 2000],
    "seeds": [42, 123],
}

#: Largest preset that is still safe on 16 GB RAM.
#: 4×4×3×2 = 96 total configs.
#: Estimated time: ~40 min on M4 (MPS).
#: Use via explicit flags: --m-values 2 4 8 16 --k-values 16 32 64 128
#:                         --data-sizes 500 2000 5000 --seeds 42 123
_PRESET_REAL_MEDIUM: dict[str, list[int]] = {
    "expansion_factors": [2, 4, 8, 16],
    "k_values": [16, 32, 64, 128],
    "data_sizes": [500, 2000, 5000],
    "seeds": [42, 123],
}


def _get_device() -> torch.device:
    """Return the best available device: CUDA → MPS (Apple Silicon) → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ScalingConfig:
    """Grid parameters and hyper-parameters for the scaling law sweep.

    Attributes
    ----------
    expansion_factors:
        Values of m (SAE width = m × hidden_dim) to sweep.
    k_values:
        Values of k (TopK sparsity) to sweep.
    data_sizes:
        Number of activation vectors in the training set to sweep.
    seeds:
        Random seeds; results are averaged across seeds for error bars.
    hidden_dim:
        Activation dimensionality (3072 for ACE-Step layer 7).
    num_train_steps:
        Gradient steps per (m, k, n, seed) configuration in real mode.
    batch_size:
        Mini-batch size for training and evaluation.
    lr:
        Adam learning rate for the quick training loop.
    val_fraction:
        Fraction of data held out for validation metrics.
    dry_run_hidden_dim:
        Smaller hidden_dim used in dry-run mode for fast CPU times.
    dry_run_steps:
        Gradient steps in dry-run mode.
    clap_threshold:
        CLAP cosine similarity above which a feature is "interpretable".
    top_features_for_interp:
        How many most-active features to score for interpretability.
    eval_concept:
        Concept used for ΔAlignment CLAP evaluation (requires model).
    eval_alpha:
        Alpha value used for ΔAlignment CLAP evaluation.
    eval_n_samples:
        Number of generated audio clips for CLAP evaluation.
    """

    # Experiment grid (paper values: m=4, k=64)
    expansion_factors: list[int] = field(
        default_factory=lambda: [2, 4, 8, 16, 32]
    )
    k_values: list[int] = field(
        default_factory=lambda: [8, 16, 32, 64, 128]
    )
    data_sizes: list[int] = field(
        default_factory=lambda: [100, 500, 1000, 5000]
    )
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])

    # Model
    hidden_dim: int = 3072  # ACE-Step layer-7 hidden dimension

    # Training
    num_train_steps: int = 500
    batch_size: int = 256
    lr: float = 1e-3
    val_fraction: float = 0.1

    # Dry-run overrides (small grid, tiny synthetic activations)
    dry_run_hidden_dim: int = 64
    dry_run_steps: int = 50

    # Interpretability scoring
    clap_threshold: float = 0.4
    top_features_for_interp: int = 50

    # Steering evaluation (skipped in dry-run; requires model weights)
    eval_concept: str = "tempo"
    eval_alpha: float = 50.0
    eval_n_samples: int = 32


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ScalingResult:
    """Metrics for one (m, k, data_size, seed) configuration.

    Attributes
    ----------
    expansion_factor, k, data_size, seed:
        Grid coordinates.
    num_latents:
        expansion_factor × hidden_dim (actual SAE width).
    fvu:
        Fraction of Variance Unexplained on the validation set [0, ∞).
    dead_feature_pct:
        Fraction of latents that never activate on the validation set [0, 1].
    mean_sparsity:
        Average L0 count over pre-activations (pre-TopK nonzero features).
    clap_delta:
        ΔAlignment CLAP at eval_alpha (-1.0 when not computed).
    lpaps:
        Audio preservation metric at eval_alpha (-1.0 when not computed).
    interpretability_score:
        Fraction of active latents labelled interpretable by CLAP
        (-1.0 when not computed).
    train_time_s:
        Wall-clock seconds for training.
    """

    expansion_factor: int
    k: int
    data_size: int
    seed: int
    num_latents: int
    fvu: float
    dead_feature_pct: float
    mean_sparsity: float
    clap_delta: float = -1.0           # sentinel: not computed
    lpaps: float = -1.0                # sentinel: not computed
    interpretability_score: float = -1.0  # sentinel: not computed
    train_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def generate_synthetic_activations(
    n: int,
    hidden_dim: int,
    seed: int,
    n_concepts: int = 16,
    concept_scale: float = 2.0,
    noise_scale: float = 1.0,
) -> torch.Tensor:
    """Generate synthetic activation vectors for dry-run training.

    Produces activations with a low-rank structured component (simulating
    concept directions in a trained transformer) overlaid with isotropic
    Gaussian noise, making the reconstruction task non-trivial and sensitive
    to SAE width / sparsity.

    Args:
        n: Number of activation vectors to generate.
        hidden_dim: Dimensionality of each vector.
        seed: Random seed for exact reproducibility.
        n_concepts: Number of latent "concept directions" to embed.
        concept_scale: Magnitude of the structured component.
        noise_scale: Magnitude of isotropic noise.

    Returns:
        Float32 tensor of shape (n, hidden_dim).
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    n_concepts = min(n_concepts, hidden_dim // 4)

    # Orthonormal concept directions
    directions = torch.randn(n_concepts, hidden_dim, generator=rng)
    directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-8)

    # Sparse concept coefficients (simulate most tokens having few active concepts)
    coefficients = torch.randn(n, n_concepts, generator=rng) * concept_scale

    # Isotropic noise
    noise = torch.randn(n, hidden_dim, generator=rng) * noise_scale

    return (coefficients @ directions + noise).to(torch.float32)  # (n, hidden_dim)


# ---------------------------------------------------------------------------
# SAE training (quick loop — no W&B, no distributed training)
# ---------------------------------------------------------------------------


def train_sae_quick(
    m: int,
    k: int,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    device: torch.device,
    seed: int,
    num_steps: int,
    batch_size: int,
    lr: float,
) -> tuple[Sae, dict[str, float]]:
    """Train a small SAE and return it with final validation metrics.

    Uses a plain Adam optimiser loop without the full SaeTrainer
    infrastructure (no W&B, no HuggingFace datasets, no distributed training).

    Args:
        m: Expansion factor (num_latents = m × d_in).
        k: TopK sparsity (clamped to num_latents if too large).
        train_data: Training activations, shape (n_train, d_in).
        val_data: Validation activations, shape (n_val, d_in).
        device: Compute device (e.g. torch.device("cpu")).
        seed: Random seed for parameter initialisation.
        num_steps: Number of Adam gradient steps.
        batch_size: Mini-batch size (clamped to n_train).
        lr: Adam learning rate.

    Returns:
        Tuple of (trained_sae, metrics_dict).
        metrics_dict contains 'fvu', 'dead_feature_pct', 'mean_sparsity'.
    """
    d_in = train_data.shape[1]
    num_latents = m * d_in
    k_actual = min(k, num_latents)  # guard against k > num_latents

    torch.manual_seed(seed)
    cfg = SaeConfig(expansion_factor=m, k=k_actual)
    sae = Sae(d_in=d_in, cfg=cfg, device=str(device))

    train_data = train_data.to(device)
    val_data = val_data.to(device)

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    effective_batch = min(batch_size, train_data.shape[0])
    rng = np.random.default_rng(seed)

    for step in range(num_steps):
        idx = rng.integers(0, train_data.shape[0], size=effective_batch)
        batch = train_data[idx].unsqueeze(1)  # (B, 1, d_in)

        optimizer.zero_grad()
        out = sae.forward(batch, dead_mask=None)
        loss = out.l2_loss
        loss.backward()

        # Remove gradient component parallel to decoder directions
        # to keep decoder on the unit hypersphere.
        if sae.W_dec is not None and sae.W_dec.grad is not None:
            sae.remove_gradient_parallel_to_decoder_directions()

        optimizer.step()

        # Re-normalise decoder columns every 10 steps
        if (step + 1) % 10 == 0:
            sae.set_decoder_norm_to_unit_norm()

    # Final normalisation
    sae.set_decoder_norm_to_unit_norm()

    # Compute validation metrics
    fvu = compute_fvu(sae, val_data)
    dead_pct = compute_dead_features(sae, val_data, batch_size=batch_size)
    mean_sparsity = compute_mean_sparsity(sae, val_data, batch_size=batch_size)

    return sae, {
        "fvu": fvu,
        "dead_feature_pct": dead_pct,
        "mean_sparsity": mean_sparsity,
    }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_fvu(
    sae: Sae,
    val_data: torch.Tensor,
    batch_size: int = 256,
) -> float:
    """Compute average Fraction of Variance Unexplained over the validation set.

    FVU = ||x - x̂||² / ||x - mean(x)||²  (per batch, averaged over batches).

    Args:
        sae: Trained SAE model.
        val_data: Validation activations, shape (n_val, d_in).
        batch_size: Evaluation batch size.

    Returns:
        Mean FVU (batch-weighted). 0.0 = perfect reconstruction; 1.0 = trivial
        (predicting the mean). Values > 1.0 are possible for very bad models.
    """
    n = val_data.shape[0]
    total_fvu = 0.0
    total_samples = 0

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = val_data[i : i + batch_size].unsqueeze(1)  # (B, 1, d_in)
            out = sae.forward(batch, dead_mask=None)
            total_fvu += float(out.fvu.item()) * batch.shape[0]
            total_samples += batch.shape[0]

    return total_fvu / max(total_samples, 1)


def compute_dead_features(
    sae: Sae,
    val_data: torch.Tensor,
    batch_size: int = 256,
) -> float:
    """Compute the fraction of latent features that never activate on val_data.

    A feature is "dead" if it does not appear in the top-k selection for any
    token in the validation set.  High dead% indicates wasted capacity.

    Args:
        sae: Trained SAE model.
        val_data: Validation activations, shape (n_val, d_in).
        batch_size: Evaluation batch size.

    Returns:
        Dead feature fraction in [0, 1].
    """
    n = val_data.shape[0]
    ever_active = torch.zeros(sae.num_latents, dtype=torch.bool, device=sae.device)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = val_data[i : i + batch_size].unsqueeze(1)  # (B, 1, d_in)
            enc = sae.encode(batch)  # top_indices: (B, k)
            ever_active[enc.top_indices.flatten()] = True

    return float((~ever_active).float().mean().item())


def compute_mean_sparsity(
    sae: Sae,
    val_data: torch.Tensor,
    batch_size: int = 256,
) -> float:
    """Compute mean L0 count — average pre-activation nonzero features per token.

    L0 is computed from pre-activations (before TopK selection), so it reflects
    how many features are above zero after ReLU.  This value is ≥ k.

    Args:
        sae: Trained SAE model.
        val_data: Validation activations, shape (n_val, d_in).
        batch_size: Evaluation batch size.

    Returns:
        Mean L0 count (≥ 0).
    """
    n = val_data.shape[0]
    total_l0 = 0.0
    total_samples = 0

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = val_data[i : i + batch_size].unsqueeze(1)  # (B, 1, d_in)
            out = sae.forward(batch, dead_mask=None)
            total_l0 += float(out.l0_loss.item()) * batch.shape[0]
            total_samples += batch.shape[0]

    return total_l0 / max(total_samples, 1)


def compute_interpretability_score(
    sae: Sae,
    val_data: torch.Tensor,
    concept_labels: list[str],
    clap_model: object | None,
    threshold: float = 0.4,
    top_n_features: int = 50,
) -> float:
    """Estimate the fraction of SAE latents that are semantically interpretable.

    A feature is "interpretable" if the top-activating clips achieve a CLAP
    cosine similarity > threshold with at least one concept label.

    In dry-run mode (clap_model is None), returns -1.0 as a sentinel.

    Args:
        sae: Trained SAE model.
        val_data: Validation activations, shape (n_val, d_in).
        concept_labels: Candidate text labels for CLAP similarity scoring.
        clap_model: CLAP model instance; None → dry-run (returns -1.0).
        threshold: CLAP similarity threshold for "interpretable".
        top_n_features: Number of most-active features to score.

    Returns:
        Interpretability fraction in [0, 1], or -1.0 in dry-run.
    """
    if clap_model is None:
        # Dry-run: cannot compute without CLAP model and audio generation.
        return -1.0

    # Collect per-feature total activation magnitude to rank by activity.
    n = val_data.shape[0]
    feature_act_sum = torch.zeros(sae.num_latents, device=sae.device)

    with torch.no_grad():
        for i in range(0, n, 256):
            batch = val_data[i : i + 256].unsqueeze(1)
            enc = sae.encode(batch)
            feature_act_sum.scatter_add_(
                0,
                enc.top_indices.flatten(),
                enc.top_acts.flatten().float(),
            )

    n_active = int((feature_act_sum > 0).sum().item())
    if n_active == 0:
        return 0.0

    top_feature_indices = feature_act_sum.topk(
        min(top_n_features, n_active)
    ).indices.tolist()

    # Full scoring requires audio generation + CLAP — stub for future use.
    raise NotImplementedError(
        f"Full interpretability scoring requires CLAP model and audio generation. "
        f"Top {len(top_feature_indices)} features identified by activation frequency; "
        "pass clap_model=None for dry-run (returns -1.0)."
    )


# ---------------------------------------------------------------------------
# Single configuration runner
# ---------------------------------------------------------------------------


def run_single_config(
    m: int,
    k: int,
    n: int,
    seed: int,
    cfg: ScalingConfig,
    dry_run: bool,
    activation_cache: Optional[Path] = None,
) -> ScalingResult:
    """Train and evaluate one (m, k, n, seed) configuration.

    Args:
        m: Expansion factor.
        k: TopK sparsity.
        n: Total data size (train + val combined).
        seed: Random seed.
        cfg: Scaling configuration (batch_size, lr, val_fraction, etc.).
        dry_run: If True, use synthetic activations and cfg.dry_run_hidden_dim.
        activation_cache: Path to cached activation Arrow dataset (real mode).

    Returns:
        ScalingResult with all computed metrics.
    """
    hidden_dim = cfg.dry_run_hidden_dim if dry_run else cfg.hidden_dim
    num_steps = cfg.dry_run_steps if dry_run else cfg.num_train_steps
    device = _get_device()

    # --- Data ---
    if dry_run or activation_cache is None:
        all_data = generate_synthetic_activations(
            n=n, hidden_dim=hidden_dim, seed=seed
        )
    else:
        all_data = _load_cached_activations(activation_cache, n=n, seed=seed)
        hidden_dim = all_data.shape[1]  # use real hidden_dim from cache

    n_val = max(1, int(n * cfg.val_fraction))
    n_train = n - n_val

    # Deterministic train/val split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    train_data = all_data[perm[:n_train]]
    val_data = all_data[perm[n_train:]]

    num_latents = m * hidden_dim

    # --- Train ---
    t0 = time.perf_counter()
    _, metrics = train_sae_quick(
        m=m,
        k=k,
        train_data=train_data,
        val_data=val_data,
        device=device,
        seed=seed,
        num_steps=num_steps,
        batch_size=min(cfg.batch_size, n_train),
        lr=cfg.lr,
    )
    elapsed = time.perf_counter() - t0

    logger.debug(
        "m=%d k=%d n=%d seed=%d | FVU=%.4f dead=%.1f%% sparsity=%.1f | %.2fs",
        m,
        k,
        n,
        seed,
        metrics["fvu"],
        metrics["dead_feature_pct"] * 100,
        metrics["mean_sparsity"],
        elapsed,
    )

    return ScalingResult(
        expansion_factor=m,
        k=k,
        data_size=n,
        seed=seed,
        num_latents=num_latents,
        fvu=metrics["fvu"],
        dead_feature_pct=metrics["dead_feature_pct"],
        mean_sparsity=metrics["mean_sparsity"],
        clap_delta=-1.0,           # TODO: steering eval requires model weights
        lpaps=-1.0,                # TODO: steering eval requires model weights
        interpretability_score=-1.0,  # TODO: requires CLAP + audio generation
        train_time_s=elapsed,
    )


def _load_cached_activations(
    cache_path: Path,
    n: int,
    seed: int,
) -> torch.Tensor:
    """Load activation vectors from a HuggingFace Arrow dataset cache.

    Args:
        cache_path: Directory containing the cached Arrow dataset shards.
        n: Maximum number of vectors to load.
        seed: Random seed for shuffling before truncation.

    Returns:
        Float32 tensor of shape (min(n_loaded, n), hidden_dim).

    Raises:
        FileNotFoundError: If cache_path does not exist.
        ImportError: If the ``datasets`` library is unavailable.
    """
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Activation cache not found: {cache_path}. "
            "Run sae/sae_src/sae/cache_activations_runner_ace.py first."
        )

    try:
        from datasets import load_from_disk  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The `datasets` library is required for real-mode training. "
            "Install it with: pip install datasets"
        ) from exc

    dataset = load_from_disk(str(cache_path))
    dataset = dataset.shuffle(seed=seed)
    n_load = min(n, len(dataset))
    acts = torch.tensor(
        np.array(dataset[:n_load]["activations"]), dtype=torch.float32
    )
    # Flatten (n, seq, d_in) → (n*seq, d_in) if needed
    if acts.ndim == 3:
        acts = acts.reshape(-1, acts.shape[-1])
    return acts[:n]


# ---------------------------------------------------------------------------
# Power law fitting
# ---------------------------------------------------------------------------


def fit_power_law(results: list[ScalingResult]) -> dict[str, float]:
    """Fit FVU ~ C × m^(−a) × k^(−b) via log-log linear regression.

    Uses only results where FVU > 0 to avoid log(0).

    Args:
        results: All ScalingResult objects to fit against.

    Returns:
        Dict with keys 'C', 'a', 'b', 'r2' (R² in log space).
        Returns zero-valued dict if fewer than 3 valid data points.
    """
    valid = [
        (r.expansion_factor, r.k, r.fvu) for r in results if r.fvu > 0
    ]
    if len(valid) < 3:
        logger.warning(
            "Too few valid FVU values for power law fit (need ≥ 3, got %d).",
            len(valid),
        )
        return {"C": 0.0, "a": 0.0, "b": 0.0, "r2": 0.0}

    ms = np.array([v[0] for v in valid], dtype=float)
    ks = np.array([v[1] for v in valid], dtype=float)
    fvus = np.array([v[2] for v in valid], dtype=float)

    # log(FVU) = log(C) - a*log(m) - b*log(k)
    log_fvu = np.log(fvus)
    X = np.column_stack([np.ones_like(ms), np.log(ms), np.log(ks)])  # (N, 3)
    coeffs, _, _, _ = np.linalg.lstsq(X, log_fvu, rcond=None)

    log_C, neg_a, neg_b = coeffs
    pred = X @ coeffs
    ss_res = float(np.sum((log_fvu - pred) ** 2))
    ss_tot = float(np.sum((log_fvu - log_fvu.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "C": float(np.exp(log_C)),
        "a": float(-neg_a),
        "b": float(-neg_b),
        "r2": float(r2),
    }


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------


def find_pareto_frontier(
    results: list[ScalingResult],
    objectives: tuple[str, ...] = ("fvu", "dead_feature_pct"),
) -> list[ScalingResult]:
    """Return the Pareto-optimal subset of results (minimising all objectives).

    Config A dominates Config B if A is at least as good on every objective
    and strictly better on at least one.  Results are first aggregated over
    seeds (mean), then compared.

    Args:
        results: All ScalingResult objects.
        objectives: Attribute names to minimise; skip those with value −1.0.

    Returns:
        Non-dominated ScalingResult objects (one per (m, k, n) key).
    """
    # Average over seeds for each (m, k, n) combination
    grouped: dict[tuple, list[ScalingResult]] = defaultdict(list)
    for r in results:
        grouped[(r.expansion_factor, r.k, r.data_size)].append(r)

    representatives: list[tuple[tuple, dict[str, float]]] = []
    for key, group in grouped.items():
        means = {
            obj: float(np.mean([getattr(g, obj) for g in group]))
            for obj in objectives
        }
        representatives.append((key, means))

    # Keep only objectives with valid (non-sentinel) values
    active_objectives = [
        obj
        for obj in objectives
        if any(v[obj] >= 0 for _, v in representatives)
    ]
    if not active_objectives:
        active_objectives = list(objectives)

    # Identify non-dominated configurations
    pareto_keys: set[tuple] = set()
    for i, (key_i, vals_i) in enumerate(representatives):
        dominated = False
        for j, (_, vals_j) in enumerate(representatives):
            if i == j:
                continue
            if all(
                vals_j[obj] <= vals_i[obj] for obj in active_objectives
            ) and any(
                vals_j[obj] < vals_i[obj] for obj in active_objectives
            ):
                dominated = True
                break
        if not dominated:
            pareto_keys.add(key_i)

    # Return one representative result per Pareto key (lowest seed)
    pareto_results = []
    for key in pareto_keys:
        group = sorted(grouped[key], key=lambda r: r.seed)
        pareto_results.append(group[0])

    return sorted(pareto_results, key=lambda r: (r.fvu, r.dead_feature_pct))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_fvu_vs_expansion(
    results: list[ScalingResult],
    output_dir: Path,
) -> None:
    """Log-log line plot: mean FVU vs. expansion factor m, one line per k.

    Reveals the power law relationship FVU ~ m^(−a).

    Args:
        results: All ScalingResult objects.
        output_dir: Directory to write fvu_vs_expansion.png.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate: by_k[k][m] = [fvu values]
    by_k: dict[int, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in results:
        if r.fvu >= 0:
            by_k[r.k][r.expansion_factor].append(r.fvu)

    fig, ax = plt.subplots(figsize=(8, 5))
    all_k = sorted(by_k.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(all_k), 1)))

    for color, k_val in zip(colors, all_k):
        ms = sorted(by_k[k_val].keys())
        fvus = [float(np.mean(by_k[k_val][m])) for m in ms]
        ax.plot(ms, fvus, marker="o", label=f"k={k_val}", color=color)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Expansion factor m", fontsize=12)
    ax.set_ylabel("FVU (log scale)", fontsize=12)
    ax.set_title("SAE Scaling: FVU vs. Expansion Factor", fontsize=14)
    ax.legend(title="Sparsity k", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    # Annotate paper config (m=4, k=64) if data exists
    paper_m, paper_k = 4, 64
    if paper_k in by_k and paper_m in by_k[paper_k] and by_k[paper_k][paper_m]:
        paper_fvu = float(np.mean(by_k[paper_k][paper_m]))
        ax.annotate(
            "Paper\n(m=4,k=64)",
            xy=(paper_m, paper_fvu),
            xytext=(paper_m * 2.5, paper_fvu * 2.0),
            arrowprops=dict(arrowstyle="->", color="red"),
            color="red",
            fontsize=8,
        )

    fig.tight_layout()
    out_path = output_dir / "fvu_vs_expansion.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_alignment_vs_k(
    results: list[ScalingResult],
    output_dir: Path,
) -> None:
    """Line plot: mean ΔAlignment CLAP vs. k, one line per expansion factor.

    If all clap_delta values are −1.0 (dry-run), saves a placeholder figure.

    Args:
        results: All ScalingResult objects.
        output_dir: Directory to write alignment_vs_k.png.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    real_results = [r for r in results if r.clap_delta >= 0]

    if not real_results:
        ax.text(
            0.5,
            0.5,
            "ΔAlignment CLAP not available in dry-run mode.\n"
            "Run with --activation-cache and model weights to compute.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color="gray",
        )
        ax.set_title(
            "SAE Scaling: ΔAlignment CLAP vs. k  [placeholder — requires model]",
            fontsize=12,
        )
    else:
        by_m: dict[int, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for r in real_results:
            by_m[r.expansion_factor][r.k].append(r.clap_delta)

        all_m = sorted(by_m.keys())
        colors = plt.cm.plasma(np.linspace(0, 1, max(len(all_m), 1)))

        for color, m_val in zip(colors, all_m):
            ks = sorted(by_m[m_val].keys())
            deltas = [float(np.mean(by_m[m_val][k])) for k in ks]
            ax.plot(ks, deltas, marker="s", label=f"m={m_val}", color=color)

        ax.set_xlabel("Sparsity k", fontsize=12)
        ax.set_ylabel("ΔAlignment CLAP", fontsize=12)
        ax.set_title("SAE Scaling: ΔAlignment CLAP vs. k", fontsize=14)
        ax.legend(title="Expansion m", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "alignment_vs_k.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_pareto_frontier(
    results: list[ScalingResult],
    output_dir: Path,
) -> None:
    """Scatter plot: FVU vs. dead feature %, Pareto frontier highlighted.

    Each point is the mean over seeds for one (m, k) configuration.
    The paper config (m=4, k=64) is annotated with a star marker.

    Args:
        results: All ScalingResult objects.
        output_dir: Directory to write pareto_frontier.png.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pareto = find_pareto_frontier(results)
    pareto_keys = {(r.expansion_factor, r.k) for r in pareto}

    # Aggregate per (m, k) over all seeds and data_sizes
    agg: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for r in results:
        agg[(r.expansion_factor, r.k)].append(
            (r.fvu, r.dead_feature_pct)
        )

    fig, ax = plt.subplots(figsize=(8, 6))

    for (m, k), vals in agg.items():
        mean_fvu = float(np.mean([v[0] for v in vals]))
        mean_dead = float(np.mean([v[1] for v in vals]))
        is_pareto = (m, k) in pareto_keys
        is_paper = m == 4 and k == 64

        color = "red" if is_paper else ("forestgreen" if is_pareto else "steelblue")
        marker = "*" if is_paper else ("D" if is_pareto else "o")
        size = 250 if is_paper else (100 if is_pareto else 40)
        zorder = 5 if is_paper else (4 if is_pareto else 3)

        ax.scatter(
            mean_fvu,
            mean_dead * 100,
            c=color,
            marker=marker,
            s=size,
            zorder=zorder,
            edgecolors="black" if is_paper else "none",
            linewidths=0.8,
        )
        ax.annotate(
            f"m={m},k={k}",
            xy=(mean_fvu, mean_dead * 100),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    # Draw Pareto frontier step line (sorted by FVU)
    if len(pareto) >= 2:
        pareto_pts = sorted(
            [
                (
                    float(np.mean([v[0] for v in agg[(r.expansion_factor, r.k)]])),
                    float(np.mean([v[1] for v in agg[(r.expansion_factor, r.k)]])) * 100,
                )
                for r in pareto
                if (r.expansion_factor, r.k) in agg
            ]
        )
        px, py = zip(*pareto_pts)
        ax.plot(px, py, "g--", alpha=0.6, linewidth=1.5, label="Pareto frontier")

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="red",
            markersize=14,
            markeredgecolor="black",
            label="Paper (m=4, k=64)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="forestgreen",
            markersize=10,
            label="Pareto-optimal",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="steelblue",
            markersize=8,
            label="Other config",
        ),
    ]
    ax.legend(handles=legend_elements)
    ax.set_xlabel("FVU (↓ better)", fontsize=12)
    ax.set_ylabel("Dead feature % (↓ better)", fontsize=12)
    ax.set_title("SAE Configuration Pareto Frontier", fontsize=14)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "pareto_frontier.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def save_results_csv(
    results: list[ScalingResult], output_dir: Path
) -> Path:
    """Save all ScalingResult objects to a CSV file.

    Args:
        results: All scaling results.
        output_dir: Directory to write all_results.csv.

    Returns:
        Path of the written CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "all_results.csv"

    if not results:
        logger.warning("No results to save.")
        return out_path

    fieldnames = list(asdict(results[0]).keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    logger.info("Saved %d results to %s", len(results), out_path)
    return out_path


def save_summary_table(
    results: list[ScalingResult],
    pareto: list[ScalingResult],
    power_law: dict[str, float],
    output_dir: Path,
) -> Path:
    """Write a LaTeX-ready Markdown summary of scaling law findings.

    Includes: power law coefficients, paper config evaluation, Pareto table,
    and metric range summary.

    Args:
        results: All ScalingResult objects.
        pareto: Pareto-optimal results.
        power_law: Dict with keys 'C', 'a', 'b', 'r2'.
        output_dir: Directory to write summary_table.md.

    Returns:
        Path of the written Markdown file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "summary_table.md"

    paper_results = [
        r for r in results if r.expansion_factor == 4 and r.k == 64
    ]
    paper_fvu = (
        float(np.mean([r.fvu for r in paper_results]))
        if paper_results
        else float("nan")
    )
    paper_dead = (
        float(np.mean([r.dead_feature_pct for r in paper_results]))
        if paper_results
        else float("nan")
    )
    paper_pareto = any(r.expansion_factor == 4 and r.k == 64 for r in pareto)

    lines = [
        "# SAE Scaling Law Summary",
        "",
        f"Total configurations evaluated: **{len(results)}**  ",
        f"Seeds per config: {len(set(r.seed for r in results))}  ",
        f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}  ",
        "",
        "## Power Law Fit",
        "",
        "```",
        # Actual exponent of m is (−a); use sign-explicit format to avoid "^(--x)".
        f"FVU ~ {power_law['C']:.4f} × m^({-power_law['a']:+.3f}) × k^({-power_law['b']:+.3f})",
        f"R² (log space) = {power_law['r2']:.4f}",
        "```",
        "",
        "Interpretation:",
        f"- **a = {power_law['a']:.3f}**: doubling m "
        f"{'reduces' if power_law['a'] > 0 else 'increases'} FVU by "
        f"≈{2**abs(power_law['a']):.2f}×",
        f"- **b = {power_law['b']:.3f}**: doubling k "
        f"{'reduces' if power_law['b'] > 0 else 'increases'} FVU by "
        f"≈{2**abs(power_law['b']):.2f}×",
        "",
        "## Paper Configuration (m=4, k=64)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Mean FVU | {paper_fvu:.4f} |",
        f"| Dead feature % | {paper_dead * 100:.1f}% |",
        f"| Pareto-optimal | {'✓ Yes' if paper_pareto else '✗ No'} |",
        "",
        "## Pareto-Optimal Configurations",
        "",
        "*(Minimising FVU and dead feature %)*",
        "",
        "| m | k | FVU | Dead % | ΔAlignment CLAP |",
        "|---|---|-----|--------|-----------------|",
    ]

    # Aggregate Pareto results over seeds
    pareto_agg: dict[tuple, list[ScalingResult]] = defaultdict(list)
    for r in pareto:
        pareto_agg[(r.expansion_factor, r.k)].append(r)

    for (m, k), group in sorted(pareto_agg.items()):
        mean_fvu = float(np.mean([r.fvu for r in group]))
        mean_dead = float(np.mean([r.dead_feature_pct for r in group]))
        clap_vals = [r.clap_delta for r in group if r.clap_delta >= 0]
        clap_str = f"{np.mean(clap_vals):.3f}" if clap_vals else "n/a"
        marker = " ← paper" if (m == 4 and k == 64) else ""
        lines.append(
            f"| {m} | {k} | {mean_fvu:.4f} | {mean_dead * 100:.1f}% "
            f"| {clap_str} |{marker}"
        )

    lines += [
        "",
        "## Metric Ranges",
        "",
        "| Metric | Min | Max | Mean |",
        "|--------|-----|-----|------|",
    ]

    for metric, label in [
        ("fvu", "FVU"),
        ("dead_feature_pct", "Dead %"),
        ("mean_sparsity", "Sparsity (L0)"),
    ]:
        vals = [getattr(r, metric) for r in results if getattr(r, metric) >= 0]
        if vals:
            lines.append(
                f"| {label} | {min(vals):.4f} | {max(vals):.4f} "
                f"| {float(np.mean(vals)):.4f} |"
            )

    lines += [
        "",
        "---",
        "*Generated by `experiments/sae_scaling.py` — "
        "Phase 3.2 of the TADA roadmap (arXiv 2602.11910).*",
    ]

    out_path.write_text("\n".join(lines))
    logger.info("Saved summary table to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Grid resolution
# ---------------------------------------------------------------------------

import argparse as _argparse  # already imported at top; alias for type hint only


def resolve_grid(
    args: _argparse.Namespace,
    cfg: "ScalingConfig",
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Resolve the final sweep grid from CLI args and the base ScalingConfig.

    Priority (highest → lowest):
      1. ``--smoke-test``        → fixed single config (immutable).
      2. Explicit CLI flags      → ``--m-values`` / ``--k-values`` /
                                   ``--data-sizes`` / ``--seeds``.
      3. ``--preset-real-small`` → RAM-safe preset for 16 GB Apple Silicon.
      4. ``--dry-run`` (small)   → compact 2×2×1×1 grid for quick iteration.
      5. Full ``cfg`` defaults   → ``cfg.expansion_factors`` etc.
                                   (applies when ``--full-grid`` or real mode).

    Args:
        args: Parsed :class:`argparse.Namespace`.
        cfg: Base :class:`ScalingConfig` providing default full-grid values.

    Returns:
        Tuple ``(expansion_factors, k_values, data_sizes, seeds)``.
    """
    # 1. Smoke test always wins — return immediately.
    if getattr(args, "smoke_test", False):
        return [2], [8], [100], [42]

    # 2. Start from mode defaults.
    if getattr(args, "dry_run", False) and not getattr(args, "full_grid", False):
        m_vals: list[int] = [2, 4]
        k_vals: list[int] = [8, 32]
        n_vals: list[int] = [100]
        s_vals: list[int] = [42]
    else:
        # Real mode or --full-grid: use the config defaults.
        m_vals = list(cfg.expansion_factors)
        k_vals = list(cfg.k_values)
        n_vals = list(cfg.data_sizes)
        s_vals = list(cfg.seeds)

    # 3. Apply named preset (overrides mode defaults, but not explicit flags).
    if getattr(args, "preset_real_small", False):
        m_vals = list(_PRESET_REAL_SMALL["expansion_factors"])
        k_vals = list(_PRESET_REAL_SMALL["k_values"])
        n_vals = list(_PRESET_REAL_SMALL["data_sizes"])
        s_vals = list(_PRESET_REAL_SMALL["seeds"])

    # 4. Apply explicit CLI overrides (highest priority after smoke-test).
    if getattr(args, "m_values", None):
        m_vals = list(args.m_values)
    if getattr(args, "k_values", None):
        k_vals = list(args.k_values)
    if getattr(args, "data_sizes", None):
        n_vals = list(args.data_sizes)
    if getattr(args, "seeds", None):
        s_vals = list(args.seeds)

    return m_vals, k_vals, n_vals, s_vals


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_scaling_experiment(
    cfg: ScalingConfig,
    output_dir: Path,
    dry_run: bool,
    smoke_test: bool = False,
    full_grid: bool = False,  # kept for API compatibility; grid now lives in cfg
    activation_cache: Optional[Path] = None,
) -> list[ScalingResult]:
    """Run the full scaling law sweep and write all outputs.

    The sweep grid is taken directly from ``cfg.expansion_factors``,
    ``cfg.k_values``, ``cfg.data_sizes``, and ``cfg.seeds``.  When called
    via :func:`main`, those fields are pre-resolved by :func:`resolve_grid`
    (applying any ``--m-values`` / ``--preset-real-small`` / etc. overrides).

    Args:
        cfg: Scaling configuration.  Grid fields must already be resolved.
        output_dir: Root directory for CSV, plots, and summary table.
        dry_run: Use synthetic activations and ``cfg.dry_run_hidden_dim``.
        smoke_test: Run only the single smallest config (m=2, k=8, n=100,
            seed=42), ignoring all grid fields in ``cfg``.
        full_grid: Kept for backward compatibility; has no effect when the
            grid has been resolved into ``cfg`` by :func:`main`.
        activation_cache: Path to real activation Arrow dataset.

    Returns:
        List of all :class:`ScalingResult` objects.
    """
    # Smoke test is a hard override that bypasses the resolved cfg grid.
    if smoke_test:
        expansion_factors: list[int] = [2]
        k_values: list[int] = [8]
        data_sizes: list[int] = [100]
        seeds: list[int] = [42]
    else:
        # Use whatever the caller (main → resolve_grid) stored in cfg.
        expansion_factors = list(cfg.expansion_factors)
        k_values = list(cfg.k_values)
        data_sizes = list(cfg.data_sizes)
        seeds = list(cfg.seeds)

    total = (
        len(expansion_factors)
        * len(k_values)
        * len(data_sizes)
        * len(seeds)
    )

    # Timing estimates: ~0.5 s/config (dry-run, CPU), ~60 s/config (real, GPU/MPS).
    secs_per_config = 0.5 if dry_run else 60.0
    secs = total * secs_per_config
    device_label = str(_get_device())
    est_str = (
        f"~{secs:.0f}s on {device_label} (dry-run)"
        if dry_run
        else f"~{secs / 3600:.1f} h on {device_label}"
    )
    print(f"Estimated: {est_str}  ({total} configs total)")
    logger.info(
        "Starting scaling sweep: %d configs total  [%s]",
        total,
        est_str,
    )
    logger.info(
        "  expansion_factors (m) : %s",
        expansion_factors,
    )
    logger.info(
        "  k_values              : %s",
        k_values,
    )
    logger.info(
        "  data_sizes (n)        : %s",
        data_sizes,
    )
    logger.info(
        "  seeds                 : %s",
        seeds,
    )

    results: list[ScalingResult] = []
    completed = 0
    log_every = max(1, total // 10)

    for m in expansion_factors:
        for k in k_values:
            for n in data_sizes:
                for seed in seeds:
                    try:
                        result = run_single_config(
                            m=m,
                            k=k,
                            n=n,
                            seed=seed,
                            cfg=cfg,
                            dry_run=dry_run,
                            activation_cache=activation_cache,
                        )
                        results.append(result)
                    except Exception as exc:
                        logger.error(
                            "Config m=%d k=%d n=%d seed=%d failed: %s",
                            m,
                            k,
                            n,
                            seed,
                            exc,
                            exc_info=True,
                        )
                    completed += 1
                    if completed % log_every == 0:
                        logger.info(
                            "Progress: %d/%d (%.0f%%)",
                            completed,
                            total,
                            100.0 * completed / total,
                        )

    if not results:
        logger.error("No results collected — check configuration and logs.")
        return results

    # --- Analysis ---
    power_law = fit_power_law(results)
    logger.info(
        "Power law fit: FVU ~ %.4f × m^(-%.3f) × k^(-%.3f)  R²=%.4f",
        power_law["C"],
        power_law["a"],
        power_law["b"],
        power_law["r2"],
    )

    pareto = find_pareto_frontier(results)
    logger.info("Pareto-optimal configs: %d", len(pareto))
    for r in pareto:
        logger.info(
            "  m=%d k=%d: FVU=%.4f dead=%.1f%%",
            r.expansion_factor,
            r.k,
            r.fvu,
            r.dead_feature_pct * 100,
        )

    # --- Output files ---
    save_results_csv(results, output_dir)
    plot_fvu_vs_expansion(results, output_dir)
    plot_alignment_vs_k(results, output_dir)
    plot_pareto_frontier(results, output_dir)
    save_summary_table(results, pareto, power_law, output_dir)

    logger.info("All outputs written to %s", output_dir)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    default_out = (
        Path(os.environ.get("TADA_WORKDIR", str(Path.home() / "tada_outputs")))
        / "results"
        / "scaling"
    )
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help="Output directory for CSV, plots, and summary table.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Use synthetic activations and a small 2×2 grid "
            "(no GPU or model weights needed)."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run only the single smallest config "
            "(m=2, k=8, n=100, seed=42).  Implies --dry-run."
        ),
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help=(
            "Run the full 5×5×4×3 grid.  "
            "Use with --dry-run for synthetic data or with "
            "--activation-cache for real activations."
        ),
    )
    parser.add_argument(
        "--activation-cache",
        type=Path,
        default=None,
        help=(
            "Path to a cached ACE-Step activation Arrow dataset "
            "(output of cache_activations_runner_ace.py).  "
            "Enables real-mode training."
        ),
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Override hidden dimension (default: 3072 real / 64 dry-run).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override gradient steps per config.",
    )
    # ------------------------------------------------------------------
    # Grid override flags — all optional, each overrides the corresponding
    # dimension of whatever mode / preset would otherwise apply.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=None,
        metavar="M",
        help=(
            "Explicit list of expansion factors m to sweep "
            "(overrides --dry-run default, --full-grid, and --preset-real-small). "
            "Example: --m-values 2 4 8"
        ),
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=None,
        metavar="K",
        help=(
            "Explicit list of TopK sparsity values k to sweep. "
            "Example: --k-values 32 64 128"
        ),
    )
    parser.add_argument(
        "--data-sizes",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Explicit list of training set sizes n to sweep. "
            "Example: --data-sizes 500 2000 5000"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        metavar="S",
        help=(
            "Explicit list of random seeds. "
            "Example: --seeds 42 123"
        ),
    )
    parser.add_argument(
        "--preset-real-small",
        action="store_true",
        help=(
            "Use the RAM-safe preset for 16 GB Apple Silicon M-series: "
            f"m={_PRESET_REAL_SMALL['expansion_factors']}, "
            f"k={_PRESET_REAL_SMALL['k_values']}, "
            f"n={_PRESET_REAL_SMALL['data_sizes']}, "
            f"seeds={_PRESET_REAL_SMALL['seeds']}  "
            "(3×3×2×2 = 36 configs). "
            "Explicit --m-values / --k-values / --data-sizes / --seeds "
            "still override individual dimensions."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the SAE scaling law experiment."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = ScalingConfig()
    if args.hidden_dim is not None:
        cfg.hidden_dim = args.hidden_dim
    if args.num_steps is not None:
        cfg.num_train_steps = args.num_steps
        cfg.dry_run_steps = args.num_steps

    # --smoke-test implies --dry-run; no activation cache also implies dry-run.
    dry_run = args.dry_run or args.smoke_test or (args.activation_cache is None)

    # Resolve the final sweep grid (preset + explicit overrides applied here).
    m_vals, k_vals, n_vals, s_vals = resolve_grid(args, cfg)
    cfg.expansion_factors = m_vals
    cfg.k_values = k_vals
    cfg.data_sizes = n_vals
    cfg.seeds = s_vals

    run_scaling_experiment(
        cfg=cfg,
        output_dir=args.out_dir,
        dry_run=dry_run,
        smoke_test=args.smoke_test,
        full_grid=args.full_grid,
        activation_cache=args.activation_cache,
    )


if __name__ == "__main__":
    main()
