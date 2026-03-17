"""
Shared pytest fixtures for the steer-audio test suite.

All random tensors use torch.manual_seed(42) for reproducibility.
All fixtures create objects that run on CPU so no GPU is required.
"""

import sys
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Ensure repo root and sae/ directory are on sys.path so that
# sae_src, src, and steering packages are importable without installation.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAE_ROOT = _REPO_ROOT / "sae"

for _p in [str(_REPO_ROOT), str(_SAE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Tensor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_activation_tensor() -> torch.Tensor:
    """Return a (batch=2, seq=16, dim=64) activation tensor.

    Uses torch.manual_seed(42) for reproducibility.
    """
    torch.manual_seed(42)
    return torch.randn(2, 16, 64)


# ---------------------------------------------------------------------------
# SAE config & model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sae_config():
    """Return a minimal SaeConfig with small k for fast CPU tests.

    Uses paper defaults for expansion_factor (4) but k=4 to keep
    the number of nonzero features tractable in unit tests.
    """
    from sae_src.sae.config import SaeConfig

    return SaeConfig(expansion_factor=4, k=4)


@pytest.fixture
def mock_sae(mock_sae_config):
    """Return a minimal Sae(d_in=64) model on CPU, seeded for reproducibility."""
    from sae_src.sae.sae import Sae

    torch.manual_seed(42)
    return Sae(d_in=64, cfg=mock_sae_config, device="cpu")


# ---------------------------------------------------------------------------
# ACE-Step config fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ace_step_config() -> dict:
    """Return the ACE-Step model config loaded from configs/model/ace_step.yaml.

    Returns a plain dict so tests don't need OmegaConf installed.
    """
    from omegaconf import OmegaConf

    cfg_path = _REPO_ROOT / "configs" / "model" / "ace_step.yaml"
    cfg = OmegaConf.load(str(cfg_path))
    return OmegaConf.to_container(cfg, resolve=False)
