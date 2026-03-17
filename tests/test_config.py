"""
Unit tests for the configuration system.

Validates paper-correct SaeConfig defaults, ACE-Step model config values,
and the TADA_WORKDIR environment variable override mechanism.
"""

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAE_ROOT = _REPO_ROOT / "sae"
for _p in [str(_REPO_ROOT), str(_SAE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# 1. SaeConfig paper defaults (expansion_factor=4, k=64)
# ---------------------------------------------------------------------------


def test_sae_config_paper_defaults():
    """SaeConfig defaults match the paper values: expansion_factor=4, k=64.

    These were corrected in Prompt 1.1 (previously 32 and 32).
    Reference: arXiv 2602.11910 Table 3.
    """
    from sae_src.sae.config import SaeConfig

    cfg = SaeConfig()
    assert cfg.expansion_factor == 4, (
        f"expansion_factor should be 4 (paper default), got {cfg.expansion_factor}"
    )
    assert cfg.k == 64, (
        f"k should be 64 (paper default), got {cfg.k}"
    )


# ---------------------------------------------------------------------------
# 2. ACE-Step functional layers are [6, 7]
# ---------------------------------------------------------------------------


def test_functional_layers_ace_step(mock_ace_step_config):
    """ACE-Step model config specifies functional_layers=[6, 7].

    These are the layers where musical concepts are most strongly localized
    according to the activation patching results (arXiv 2602.11910 Fig. 3).
    """
    fl = mock_ace_step_config["functional_layers"]
    assert list(fl) == [6, 7], (
        f"Expected functional_layers=[6, 7], got {fl}"
    )


# ---------------------------------------------------------------------------
# 3. TADA_WORKDIR env var overrides the default workdir
# ---------------------------------------------------------------------------


def test_workdir_env_override(monkeypatch, tmp_path):
    """Setting TADA_WORKDIR changes the resolved workdir path.

    All scripts use os.environ.get('TADA_WORKDIR', fallback) so that users
    can redirect all outputs to a custom location without editing code.
    """
    custom_dir = str(tmp_path / "my_tada_outputs")
    monkeypatch.setenv("TADA_WORKDIR", custom_dir)

    resolved = os.environ.get("TADA_WORKDIR", str(Path.home() / "tada_outputs"))
    assert resolved == custom_dir, (
        f"Expected TADA_WORKDIR={custom_dir!r}, got {resolved!r}"
    )


def test_workdir_default_when_env_unset(monkeypatch):
    """When TADA_WORKDIR is not set, the fallback default is used."""
    monkeypatch.delenv("TADA_WORKDIR", raising=False)

    default_workdir = str(Path.home() / "tada_outputs")
    resolved = os.environ.get("TADA_WORKDIR", default_workdir)
    assert resolved == default_workdir, (
        f"Expected default workdir {default_workdir!r}, got {resolved!r}"
    )
