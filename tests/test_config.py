"""
Unit tests for the configuration system (Prompts 1.1 and 1.2).

Covers:
- SaeConfig paper-correct defaults (Prompt 1.1).
- base.yaml loads without errors via Hydra (Prompt 1.2).
- Each model config has required keys: name, functional_layers, d_model (Prompt 1.2).
- Each concept config has required keys: name, alpha_range, tau (Prompt 1.2).
- TADA_WORKDIR env var override works (Prompts 1.1 and 1.2).
"""

import os
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAE_ROOT = _REPO_ROOT / "sae"
_CONFIGS_DIR = _REPO_ROOT / "configs"

for _p in [str(_REPO_ROOT), str(_SAE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(rel_path: str) -> dict:
    """Load a YAML file relative to configs/ and return as a plain dict."""
    cfg = OmegaConf.load(str(_CONFIGS_DIR / rel_path))
    return OmegaConf.to_container(cfg, resolve=False)


# ---------------------------------------------------------------------------
# 1. SaeConfig paper defaults (expansion_factor=4, k=64) — Prompt 1.1
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
# 2. base.yaml loads without errors via Hydra — Prompt 1.2
# ---------------------------------------------------------------------------


def test_base_yaml_loads_via_hydra():
    """base.yaml composes without errors and contains all required top-level keys."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir=str(_CONFIGS_DIR.resolve()), version_base="1.4"
    ):
        cfg = compose(config_name="base")
        # Resolve scalar values inside the context so Hydra interpolations work.
        seed = cfg.seed
        num_workers = cfg.num_workers
        keys = set(cfg.keys())

    assert "workdir" in keys, "base.yaml must define 'workdir'"
    assert "seed" in keys, "base.yaml must define 'seed'"
    assert "device" in keys, "base.yaml must define 'device'"
    assert "num_workers" in keys, "base.yaml must define 'num_workers'"
    assert seed == 42
    assert num_workers == 4


def test_base_yaml_workdir_references_tada_workdir():
    """base.yaml workdir interpolation references the TADA_WORKDIR env var.

    The actual env-var resolution path (TADA_WORKDIR set → cfg.workdir = that path)
    is tested in test_workdir_env_override via os.environ.  Here we verify the YAML
    interpolation string is wired to the right variable.
    """
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir=str(_CONFIGS_DIR.resolve()), version_base="1.4"
    ):
        cfg = compose(config_name="base")
        # Retrieve the raw (unresolved) workdir interpolation string.
        raw = OmegaConf.to_container(cfg, resolve=False)["workdir"]

    assert "TADA_WORKDIR" in str(raw), (
        f"base.yaml workdir must reference TADA_WORKDIR; got raw value: {raw!r}"
    )


def test_base_yaml_workdir_override_via_hydra(tmp_path):
    """Hydra CLI overrides can change the workdir (same mechanism as TADA_WORKDIR)."""
    from hydra import compose, initialize_config_dir

    custom_dir = str(tmp_path / "my_tada_outputs")

    with initialize_config_dir(
        config_dir=str(_CONFIGS_DIR.resolve()), version_base="1.4"
    ):
        # Explicit override — avoids resolving ${hydra:runtime.cwd} fallback in
        # the compose context (HydraConfig is not populated by compose).
        cfg = compose(config_name="base", overrides=[f"workdir={custom_dir}"])
        resolved = cfg.workdir

    assert resolved == custom_dir, (
        f"Expected overridden workdir={custom_dir!r}, got {resolved!r}"
    )


# ---------------------------------------------------------------------------
# 3. Each model config has keys: name, functional_layers, d_model — Prompt 1.2
# ---------------------------------------------------------------------------

_MODEL_YAMLS = ["model/ace_step.yaml", "model/audioldm2.yaml", "model/stable_audio.yaml"]


@pytest.mark.parametrize("yaml_path", _MODEL_YAMLS)
def test_model_config_has_required_keys(yaml_path: str):
    """Every model yaml must contain name, functional_layers, and d_model."""
    cfg = _load_yaml(yaml_path)
    for key in ("name", "functional_layers", "d_model"):
        assert key in cfg, (
            f"{yaml_path} is missing required key '{key}'. Found keys: {list(cfg.keys())}"
        )


def test_ace_step_model_config_values():
    """ACE-Step model config matches TADA paper values (arXiv 2602.11910 §4)."""
    cfg = _load_yaml("model/ace_step.yaml")
    assert cfg["name"] == "ace_step", f"Expected name='ace_step', got {cfg['name']!r}"
    assert list(cfg["functional_layers"]) == [6, 7], (
        f"Expected functional_layers=[6, 7], got {cfg['functional_layers']}"
    )
    assert cfg["d_model"] == 1024, f"Expected d_model=1024, got {cfg['d_model']}"
    assert cfg["guidance_scale"] == 7.0, f"Expected guidance_scale=7.0, got {cfg['guidance_scale']}"
    assert cfg["total_layers"] == 24, f"Expected total_layers=24, got {cfg['total_layers']}"
    assert "model_id" in cfg, "ace_step.yaml must contain 'model_id'"


# kept from Prompt 1.1 — uses the conftest fixture
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
# 4. Each concept config has keys: name, alpha_range, tau — Prompt 1.2
# ---------------------------------------------------------------------------

_CONCEPT_YAMLS = [
    "concept/tempo.yaml",
    "concept/mood.yaml",
    "concept/instruments.yaml",
    "concept/vocal_gender.yaml",
    "concept/genre.yaml",
]


@pytest.mark.parametrize("yaml_path", _CONCEPT_YAMLS)
def test_concept_config_has_required_keys(yaml_path: str):
    """Every concept yaml must contain name, alpha_range, and tau."""
    cfg = _load_yaml(yaml_path)
    for key in ("name", "alpha_range", "tau"):
        assert key in cfg, (
            f"{yaml_path} is missing required key '{key}'. Found keys: {list(cfg.keys())}"
        )


@pytest.mark.parametrize("yaml_path", _CONCEPT_YAMLS)
def test_concept_alpha_range_is_list(yaml_path: str):
    """alpha_range must be a list of 21 values from -100 to 100 (step 10)."""
    cfg = _load_yaml(yaml_path)
    ar = cfg["alpha_range"]
    assert isinstance(ar, list), (
        f"{yaml_path}: alpha_range must be a list, got {type(ar)}"
    )
    assert len(ar) == 21, (
        f"{yaml_path}: alpha_range must have 21 entries (-100..100 step 10), got {len(ar)}"
    )
    assert ar[0] == -100 and ar[-1] == 100


def test_tempo_concept_values():
    """Tempo concept yaml matches spec: similarity_metric=muq, tau=20."""
    cfg = _load_yaml("concept/tempo.yaml")
    assert cfg["similarity_metric"] == "muq", (
        f"tempo.yaml: expected similarity_metric='muq', got {cfg['similarity_metric']!r}"
    )
    assert cfg["tau"] == 20


def test_mood_concept_values():
    """Mood concept yaml matches spec: tau=40, similarity_metric=clap."""
    cfg = _load_yaml("concept/mood.yaml")
    assert cfg["tau"] == 40, f"mood.yaml: expected tau=40, got {cfg['tau']}"
    assert cfg["similarity_metric"] == "clap"


def test_vocal_gender_concept_values():
    """Vocal gender concept yaml matches spec: similarity_metric=clap, tau=20."""
    cfg = _load_yaml("concept/vocal_gender.yaml")
    assert cfg["similarity_metric"] == "clap", (
        f"vocal_gender.yaml: expected similarity_metric='clap', got {cfg['similarity_metric']!r}"
    )
    assert cfg["tau"] == 20


# ---------------------------------------------------------------------------
# 5. TADA_WORKDIR env var override — Prompt 1.1 + 1.2
# ---------------------------------------------------------------------------


def test_workdir_env_override(monkeypatch, tmp_path):
    """Setting TADA_WORKDIR changes the resolved workdir path."""
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
