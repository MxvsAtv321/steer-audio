"""
Tests for the `tada` CLI (steer_audio/cli.py).

All tests use Click's CliRunner so no subprocesses are spawned.
A temporary directory is used as a fake TADA_WORKDIR throughout.

Covers:
  - `tada --help` exits 0
  - `tada status` exits 0 and prints a table (mocked workdir)
  - `tada list-vectors` on an empty directory prints "No vectors found"
  - `tada list-vectors` with vectors prints a table
  - `tada evaluate` without prior vectors exits 1 with a clear error
  - `tada compute-vectors --dry-run` exits 0 and prints dry-run notice
  - `tada localize --dry-run` exits 0 and prints dry-run notice
  - `tada generate --dry-run` exits 0 and prints dry-run notice
  - `tada train-sae --dry-run` exits 0 and prints dry-run notice

Reference: TADA roadmap Prompt 1.4.
"""

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from steer_audio.cli import main


# ---------------------------------------------------------------------------
# Fixture: temporary workdir with fake artefacts
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_workdir(tmp_path: Path) -> Path:
    """Return a temp directory pre-populated with a few artefacts."""
    # vectors/tempo/sv.safetensors
    vec_dir = tmp_path / "vectors" / "tempo"
    vec_dir.mkdir(parents=True)
    (vec_dir / "sv.safetensors").write_bytes(b"\x00" * 64)

    # eval/tempo/results.csv (has_eval = True)
    eval_dir = tmp_path / "eval" / "tempo"
    eval_dir.mkdir(parents=True)
    (eval_dir / "results.csv").write_text("alpha,clap\n0,0.5\n")

    # patching/mood/ (no vectors, no eval)
    (tmp_path / "patching" / "mood").mkdir(parents=True)

    return tmp_path


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# 1. `tada --help` exits 0
# ---------------------------------------------------------------------------


def test_help_exits_zero(runner):
    """`tada --help` should print usage information and exit with code 0."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0, (
        f"Expected exit 0 for --help; got {result.exit_code}.\n{result.output}"
    )
    assert "Usage" in result.output or "tada" in result.output.lower()


# ---------------------------------------------------------------------------
# 2. `tada status` exits 0 and prints a table
# ---------------------------------------------------------------------------


def test_status_exits_zero(runner, fake_workdir, monkeypatch):
    """`tada status` exits 0 and emits at least one concept row."""
    monkeypatch.setenv("TADA_WORKDIR", str(fake_workdir))
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0, (
        f"Expected exit 0 for `tada status`; got {result.exit_code}.\n{result.output}"
    )
    # Should mention the 'tempo' concept that has a vector
    assert "tempo" in result.output or "TADA" in result.output


def test_status_empty_workdir(runner, tmp_path, monkeypatch):
    """`tada status` on an empty workdir exits 0 without crashing."""
    monkeypatch.setenv("TADA_WORKDIR", str(tmp_path))
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0, (
        f"Expected exit 0 for empty workdir; got {result.exit_code}.\n{result.output}"
    )


# ---------------------------------------------------------------------------
# 3. `tada list-vectors` on empty directory prints "No vectors found"
# ---------------------------------------------------------------------------


def test_list_vectors_empty(runner, tmp_path, monkeypatch):
    """`tada list-vectors` on an empty workdir prints 'No vectors found'."""
    monkeypatch.setenv("TADA_WORKDIR", str(tmp_path))
    result = runner.invoke(main, ["list-vectors"])
    assert result.exit_code == 0, (
        f"Expected exit 0; got {result.exit_code}.\n{result.output}"
    )
    assert "No vectors found" in result.output


def test_list_vectors_empty_vectors_dir(runner, tmp_path, monkeypatch):
    """`tada list-vectors` when vectors/ exists but holds no .safetensors."""
    (tmp_path / "vectors" / "tempo").mkdir(parents=True)
    monkeypatch.setenv("TADA_WORKDIR", str(tmp_path))
    result = runner.invoke(main, ["list-vectors"])
    assert result.exit_code == 0
    assert "No vectors found" in result.output


# ---------------------------------------------------------------------------
# 4. `tada list-vectors` with vectors prints a table
# ---------------------------------------------------------------------------


def test_list_vectors_with_vectors(runner, fake_workdir, monkeypatch):
    """`tada list-vectors` lists the tempo vector present in fake_workdir."""
    monkeypatch.setenv("TADA_WORKDIR", str(fake_workdir))
    result = runner.invoke(main, ["list-vectors"])
    assert result.exit_code == 0, result.output
    assert "tempo" in result.output
    assert "sv.safetensors" in result.output


# ---------------------------------------------------------------------------
# 5. `tada evaluate` without prior vectors exits 1 with a clear error
# ---------------------------------------------------------------------------


def test_evaluate_exits_1_without_vectors(runner, tmp_path, monkeypatch):
    """`tada evaluate` exits 1 and emits an error when no vectors exist."""
    monkeypatch.setenv("TADA_WORKDIR", str(tmp_path))
    result = runner.invoke(main, ["evaluate", "--concept", "tempo"])
    assert result.exit_code == 1, (
        f"Expected exit 1 when vectors are missing; got {result.exit_code}.\n"
        f"{result.output}"
    )
    combined = result.output + (result.stderr if result.stderr else "")
    assert "tempo" in combined or "vector" in combined.lower()


def test_evaluate_dry_run_skips_vector_check(runner, tmp_path, monkeypatch):
    """`tada evaluate --dry-run` exits 0 even when no vectors exist."""
    monkeypatch.setenv("TADA_WORKDIR", str(tmp_path))
    result = runner.invoke(main, ["evaluate", "--concept", "tempo", "--dry-run"])
    assert result.exit_code == 0, (
        f"Expected exit 0 for dry-run evaluate; got {result.exit_code}.\n{result.output}"
    )
    assert "dry-run" in result.output.lower() or "Would" in result.output


# ---------------------------------------------------------------------------
# 6. Dry-run flags print notices and exit 0 without spawning subprocesses
# ---------------------------------------------------------------------------


def test_compute_vectors_dry_run(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("TADA_WORKDIR", str(tmp_path))
    result = runner.invoke(main, ["compute-vectors", "--concept", "tempo", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output.lower() or "Would" in result.output


def test_localize_dry_run(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("TADA_WORKDIR", str(tmp_path))
    result = runner.invoke(main, ["localize", "--concept", "tempo", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output.lower() or "Would" in result.output


def test_generate_dry_run(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("TADA_WORKDIR", str(tmp_path))
    result = runner.invoke(main, ["generate", "--concept", "tempo", "--alpha", "30", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output.lower() or "Would" in result.output


def test_train_sae_dry_run(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("TADA_WORKDIR", str(tmp_path))
    result = runner.invoke(main, ["train-sae", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output.lower() or "Would" in result.output
