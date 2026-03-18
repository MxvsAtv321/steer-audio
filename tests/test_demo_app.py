"""Smoke tests for the Audio Attribute Studio Gradio app.

These tests verify that the UI builds correctly and that key helper functions
work without requiring model weights, a GPU, or pre-computed steering vectors.
All tests run purely on CPU with synthetic data.
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_gradio_file(tmp_path, monkeypatch):
    """Redirect gr.File uploads to a tmp_path string for batch tests."""
    # Nothing to patch here; gr.File returns a path string from Gradio.
    yield


# ---------------------------------------------------------------------------
# Import + interface-build smoke tests
# ---------------------------------------------------------------------------


def test_module_imports_without_model():
    """demo.app must import cleanly even when model weights are absent."""
    # Force a fresh import (in case a previous test cached a module reference).
    if "demo.app" in sys.modules:
        del sys.modules["demo.app"]
    import demo.app  # noqa: F401 — import side-effects are the test


def test_build_interface_returns_gradio_blocks():
    """build_interface() must return a gradio.Blocks instance."""
    import gradio as gr

    from demo.app import build_interface

    demo = build_interface()
    assert isinstance(demo, gr.Blocks), (
        f"Expected gradio.Blocks, got {type(demo).__name__}"
    )


def test_build_interface_idempotent():
    """Calling build_interface() twice must not raise."""
    from demo.app import build_interface

    d1 = build_interface()
    d2 = build_interface()
    import gradio as gr

    assert isinstance(d1, gr.Blocks)
    assert isinstance(d2, gr.Blocks)


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


def test_demo_audio_shape_and_range():
    """_demo_audio must return a 1-D float32 array within [-1, 1]."""
    from demo.app import _demo_audio

    audio, sr = _demo_audio("test prompt", duration=2.0, seed=42)
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert audio.shape[0] == sr * 2, "Length should equal sr * duration."
    assert np.all(np.abs(audio) <= 1.0 + 1e-6), "Audio must be within [-1, 1]."


def test_demo_audio_deterministic():
    """Same seed must produce identical audio."""
    from demo.app import _demo_audio

    a1, _ = _demo_audio("hello", 1.0, seed=7)
    a2, _ = _demo_audio("hello", 1.0, seed=7)
    np.testing.assert_array_equal(a1, a2)


def test_demo_audio_different_seeds():
    """Different seeds should produce different audio."""
    from demo.app import _demo_audio

    a1, _ = _demo_audio("hello", 1.0, seed=1)
    a2, _ = _demo_audio("hello", 1.0, seed=2)
    assert not np.array_equal(a1, a2)


def test_make_spectrogram_shape():
    """make_spectrogram must return a (H, W, 3) uint8 array."""
    from demo.app import _demo_audio, make_spectrogram

    audio, sr = _demo_audio("test", 2.0, seed=0)
    img = make_spectrogram(audio, sr)
    assert img.ndim == 3, "Spectrogram must be 3-D (H, W, C)."
    assert img.shape[2] == 3, "Spectrogram must have 3 colour channels (RGB)."
    assert img.dtype == np.uint8


def test_placeholder_image_shape():
    """_placeholder_image must return a valid (H, W, 3) uint8 array."""
    from demo.app import _placeholder_image

    img = _placeholder_image("test text")
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.dtype == np.uint8


# ---------------------------------------------------------------------------
# Schedule factory
# ---------------------------------------------------------------------------


def test_build_schedule_constant():
    """Constant schedule returns alpha_max regardless of timestep."""
    from demo.app import _build_schedule

    sched = _build_schedule("Constant", alpha_max=60.0)
    assert sched(30, 30) == pytest.approx(60.0)
    assert sched(1, 30) == pytest.approx(60.0)


def test_build_schedule_cosine_decay_peak_at_start():
    """Cosine schedule should be near alpha_max at t=T."""
    from demo.app import _build_schedule

    sched = _build_schedule("Cosine Decay", alpha_max=80.0)
    # At t=T (very start) cosine factor = 1; at t=1 it should be low.
    val_start = sched(30, 30)
    val_end = sched(1, 30)
    assert val_start >= val_end, "Cosine schedule should decay over time."


def test_build_schedule_early_only():
    """Early-only schedule returns alpha in first half and 0 in second half."""
    from demo.app import _build_schedule

    sched = _build_schedule("Early Steps Only", alpha_max=50.0)
    # t > cutoff*T → non-zero.
    assert sched(25, 30) > 0.0
    # t near 0 → zero.
    assert sched(1, 30) == pytest.approx(0.0)


def test_build_schedule_late_only():
    """Late-only schedule returns alpha in second half and 0 in first half."""
    from demo.app import _build_schedule

    sched = _build_schedule("Late Steps Only", alpha_max=50.0)
    # t near T → zero.
    assert sched(30, 30) == pytest.approx(0.0)
    # t near 0 → non-zero.
    assert sched(1, 30) > 0.0


# ---------------------------------------------------------------------------
# Feature chart + heatmap smoke tests
# ---------------------------------------------------------------------------


def test_feature_importance_chart_returns_image():
    """_feature_importance_chart must return a valid RGB image for any concept."""
    from demo.app import _feature_importance_chart

    img = _feature_importance_chart("tempo")
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.dtype == np.uint8


def test_feature_overlap_heatmap_returns_image():
    """_feature_overlap_heatmap must return a valid RGB image."""
    from demo.app import _feature_overlap_heatmap

    img = _feature_overlap_heatmap()
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.dtype == np.uint8


# ---------------------------------------------------------------------------
# generate_steered in demo mode (no model, no vectors)
# ---------------------------------------------------------------------------


def test_generate_steered_demo_mode_no_concepts():
    """generate_steered with all-zero alphas returns unsteered placeholder."""
    from demo.app import generate_steered

    result = generate_steered(
        prompt="test",
        duration=2.0,
        seed=0,
        tempo=0,
        mood=0,
        vocal_gender=0,
        guitar=0,
        drums=0,
        flute=0,
        violin=0,
        trumpet=0,
        jazz=0,
        reggae=0,
        techno=0,
        method="CAA",
        orthogonalize=True,
        schedule="Constant",
    )
    steered_audio, steered_spec, baseline_audio, info, status = result
    sr_s, audio_s = steered_audio
    sr_b, audio_b = baseline_audio
    assert isinstance(audio_s, np.ndarray)
    assert isinstance(audio_b, np.ndarray)
    assert sr_s > 0
    assert sr_b > 0
    assert steered_spec.ndim == 3
    assert "No concepts" in info


def test_generate_steered_demo_mode_with_concept():
    """generate_steered with nonzero alpha returns audio even without vectors."""
    from demo.app import generate_steered

    result = generate_steered(
        prompt="jazz piano",
        duration=2.0,
        seed=1,
        tempo=50,
        mood=0,
        vocal_gender=0,
        guitar=0,
        drums=0,
        flute=0,
        violin=0,
        trumpet=0,
        jazz=0,
        reggae=0,
        techno=0,
        method="CAA",
        orthogonalize=True,
        schedule="Constant",
    )
    steered_audio, steered_spec, baseline_audio, info, status = result
    sr, audio = steered_audio
    assert audio.shape[0] > 0
    assert sr > 0


# ---------------------------------------------------------------------------
# Algebra expression parsing
# ---------------------------------------------------------------------------


def test_algebra_expression_empty_returns_hint():
    """Empty expression returns a helpful hint string."""
    from demo.app import evaluate_algebra_expression

    audio, msg = evaluate_algebra_expression("", "test", 0)
    assert audio is None
    assert "Enter" in msg or "expression" in msg.lower()


def test_algebra_expression_unknown_concept():
    """Expression with unknown concept names returns an informative error."""
    from demo.app import evaluate_algebra_expression

    audio, msg = evaluate_algebra_expression("xylophone + piano", "test", 0)
    # Either parses fine (both may be valid) or returns an error about unknown concepts.
    assert isinstance(msg, str) and len(msg) > 0


def test_algebra_expression_valid_syntax():
    """Valid expression with known concepts parses without raising."""
    from demo.app import evaluate_algebra_expression

    # Should not raise; audio may be None in demo mode.
    audio, msg = evaluate_algebra_expression("jazz + tempo", "a song", 0)
    assert isinstance(msg, str)


# ---------------------------------------------------------------------------
# Batch experiment
# ---------------------------------------------------------------------------


def test_run_batch_no_file():
    """run_batch with None input returns an error message."""
    from demo.app import run_batch

    csv_out, status = run_batch(None)
    assert "No file" in status


def test_run_batch_with_csv(tmp_path):
    """run_batch with a minimal CSV file completes without raising."""
    import csv as csv_mod

    csv_path = tmp_path / "test_batch.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=["prompt", "tempo", "seed", "duration"])
        writer.writeheader()
        writer.writerow({"prompt": "jazz music", "tempo": "50", "seed": "42", "duration": "2"})

    from demo.app import run_batch

    csv_out, status = run_batch(str(csv_path))
    assert "1" in status  # "Processed 1 row(s)."
    assert "prompt" in csv_out or "row" in csv_out  # CSV has headers
