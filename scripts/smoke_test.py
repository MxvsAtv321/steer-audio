#!/usr/bin/env python3
"""
smoke_test.py — Minimal end-to-end test for ACE-Step steering on RunPod A40.

Runs in < 5 minutes.  Faithfully copies the working generation pattern from
run_phase5_1.py — same controller setup, same register_vector_control call,
same generate() args.

Usage:
    python scripts/smoke_test.py

Environment:
    ACEMODEL_PATH  — path to ACE-Step weights (default: /workspace/ACE-Step)
    TADA_WORKDIR   — workdir containing vectors/ (default: /workspace/steer-audio/outputs)
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Repo path setup (mirrors run_phase5_1.py)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_SAE_ROOT = _REPO_ROOT / "sae"
_ACE_ROOT = _SRC_ROOT / "models" / "ace_step"

for _p in [str(_REPO_ROOT), str(_SAE_ROOT), str(_SAE_ROOT / "sae_src"), str(_SRC_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ACE_SUBMODULE = _ACE_ROOT / "ACE"
if _ACE_SUBMODULE.exists() and str(_ACE_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(_ACE_SUBMODULE))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = Path(os.environ.get("ACEMODEL_PATH", "/workspace/ACE-Step"))
WORKDIR = Path(os.environ.get("TADA_WORKDIR", "/workspace/steer-audio/outputs"))
SV_PATH = WORKDIR / "vectors" / "ace_piano_passes2_allTrue" / "sv.pkl"
OUT_WAV = Path("/tmp/smoke_test.wav")

CONCEPT = "piano"
ALPHA = 1.0
AUDIO_DURATION = 10.0
INFER_STEPS = 30
SAMPLE_RATE = 44100
LAYERS = ["tf6", "tf7"]
PROMPT = "an upbeat electronic track with synths"
CLAP_PROMPT = "piano music"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device       : {device}")
    print(f"model path   : {MODEL_PATH}")
    print(f"sv.pkl path  : {SV_PATH}")
    print()

    # ------------------------------------------------------------------
    # Step 1 — load pipeline
    # ------------------------------------------------------------------
    print("Step 1: Loading ACE-Step pipeline ...")
    from src.models.ace_step.pipeline_ace import SimpleACEStepPipeline  # type: ignore

    pipe = SimpleACEStepPipeline(device=device)
    pipe.load()
    print("  Pipeline loaded.\n")

    # ------------------------------------------------------------------
    # Step 2 — load sv.pkl
    # ------------------------------------------------------------------
    print(f"Step 2: Loading sv.pkl from {SV_PATH} ...")
    if not SV_PATH.exists():
        print(f"  ERROR: {SV_PATH} not found.")
        sys.exit(1)

    with open(SV_PATH, "rb") as f:
        steering_vectors = pickle.load(f)

    # Step 3 — print sv.pkl structure
    top_keys = sorted(steering_vectors.keys())
    print(f"  Top-level keys ({len(top_keys)} steps): first={top_keys[0]!r}  last={top_keys[-1]!r}")
    first_step = steering_vectors[top_keys[0]]
    layer_keys = list(first_step.keys())
    print(f"  Layer keys: {layer_keys}")
    for ln in layer_keys:
        vecs = first_step[ln]
        shape = np.array(vecs[0]).shape if vecs else "(empty)"
        print(f"    {ln}: {len(vecs)} vector(s), shape={shape}")
    print()

    # ------------------------------------------------------------------
    # Step 4 — generate 1 steered audio clip
    # (exact pattern from run_phase5_1.py generate_steered_audio)
    # ------------------------------------------------------------------
    print(f"Step 4: Generating steered audio (alpha={ALPHA}, concept={CONCEPT}) ...")

    from src.models.ace_step.ace_steering.controller import (  # type: ignore
        VectorStore,
        compute_num_cfg_passes,
        register_vector_control,
    )

    num_cfg_passes = compute_num_cfg_passes(0.0, 0.0)

    controller = VectorStore(
        device=device,
        save_only_cond=True,
        num_cfg_passes=num_cfg_passes,
    )
    controller.steer = True
    controller.alpha = ALPHA
    # Pass the full steering_vectors dict (all steps x all layers).
    # Layer filtering is handled by explicit_layers in register_vector_control,
    # which only hooks the target layers so the controller is never called
    # with a layer name that isn't present in steering_vectors.
    controller.steering_vectors = steering_vectors

    register_vector_control(
        pipe.ace_step_transformer,
        controller,
        explicit_layers=LAYERS,
    )

    audio_output = pipe.generate(
        prompt=PROMPT,
        audio_duration=AUDIO_DURATION,
        infer_step=INFER_STEPS,
        manual_seed=42,
        return_type="audio",
        use_erg_lyric=False,
        guidance_scale_text=0.0,
        guidance_scale_lyric=0.0,
        guidance_scale=3.0,
        guidance_interval=1.0,
        guidance_interval_decay=0.0,
    )
    controller.reset()

    # Step 5 — save WAV
    import torchaudio  # type: ignore

    audio_tensor = audio_output.cpu()
    if audio_tensor.ndim == 3:
        audio_tensor = audio_tensor.squeeze(0)
    torchaudio.save(str(OUT_WAV), audio_tensor, SAMPLE_RATE)
    print(f"  Saved to {OUT_WAV}  (shape={tuple(audio_tensor.shape)})\n")

    # ------------------------------------------------------------------
    # Step 6 — CLAP score
    # ------------------------------------------------------------------
    print(f"Step 6: Scoring with CLAP against prompt '{CLAP_PROMPT}' ...")
    clap_score: float = -1.0
    try:
        import laion_clap  # type: ignore
        import torchaudio.functional as taf  # type: ignore

        clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        clap_model.load_ckpt()

        waveform, sr = torchaudio.load(str(OUT_WAV))
        if sr != 48000:
            waveform = taf.resample(waveform, sr, 48000)
        audio_data = waveform.mean(0).numpy()

        a_emb = clap_model.get_audio_embedding_from_data([audio_data], use_tensor=False)
        t_emb = clap_model.get_text_embedding([CLAP_PROMPT])
        clap_score = float(
            np.dot(a_emb[0], t_emb[0])
            / (np.linalg.norm(a_emb[0]) * np.linalg.norm(t_emb[0]) + 1e-8)
        )
        print(f"  CLAP score: {clap_score:.4f}")
    except ImportError:
        print("  laion_clap not installed — skipping CLAP score.")
    except Exception as exc:
        import traceback
        print(f"  CLAP failed: {exc}")
        traceback.print_exc()

    # ------------------------------------------------------------------
    # Step 7 — PASS / FAIL
    # ------------------------------------------------------------------
    print()
    wav_ok = OUT_WAV.exists() and OUT_WAV.stat().st_size > 0
    if wav_ok:
        print("PASS — audio file written successfully.")
        if clap_score >= 0:
            print(f"       CLAP({CLAP_PROMPT!r}) = {clap_score:.4f}")
    else:
        print("FAIL — audio file missing or empty.")
        sys.exit(1)


if __name__ == "__main__":
    main()
