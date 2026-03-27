#!/usr/bin/env python3
"""
run_paper_experiments.py — ISMIR 2026 TADA paper experiment runner.

Single, self-contained runner for the full paper pipeline on a RunPod A40 GPU
at /workspace/steer-audio.

Usage:
    # All experiments
    python scripts/run_paper_experiments.py

    # Specific experiments
    python scripts/run_paper_experiments.py --experiments 1 2 5

    # Fast synthetic dry-run (no GPU, no model)
    python scripts/run_paper_experiments.py --dry-run

    # Subset of concepts
    python scripts/run_paper_experiments.py --concepts piano jazz

Environment variables:
    TADA_WORKDIR     — workdir for intermediate files (default: <repo>/outputs)
    ACEMODEL_PATH    — path to ACE-Step model weights (default: /workspace/ACE-Step)
    TADA_DEVICE      — 'cuda' or 'cpu' (default: cuda if available)

Reference: TADA arXiv 2602.11910, targeting ISMIR 2026.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import pickle
import random
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("paper_exp")

# ---------------------------------------------------------------------------
# Repo path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAE_ROOT = _REPO_ROOT / "sae"
_SRC_ROOT = _REPO_ROOT / "src"
_ACE_ROOT = _SRC_ROOT / "models" / "ace_step"

for _p in [
    str(_REPO_ROOT),
    str(_SAE_ROOT),
    str(_SAE_ROOT / "sae_src"),
    str(_SRC_ROOT),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ACE_SUBMODULE = _ACE_ROOT / "ACE"
if _ACE_SUBMODULE.exists() and str(_ACE_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(_ACE_SUBMODULE))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_CONCEPTS: List[str] = ["piano", "tempo", "mood", "drums", "jazz"]
ALPHA_VALUES: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
N_PAIRS: int = 256       # prompt pairs for CAA vector computation
N_TEST: int = 50         # separate test prompts for evaluation
INFER_STEPS: int = 30
AUDIO_DURATION: float = 12.0
SAMPLE_RATE: int = 44100
FUNCTIONAL_LAYERS: List[str] = ["tf6", "tf7"]

# Dry-run uses tiny counts for fast iteration.
DRY_N_PAIRS: int = 4
DRY_N_TEST: int = 3
DRY_ALPHA_VALUES: List[float] = [0.0, 1.0]

# Results root (populated after parse_args).
RESULTS_DIR: Path = _REPO_ROOT / "results" / "paper"

# ---------------------------------------------------------------------------
# Prompt generation helpers
# ---------------------------------------------------------------------------

_NEUTRAL_POOL = [
    "a song", "a melody", "music", "a tune", "a track", "a composition",
    "instrumental music", "a piece of music", "background music",
    "a musical performance", "an upbeat song", "a slow song",
    "a fast-paced track", "electronic music", "acoustic music",
    "orchestral music", "a pop song", "a rock song", "a jazz piece",
    "a classical piece", "hip hop music", "country music", "blues music",
    "folk music", "reggae music", "metal music", "punk rock", "dance music",
    "ambient music", "lofi music", "a ballad", "a love song", "a happy song",
    "a sad song", "energetic music", "calm music", "dramatic music",
    "cheerful music", "melancholic music", "aggressive music",
    "gentle music", "powerful music", "soft music", "loud music",
    "rhythmic music", "harmonious music", "simple music", "complex music",
    "minimalist music", "a nocturne", "a symphony",
]

_TEST_PROMPTS_POOL = [
    "Upbeat indie pop track with jangly guitars and handclaps, summer road trip vibes",
    "Melancholic jazz ballad with smooth saxophone, walking bassline, late night atmosphere",
    "Energetic drum and bass with synth arpeggios, electronic bleeps, futuristic feel",
    "Acoustic folk song with fingerpicked guitar, gentle harmonies, campfire warmth",
    "Dark synthwave anthem with pulsing bass, retro analog synths, neon city nights",
    "Cheerful bossa nova with nylon string guitar, soft percussion, breezy coastal mood",
    "Heavy metal riff with distorted guitars, thundering drums, aggressive energy",
    "Ambient electronic soundscape with ethereal pads, distant chimes, meditative calm",
    "Funky disco groove with slap bass, wah-wah guitar, rhythmic strings, danceable beat",
    "Minimalist lo-fi hip hop beat with vinyl crackle, mellow Rhodes, jazzy drum loop",
    "Traditional Irish jig with fiddle, tin whistle, bodhran drums, celtic celebration",
    "Afrobeat groove with polyrhythmic percussion, brass stabs, hypnotic guitar riff",
    "Flamenco piece with passionate acoustic guitar, handclaps, percussive footwork energy",
    "Indian classical fusion with sitar melody, tabla rhythms, meditative drone",
    "Brazilian samba with surdo drums, cavaquinho strumming, carnival energy",
    "Middle Eastern dub with oud melody, electronic beats, desert mysticism",
    "Reggae roots track with offbeat guitar skank, deep bass, one drop drums",
    "Japanese city pop with bright synths, punchy drums, nostalgic 80s shimmer",
    "Balkan brass band with energetic trumpets, tuba bass, wedding celebration chaos",
    "Hawaiian slack key guitar with ukulele, gentle waves, tropical serenity",
    "Techno industrial with distorted kicks, metallic textures, relentless machine rhythm",
    "Chillwave with washed-out synths, reverbed vocals, hazy summer nostalgia",
    "Hardstyle anthem with reverse bass, euphoric lead, stadium energy",
    "Glitch hop with chopped samples, wonky bass, playful digital chaos",
    "Progressive house with building arpeggios, lush pads, sunrise festival moment",
    "Dubstep with massive wobble bass, half-time drums, dark underground energy",
    "Vaporwave with slowed samples, reverbed mall music, retro consumerist dreamscape",
    "Breakbeat with chopped drums, funky samples, high energy street dance",
    "Trance anthem with soaring lead, four-on-floor kick, euphoric breakdown",
    "IDM with complex polyrhythms, granular textures, experimental soundscape",
    "90s grunge with fuzzy guitars, angst-filled dynamics, Seattle rain melancholy",
    "Surf rock with reverbed twangy guitar, driving drums, California wave energy",
    "Post-rock crescendo with layered guitars, building dynamics, cinematic emotion",
    "Psychedelic rock with phaser guitars, swirling organs, mind-expanding journey",
    "Punk rock blast with fast power chords, shouted energy, rebellious spirit",
    "Southern rock with slide guitar, boogie rhythm, dusty highway freedom",
    "Shoegaze wall with heavily effected guitars, dreamy vocals, blurred beauty",
    "Progressive rock odyssey with odd time signatures, synth solos, epic storytelling",
    "Stoner rock with downtuned riffs, slow heavy groove, hazy desert vibes",
    "Math rock with intricate tapping, angular rhythms, precise chaos",
    "Bebop jazz with fast changes, virtuosic piano, swinging drums, smoky club",
    "Delta blues with slide guitar, stomping rhythm, front porch storytelling",
    "Cool jazz with muted trumpet, brushed drums, sophisticated restraint",
    "Chicago electric blues with wailing harmonica, gritty guitar, juke joint energy",
    "Latin jazz with congas, piano montuno, energetic brass, dance floor heat",
    "Free jazz with avant-garde saxophone, chaotic drums, boundary-pushing expression",
    "Smooth jazz with polished production, soft electric piano, easy listening warmth",
    "Gypsy jazz with acoustic guitar runs, violin melody, Parisian café charm",
    "Soul jazz with funky organ, tenor sax, head-nodding groove",
    "Blues rock with overdriven guitar, shuffling drums, roadhouse swagger",
]


def _cycle_to(lst: List[str], n: int) -> List[str]:
    """Return exactly *n* items by cycling through *lst*."""
    if not lst:
        return [f"item_{i}" for i in range(n)]
    result = []
    while len(result) < n:
        result.extend(lst)
    return result[:n]


def get_test_prompts(n: int = N_TEST) -> List[str]:
    """Return *n* test prompts (not used in vector computation)."""
    return _cycle_to(_TEST_PROMPTS_POOL, n)


_CONCEPT_POSITIVE_KEYWORD: Dict[str, str] = {
    "piano": "with piano",
    "tempo": "fast tempo",
    "mood": "happy mood",
    "drums": "with prominent drums",
    "jazz": "jazz style",
    "electric_guitar": "with electric guitar",
    "fast_tempo": "fast tempo",
    "female_vocals": "with female vocals",
    "violin": "with violin",
}

_CONCEPT_NEGATIVE_KEYWORD: Dict[str, str] = {
    "piano": "without piano",
    "tempo": "slow tempo",
    "mood": "sad mood",
    "drums": "without drums",
    "jazz": "non-jazz",
    "electric_guitar": "without electric guitar",
    "fast_tempo": "slow tempo",
    "female_vocals": "without female vocals",
    "violin": "without violin",
}


def get_prompt_pairs(concept: str, n: int) -> Tuple[List[str], List[str]]:
    """Return (neg_prompts, pos_prompts) of length *n* for CAA computation."""
    pos_kw = _CONCEPT_POSITIVE_KEYWORD.get(concept, f"with {concept}")
    neg_kw = _CONCEPT_NEGATIVE_KEYWORD.get(concept, f"without {concept}")

    # Try loading from sae_src steer_prompts
    try:
        from sae_src.configs import steer_prompts  # type: ignore

        fn_map = {
            "piano": "get_prompts_piano",
            "mood": "get_prompts_mood",
            "tempo": "get_prompts_tempo",
            "drums": "get_prompts_drums",
        }
        fn_name = fn_map.get(concept)
        if fn_name and hasattr(steer_prompts, fn_name):
            neg_raw, pos_raw, _ = getattr(steer_prompts, fn_name)(num=n)
            return _cycle_to(neg_raw, n), _cycle_to(pos_raw, n)
    except Exception:
        pass

    neutrals = _cycle_to(_NEUTRAL_POOL, n)
    pos = [f"{p} {pos_kw}" for p in neutrals]
    neg = [p for p in neutrals]
    return neg, pos


# ---------------------------------------------------------------------------
# Dependency detection helpers
# ---------------------------------------------------------------------------


def _try_import_clap():
    """Return laion_clap module or None."""
    try:
        import laion_clap  # type: ignore
        return laion_clap
    except ImportError:
        log.warning("laion_clap not installed — CLAP scores will be -1")
        return None


def _try_import_muq():
    """Return muq module or None. Tries to install muq==0.1.0 if missing."""
    try:
        import muq as _muq  # type: ignore
        return _muq
    except ImportError:
        log.warning("muq not installed — attempting pip install muq==0.1.0")
        try:
            import subprocess
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "muq==0.1.0", "-q"],
                check=True, capture_output=True,
            )
            import muq as _muq  # type: ignore
            return _muq
        except Exception as e:
            log.warning("muq install failed (%s) — MuQ scores will be -1", e)
            return None


def _try_import_fadtk():
    """Return fadtk module or None."""
    try:
        import fadtk  # type: ignore
        return fadtk
    except ImportError:
        log.warning("fadtk not installed — FAD scores will be -1")
        return None


# ---------------------------------------------------------------------------
# CLAP loading (singleton)
# ---------------------------------------------------------------------------

_CLAP_MODEL = None


def get_clap_model(laion_clap_mod):
    """Load CLAP model once and cache it."""
    global _CLAP_MODEL
    if _CLAP_MODEL is None and laion_clap_mod is not None:
        try:
            model = laion_clap_mod.CLAP_Module(enable_fusion=False)
            model.load_ckpt()
            _CLAP_MODEL = model
        except Exception as e:
            log.warning("Failed to load CLAP model: %s", e)
    return _CLAP_MODEL


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------


def compute_clap_scores(
    audio_paths: List[Path],
    prompts: List[str],
    clap_mod=None,
    dry_run: bool = False,
) -> List[float]:
    """Return per-file CLAP scores. Returns -1.0 on failure."""
    if dry_run or clap_mod is None:
        return [-1.0] * len(audio_paths)

    model = get_clap_model(clap_mod)
    if model is None:
        return [-1.0] * len(audio_paths)

    scores = []
    try:
        import torchaudio  # type: ignore
        import torchaudio.functional as taf  # type: ignore
        _CLAP_SAMPLE_RATE = 48000  # laion_clap is trained at 48 kHz
        for wav_path, prompt in zip(audio_paths, prompts):
            try:
                if not wav_path.exists() or wav_path.stat().st_size == 0:
                    log.warning("CLAP skip — missing or empty: %s", wav_path)
                    scores.append(-1.0)
                    continue
                waveform, sr = torchaudio.load(str(wav_path))
                # Resample to 48 kHz — CLAP is trained at this rate; passing
                # 44100 Hz audio without resampling produces garbage embeddings.
                if sr != _CLAP_SAMPLE_RATE:
                    waveform = taf.resample(waveform, sr, _CLAP_SAMPLE_RATE)
                audio_data = waveform.mean(0).numpy()
                a_emb = model.get_audio_embedding_from_data([audio_data], use_tensor=False)
                t_emb = model.get_text_embedding([prompt])
                cos = float(
                    np.dot(a_emb[0], t_emb[0])
                    / (np.linalg.norm(a_emb[0]) * np.linalg.norm(t_emb[0]) + 1e-8)
                )
                scores.append(cos)
            except Exception as exc:
                import traceback
                log.warning(
                    "CLAP failed for %s — score set to -1.0\n  exception: %s\n%s",
                    wav_path.name, exc, traceback.format_exc(),
                )
                scores.append(-1.0)
    except ImportError as e:
        log.warning("torchaudio not available for CLAP scoring: %s", e)
        scores = [-1.0] * len(audio_paths)
    return scores


def compute_clap_concept_score(
    audio_paths: List[Path],
    concept_keyword: str,
    clap_mod=None,
    dry_run: bool = False,
) -> float:
    """CLAP score against concept keyword (e.g. 'piano music')."""
    prompts = [f"{concept_keyword} music"] * len(audio_paths)
    scores = compute_clap_scores(audio_paths, prompts, clap_mod, dry_run)
    valid = [s for s in scores if s >= 0]
    return float(np.mean(valid)) if valid else -1.0


def compute_muq_scores(
    audio_paths: List[Path],
    muq_mod=None,
    dry_run: bool = False,
) -> float:
    """Return mean MuQ score. Returns -1.0 on failure."""
    if dry_run or muq_mod is None:
        return -1.0
    try:
        scores = []
        for p in audio_paths:
            if not p.exists() or p.stat().st_size == 0:
                continue
            try:
                score = muq_mod.compute(str(p))
                if isinstance(score, (int, float)):
                    scores.append(float(score))
                elif hasattr(score, "__iter__"):
                    scores.extend([float(x) for x in score])
            except Exception as exc:
                log.warning("MuQ failed for %s: %s", p.name, exc)
        return float(np.mean(scores)) if scores else -1.0
    except Exception as exc:
        log.warning("MuQ computation error: %s", exc)
        return -1.0


def compute_fad_score(
    audio_paths: List[Path],
    fadtk_mod=None,
    dry_run: bool = False,
) -> float:
    """Return FAD score. Returns -1.0 on failure or if fadtk unavailable."""
    if dry_run or fadtk_mod is None:
        return -1.0
    try:
        # fadtk expects a directory of WAV files
        import tempfile, shutil
        with tempfile.TemporaryDirectory() as tmpdir:
            for p in audio_paths:
                if p.exists():
                    shutil.copy(str(p), tmpdir)
            score = fadtk_mod.compute_fad(tmpdir)
            return float(score)
    except Exception as exc:
        log.warning("FAD computation failed: %s", exc)
        return -1.0


def compute_cosine_interference(sv_a: Dict, sv_b: Dict) -> float:
    """Mean cosine similarity between two steering vector dicts (across layers/steps)."""
    cosines = []
    for step_key in sv_a:
        if step_key not in sv_b:
            continue
        for layer_name in sv_a[step_key]:
            if layer_name not in sv_b[step_key]:
                continue
            va = sv_a[step_key][layer_name]
            vb = sv_b[step_key][layer_name]
            arr_a = va[0] if va else None
            arr_b = vb[0] if vb else None
            if arr_a is None or arr_b is None:
                continue
            na = np.linalg.norm(arr_a)
            nb = np.linalg.norm(arr_b)
            if na > 0 and nb > 0:
                cosines.append(float(np.dot(arr_a.flatten(), arr_b.flatten()) / (na * nb)))
    return float(np.mean(cosines)) if cosines else 0.0


# ---------------------------------------------------------------------------
# Gram-Schmidt orthogonalization
# ---------------------------------------------------------------------------


def gram_schmidt_orthogonalize(vectors: List[np.ndarray]) -> List[np.ndarray]:
    """Gram-Schmidt orthogonalization of a list of 1-D numpy arrays.

    Returns orthonormal vectors. If a vector's residual norm < 1e-6, it is
    replaced by a zero vector and a warning is emitted.
    """
    result: List[np.ndarray] = []
    for i, v in enumerate(vectors):
        v = v.copy().astype(np.float64).flatten()
        for u in result:
            v = v - np.dot(v, u) * u
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            log.warning(
                "Vector %d has near-zero residual after GS orthogonalization (norm=%.2e)."
                " Concepts may be nearly collinear.",
                i, norm,
            )
            result.append(np.zeros_like(v))
        else:
            result.append(v / norm)
    return result


def combine_sv_dicts(
    sv_list: List[Dict],
    weights: List[float],
    orthogonalize: bool = False,
) -> Dict:
    """Linearly combine steering-vector dicts with optional GS orthogonalization.

    Args:
        sv_list: List of sv.pkl dicts (one per concept).
        weights: Scalar weight per concept.
        orthogonalize: If True, apply Gram-Schmidt before combining.

    Returns:
        Combined sv.pkl dict with the same step/layer structure.
    """
    if not sv_list:
        return {}

    combined: Dict = defaultdict(lambda: defaultdict(list))
    all_step_keys = sorted(set().union(*[sv.keys() for sv in sv_list]))

    for step_key in all_step_keys:
        all_layer_names = sorted(set().union(*[
            sv.get(step_key, {}).keys() for sv in sv_list
        ]))
        for layer_name in all_layer_names:
            arrs = []
            ws = []
            for sv, w in zip(sv_list, weights):
                vecs = sv.get(step_key, {}).get(layer_name, [])
                if vecs:
                    arrs.append(vecs[0].flatten().astype(np.float64))
                    ws.append(w)

            if not arrs:
                continue

            if orthogonalize:
                orth_arrs = gram_schmidt_orthogonalize(arrs)
                combined_arr = sum(w * a for w, a in zip(ws, orth_arrs))
            else:
                combined_arr = sum(w * a for w, a in zip(ws, arrs))

            norm = np.linalg.norm(combined_arr)
            if norm > 0:
                combined_arr = combined_arr / norm

            # Reshape back to original shape
            orig_shape = sv_list[0].get(step_key, {}).get(layer_name, [arrs[0]])[0].shape
            combined[step_key][layer_name] = [combined_arr.reshape(orig_shape)]

    return dict(combined)


# ---------------------------------------------------------------------------
# Schedule functions
# ---------------------------------------------------------------------------


def get_schedule_fn(schedule_type: str) -> Callable[[int, int], float]:
    """Return f(step_idx, total_steps) -> float ∈ [0, 1]."""
    if schedule_type == "constant":
        return lambda step, total: 1.0
    elif schedule_type == "cosine":
        def _cosine(step: int, total: int) -> float:
            if total <= 0:
                return 1.0
            return (1.0 + math.cos(math.pi * min(step, total) / total)) / 2.0
        return _cosine
    elif schedule_type == "early_only":
        return lambda step, total: 1.0 if (total > 0 and step < total * 0.4) else 0.0
    elif schedule_type == "late_only":
        return lambda step, total: 0.0 if (total > 0 and step < total * 0.6) else 1.0
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type!r}")


def apply_schedule_to_sv(sv: Dict, schedule_fn: Callable, total_steps: int) -> Dict:
    """Pre-scale a steering-vector dict by a timestep schedule."""
    step_keys = sorted(sv.keys())
    n_steps = len(step_keys)
    scaled: Dict = {}
    for step_idx, step_key in enumerate(step_keys):
        scale = schedule_fn(step_idx, n_steps if n_steps > 0 else total_steps)
        layers = sv[step_key]
        scaled[step_key] = {}
        for layer_name, vecs in layers.items():
            arr = vecs[0] if vecs else None
            if arr is not None:
                scaled[step_key][layer_name] = [arr * scale]
    return scaled


def apply_window_to_sv(sv: Dict, start_frac: float, end_frac: float) -> Dict:
    """Apply steering only within [start_frac, end_frac] fraction of steps."""
    def _window_fn(step: int, total: int) -> float:
        if total <= 0:
            return 0.0
        frac = step / total
        return 1.0 if start_frac <= frac < end_frac else 0.0
    return apply_schedule_to_sv(sv, _window_fn, len(sv))


# ---------------------------------------------------------------------------
# CAA vector computation
# ---------------------------------------------------------------------------


def _sv_cache_path(vectors_dir: Path, concept: str) -> Path:
    """Return directory path for pre-computed vectors of *concept*."""
    return vectors_dir / f"ace_{concept}_passes2_allTrue"


def compute_caa_vector(
    pipe,
    concept: str,
    device: str,
    vectors_dir: Path,
    n_pairs: int = N_PAIRS,
    infer_steps: int = INFER_STEPS,
    audio_duration: float = AUDIO_DURATION,
    dry_run: bool = False,
) -> Optional[Path]:
    """Compute and cache CAA steering vectors for *concept*.

    Skips recomputation if sv.pkl already exists.  Returns the vector
    directory, or None on failure.
    """
    save_dir = _sv_cache_path(vectors_dir, concept)
    save_dir.mkdir(parents=True, exist_ok=True)
    sv_path = save_dir / "sv.pkl"
    config_path = save_dir / "config.json"

    if sv_path.exists() and config_path.exists():
        log.info("[%s] Cached CAA vectors found at %s — skipping.", concept, save_dir)
        return save_dir

    log.info("[%s] Computing CAA vectors (%d pairs) ...", concept, n_pairs)

    if dry_run:
        # Write synthetic vectors with integer step keys — matches controller expectation
        # (controller uses actual_denoising_step integer as dict key, NOT layer-name strings).
        # Real sv.pkl has layers tf0-tf23 all shape (2560,); dry-run only needs FUNCTIONAL_LAYERS
        # since generate_audio returns early in dry-run mode before ever reading the sv.
        rng = np.random.default_rng(abs(hash(concept)) % (2**31))
        d_model = 2560  # matches real ACE-Step hidden dim for tf0-tf23
        sv: Dict = {}
        for step_idx in range(infer_steps):
            sv[step_idx] = {}
            for layer in FUNCTIONAL_LAYERS:
                vec = rng.standard_normal(d_model).astype(np.float32)
                vec /= (np.linalg.norm(vec) + 1e-8)
                sv[step_idx][layer] = [vec]
        sv[infer_steps] = sv[infer_steps - 1]
        with open(sv_path, "wb") as f:
            pickle.dump(sv, f)
        _write_json(config_path, {"concept": concept, "n_pairs": n_pairs, "dry_run": True})
        log.info("[dry-run] Wrote synthetic vectors for '%s'.", concept)
        return save_dir

    # --- Real computation ---
    try:
        from src.models.ace_step.ace_steering.controller import (  # type: ignore
            VectorStore,
            compute_num_cfg_passes,
            register_vector_control,
        )
    except ImportError as e:
        log.error("Cannot import ACE-Step controller: %s", e)
        return None

    neg_prompts, pos_prompts = get_prompt_pairs(concept, n_pairs)
    num_cfg_passes = compute_num_cfg_passes(0.0, 0.0)

    pos_vectors: List[Dict] = []
    neg_vectors: List[Dict] = []

    for prompt_pos, prompt_neg in zip(pos_prompts, neg_prompts):
        for polarity, prompt in [("pos", prompt_pos), ("neg", prompt_neg)]:
            ctrl = VectorStore(
                device=device,
                save_only_cond=True,
                num_cfg_passes=num_cfg_passes,
            )
            ctrl.steer = False
            # Use explicit_layers here too — hooks only the SAME blocks that
            # generation will register, so sv.pkl layer keys and vector dimensions
            # are guaranteed to match.  Without this, every LinearTransformerBlock
            # in the model is hooked, including blocks at other layers that may have
            # different hidden dimensions (e.g. 2560 vs 1024), causing a size
            # mismatch at generation time.
            register_vector_control(
                pipe.ace_step_transformer, ctrl, explicit_layers=FUNCTIONAL_LAYERS
            )
            pipe.generate(
                prompt=prompt,
                audio_duration=audio_duration,
                infer_step=infer_steps,
                manual_seed=42,
                return_type="latent",
                use_erg_lyric=False,
                guidance_scale_text=0.0,
                guidance_scale_lyric=0.0,
                guidance_scale=3.0,
                guidance_interval=1.0,
                guidance_interval_decay=0.0,
            )
            if polarity == "pos":
                pos_vectors.append(ctrl.vector_store)
            else:
                neg_vectors.append(ctrl.vector_store)
            ctrl.reset()

    # Compute mean difference (CAA)
    sv: Dict = defaultdict(lambda: defaultdict(list))
    step_keys = list(pos_vectors[0].keys())
    layer_names = list(pos_vectors[0][step_keys[0]].keys())

    for sk in step_keys:
        for ln in layer_names:
            pos_avg = np.mean([pos_vectors[i][sk][ln][0] for i in range(len(pos_vectors))], axis=0)
            neg_avg = np.mean([neg_vectors[i][sk][ln][0] for i in range(len(neg_vectors))], axis=0)
            diff = pos_avg - neg_avg
            norm = np.linalg.norm(diff)
            if norm > 0:
                diff = diff / norm
            sv[sk][ln] = [diff]

    with open(sv_path, "wb") as f:
        pickle.dump(dict(sv), f)
    _write_json(config_path, {
        "concept": concept, "n_pairs": n_pairs,
        "infer_steps": infer_steps, "audio_duration": audio_duration, "device": device,
    })
    log.info("[%s] CAA vectors saved to %s.", concept, save_dir)
    return save_dir


def _normalize_sv_keys(sv: Dict) -> Dict:
    """Convert sv.pkl step keys to integers.

    Existing sv.pkl files may store step keys as strings ("0", "1", ...,
    "step_0", ...) rather than integers.  The controller looks up steps via
    ``self.actual_denoising_step`` which is always an integer, so all
    downstream code must use integer keys.  This function normalises at
    load time so the rest of the pipeline never has to care.

    Strategy:
      1. If keys are already ints → return unchanged.
      2. If string keys are digit strings ("0", "1", ...) → cast to int.
      3. Otherwise (e.g. "step_0") → sort lexicographically and re-index
         as 0, 1, 2, ... preserving the order.
    """
    if not sv:
        return sv
    first_key = next(iter(sv))
    if isinstance(first_key, int):
        return sv  # already integers — nothing to do

    # Try direct int() cast (covers "0", "1", ..., "29")
    try:
        converted = {int(k): v for k, v in sv.items()}
        log.debug("Normalised sv keys from str→int (e.g. %r → %d)", first_key, int(first_key))
        return converted
    except (ValueError, TypeError):
        pass

    # Fallback: sort string keys and assign sequential integer indices
    sorted_keys = sorted(sv.keys())
    log.warning(
        "sv.pkl has non-numeric string keys (e.g. %r) — re-indexing as 0..%d.",
        sorted_keys[0], len(sorted_keys) - 1,
    )
    return {i: sv[k] for i, k in enumerate(sorted_keys)}


def load_sv(sv_dir: Path) -> Optional[Dict]:
    """Load sv.pkl from *sv_dir* and normalise step keys to integers."""
    sv_path = sv_dir / "sv.pkl"
    if not sv_path.exists():
        log.warning("sv.pkl not found at %s", sv_path)
        return None
    try:
        with open(sv_path, "rb") as f:
            sv = pickle.load(f)
        return _normalize_sv_keys(sv)
    except Exception as exc:
        log.error("Failed to load sv.pkl from %s: %s", sv_path, exc)
        return None


# ---------------------------------------------------------------------------
# sv.pkl layer validation helpers
# ---------------------------------------------------------------------------


def _get_sv_layer_keys(sv: Dict) -> List[str]:
    """Return the layer-name keys stored in sv.pkl (from the first step entry)."""
    if not sv:
        return []
    first_step_val = next(iter(sv.values()))
    return list(first_step_val.keys())


def _validate_sv_layers(
    sv: Dict,
    requested_layers: List[str],
    label: str = "",
) -> List[str]:
    """Check sv.pkl layer keys against *requested_layers* and return the safe intersection.

    Logs every layer key found in sv together with its vector shape.  Warns when
    requested layers are absent.  Raises ValueError when the intersection is empty
    (no usable layers), pointing the user to delete the stale cache.

    Args:
        sv: Normalised sv.pkl dict.
        requested_layers: Layers the caller wants to use (e.g. FUNCTIONAL_LAYERS).
        label: Optional label for log messages (e.g. concept name).

    Returns:
        Sorted list of layer names that are both in sv and in requested_layers.
    """
    sv_layers = _get_sv_layer_keys(sv)

    # Log every layer present in the sv and the shape of its stored vector.
    first_step = next(iter(sv.values())) if sv else {}
    shape_info = {}
    for ln in sv_layers:
        vecs = first_step.get(ln, [])
        shape_info[ln] = tuple(vecs[0].shape) if vecs else "(empty)"
    log.info("[%s] sv.pkl layer keys + shapes: %s", label, shape_info)

    intersection = [l for l in requested_layers if l in sv_layers]
    missing = [l for l in requested_layers if l not in sv_layers]
    extra = [l for l in sv_layers if l not in requested_layers]

    if missing:
        log.warning(
            "[%s] Requested layers %s are NOT in sv.pkl (available: %s). "
            "If this is a stale cached vector file, delete %s and rerun to recompute.",
            label, missing, sv_layers, label,
        )
    if extra:
        log.debug("[%s] sv.pkl has extra layers not requested: %s — ignoring.", label, extra)
    if not intersection:
        raise ValueError(
            f"[{label}] No overlap between requested layers {requested_layers} "
            f"and sv.pkl layers {sv_layers}. "
            f"Delete the cached vector directory for '{label}' and rerun so vectors "
            f"are recomputed with the correct layer convention."
        )

    log.info("[%s] Using layers for generation: %s", label, intersection)
    return sorted(intersection)


# ---------------------------------------------------------------------------
# Audio generation
# ---------------------------------------------------------------------------


def generate_audio(
    pipe,
    sv_dict: Dict,
    out_dir: Path,
    device: str,
    alpha: float,
    prompts: List[str],
    infer_steps: int = INFER_STEPS,
    audio_duration: float = AUDIO_DURATION,
    layers: List[str] = FUNCTIONAL_LAYERS,
    steer_mode: str = "cond_only",
    dry_run: bool = False,
    seed: int = 42,
) -> List[Path]:
    """Generate audio steered by *sv_dict* and save WAVs.

    Returns list of WAV paths (one per prompt).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        paths = []
        for i in range(len(prompts)):
            wav = out_dir / f"p{i:03d}.wav"
            wav.touch()
            paths.append(wav)
        return paths

    try:
        from src.models.ace_step.ace_steering.controller import (  # type: ignore
            VectorStore,
            compute_num_cfg_passes,
            register_vector_control,
        )
        import torchaudio  # type: ignore
    except ImportError as e:
        log.error("Import error during audio generation: %s", e)
        return []

    num_cfg_passes = compute_num_cfg_passes(0.0, 0.0)
    paths: List[Path] = []

    for i, prompt in enumerate(prompts):
        wav_path = out_dir / f"p{i:03d}.wav"
        if wav_path.exists() and wav_path.stat().st_size > 0:
            paths.append(wav_path)
            continue

        ctrl = VectorStore(
            device=device,
            save_only_cond=(steer_mode == "cond_only"),
            num_cfg_passes=num_cfg_passes,
        )
        # Mirror smoke_test.py pattern: steer whenever sv is present.
        # Callers already pass sv_dict=None for unsteered/baseline runs.
        ctrl.steer = sv_dict is not None
        ctrl.alpha = alpha
        # Pass the full sv dict (all steps × all layers).
        # Layer filtering is handled by explicit_layers in register_vector_control,
        # which only hooks the target layers so the controller is never called
        # with a layer name that isn't present in sv_dict.
        if sv_dict is not None:
            # Pad the sv dict so the controller never sees a KeyError if the
            # pipeline runs one extra forward pass beyond infer_steps (e.g. step 30
            # for a 30-step run).  _pad_sv_for_extra_steps also normalises keys to
            # integers, so combine_sv_dicts outputs are handled safely too.
            ctrl.steering_vectors = _pad_sv_for_extra_steps(sv_dict)

        # Always register a fresh controller — even for unsteered runs — to
        # prevent a stale controller from a previous call remaining hooked
        # with its incremented actual_denoising_step counter.
        register_vector_control(pipe.ace_step_transformer, ctrl, explicit_layers=layers)

        audio_out = pipe.generate(
            prompt=prompt,
            audio_duration=audio_duration,
            infer_step=infer_steps,
            manual_seed=seed,
            return_type="audio",
            use_erg_lyric=False,
            guidance_scale_text=0.0,
            guidance_scale_lyric=0.0,
            guidance_scale=3.0,
            guidance_interval=1.0,
            guidance_interval_decay=0.0,
        )
        ctrl.reset()

        audio_tensor = audio_out.cpu()
        if audio_tensor.ndim == 3:
            audio_tensor = audio_tensor.squeeze(0)
        torchaudio.save(str(wav_path), audio_tensor, SAMPLE_RATE)
        log.info("  Saved %s (alpha=%.2f)", wav_path.name, alpha)
        paths.append(wav_path)

    return paths


def _pad_sv_for_extra_steps(sv: Dict, extra: int = 2) -> Dict:
    """Duplicate the final step entry so the controller never sees a KeyError.

    ACE-Step's scheduler sometimes runs N+1 forward passes for N inference
    steps (e.g. an initial or final auxiliary call).  Padding the sv dict with
    copies of the last step is a safe no-op: if the extra step is never reached
    the padding is unused, and if it is reached the steering just continues with
    the final-step vector instead of crashing.

    Always normalises keys to integers first so the arithmetic ``max_key + i``
    never hits a TypeError on string keys.
    """
    if not sv:
        return sv
    sv = _normalize_sv_keys(sv)
    max_key = max(sv.keys())          # guaranteed int after normalisation
    padded = dict(sv)
    for i in range(1, extra + 1):
        padded[max_key + i] = sv[max_key]
    return padded


def generate_random_vector_sv(sv_ref: Dict, seed: int = 0) -> Dict:
    """Generate a sv-dict of random unit vectors with the same structure as *sv_ref*."""
    rng = np.random.default_rng(seed)
    rv: Dict = {}
    for sk, layers in sv_ref.items():
        rv[sk] = {}
        for ln, vecs in layers.items():
            arr = vecs[0] if vecs else None
            if arr is not None:
                rand = rng.standard_normal(arr.shape).astype(arr.dtype)
                norm = np.linalg.norm(rand)
                rv[sk][ln] = [rand / norm if norm > 0 else rand]
    return rv


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, rows: List[Dict], fieldnames: Optional[List[str]] = None) -> None:
    """Write *rows* to *path* as CSV with optional explicit field ordering."""
    if not rows:
        log.warning("No rows to write for %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), path)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _init_matplotlib():
    """Import and configure matplotlib for paper-quality output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.figsize": (8, 5),
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })
    return plt


# ---------------------------------------------------------------------------
# ACE-Step pipeline loader
# ---------------------------------------------------------------------------


def load_ace_pipeline(device: str, model_path: Path):
    """Load SimpleACEStepPipeline from local checkpoint."""
    try:
        from src.models.ace_step.pipeline_ace import SimpleACEStepPipeline  # type: ignore
    except ImportError as e:
        log.error("Cannot import SimpleACEStepPipeline: %s", e)
        raise

    log.info("Loading ACE-Step pipeline from %s ...", model_path)
    t0 = time.time()
    pipe = SimpleACEStepPipeline(device=device)
    pipe.load()
    log.info("ACE-Step loaded in %.1f s", time.time() - t0)
    return pipe


# ---------------------------------------------------------------------------
# EXPERIMENT 1 — Baseline single-concept steering
# ---------------------------------------------------------------------------


def run_exp1(
    pipe,
    device: str,
    concepts: List[str],
    vectors_dir: Path,
    audio_base: Path,
    results_dir: Path,
    n_pairs: int = N_PAIRS,
    n_test: int = N_TEST,
    alpha_values: List[float] = ALPHA_VALUES,
    dry_run: bool = False,
) -> List[Dict]:
    """Experiment 1: Baseline single-concept steering.

    Tests each concept at multiple alpha values with three baselines:
      (a) unsteered (alpha=0)
      (b) text prompt with concept word appended
      (c) random unit vector steering
    Metrics: CLAP, MuQ, FAD, LPAPS
    Output: results/paper/exp1_baseline.csv
    """
    log.info("=== EXP 1: Baseline Single-Concept Steering ===")
    clap_mod = _try_import_clap()
    muq_mod = _try_import_muq()
    fadtk_mod = _try_import_fadtk()

    test_prompts = get_test_prompts(n_test)
    rows: List[Dict] = []

    for concept in concepts:
        log.info("  [exp1] concept=%s", concept)

        # Ensure CAA vectors are computed / cached
        sv_dir = compute_caa_vector(
            pipe, concept, device, vectors_dir, n_pairs, dry_run=dry_run
        )
        if sv_dir is None and not dry_run:
            log.error("Failed to get vectors for %s — skipping.", concept)
            continue

        sv = load_sv(sv_dir) if sv_dir else None
        random_sv = generate_random_vector_sv(sv, seed=42) if sv else None

        pos_kw = _CONCEPT_POSITIVE_KEYWORD.get(concept, concept)

        # Alpha sweep (main steering)
        for alpha in alpha_values:
            out_dir = audio_base / "exp1" / concept / f"alpha_{alpha}"
            sv_to_use = sv if alpha != 0.0 else None
            paths = generate_audio(
                pipe, sv_to_use, out_dir, device, alpha, test_prompts,
                dry_run=dry_run,
            )
            clap_scores = compute_clap_scores(paths, test_prompts, clap_mod, dry_run)
            muq = compute_muq_scores(paths, muq_mod, dry_run)
            fad = compute_fad_score(paths, fadtk_mod, dry_run)
            rows.append({
                "concept": concept,
                "condition": "steered",
                "alpha": alpha,
                "clap": float(np.mean([s for s in clap_scores if s >= 0]) if any(s >= 0 for s in clap_scores) else -1.0),
                "muq": muq,
                "fad": fad,
                "lpaps": -1.0,  # LPAPS unavailable
            })

        # Baseline (b): text prompt with concept appended
        baseline_prompts = [f"{p} {pos_kw}" for p in test_prompts]
        out_dir_tb = audio_base / "exp1" / concept / "text_baseline"
        paths_tb = generate_audio(
            pipe, None, out_dir_tb, device, 0.0, baseline_prompts, dry_run=dry_run
        )
        clap_tb = compute_clap_scores(paths_tb, test_prompts, clap_mod, dry_run)
        muq_tb = compute_muq_scores(paths_tb, muq_mod, dry_run)
        rows.append({
            "concept": concept, "condition": "text_baseline", "alpha": 0.0,
            "clap": float(np.mean([s for s in clap_tb if s >= 0]) if any(s >= 0 for s in clap_tb) else -1.0),
            "muq": muq_tb, "fad": -1.0, "lpaps": -1.0,
        })

        # Baseline (c): random unit vector
        if random_sv is not None:
            out_dir_rv = audio_base / "exp1" / concept / "random_vector"
            paths_rv = generate_audio(
                pipe, random_sv, out_dir_rv, device, 1.0, test_prompts, dry_run=dry_run
            )
            clap_rv = compute_clap_scores(paths_rv, test_prompts, clap_mod, dry_run)
            muq_rv = compute_muq_scores(paths_rv, muq_mod, dry_run)
            rows.append({
                "concept": concept, "condition": "random_vector", "alpha": 1.0,
                "clap": float(np.mean([s for s in clap_rv if s >= 0]) if any(s >= 0 for s in clap_rv) else -1.0),
                "muq": muq_rv, "fad": -1.0, "lpaps": -1.0,
            })

    out_path = results_dir / "exp1_baseline.csv"
    write_csv(out_path, rows, ["concept", "condition", "alpha", "clap", "muq", "fad", "lpaps"])
    log.info("EXP 1 done. Results: %s", out_path)
    return rows


# ---------------------------------------------------------------------------
# EXPERIMENT 2 — Timestep commitment (THE KEY FINDING)
# ---------------------------------------------------------------------------


def run_exp2(
    pipe,
    device: str,
    concepts: List[str],
    vectors_dir: Path,
    audio_base: Path,
    results_dir: Path,
    n_test: int = N_TEST,
    dry_run: bool = False,
) -> List[Dict]:
    """Experiment 2: Timestep commitment — WHEN does the model encode semantics?

    Applies steering only within 5 time windows and measures CLAP delta vs
    unsteered baseline.  One line per concept in the figure.
    Output: results/paper/exp2_timestep_commitment.csv
            results/paper/figures/exp2_commitment_curve.png
    """
    log.info("=== EXP 2: Timestep Commitment ===")
    clap_mod = _try_import_clap()
    test_prompts = get_test_prompts(n_test)
    alpha = 1.0
    windows = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

    rows: List[Dict] = []

    # First compute unsteered baseline CLAP per concept
    unsteered_clap: Dict[str, float] = {}
    for concept in concepts:
        sv_dir = compute_caa_vector(pipe, concept, device, vectors_dir, DRY_N_PAIRS if dry_run else N_PAIRS, dry_run=dry_run)
        out_dir = audio_base / "exp2" / concept / "unsteered"
        paths = generate_audio(pipe, None, out_dir, device, 0.0, test_prompts, dry_run=dry_run)
        clap_scores = compute_clap_scores(paths, test_prompts, clap_mod, dry_run)
        valid = [s for s in clap_scores if s >= 0]
        unsteered_clap[concept] = float(np.mean(valid)) if valid else -1.0

    for concept in concepts:
        sv_dir = _sv_cache_path(vectors_dir, concept)
        sv = load_sv(sv_dir)
        if sv is None and not dry_run:
            log.warning("[exp2] No sv for %s — skipping.", concept)
            continue
        if sv is None:
            sv = {}  # dry_run: windowed svs will be empty but generate_audio handles it

        for start_frac, end_frac in windows:
            window_label = f"{start_frac:.1f}-{end_frac:.1f}"
            windowed_sv = apply_window_to_sv(sv, start_frac, end_frac) if sv else None
            out_dir = audio_base / "exp2" / concept / f"window_{window_label}"
            paths = generate_audio(
                pipe, windowed_sv, out_dir, device, alpha, test_prompts, dry_run=dry_run
            )
            clap_scores = compute_clap_scores(paths, test_prompts, clap_mod, dry_run)
            valid = [s for s in clap_scores if s >= 0]
            steered_clap = float(np.mean(valid)) if valid else -1.0
            base = unsteered_clap.get(concept, -1.0)
            delta = (steered_clap - base) if (steered_clap >= 0 and base >= 0) else -1.0
            rows.append({
                "concept": concept,
                "window": window_label,
                "window_start": start_frac,
                "window_end": end_frac,
                "clap_steered": steered_clap,
                "clap_unsteered": base,
                "clap_delta": delta,
            })
            log.info("  [exp2] %s window=%s  CLAP_delta=%.4f", concept, window_label, delta)

    out_path = results_dir / "exp2_timestep_commitment.csv"
    write_csv(out_path, rows, ["concept", "window", "window_start", "window_end",
                               "clap_steered", "clap_unsteered", "clap_delta"])

    # Plot
    _plot_exp2(rows, results_dir / "figures" / "exp2_commitment_curve.png", concepts)
    log.info("EXP 2 done. Results: %s", out_path)
    return rows


def _plot_exp2(rows: List[Dict], out_path: Path, concepts: List[str]) -> None:
    """One line per concept, x=time window center, y=CLAP delta."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt = _init_matplotlib()
        fig, ax = plt.subplots(figsize=(8, 5))
        markers = ["o", "s", "^", "D", "v"]
        window_centers = [0.1, 0.3, 0.5, 0.7, 0.9]
        window_labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]

        for i, concept in enumerate(concepts):
            concept_rows = [r for r in rows if r["concept"] == concept]
            concept_rows.sort(key=lambda r: float(r["window_start"]))
            deltas = [float(r["clap_delta"]) for r in concept_rows]
            if all(d < 0 for d in deltas):
                continue  # no data
            ax.plot(
                window_centers[:len(deltas)], deltas,
                marker=markers[i % len(markers)],
                label=concept,
            )

        ax.set_xlabel("Timestep window (fraction of denoising)")
        ax.set_ylabel("CLAP delta vs unsteered baseline")
        ax.set_title("Timestep Commitment: When Does the Model Encode Semantic Concepts?")
        ax.set_xticks(window_centers)
        ax.set_xticklabels(window_labels, rotation=15)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved exp2 figure: %s", out_path)
    except Exception as exc:
        log.warning("Could not create exp2 figure: %s", exc)


# ---------------------------------------------------------------------------
# EXPERIMENT 3 — Concept algebra
# ---------------------------------------------------------------------------


def run_exp3(
    pipe,
    device: str,
    vectors_dir: Path,
    audio_base: Path,
    results_dir: Path,
    n_test: int = N_TEST,
    dry_run: bool = False,
) -> List[Dict]:
    """Experiment 3: Concept algebra — addition, subtraction, interpolation.

    Compares naive additive vs Gram-Schmidt orthogonalized vectors.
    Measures BOTH concept CLAP scores simultaneously.
    Output: results/paper/exp3_concept_algebra.csv
    """
    log.info("=== EXP 3: Concept Algebra ===")
    clap_mod = _try_import_clap()
    test_prompts = get_test_prompts(n_test)

    # Define algebra operations: (label, concept_a, weight_a, concept_b, weight_b)
    operations: List[Dict] = [
        # Addition pairs
        {"label": "piano+jazz",       "ca": "piano", "wa": 1.0, "cb": "jazz",  "wb": 1.0},
        {"label": "tempo+mood",       "ca": "tempo", "wa": 1.0, "cb": "mood",  "wb": 1.0},
        {"label": "drums+jazz",       "ca": "drums", "wa": 1.0, "cb": "jazz",  "wb": 1.0},
        # Subtraction pairs
        {"label": "piano-drums",      "ca": "piano", "wa": 1.0, "cb": "drums", "wb": -1.0},
        {"label": "mood-fast_tempo",  "ca": "mood",  "wa": 1.0, "cb": "tempo", "wb": -1.0},
        {"label": "jazz-electric_guitar", "ca": "jazz", "wa": 1.0, "cb": "electric_guitar", "wb": -1.0},
        # Interpolation: piano + jazz
        {"label": "0.25piano+0.75jazz", "ca": "piano", "wa": 0.25, "cb": "jazz", "wb": 0.75},
        {"label": "0.50piano+0.50jazz", "ca": "piano", "wa": 0.50, "cb": "jazz", "wb": 0.50},
        {"label": "0.75piano+0.25jazz", "ca": "piano", "wa": 0.75, "cb": "jazz", "wb": 0.25},
    ]

    rows: List[Dict] = []

    for op in operations:
        label = op["label"]
        ca, wa, cb, wb = op["ca"], op["wa"], op["cb"], op["wb"]
        log.info("  [exp3] %s", label)

        # Ensure vectors exist
        n_pairs = DRY_N_PAIRS if dry_run else N_PAIRS
        sv_dir_a = compute_caa_vector(pipe, ca, device, vectors_dir, n_pairs, dry_run=dry_run)
        sv_dir_b = compute_caa_vector(pipe, cb, device, vectors_dir, n_pairs, dry_run=dry_run)
        sv_a = load_sv(sv_dir_a) if sv_dir_a else None
        sv_b = load_sv(sv_dir_b) if sv_dir_b else None

        if sv_a is None or sv_b is None:
            log.warning("[exp3] Missing vectors for %s — using zeros.", label)
            sv_a = sv_a or {}
            sv_b = sv_b or {}

        for method in ["naive", "gramschmidt"]:
            orthogonalize = (method == "gramschmidt")
            combined = combine_sv_dicts([sv_a, sv_b], [wa, wb], orthogonalize=orthogonalize)

            out_dir = audio_base / "exp3" / label / method
            paths = generate_audio(
                pipe, combined, out_dir, device, 1.0, test_prompts, dry_run=dry_run
            )
            clap_a = compute_clap_concept_score(paths, ca, clap_mod, dry_run)
            clap_b = compute_clap_concept_score(paths, cb, clap_mod, dry_run)
            rows.append({
                "label": label,
                "concept_a": ca, "weight_a": wa,
                "concept_b": cb, "weight_b": wb,
                "method": method,
                "clap_concept_a": clap_a,
                "clap_concept_b": clap_b,
            })
            log.info(
                "    method=%-12s  CLAP(%s)=%.4f  CLAP(%s)=%.4f",
                method, ca, clap_a, cb, clap_b,
            )

    out_path = results_dir / "exp3_concept_algebra.csv"
    write_csv(out_path, rows, [
        "label", "concept_a", "weight_a", "concept_b", "weight_b",
        "method", "clap_concept_a", "clap_concept_b",
    ])
    log.info("EXP 3 done. Results: %s", out_path)
    return rows


# ---------------------------------------------------------------------------
# EXPERIMENT 4 — Gram-Schmidt ablation
# ---------------------------------------------------------------------------


def run_exp4(
    pipe,
    device: str,
    vectors_dir: Path,
    audio_base: Path,
    results_dir: Path,
    n_test: int = N_TEST,
    dry_run: bool = False,
) -> List[Dict]:
    """Experiment 4: Gram-Schmidt ablation.

    Compares multi-concept steering WITH vs WITHOUT Gram-Schmidt.
    Measures per-concept CLAP and interference score (cosine similarity).
    Output: results/paper/exp4_gramschmidt_ablation.csv
    """
    log.info("=== EXP 4: Gram-Schmidt Ablation ===")
    clap_mod = _try_import_clap()
    test_prompts = get_test_prompts(n_test)
    n_pairs = DRY_N_PAIRS if dry_run else N_PAIRS

    concept_pairs = [("piano", "jazz"), ("tempo", "mood"), ("drums", "piano")]
    rows: List[Dict] = []

    for ca, cb in concept_pairs:
        sv_dir_a = compute_caa_vector(pipe, ca, device, vectors_dir, n_pairs, dry_run=dry_run)
        sv_dir_b = compute_caa_vector(pipe, cb, device, vectors_dir, n_pairs, dry_run=dry_run)
        sv_a = load_sv(sv_dir_a) if sv_dir_a else {}
        sv_b = load_sv(sv_dir_b) if sv_dir_b else {}
        interference = compute_cosine_interference(sv_a, sv_b)

        pair_label = f"{ca}+{cb}"
        log.info("  [exp4] pair=%s  interference(cosine)=%.4f", pair_label, interference)

        for method in ["without_gs", "with_gs"]:
            orthogonalize = (method == "with_gs")
            combined = combine_sv_dicts([sv_a, sv_b], [1.0, 1.0], orthogonalize=orthogonalize)

            out_dir = audio_base / "exp4" / pair_label / method
            paths = generate_audio(
                pipe, combined, out_dir, device, 1.0, test_prompts, dry_run=dry_run
            )
            clap_a = compute_clap_concept_score(paths, ca, clap_mod, dry_run)
            clap_b = compute_clap_concept_score(paths, cb, clap_mod, dry_run)
            rows.append({
                "pair": pair_label,
                "concept_a": ca, "concept_b": cb,
                "method": method,
                "clap_concept_a": clap_a,
                "clap_concept_b": clap_b,
                "interference_cosine": interference,
            })
            log.info(
                "    %-12s  CLAP(%s)=%.4f  CLAP(%s)=%.4f",
                method, ca, clap_a, cb, clap_b,
            )

    out_path = results_dir / "exp4_gramschmidt_ablation.csv"
    write_csv(out_path, rows, [
        "pair", "concept_a", "concept_b", "method",
        "clap_concept_a", "clap_concept_b", "interference_cosine",
    ])
    log.info("EXP 4 done. Results: %s", out_path)
    return rows


# ---------------------------------------------------------------------------
# EXPERIMENT 5 — Schedule comparison
# ---------------------------------------------------------------------------


def run_exp5(
    pipe,
    device: str,
    concepts: List[str],
    vectors_dir: Path,
    audio_base: Path,
    results_dir: Path,
    n_test: int = N_TEST,
    dry_run: bool = False,
) -> List[Dict]:
    """Experiment 5: Steering schedule comparison.

    Compares constant, cosine, early_only, late_only schedules.
    Output: results/paper/exp5_schedules.csv
            results/paper/figures/exp5_schedule_comparison.png
    """
    log.info("=== EXP 5: Schedule Comparison ===")
    clap_mod = _try_import_clap()
    muq_mod = _try_import_muq()
    test_prompts = get_test_prompts(n_test)
    schedules = ["constant", "cosine", "early_only", "late_only"]
    alpha = 1.0
    n_pairs = DRY_N_PAIRS if dry_run else N_PAIRS

    rows: List[Dict] = []

    for concept in concepts:
        sv_dir = compute_caa_vector(pipe, concept, device, vectors_dir, n_pairs, dry_run=dry_run)
        sv = load_sv(sv_dir) if sv_dir else {}

        for schedule in schedules:
            log.info("  [exp5] concept=%s schedule=%s", concept, schedule)
            sched_fn = get_schedule_fn(schedule)
            scheduled_sv = apply_schedule_to_sv(sv, sched_fn, INFER_STEPS) if sv else {}

            out_dir = audio_base / "exp5" / concept / schedule
            paths = generate_audio(
                pipe, scheduled_sv, out_dir, device, alpha, test_prompts, dry_run=dry_run
            )
            clap_scores = compute_clap_scores(paths, test_prompts, clap_mod, dry_run)
            valid_clap = [s for s in clap_scores if s >= 0]
            muq = compute_muq_scores(paths, muq_mod, dry_run)
            rows.append({
                "concept": concept,
                "schedule": schedule,
                "alpha": alpha,
                "clap": float(np.mean(valid_clap)) if valid_clap else -1.0,
                "muq": muq,
            })
            log.info(
                "    CLAP=%.4f  MuQ=%.4f",
                rows[-1]["clap"], muq,
            )

    out_path = results_dir / "exp5_schedules.csv"
    write_csv(out_path, rows, ["concept", "schedule", "alpha", "clap", "muq"])
    _plot_exp5(rows, results_dir / "figures" / "exp5_schedule_comparison.png", concepts, schedules)
    log.info("EXP 5 done. Results: %s", out_path)
    return rows


def _plot_exp5(
    rows: List[Dict],
    out_path: Path,
    concepts: List[str],
    schedules: List[str],
) -> None:
    """Grouped bar chart: CLAP by schedule, one group per concept."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt = _init_matplotlib()
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(concepts))
        n_sched = len(schedules)
        width = 0.8 / n_sched
        hatches = ["/", "\\", "x", "."]
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        for i, sched in enumerate(schedules):
            clap_vals = []
            for concept in concepts:
                matched = [r for r in rows if r["concept"] == concept and r["schedule"] == sched]
                clap_vals.append(float(matched[0]["clap"]) if matched and float(matched[0]["clap"]) >= 0 else 0.0)
            offset = (i - n_sched / 2 + 0.5) * width
            ax.bar(
                x + offset, clap_vals, width,
                label=sched, hatch=hatches[i % len(hatches)],
                color=colors[i % len(colors)], alpha=0.85,
            )

        ax.set_xlabel("Concept")
        ax.set_ylabel("CLAP score")
        ax.set_title("Schedule Comparison: CLAP by Concept and Schedule Type")
        ax.set_xticks(x)
        ax.set_xticklabels(concepts)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved exp5 figure: %s", out_path)
    except Exception as exc:
        log.warning("Could not create exp5 figure: %s", exc)


# ---------------------------------------------------------------------------
# EXPERIMENT 6 — Human evaluation preparation
# ---------------------------------------------------------------------------


def run_exp6(
    pipe,
    device: str,
    concepts: List[str],
    vectors_dir: Path,
    results_dir: Path,
    n_pairs_per_concept: int = 10,
    dry_run: bool = False,
) -> None:
    """Experiment 6: Human evaluation preparation.

    Generates matched audio pairs: unsteered vs steered at alpha=1.0.
    Also generates 5 concept algebra pairs.
    Output: results/paper/human_eval/{concept}/pair_{i}_{condition}.wav
            results/paper/human_eval/instructions.md
    """
    log.info("=== EXP 6: Human Evaluation Preparation ===")
    alpha = 1.0
    human_eval_dir = results_dir / "human_eval"
    n_pairs = DRY_N_PAIRS if dry_run else N_PAIRS
    test_prompts = get_test_prompts(N_TEST)

    for concept in concepts:
        log.info("  [exp6] Generating pairs for concept=%s", concept)
        sv_dir = compute_caa_vector(pipe, concept, device, vectors_dir, n_pairs, dry_run=dry_run)
        sv = load_sv(sv_dir) if sv_dir else None
        concept_dir = human_eval_dir / concept
        concept_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_pairs_per_concept):
            prompt = test_prompts[i % len(test_prompts)]
            seed_i = 42 + i

            # Unsteered
            unsteered_path = concept_dir / f"pair_{i:02d}_unsteered.wav"
            if not unsteered_path.exists() or unsteered_path.stat().st_size == 0:
                unst_dir = concept_dir / f"_tmp_unst_{i}"
                paths = generate_audio(
                    pipe, None, unst_dir, device, 0.0, [prompt],
                    dry_run=dry_run, seed=seed_i,
                )
                if paths:
                    paths[0].rename(unsteered_path) if not dry_run else unsteered_path.touch()

            # Steered
            steered_path = concept_dir / f"pair_{i:02d}_steered.wav"
            if not steered_path.exists() or steered_path.stat().st_size == 0:
                st_dir = concept_dir / f"_tmp_st_{i}"
                paths = generate_audio(
                    pipe, sv, st_dir, device, alpha, [prompt],
                    dry_run=dry_run, seed=seed_i,
                )
                if paths:
                    paths[0].rename(steered_path) if not dry_run else steered_path.touch()

        log.info("    Wrote %d pairs to %s", n_pairs_per_concept, concept_dir)

    # Concept algebra pairs: piano vs piano-drums, jazz vs jazz-drums, etc.
    algebra_pairs = [
        ("piano", "piano", 1.0, "drums", -1.0),
        ("jazz", "jazz", 1.0, "drums", -1.0),
        ("mood", "mood", 1.0, "tempo", -1.0),
        ("tempo", "tempo", 1.0, "piano", -1.0),
        ("piano+jazz", "piano", 1.0, "jazz", 1.0),
    ]
    algebra_dir = human_eval_dir / "algebra"
    algebra_dir.mkdir(parents=True, exist_ok=True)

    for idx, (label, ca, wa, cb, wb) in enumerate(algebra_pairs):
        log.info("  [exp6] Algebra pair %d: %s", idx, label)
        n_p = DRY_N_PAIRS if dry_run else N_PAIRS
        sv_dir_a = compute_caa_vector(pipe, ca, device, vectors_dir, n_p, dry_run=dry_run)
        sv_dir_b = compute_caa_vector(pipe, cb, device, vectors_dir, n_p, dry_run=dry_run)
        sv_a = load_sv(sv_dir_a) if sv_dir_a else {}
        sv_b = load_sv(sv_dir_b) if sv_dir_b else {}
        combined = combine_sv_dicts([sv_a, sv_b], [wa, wb])

        prompt = test_prompts[idx % len(test_prompts)]
        paths = generate_audio(
            pipe, combined, algebra_dir / f"pair_{idx:02d}_{label}", device, 1.0,
            [prompt], dry_run=dry_run, seed=42 + idx,
        )
        if paths and paths[0].exists():
            dest = algebra_dir / f"pair_{idx:02d}_{label}.wav"
            if not dest.exists():
                if dry_run:
                    dest.touch()
                else:
                    paths[0].rename(dest)

    # Write instructions
    _write_human_eval_instructions(human_eval_dir, concepts)
    log.info("EXP 6 done. Human eval materials: %s", human_eval_dir)


def _write_human_eval_instructions(human_eval_dir: Path, concepts: List[str]) -> None:
    """Write instructions.md for human evaluators."""
    instructions = f"""# Human Evaluation — TADA Audio Steering Study

## Overview

This listening test evaluates the effectiveness of activation steering in audio
diffusion models (TADA paper, ISMIR 2026). You will hear pairs of audio clips
and rate them on perceptual quality and concept adherence.

## Concepts Evaluated

{", ".join(concepts)}

## Task 1: Single-Concept Steering

For each pair of audio clips (unsteered vs steered):
1. Listen to both clips completely.
2. Rate each clip on:
   - **Concept adherence** (1–5): Does the audio clearly exhibit the concept?
   - **Audio quality** (1–5): Is the audio pleasant and free of artifacts?
3. Indicate which clip better matches the concept.

## Task 2: Concept Algebra Pairs

For each algebra pair:
1. Listen to both clips.
2. Rate how well each achieves the intended concept combination.
3. Note any artifacts introduced by the algebra operation.

## File Naming

- `pair_00_unsteered.wav` — baseline generation (no steering)
- `pair_00_steered.wav` — steered at α=1.0
- `algebra/pair_00_<label>.wav` — concept algebra result

## Notes

- All clips were generated with the same text prompt and seed.
- Only the activation steering differed between conditions.
- Rate independently; do not let the file order influence your ratings.

---
Generated by TADA paper experiment runner.
"""
    (human_eval_dir / "instructions.md").write_text(instructions)
    log.info("Wrote human eval instructions: %s", human_eval_dir / "instructions.md")


# ---------------------------------------------------------------------------
# EXPERIMENT 7 — Ablation studies
# ---------------------------------------------------------------------------


def run_exp7(
    pipe,
    device: str,
    vectors_dir: Path,
    audio_base: Path,
    results_dir: Path,
    n_test: int = N_TEST,
    dry_run: bool = False,
) -> List[Dict]:
    """Experiment 7: Ablation studies.

    7a: Layer ablation — test steering at tf4, tf5, tf6, tf7, tf8, tf6+tf7
    7b: Prompt pair count ablation — 5, 25, 50, 100, 256 pairs
    Concept: piano
    Output: results/paper/exp7_ablations.csv
            results/paper/figures/exp7_npairs_curve.png
    """
    log.info("=== EXP 7: Ablation Studies ===")
    clap_mod = _try_import_clap()
    test_prompts = get_test_prompts(n_test)
    concept = "piano"
    alpha = 1.0
    rows: List[Dict] = []

    # --- 7a: Layer ablation ---
    layer_configs = [
        ("tf4", ["tf4"]),
        ("tf5", ["tf5"]),
        ("tf6", ["tf6"]),
        ("tf7", ["tf7"]),
        ("tf8", ["tf8"]),
        ("tf6+tf7", ["tf6", "tf7"]),
    ]
    n_pairs = DRY_N_PAIRS if dry_run else N_PAIRS

    # Use default vectors (tf6+tf7) for layer ablation testing
    sv_dir = compute_caa_vector(pipe, concept, device, vectors_dir, n_pairs, dry_run=dry_run)
    sv = load_sv(sv_dir) if sv_dir else None

    for layer_label, layer_list in layer_configs:
        log.info("  [exp7a] layers=%s", layer_label)
        out_dir = audio_base / "exp7" / "layers" / layer_label
        paths = generate_audio(
            pipe, sv, out_dir, device, alpha, test_prompts,
            layers=layer_list, dry_run=dry_run,
        )
        clap_scores = compute_clap_scores(paths, test_prompts, clap_mod, dry_run)
        valid = [s for s in clap_scores if s >= 0]
        clap_mean = float(np.mean(valid)) if valid else -1.0
        rows.append({
            "ablation_type": "layer",
            "value": layer_label,
            "piano_clap": clap_mean,
        })
        log.info("    layers=%-8s  CLAP=%.4f", layer_label, clap_mean)

    # --- 7b: Prompt pair count ablation ---
    pair_counts = [5, 25, 50, 100, 256]
    if dry_run:
        pair_counts = [p for p in pair_counts if p <= DRY_N_PAIRS * 2]
        pair_counts = pair_counts if pair_counts else [DRY_N_PAIRS]

    for n_p in pair_counts:
        log.info("  [exp7b] n_pairs=%d", n_p)
        # Compute fresh vectors for this pair count (bypass cache to test count ablation)
        ablation_vectors_dir = vectors_dir / f"ablation_npairs"
        sv_dir_np = ablation_vectors_dir / f"ace_{concept}_n{n_p}_passes2_allTrue"
        sv_dir_np.mkdir(parents=True, exist_ok=True)
        sv_path_np = sv_dir_np / "sv.pkl"
        config_path_np = sv_dir_np / "config.json"

        if not sv_path_np.exists() or not config_path_np.exists():
            compute_caa_vector(
                pipe, concept, device, ablation_vectors_dir,
                n_p, dry_run=dry_run,
            )
            # rename the default cache entry to our n_p specific dir
            default_dir = _sv_cache_path(ablation_vectors_dir, concept)
            if default_dir.exists() and default_dir != sv_dir_np:
                sv_ref = load_sv(default_dir)
                if sv_ref is not None:
                    with open(sv_path_np, "wb") as f:
                        pickle.dump(sv_ref, f)
                    _write_json(config_path_np, {"concept": concept, "n_pairs": n_p})

        sv_np = load_sv(sv_dir_np)
        out_dir = audio_base / "exp7" / "npairs" / f"n{n_p}"
        paths = generate_audio(
            pipe, sv_np, out_dir, device, alpha, test_prompts, dry_run=dry_run
        )
        clap_scores = compute_clap_scores(paths, test_prompts, clap_mod, dry_run)
        valid = [s for s in clap_scores if s >= 0]
        clap_mean = float(np.mean(valid)) if valid else -1.0
        rows.append({
            "ablation_type": "npairs",
            "value": str(n_p),
            "piano_clap": clap_mean,
        })
        log.info("    n_pairs=%-4d  CLAP=%.4f", n_p, clap_mean)

    out_path = results_dir / "exp7_ablations.csv"
    write_csv(out_path, rows, ["ablation_type", "value", "piano_clap"])
    _plot_exp7_npairs(rows, results_dir / "figures" / "exp7_npairs_curve.png")
    log.info("EXP 7 done. Results: %s", out_path)
    return rows


def _plot_exp7_npairs(rows: List[Dict], out_path: Path) -> None:
    """Line plot: CLAP vs number of prompt pairs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt = _init_matplotlib()
        npairs_rows = [r for r in rows if r["ablation_type"] == "npairs"]
        npairs_rows.sort(key=lambda r: int(r["value"]))
        xs = [int(r["value"]) for r in npairs_rows]
        ys = [float(r["piano_clap"]) for r in npairs_rows]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, marker="o", color="#4C72B0")
        ax.set_xlabel("Number of prompt pairs used for CAA vector computation")
        ax.set_ylabel("CLAP score (piano concept)")
        ax.set_title("Effect of Prompt Pair Count on Steering Quality (Piano)")
        ax.grid(True, alpha=0.3)
        if len(xs) > 1:
            ax.set_xscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels([str(x) for x in xs])
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved exp7 npairs figure: %s", out_path)
    except Exception as exc:
        log.warning("Could not create exp7 figure: %s", exc)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary_table(results_dir: Path) -> None:
    """Read all CSV results and print a human-readable summary table."""
    print("\n" + "=" * 70)
    print("TADA PAPER EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)

    csv_files = [
        ("exp1_baseline.csv",            "Exp 1 — Baseline Steering"),
        ("exp2_timestep_commitment.csv", "Exp 2 — Timestep Commitment"),
        ("exp3_concept_algebra.csv",     "Exp 3 — Concept Algebra"),
        ("exp4_gramschmidt_ablation.csv","Exp 4 — GS Ablation"),
        ("exp5_schedules.csv",           "Exp 5 — Schedule Comparison"),
        ("exp7_ablations.csv",           "Exp 7 — Ablations"),
    ]

    for fname, title in csv_files:
        path = results_dir / fname
        if not path.exists():
            print(f"\n{title}: NOT RUN")
            continue
        print(f"\n{title}")
        print("-" * 50)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            print("  (empty)")
            continue
        fields = list(rows[0].keys())
        # Print header
        header = "  " + "  ".join(f"{fn[:14]:<14}" for fn in fields)
        print(header)
        print("  " + "-" * (16 * len(fields)))
        for row in rows[:20]:  # cap at 20 lines per table
            line = "  " + "  ".join(f"{str(row.get(fn,''))[:14]:<14}" for fn in fields)
            print(line)
        if len(rows) > 20:
            print(f"  ... ({len(rows) - 20} more rows)")

    print("\n" + "=" * 70)
    print(f"Results saved to: {results_dir}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TADA ISMIR 2026 paper experiment runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--experiments", "-e",
        nargs="+",
        type=int,
        default=list(range(1, 8)),
        metavar="N",
        help="Which experiments to run (1–7). Default: all.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Fast synthetic run without GPU or model inference.",
    )
    p.add_argument(
        "--concepts",
        nargs="+",
        default=ALL_CONCEPTS,
        metavar="CONCEPT",
        help="Concepts to use. Default: piano tempo mood drums jazz.",
    )
    p.add_argument(
        "--n-pairs",
        type=int,
        default=N_PAIRS,
        help="Number of prompt pairs per concept for CAA computation.",
    )
    p.add_argument(
        "--n-test",
        type=int,
        default=N_TEST,
        help="Number of test prompts for evaluation.",
    )
    p.add_argument(
        "--alpha-values",
        nargs="+",
        type=float,
        default=ALPHA_VALUES,
        metavar="ALPHA",
        help="Alpha values for exp1 sweep.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Where to write results. Default: <repo>/results/paper.",
    )
    p.add_argument(
        "--infer-steps",
        type=int,
        default=INFER_STEPS,
        help="Number of diffusion inference steps.",
    )
    p.add_argument(
        "--audio-duration",
        type=float,
        default=AUDIO_DURATION,
        help="Audio clip duration in seconds.",
    )
    return p.parse_args()


def run_clap_diagnostic() -> None:
    """Startup diagnostic: generate 3 s of silence, score it with CLAP.

    Prints the score (should be a small positive float for 'piano music').
    If it returns -1.0, the full exception is printed so the root cause is visible.
    """
    log.info("=== CLAP diagnostic ===")
    diag_path = Path("/tmp/test_clap_diag.wav")
    try:
        import torchaudio  # type: ignore
        silence = torch.zeros(1, 44100 * 3)
        torchaudio.save(str(diag_path), silence, 44100)
        log.info("  Wrote 3 s silence to %s", diag_path)
    except Exception as e:
        log.warning("  Could not write diagnostic WAV: %s", e)
        return

    clap_mod = _try_import_clap()
    if clap_mod is None:
        log.warning("  CLAP not available — skipping diagnostic.")
        return

    scores = compute_clap_scores([diag_path], ["piano music"], clap_mod, dry_run=False)
    score = scores[0] if scores else -1.0
    if score >= 0:
        log.info("  CLAP diagnostic score (silence vs 'piano music') = %.4f  [OK]", score)
    else:
        log.warning(
            "  CLAP diagnostic returned -1.0 for silence — check exceptions above. "
            "Likely cause: wrong sample rate (need 48 kHz) or missing model weights."
        )


def _preflight(dry_run: bool) -> Tuple[str, Path]:
    """Check environment. Returns (device, model_path)."""
    model_path = Path(os.environ.get("ACEMODEL_PATH", "/workspace/ACE-Step"))
    device_env = os.environ.get("TADA_DEVICE", "")
    device = device_env if device_env else ("cuda" if torch.cuda.is_available() else "cpu")

    log.info("Python    : %s.%s.%s", *sys.version_info[:3])
    log.info("Device    : %s", device)
    if device == "cuda":
        log.info("GPU       : %s (%.1f GB VRAM)",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)
    log.info("Model path: %s", model_path)

    if not dry_run:
        if device == "cpu":
            log.warning("No CUDA device — consider using --dry-run. Proceeding on CPU.")
        if not model_path.exists():
            log.error(
                "ACE-Step weights not found at %s. "
                "Set ACEMODEL_PATH or use --dry-run.",
                model_path,
            )
            sys.exit(1)
        py = sys.version_info
        if py >= (3, 13):
            log.error(
                "Python %d.%d.%d detected — ACE-Step requires Python < 3.13.",
                *py[:3],
            )
            sys.exit(1)

    return device, model_path


def main() -> None:
    args = parse_args()

    # Override dry-run counts
    n_pairs = DRY_N_PAIRS if args.dry_run else args.n_pairs
    n_test = DRY_N_TEST if args.dry_run else args.n_test
    alpha_values = DRY_ALPHA_VALUES if args.dry_run else args.alpha_values

    workdir = Path(os.environ.get("TADA_WORKDIR", str(_REPO_ROOT / "outputs")))
    results_dir = args.results_dir or RESULTS_DIR
    vectors_dir = workdir / "vectors"
    audio_base = workdir / "audio"

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    audio_base.mkdir(parents=True, exist_ok=True)

    log.info("=== TADA ISMIR 2026 Paper Experiment Runner ===")
    log.info("Experiments : %s", args.experiments)
    log.info("Concepts    : %s", args.concepts)
    log.info("Dry-run     : %s", args.dry_run)
    log.info("Results dir : %s", results_dir)
    log.info("Vectors dir : %s", vectors_dir)
    log.info("n_pairs     : %d", n_pairs)
    log.info("n_test      : %d", n_test)

    device, model_path = _preflight(args.dry_run)

    # CLAP diagnostic — run before anything else so failures surface immediately
    if not args.dry_run:
        run_clap_diagnostic()

    # Load pipeline once (reused across all experiments)
    pipe = None
    if not args.dry_run:
        pipe = load_ace_pipeline(device=device, model_path=model_path)

    t_start = time.time()
    exp_results: Dict[int, Any] = {}

    if 1 in args.experiments:
        exp_results[1] = run_exp1(
            pipe, device, args.concepts, vectors_dir, audio_base, results_dir,
            n_pairs=n_pairs, n_test=n_test, alpha_values=alpha_values,
            dry_run=args.dry_run,
        )

    if 2 in args.experiments:
        exp_results[2] = run_exp2(
            pipe, device, args.concepts, vectors_dir, audio_base, results_dir,
            n_test=n_test, dry_run=args.dry_run,
        )

    if 3 in args.experiments:
        exp_results[3] = run_exp3(
            pipe, device, vectors_dir, audio_base, results_dir,
            n_test=n_test, dry_run=args.dry_run,
        )

    if 4 in args.experiments:
        exp_results[4] = run_exp4(
            pipe, device, vectors_dir, audio_base, results_dir,
            n_test=n_test, dry_run=args.dry_run,
        )

    if 5 in args.experiments:
        exp_results[5] = run_exp5(
            pipe, device, args.concepts, vectors_dir, audio_base, results_dir,
            n_test=n_test, dry_run=args.dry_run,
        )

    if 6 in args.experiments:
        run_exp6(
            pipe, device, args.concepts, vectors_dir, results_dir,
            n_pairs_per_concept=3 if args.dry_run else 10,
            dry_run=args.dry_run,
        )

    if 7 in args.experiments:
        exp_results[7] = run_exp7(
            pipe, device, vectors_dir, audio_base, results_dir,
            n_test=n_test, dry_run=args.dry_run,
        )

    elapsed = time.time() - t_start
    log.info("All experiments completed in %.1f s (%.1f min).", elapsed, elapsed / 60)
    print_summary_table(results_dir)


if __name__ == "__main__":
    main()
