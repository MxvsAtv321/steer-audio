#!/usr/bin/env python3
"""
Phase 5.1 — First Real Runs on ACE-Step (CUDA GPU required).

Runs the full TADA Phase 5.1 pipeline:
  1. Compute real CAA steering vectors for 5 concepts (piano, tempo, mood,
     female_vocals, drums) using prompt pairs from steer_prompts.py.
  2. Generate steered audio clips at alpha ∈ {0, 0.5, 1.0, 2.0} for each concept.
  3. Compute per-alpha MUQ-T CLAP similarity scores.
  4. Run `experiments/sae_scaling.py --preset-real-small` with real activations.
  5. Run `experiments/vector_geometry.py` with real vectors.
  6. Write results to experiments/results/ and results/eval/{concept}/.
  7. Update docs/results_summary.md with real numbers.

Usage (on RunPod A40 or any CUDA machine with ACE-Step weights):
  export TADA_WORKDIR=/workspace/steer-audio/outputs
  export ACEMODEL_PATH=/workspace/ACE-Step
  python scripts/run_phase5_1.py [--concepts piano tempo mood] [--dry-run]

Requirements:
  - Python 3.10–3.12 (ACE-Step / spacy incompatible with 3.13)
  - CUDA GPU
  - /workspace/ACE-Step/ containing ACE-Step model weights
  - pip install -r requirements/requirements_1.txt
  - pip install -r requirements/requirements_2.txt --no-deps

Reference: TADA roadmap Prompt 5.1 (arXiv 2602.11910).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repo path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAE_ROOT = _REPO_ROOT / "sae"
_SRC_ROOT = _REPO_ROOT / "src"
_ACE_ROOT = _SRC_ROOT / "models" / "ace_step"

for _p in [str(_REPO_ROOT), str(_SAE_ROOT), str(_SAE_ROOT / "sae_src"), str(_SRC_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ACE-Step submodule path (may be a git submodule or a checked-out copy)
_ACE_SUBMODULE = _ACE_ROOT / "ACE"
if _ACE_SUBMODULE.exists() and str(_ACE_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(_ACE_SUBMODULE))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONCEPTS_5 = ["piano", "tempo", "mood", "female_vocals", "drums"]

# Keep small for Phase 5.1 first runs — use 5 prompt pairs (out of 50) so
# vector computation takes ~3 min/concept on an A40.
N_PROMPT_PAIRS: int = 5

# Audio duration in seconds — short clips to fit in memory and time budget.
AUDIO_DURATION: float = 12.0

# Inference steps — 30 matches the eval scripts default.
INFER_STEPS: int = 30

# Alpha values for steering evaluation.
ALPHA_VALUES: list[float] = [0.0, 0.5, 1.0, 2.0]

# Functional layers per TADA paper (§ 4.1).
FUNCTIONAL_LAYERS: list[str] = ["tf6", "tf7"]

# Number of test prompts for audio generation per alpha sweep.
N_TEST_PROMPTS: int = 3

TEST_PROMPTS = [
    "an upbeat electronic track with synths",
    "a calm acoustic guitar melody",
    "a fast-paced jazz piano trio",
]

SAMPLE_RATE: int = 44100


# ---------------------------------------------------------------------------
# Step 0 — preflight checks
# ---------------------------------------------------------------------------


def check_environment() -> dict[str, Any]:
    """Verify CUDA, ACE-Step availability, and key paths."""
    info: dict[str, Any] = {}

    # CUDA
    info["cuda"] = torch.cuda.is_available()
    info["device"] = "cuda" if info["cuda"] else "cpu"
    if info["cuda"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        info["gpu_name"] = "n/a"
        info["vram_gb"] = 0.0
    log.info("GPU: %s  VRAM: %.1f GB  CUDA: %s", info["gpu_name"], info["vram_gb"], info["cuda"])

    # ACE-Step model weights
    model_path = Path(os.environ.get("ACEMODEL_PATH", "/workspace/ACE-Step"))
    info["model_path"] = model_path
    info["model_exists"] = model_path.exists()
    if not info["model_exists"]:
        log.warning("ACE-Step model not found at %s; set ACEMODEL_PATH env var.", model_path)

    # Python version (ACE-Step requires < 3.13)
    py_ver = sys.version_info
    info["python_version"] = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    info["python_ok"] = py_ver < (3, 13)
    if not info["python_ok"]:
        log.error(
            "Python %s detected — ACE-Step requires Python < 3.13 due to spacy dependency.",
            info["python_version"],
        )

    return info


# ---------------------------------------------------------------------------
# Step 1 — load ACE-Step pipeline
# ---------------------------------------------------------------------------


def load_ace_pipeline(device: str, model_path: Path):
    """Load SimpleACEStepPipeline from the local checkpoint."""
    from src.models.ace_step.pipeline_ace import SimpleACEStepPipeline  # type: ignore

    log.info("Loading ACE-Step pipeline from %s ...", model_path)
    t0 = time.time()
    pipe = SimpleACEStepPipeline(device=device)
    pipe.load()
    log.info("ACE-Step loaded in %.1f s", time.time() - t0)
    return pipe


# ---------------------------------------------------------------------------
# Step 2 — compute CAA steering vectors
# ---------------------------------------------------------------------------


def compute_caa_vectors(
    pipe,
    concept: str,
    device: str,
    out_dir: Path,
    n_pairs: int = N_PROMPT_PAIRS,
    infer_steps: int = INFER_STEPS,
    audio_duration: float = AUDIO_DURATION,
    dry_run: bool = False,
) -> Path:
    """Compute and save CAA steering vectors for *concept*.

    Returns the directory where vectors were saved.
    """
    from sae_src.configs.steer_prompts import CONCEPT_TO_PROMPTS  # type: ignore
    from src.models.ace_step.ace_steering.controller import (  # type: ignore
        VectorStore,
        compute_num_cfg_passes,
        register_vector_control,
    )

    save_dir = out_dir / f"ace_{concept}_passes2_allTrue"
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = save_dir / "config.json"
    if config_path.exists():
        log.info("[%s] CAA vectors already exist at %s — skipping.", concept, save_dir)
        return save_dir

    log.info("[%s] Computing CAA steering vectors (%d prompt pairs) ...", concept, n_pairs)

    prompts_neg, prompts_pos, lyrics = CONCEPT_TO_PROMPTS[concept]()
    prompts_pos = prompts_pos[:n_pairs]
    prompts_neg = prompts_neg[:n_pairs]

    num_cfg_passes = compute_num_cfg_passes(
        guidance_scale_text=0.0,
        guidance_scale_lyric=0.0,
    )

    if dry_run:
        log.info("[dry-run] Would compute %d prompt pairs for '%s'.", n_pairs, concept)
        # Write placeholder config.
        _write_json(
            config_path,
            {
                "concept": concept,
                "n_pairs": n_pairs,
                "dry_run": True,
                "layers": FUNCTIONAL_LAYERS,
            },
        )
        return save_dir

    pos_vectors: list[Any] = []
    neg_vectors: list[Any] = []

    for prompt_pos, prompt_neg in zip(prompts_pos, prompts_neg):
        for polarity, prompt in [("pos", prompt_pos), ("neg", prompt_neg)]:
            controller = VectorStore(
                device=device,
                save_only_cond=True,
                num_cfg_passes=num_cfg_passes,
            )
            controller.steer = False
            register_vector_control(pipe.ace_step_transformer, controller)

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
                pos_vectors.append(controller.vector_store)
            else:
                neg_vectors.append(controller.vector_store)
            controller.reset()

    # Compute mean difference (CAA).
    steering_vectors: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    all_step_keys = list(pos_vectors[0].keys())
    layer_names = list(pos_vectors[0][all_step_keys[0]].keys())

    for step_key in all_step_keys:
        for layer_name in layer_names:
            pos_avg = np.mean(
                [pos_vectors[i][step_key][layer_name][0] for i in range(len(pos_vectors))],
                axis=0,
            )
            neg_avg = np.mean(
                [neg_vectors[i][step_key][layer_name][0] for i in range(len(neg_vectors))],
                axis=0,
            )
            sv = pos_avg - neg_avg
            norm = np.linalg.norm(sv)
            if norm > 0:
                sv = sv / norm
            steering_vectors[step_key][layer_name].append(sv)

    with open(save_dir / "sv.pkl", "wb") as f:
        pickle.dump(dict(steering_vectors), f)
    with open(save_dir / "pos_vectors.pkl", "wb") as f:
        pickle.dump(pos_vectors, f)
    with open(save_dir / "neg_vectors.pkl", "wb") as f:
        pickle.dump(neg_vectors, f)

    _write_json(
        config_path,
        {
            "concept": concept,
            "n_pairs": n_pairs,
            "audio_duration": audio_duration,
            "infer_steps": infer_steps,
            "device": device,
            "layers": FUNCTIONAL_LAYERS,
        },
    )
    log.info("[%s] Vectors saved to %s", concept, save_dir)
    return save_dir


# ---------------------------------------------------------------------------
# Step 3 — generate steered audio
# ---------------------------------------------------------------------------


def generate_steered_audio(
    pipe,
    concept: str,
    sv_dir: Path,
    out_dir: Path,
    device: str,
    alphas: list[float] = ALPHA_VALUES,
    test_prompts: list[str] = TEST_PROMPTS,
    infer_steps: int = INFER_STEPS,
    audio_duration: float = AUDIO_DURATION,
    layers: list[str] = FUNCTIONAL_LAYERS,
    steer_mode: str = "cond_only",
    dry_run: bool = False,
) -> dict[float, list[Path]]:
    """Generate steered audio for *concept* at each alpha value.

    Returns a dict mapping alpha → list of generated WAV paths.
    """
    from src.models.ace_step.ace_steering.controller import (  # type: ignore
        VectorStore,
        compute_num_cfg_passes,
        register_vector_control,
    )

    out_paths: dict[float, list[Path]] = {}

    if dry_run:
        for alpha in alphas:
            alpha_dir = out_dir / f"alpha_{alpha}"
            alpha_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for p_idx in range(len(test_prompts)):
                wav = alpha_dir / f"p{p_idx}.wav"
                wav.touch()
                paths.append(wav)
            out_paths[alpha] = paths
        log.info("[dry-run] Skipped audio generation for concept=%s", concept)
        return out_paths

    # Load pre-computed SVs.
    sv_path = sv_dir / "sv.pkl"
    if not sv_path.exists():
        log.error("No sv.pkl found at %s — skipping audio generation.", sv_path)
        return {}

    with open(sv_path, "rb") as f:
        steering_vectors = pickle.load(f)

    num_cfg_passes = compute_num_cfg_passes(0.0, 0.0)

    for alpha in alphas:
        alpha_dir = out_dir / f"alpha_{alpha}"
        alpha_dir.mkdir(parents=True, exist_ok=True)
        out_paths[alpha] = []

        for p_idx, prompt in enumerate(test_prompts):
            wav_path = alpha_dir / f"p{p_idx}.wav"
            if wav_path.exists():
                log.info("  [skip] %s already exists.", wav_path.name)
                out_paths[alpha].append(wav_path)
                continue

            controller = VectorStore(
                device=device,
                save_only_cond=(steer_mode == "cond_only"),
                num_cfg_passes=num_cfg_passes,
            )
            controller.steer = True
            controller.alpha = alpha
            # Pass the full steering_vectors dict (all steps × all layers).
            # Layer filtering is handled by explicit_layers in register_vector_control,
            # which only hooks the target layers so the controller is never called
            # with a layer name that isn't present in steering_vectors.
            controller.steering_vectors = steering_vectors

            register_vector_control(
                pipe.ace_step_transformer,
                controller,
                explicit_layers=layers,
            )

            audio_output = pipe.generate(
                prompt=prompt,
                audio_duration=audio_duration,
                infer_step=infer_steps,
                manual_seed=42,
                return_type="pt",
                use_erg_lyric=False,
                guidance_scale_text=0.0,
                guidance_scale_lyric=0.0,
                guidance_scale=3.0,
                guidance_interval=1.0,
                guidance_interval_decay=0.0,
            )
            controller.reset()

            import torchaudio  # type: ignore
            torchaudio.save(str(wav_path), audio_output.cpu(), SAMPLE_RATE)
            log.info("  Saved %s (alpha=%.1f)", wav_path.name, alpha)
            out_paths[alpha].append(wav_path)

    return out_paths


# ---------------------------------------------------------------------------
# Step 4 — CLAP evaluation
# ---------------------------------------------------------------------------


def evaluate_clap(
    concept: str,
    audio_paths_by_alpha: dict[float, list[Path]],
    test_prompts: list[str],
    eval_dir: Path,
    dry_run: bool = False,
) -> dict[float, float]:
    """Compute mean CLAP alignment per alpha.

    Falls back to stub values if laion_clap is not available.
    Returns dict alpha → mean_clap_score.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    scores: dict[float, float] = {}

    if dry_run:
        # Stub scores increase linearly with alpha for demo purposes.
        for alpha in ALPHA_VALUES:
            scores[alpha] = 0.30 + 0.05 * alpha
        _write_clap_csv(eval_dir / "metrics.csv", scores)
        return scores

    try:
        import laion_clap  # type: ignore

        model_clap = laion_clap.CLAP_Module(enable_fusion=False)
        model_clap.load_ckpt()

        for alpha, paths in sorted(audio_paths_by_alpha.items()):
            clap_vals: list[float] = []
            for wav_path, prompt in zip(paths, test_prompts):
                if not wav_path.exists() or wav_path.stat().st_size == 0:
                    continue
                try:
                    import torchaudio  # type: ignore
                    waveform, sr = torchaudio.load(str(wav_path))
                    audio_data = waveform.mean(0).numpy()
                    audio_embed = model_clap.get_audio_embedding_from_data(
                        [audio_data], use_tensor=False
                    )
                    text_embed = model_clap.get_text_embedding([prompt])
                    cos = float(
                        np.dot(audio_embed[0], text_embed[0])
                        / (np.linalg.norm(audio_embed[0]) * np.linalg.norm(text_embed[0]) + 1e-8)
                    )
                    clap_vals.append(cos)
                except Exception as exc:
                    log.warning("CLAP eval failed for %s: %s", wav_path, exc)
            scores[alpha] = float(np.mean(clap_vals)) if clap_vals else -1.0
            log.info("[%s] alpha=%.1f  mean_CLAP=%.4f", concept, alpha, scores[alpha])

    except ImportError:
        log.warning("laion_clap not installed — writing stub CLAP scores.")
        for alpha in sorted(audio_paths_by_alpha):
            scores[alpha] = -1.0

    _write_clap_csv(eval_dir / "metrics.csv", scores)
    return scores


def _write_clap_csv(path: Path, scores: dict[float, float]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "clap"])
        writer.writeheader()
        for alpha, clap in sorted(scores.items()):
            writer.writerow({"alpha": alpha, "clap": f"{clap:.6f}"})
    log.info("Wrote CLAP metrics to %s", path)


# ---------------------------------------------------------------------------
# Step 5 — SAE scaling (real_small preset via subprocess)
# ---------------------------------------------------------------------------


def run_sae_scaling(
    workdir: Path,
    out_dir: Path,
    dry_run: bool = False,
) -> None:
    """Run sae_scaling.py with --preset-real-small if activation cache exists."""
    cache_dir = workdir / "cache" / "layer7"
    scaling_script = _REPO_ROOT / "experiments" / "sae_scaling.py"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cache_dir.exists():
        log.warning(
            "Activation cache not found at %s. "
            "Run cache_activations_runner_ace.py first, or use --dry-run for synthetic run.",
            cache_dir,
        )
        if not dry_run:
            return
        # Fall back to dry-run mode.
        cmd = [
            sys.executable,
            str(scaling_script),
            "--dry-run",
            "--full-grid",
            "--out-dir",
            str(out_dir),
        ]
    else:
        cmd = [
            sys.executable,
            str(scaling_script),
            "--preset-real-small",
            "--activation-cache",
            str(cache_dir),
            "--out-dir",
            str(out_dir),
        ]

    log.info("Running SAE scaling: %s", " ".join(cmd))
    if dry_run and cache_dir.exists():
        log.info("[dry-run] Would run SAE scaling with real cache at %s", cache_dir)
        return

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log.error("sae_scaling.py exited with code %d", result.returncode)
    else:
        log.info("SAE scaling complete. Results in %s", out_dir)


# ---------------------------------------------------------------------------
# Step 6 — vector geometry (subprocess)
# ---------------------------------------------------------------------------


def run_vector_geometry(
    vectors_dir: Path,
    out_dir: Path,
    dry_run: bool = False,
) -> None:
    """Run vector_geometry.py with the real vectors directory."""
    geometry_script = _REPO_ROOT / "experiments" / "vector_geometry.py"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not vectors_dir.exists() or not any(vectors_dir.glob("**/sv.pkl")):
        log.warning("No vectors found at %s — skipping geometry analysis.", vectors_dir)
        return

    cmd = [
        sys.executable,
        str(geometry_script),
        "--vectors-dir",
        str(vectors_dir),
        "--out-dir",
        str(out_dir),
    ]
    if dry_run:
        cmd.append("--dry-run")

    log.info("Running vector geometry: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log.error("vector_geometry.py exited with code %d", result.returncode)
    else:
        log.info("Vector geometry complete. Results in %s", out_dir)


# ---------------------------------------------------------------------------
# Step 7 — update docs/results_summary.md
# ---------------------------------------------------------------------------


def update_results_summary(
    concept_scores: dict[str, dict[float, float]],
    docs_dir: Path,
) -> None:
    """Append a 'Real Run Results' section to docs/results_summary.md."""
    summary_path = docs_dir / "results_summary.md"
    if not summary_path.exists():
        log.warning("docs/results_summary.md not found at %s — skipping update.", summary_path)
        return

    # Build the new section.
    lines = [
        "",
        "---",
        "",
        "## Real Run Results (Phase 5.1)",
        "",
        "Generated on ACE-Step with real model weights and CUDA GPU.",
        "",
        "### CLAP Alignment vs Alpha",
        "",
        "| Concept | α=0 | α=0.5 | α=1.0 | α=2.0 |",
        "|---------|-----|-------|-------|-------|",
    ]
    for concept, scores in sorted(concept_scores.items()):
        row = f"| {concept} "
        for alpha in [0.0, 0.5, 1.0, 2.0]:
            v = scores.get(alpha, -1.0)
            row += f"| {v:.3f} " if v >= 0 else "| n/a "
        row += "|"
        lines.append(row)

    lines += [
        "",
        "> CLAP scores computed with `laion_clap` on " + str(N_TEST_PROMPTS) + " test prompts.",
        "> α=0 is the unsteered baseline; higher α = stronger steering.",
        "",
    ]

    existing = summary_path.read_text()
    # Avoid duplicate section.
    if "## Real Run Results (Phase 5.1)" in existing:
        log.info("docs/results_summary.md already contains Phase 5.1 section — skipping update.")
        return

    with open(summary_path, "a") as f:
        f.write("\n".join(lines) + "\n")
    log.info("Updated %s with Phase 5.1 real run results.", summary_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 5.1 — First Real Runs on ACE-Step.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--concepts",
        nargs="+",
        default=CONCEPTS_5,
        metavar="CONCEPT",
        help="Concepts to run. Choose from: piano tempo mood female_vocals drums.",
    )
    p.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=ALPHA_VALUES,
        metavar="ALPHA",
        help="Alpha values for steering evaluation.",
    )
    p.add_argument(
        "--n-pairs",
        type=int,
        default=N_PROMPT_PAIRS,
        help="Number of prompt pairs per concept for CAA computation.",
    )
    p.add_argument(
        "--audio-duration",
        type=float,
        default=AUDIO_DURATION,
        help="Duration of each generated audio clip in seconds.",
    )
    p.add_argument(
        "--infer-steps",
        type=int,
        default=INFER_STEPS,
        help="Number of diffusion inference steps.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip actual model inference; write placeholder files.",
    )
    p.add_argument(
        "--skip-sae",
        action="store_true",
        help="Skip SAE scaling experiment.",
    )
    p.add_argument(
        "--skip-geometry",
        action="store_true",
        help="Skip vector geometry analysis.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    workdir = Path(os.environ.get("TADA_WORKDIR", str(_REPO_ROOT / "outputs")))
    model_path = Path(os.environ.get("ACEMODEL_PATH", "/workspace/ACE-Step"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("=== Phase 5.1 — First Real Runs ===")
    log.info("TADA_WORKDIR : %s", workdir)
    log.info("ACEMODEL_PATH: %s", model_path)
    log.info("device       : %s", device)
    log.info("concepts     : %s", args.concepts)
    log.info("alphas       : %s", args.alphas)
    log.info("dry_run      : %s", args.dry_run)

    # Preflight checks.
    env_info = check_environment()
    if not args.dry_run:
        if not env_info["cuda"]:
            log.error("No CUDA device detected — aborting. Use --dry-run for a test run.")
            sys.exit(1)
        if not env_info["model_exists"]:
            log.error(
                "ACE-Step weights not found at %s. "
                "Set ACEMODEL_PATH env var or use --dry-run.",
                model_path,
            )
            sys.exit(1)
        if not env_info["python_ok"]:
            log.error(
                "Python %s detected — ACE-Step requires Python < 3.13.",
                env_info["python_version"],
            )
            sys.exit(1)

    vectors_workdir = workdir / "vectors"
    audio_workdir = workdir / "audio"
    eval_root = _REPO_ROOT / "results" / "eval"
    experiments_results = _REPO_ROOT / "experiments" / "results"

    # Load pipeline once (reused for all concepts).
    pipe = None
    if not args.dry_run:
        pipe = load_ace_pipeline(device=device, model_path=model_path)

    concept_scores: dict[str, dict[float, float]] = {}

    for concept in args.concepts:
        log.info("\n--- Concept: %s ---", concept)

        # Step 2: Compute CAA vectors.
        sv_dir = compute_caa_vectors(
            pipe=pipe,
            concept=concept,
            device=device,
            out_dir=vectors_workdir,
            n_pairs=args.n_pairs,
            infer_steps=args.infer_steps,
            audio_duration=args.audio_duration,
            dry_run=args.dry_run,
        )

        # Step 3: Generate steered audio.
        audio_out = audio_workdir / concept
        audio_paths = generate_steered_audio(
            pipe=pipe,
            concept=concept,
            sv_dir=sv_dir,
            out_dir=audio_out,
            device=device,
            alphas=args.alphas,
            infer_steps=args.infer_steps,
            audio_duration=args.audio_duration,
            dry_run=args.dry_run,
        )

        # Step 4: CLAP evaluation.
        eval_dir = eval_root / concept
        scores = evaluate_clap(
            concept=concept,
            audio_paths_by_alpha=audio_paths,
            test_prompts=TEST_PROMPTS,
            eval_dir=eval_dir,
            dry_run=args.dry_run,
        )
        concept_scores[concept] = scores

    # Step 5: SAE scaling.
    if not args.skip_sae:
        run_sae_scaling(
            workdir=workdir,
            out_dir=experiments_results / "scaling",
            dry_run=args.dry_run,
        )

    # Step 6: Vector geometry.
    if not args.skip_geometry:
        run_vector_geometry(
            vectors_dir=vectors_workdir,
            out_dir=experiments_results / "geometry",
            dry_run=args.dry_run,
        )

    # Step 7: Update results_summary.md.
    update_results_summary(
        concept_scores=concept_scores,
        docs_dir=_REPO_ROOT / "docs",
    )

    # Print summary.
    log.info("\n=== Phase 5.1 Summary ===")
    log.info("%-20s  %s", "Concept", "  ".join(f"α={a}" for a in args.alphas))
    log.info("-" * 60)
    for concept, scores in sorted(concept_scores.items()):
        row = "  ".join(
            f"{scores.get(a, -1.0):6.3f}" if scores.get(a, -1.0) >= 0 else "   n/a"
            for a in args.alphas
        )
        log.info("%-20s  %s", concept, row)

    log.info("\nOutput directories:")
    log.info("  Steering vectors : %s", vectors_workdir)
    log.info("  Steered audio    : %s", audio_workdir)
    log.info("  Eval CSV/plots   : %s", eval_root)
    log.info("  SAE scaling      : %s", experiments_results / "scaling")
    log.info("  Vector geometry  : %s", experiments_results / "geometry")
    log.info("  Docs summary     : %s", _REPO_ROOT / "docs" / "results_summary.md")
    log.info("\nDone.")


if __name__ == "__main__":
    main()
