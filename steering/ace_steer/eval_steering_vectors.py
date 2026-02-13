"""
Evaluate steering vectors for ACE-Step model.

This script generates steered audio and evaluates using MUQ-T similarity.

Usage:
    python eval_steering_vectors.py \
        --sv_path steering_vectors/ace_instrument_pospiano_negNone_passes2_allTrue \
        --concept piano \
        --layers tf7 \
        --steer_mode cond_only
"""

import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
from fire import Fire
from tqdm import tqdm

WORKDIR_PATH = "<WORKDIR_PATH>"
sys.path.append(f"{WORKDIR_PATH}")
sys.path.append(f"{WORKDIR_PATH}/src/models/ace_step/ACE")
sys.path.append(f"{WORKDIR_PATH}/sae")

from sae_src.configs.eval import CONCEPT_TO_EVAL_PROMPTS

from editing.eval_medley import get_mulan
from src.models.ace_step.ace_steering.controller import compute_num_cfg_passes
from src.models.ace_step.steering_ace import SteeredACEStepPipeline
from steering.ace_steer.prompts import LYRICS, NO_LYRICS, TEST_PROMPTS

GENERATION_SEED = 2115
GUIDANCE_SCALE = 5.0
AUDIO_LENGTH_IN_S = 30.0
NUM_INFERENCE_STEPS = 30

MULTIPLIERS = [
    -100.0,
    -90.0,
    -80.0,
    -70.0,
    -60.0,
    -50.0,
    -40.0,
    -30.0,
    -20.0,
    -10.0,
    0.0,
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    100.0,
]


# Layer configurations
LAYER_CONFIGS = {
    "all": [f"tf{i}" for i in range(24)],
    "tf6": ["tf6"],
    "tf7": ["tf7"],
    "tf6tf7": ["tf6", "tf7"],
    "no_tf6tf7": [f"tf{i}" for i in range(24) if i not in [6, 7]],
}


def load_sv_config(config_path):
    """Load steering vector config from json file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def eval_audios(save_dir_before_alphas, test_prompts, sample_rate, eval_prompts):
    """Evaluate generated audios using MUQ-T similarity."""
    if not isinstance(eval_prompts, list):
        eval_prompts = [eval_prompts]

    for idx_ep, eval_prompt in enumerate(eval_prompts):
        all_dfs = []
        for alpha in MULTIPLIERS:
            audios = [
                torchaudio.load(save_dir_before_alphas + f"/alpha_{alpha}" + f"/p{p_idx}.wav")[0]
                for p_idx in range(len(test_prompts))
            ]
            prompts = [eval_prompt] * len(audios)
            srs = [sample_rate] * len(audios)
            muqt_df_alpha = get_mulan(prompts, audios, srs, torch.device("cuda"), verbose=False)
            muqt_df_alpha["alpha"] = [alpha] * len(test_prompts)
            muqt_df_alpha["p_idx"] = list(range(len(test_prompts)))
            all_dfs.append(muqt_df_alpha)
        all_dfs = pd.concat(all_dfs)
        summary = all_dfs.groupby("alpha")["muqt_sim_p0"].agg(["mean", "std"])

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["std"],
            fmt="-o",
            capsize=5,
            color="#009FB7",
            ecolor="#AAA",
            linewidth=2,
            markersize=7,
        )
        ax.set_xlabel(r"$\alpha$", fontsize=14)
        ax.set_ylabel("MUQ-T Similarity", fontsize=14)
        pth_splitted = save_dir_before_alphas.split("/")
        ax.set_title(
            f'Similarity to prompt "{eval_prompt}" ({pth_splitted[-2]}, {pth_splitted[-1]})',
            fontsize=16,
            fontweight="bold",
        )
        ax.axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        ax.xaxis.labelpad = -10

        plt.xticks(summary.index, rotation=45, ha="right", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_dir_before_alphas + f"/muqt_sim_p{idx_ep}.png", dpi=150)
        plt.close()
        summary.to_csv(save_dir_before_alphas + f"/summary_p{idx_ep}.csv", index=True)

        print(f"  Saved plot to {save_dir_before_alphas}/muqt_sim_p{idx_ep}.png")


def main(
    sv_path: str,
    concept: str,
    layers: str = "tf7",
    steer_mode: str = "cond_only",
    save_dir: str = None,
):
    """
    Evaluate steering vectors for ACE-Step model.

    Args:
        sv_path: Path to steering vectors directory (e.g., steering_vectors/ace_instrument_pospiano_negNone_passes2_allTrue)
        concept: Concept name for evaluation prompts (e.g., 'piano', 'mood', 'tempo', 'female_vocals', 'drums')
        layers: Which layers to steer. Options: 'all', 'tf6', 'tf7', 'tf6tf7', 'no_tf6tf7'
        steer_mode: How to apply steering vectors. Options: 'cond_only', 'separate', 'both_cond', 'uncond_only', 'uncond_for_cond', 'both_uncond'
        save_dir: Directory to save outputs. If None, auto-generated based on parameters.
    """
    # Validate layers argument
    if layers not in LAYER_CONFIGS:
        raise ValueError(f"Unknown layers: {layers}. Available: {list(LAYER_CONFIGS.keys())}")

    layers_to_steer = LAYER_CONFIGS[layers]

    # Load steering vectors and config
    sv_file = os.path.join(sv_path, "sv.pkl")
    config_file = os.path.join(sv_path, "config.json")

    if not os.path.exists(sv_file):
        raise FileNotFoundError(f"Steering vectors not found at {sv_file}")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config not found at {config_file}")

    print(f"Loading steering vectors from {sv_path}")
    with open(sv_file, "rb") as f:
        steering_vectors = pickle.load(f)

    config = load_sv_config(config_file)

    # Validate critical parameters - these MUST match
    config_num_steps = config.get("num_inference_steps")
    if config_num_steps is not None and config_num_steps != NUM_INFERENCE_STEPS:
        raise ValueError(
            f"CRITICAL: num_inference_steps mismatch! "
            f"Config has {config_num_steps}, but script expects {NUM_INFERENCE_STEPS}. "
            f"Steering vectors were computed with different diffusion steps."
        )

    config_guidance = config.get("guidance_scale")
    if config_guidance is not None:
        # Check for CFG vs no-CFG mismatch (one <=1.0 means no CFG, >1.0 means CFG)
        config_uses_cfg = config_guidance > 1.0
        script_uses_cfg = GUIDANCE_SCALE > 1.0
        if config_uses_cfg != script_uses_cfg:
            raise ValueError(
                f"CRITICAL: guidance_scale CFG mismatch! "
                f"Config has {config_guidance} ({'CFG' if config_uses_cfg else 'no CFG'}), "
                f"but script expects {GUIDANCE_SCALE} ({'CFG' if script_uses_cfg else 'no CFG'}). "
                f"This would produce fundamentally different generations."
            )

    # Use script constants as primary, warn if config differs
    num_inference_steps = NUM_INFERENCE_STEPS
    guidance_scale = GUIDANCE_SCALE
    audio_duration = AUDIO_LENGTH_IN_S

    # Check for non-critical mismatches and warn
    config_audio_duration = config.get("audio_duration")
    if config_audio_duration is not None and config_audio_duration != AUDIO_LENGTH_IN_S:
        print(
            f"WARNING: audio_duration mismatch. Config has {config_audio_duration}s, "
            f"using script value {AUDIO_LENGTH_IN_S}s"
        )

    config_guidance_value = config.get("guidance_scale")
    if config_guidance_value is not None and config_guidance_value != GUIDANCE_SCALE:
        print(
            f"WARNING: guidance_scale mismatch. Config has {config_guidance_value}, "
            f"using script value {GUIDANCE_SCALE}"
        )

    # Load other params from config (these are less critical)
    num_cfg_passes = config.get("num_cfg_passes", 2)
    guidance_scale_text = config.get("guidance_scale_text", 0.0)
    guidance_scale_lyric = config.get("guidance_scale_lyric", 0.0)
    guidance_interval = config.get("guidance_interval", 1.0)
    guidance_interval_decay = config.get("guidance_interval_decay", 0.0)

    print(f"Using: {num_inference_steps} steps, guidance_scale={guidance_scale}, " f"audio_duration={audio_duration}s")

    # Get evaluation prompts
    if concept not in CONCEPT_TO_EVAL_PROMPTS:
        raise ValueError(
            f"Concept {concept} not found in CONCEPT_TO_EVAL_PROMPTS. Available: {list(CONCEPT_TO_EVAL_PROMPTS.keys())}"
        )
    eval_prompts = CONCEPT_TO_EVAL_PROMPTS[concept]

    # Setup save directory
    if save_dir is None:
        save_dir = f"/data/lstaniszewski/code/audio-interv/steering/outputs/{concept}/{steer_mode}/{layers}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving outputs to {save_dir}")

    # Save run config
    run_config = {
        "sv_path": sv_path,
        "concept": concept,
        "layers": layers,
        "layers_to_steer": layers_to_steer,
        "steer_mode": steer_mode,
        "eval_prompts": eval_prompts,
        "multipliers": MULTIPLIERS,
        "test_prompts": TEST_PROMPTS,
        "generation_seed": GENERATION_SEED,
        **config,
    }
    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # Initialize pipeline
    print("Loading ACE-Step pipeline...")
    steered_pipe = SteeredACEStepPipeline(
        device="cuda",
        steering_vectors=None,
        steer=True,
        alpha=0,
        steer_mode=steer_mode,
        num_cfg_passes=compute_num_cfg_passes(guidance_scale_text, guidance_scale_lyric),
    )
    steered_pipe.load()
    print("Pipeline loaded")

    # Setup steering
    steered_pipe.setup_steering(
        steering_vectors=steering_vectors,
        steer=True,
        alpha=0,
        steer_back=False,
        save_only_cond=False,
        steer_mode=steer_mode,
        num_cfg_passes=compute_num_cfg_passes(guidance_scale_text, guidance_scale_lyric),
        explicit_layers=layers_to_steer,
        verbose_register_layers=True,
    )
    print("Steering ready")

    # Prepare latents
    latents = steered_pipe.prepare_latents(
        batch_size=len(TEST_PROMPTS),
        audio_duration=audio_duration,
        seed=GENERATION_SEED,
    )

    lyrics = NO_LYRICS
    test_prompts = TEST_PROMPTS

    if "vocal" in concept:
        lyrics = LYRICS
        test_prompts = [f"{p}, with vocal" for p in TEST_PROMPTS]

    # Generate for each multiplier
    print(f"Generating audios for {len(MULTIPLIERS)} multipliers...")
    for multiplier in tqdm(MULTIPLIERS, desc="Generating"):
        save_directory_multiplier = os.path.join(save_dir, f"alpha_{multiplier}")
        os.makedirs(save_directory_multiplier, exist_ok=True)

        steered_pipe.controller.steer = True
        steered_pipe.controller.alpha = multiplier
        steered_pipe.controller.steer_mode = steer_mode

        audios = steered_pipe.generate(
            prompt=test_prompts,
            lyrics=lyrics,
            audio_duration=audio_duration,
            infer_step=num_inference_steps,
            manual_seed=GENERATION_SEED,
            return_type="audio",
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            guidance_scale=guidance_scale,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            latents=latents,
        )

        for i, audio in enumerate(audios):
            torchaudio.save(
                os.path.join(save_directory_multiplier, f"p{i}.wav"),
                audio.cpu(),
                steered_pipe.sample_rate,
            )

        steered_pipe.reset_controller()

    # # Evaluate
    # print("Evaluating generated audios...")
    # eval_audios(save_dir, test_prompts, steered_pipe.sample_rate, eval_prompts)

    print(f"Done! Results saved to {save_dir}")


if __name__ == "__main__":
    Fire(main)
