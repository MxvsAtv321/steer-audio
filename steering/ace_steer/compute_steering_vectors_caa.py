"""
Compute steering vectors for ACE-Step with CAA.

This script generates steering vectors by computing the difference between
activations from positive and negative prompt pairs defined in
sae.sae_src.configs.steer_prompts.CONCEPT_TO_PROMPTS.

Usage:
    python compute_steering_vectors_sae.py \
        --concept piano \
        --save_dir steering_vectors \
        --num_inference_steps 30
"""

import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from tqdm import tqdm

WORKDIR_PATH = "<WORKDIR_PATH>"

sys.path.append(f"{WORKDIR_PATH}")
sys.path.append(f"{WORKDIR_PATH}/src/models/ace_step/ACE")
sys.path.append(f"{WORKDIR_PATH}/sae")
from sae.sae_src.configs.steer_prompts import CONCEPT_TO_PROMPTS

from src.models.ace_step.ace_steering.controller import (
    VectorStore,
    compute_num_cfg_passes,
    register_vector_control,
)
from src.models.ace_step.pipeline_ace import SimpleACEStepPipeline

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SAVE_DIR = "steering_vectors"


def get_prompts_pairs(concept: str):
    """Get prompt pairs from CONCEPT_TO_PROMPTS config."""
    print(f"\nGenerating prompts for concept: {concept}")

    if concept not in CONCEPT_TO_PROMPTS:
        raise ValueError(
            f"Unknown concept: {concept}. Available concepts: {list(CONCEPT_TO_PROMPTS.keys())}"
        )

    prompts_neg, prompts_pos, lyrics = CONCEPT_TO_PROMPTS[concept]()
    print(f"Generated {len(prompts_pos)} prompt pairs")

    return prompts_pos, prompts_neg, lyrics


def generate_vectors(
    prompts_pos,
    prompts_neg,
    device: str,
    save_all_cfg_passes: bool,
    audio_duration: float,
    num_inference_steps: int,
    seed: int,
    pipe: SimpleACEStepPipeline,
    guidance_scale_text: float,
    guidance_scale_lyric: float,
    guidance_scale: float,
    guidance_interval: float,
    guidance_interval_decay: float,
):
    num_cfg_passes = compute_num_cfg_passes(
        guidance_scale_text=guidance_scale_text,
        guidance_scale_lyric=guidance_scale_lyric,
    )

    pos_vectors = []
    neg_vectors = []
    for i, (prompt_pos, prompt_neg) in tqdm(
        enumerate(zip(prompts_pos, prompts_neg)),
        total=len(prompts_pos),
        desc="Collecting activations for pairs",
    ):
        controller = VectorStore(
            device=device,
            save_only_cond=not save_all_cfg_passes,
            num_cfg_passes=num_cfg_passes,
        )
        controller.steer = False  # Just collecting activations, not steering
        register_vector_control(pipe.ace_step_transformer, controller)

        _ = pipe.generate(
            prompt=prompt_pos,
            audio_duration=audio_duration,
            infer_step=num_inference_steps,
            manual_seed=seed,
            return_type="latent",
            use_erg_lyric=False,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            guidance_scale=guidance_scale,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
        )

        pos_vectors.append(controller.vector_store)
        controller.reset()

        controller = VectorStore(
            device=device,
            save_only_cond=not save_all_cfg_passes,
            num_cfg_passes=num_cfg_passes,
        )
        controller.steer = False
        register_vector_control(pipe.ace_step_transformer, controller)

        _ = pipe.generate(
            prompt=prompt_neg,
            audio_duration=audio_duration,
            infer_step=num_inference_steps,
            manual_seed=seed,
            return_type="latent",
            use_erg_lyric=False,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            guidance_scale=guidance_scale,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
        )
        neg_vectors.append(controller.vector_store)
        controller.reset()

    return pos_vectors, neg_vectors


def compute_sv(pos_vectors, neg_vectors):
    print("\nComputing steering vectors...")
    steering_vectors = {}
    all_step_keys = list(pos_vectors[0].keys())
    layer_names = list(pos_vectors[0][all_step_keys[0]].keys())

    for step_key in all_step_keys:
        steering_vectors[step_key] = defaultdict(list)
        for layer_name in layer_names:
            pos_vectors_layer = [
                pos_vectors[i][step_key][layer_name][0] for i in range(len(pos_vectors))
            ]
            pos_vectors_avg = np.mean(pos_vectors_layer, axis=0)

            neg_vectors_layer = [
                neg_vectors[i][step_key][layer_name][0] for i in range(len(neg_vectors))
            ]
            neg_vectors_avg = np.mean(neg_vectors_layer, axis=0)

            steering_vector = pos_vectors_avg - neg_vectors_avg

            norm = np.linalg.norm(steering_vector)
            if norm > 0:
                steering_vector = steering_vector / norm

            steering_vectors[step_key][layer_name].append(steering_vector)

    return steering_vectors


def main(
    concept: str,
    num_inference_steps: int = 30,
    audio_duration: float = 30.0,
    guidance_scale_text: float = 0.0,
    guidance_scale_lyric: float = 0.0,
    guidance_scale: float = 3.0,
    guidance_interval: float = 1.0,
    guidance_interval_decay: float = 0.0,
    seed: int = 42,
    device: str = DEFAULT_DEVICE,
    save_dir: str = DEFAULT_SAVE_DIR,
    save_all_cfg_passes: bool = True,
):
    """
    Compute steering vectors for a concept defined in CONCEPT_TO_PROMPTS.

    Args:
        concept: Concept name from CONCEPT_TO_PROMPTS (e.g., 'piano', 'mood', 'tempo', 'female_vocals', 'drums')
        num_inference_steps: Number of diffusion steps
        audio_duration: Audio duration in seconds
        guidance_scale_text: Text guidance scale
        guidance_scale_lyric: Lyric guidance scale
        guidance_scale: Overall guidance scale
        guidance_interval: Guidance interval
        guidance_interval_decay: Guidance interval decay
        seed: Random seed
        device: Device to run on
        save_dir: Directory to save steering vectors
        save_all_cfg_passes: Whether to save all CFG passes
    """
    # Initialize pipeline
    print("Loading ACE-Step pipeline...")
    pipe = SimpleACEStepPipeline(
        device=device,
    )
    pipe.load()

    prompts_pos, prompts_neg, lyrics = get_prompts_pairs(concept)

    num_cfg_passes = compute_num_cfg_passes(guidance_scale_text, guidance_scale_lyric)
    pos_vectors, neg_vectors = generate_vectors(
        prompts_pos,
        prompts_neg,
        device,
        save_all_cfg_passes,
        audio_duration,
        num_inference_steps,
        seed,
        pipe,
        guidance_scale_text,
        guidance_scale_lyric,
        guidance_scale,
        guidance_interval,
        guidance_interval_decay,
    )
    steering_vectors = compute_sv(pos_vectors, neg_vectors)

    # saving steering vectors
    save_directory = Path(save_dir).resolve()
    vector_directory = f"ace_{concept}_passes{num_cfg_passes}_all{save_all_cfg_passes}"
    save_directory = (save_directory / vector_directory).resolve()
    os.makedirs(save_directory, exist_ok=True)

    with open((save_directory / "sv.pkl"), "wb") as f:
        pickle.dump(steering_vectors, f)
    with open((save_directory / "pos_vectors.pkl"), "wb") as f:
        pickle.dump(pos_vectors, f)
    with open((save_directory / "neg_vectors.pkl"), "wb") as f:
        pickle.dump(neg_vectors, f)
    with open((save_directory / "config.json"), "w") as f:
        json.dump(
            {
                "concept": concept,
                "lyrics": lyrics,
                "num_cfg_passes": num_cfg_passes,
                "save_all_cfg_passes": save_all_cfg_passes,
                "audio_duration": audio_duration,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
                "device": device,
                "save_dir": save_dir,
                "guidance_scale_text": guidance_scale_text,
                "guidance_scale_lyric": guidance_scale_lyric,
                "guidance_scale": guidance_scale,
                "guidance_interval": guidance_interval,
                "guidance_interval_decay": guidance_interval_decay,
            },
            f,
        )

    print(f"\nSteering vectors saved to: {save_directory}")
    print(f"Compatible steering modes:")
    if save_all_cfg_passes:
        print(
            f"  - 'separate': ✓ (cond steered with cond vectors, uncond with uncond vectors)"
        )
        print(f"  - 'cond_only': ✓ (only cond steered, uses cond vectors)")
        print(f"  - 'both_cond': ✓ (both steered with cond vectors)")
    else:
        print(f"  - 'separate': ✗ (need --save_all_cfg_passes)")
        print(f"  - 'cond_only': ✓ (only cond steered, uses cond vectors)")
        print(f"  - 'both_cond': ✓ (both steered with cond vectors)")


if __name__ == "__main__":
    Fire(main)
