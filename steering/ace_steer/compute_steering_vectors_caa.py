"""
Compute CAA steering vectors for ACE-Step with richer per-extraction outputs.

Outputs per concept (written inside save_dir/ace_{concept}_passes{N}/):
  sv.pkl                              – legacy: {step: {layer: [norm_array]}}
  {concept}_per_extraction.pt         – (N, D) float32 — individual h_pos−h_neg diffs
                                         (mean over timesteps, primary_layer only)
  {concept}_mean.pt                   – (D,) float32 — mean of above rows
  {concept}_{pair_id}_{seed}_all_layers.pt
                                      – dict {"tf0": (D,), …, "tf23": (D,)} per extraction
  pos_vectors.pkl / neg_vectors.pkl   – raw activation stores (for re-analysis)
  config.json                         – run metadata

CLI (Fire-based, suitable for Colab):
    python compute_steering_vectors_caa.py \\
        --concept piano \\
        --seeds '[42,43,44]' \\
        --num_inference_steps 30 \\
        --save_dir /content/steer-audio/outputs/vectors

Hydra entry point also available for HPC workflows:
    python compute_steering_vectors_caa.py concept=piano seed=42
"""

import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fire import Fire
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup: works locally, on RunPod, and in Colab
# ---------------------------------------------------------------------------
WORKDIR_PATH = os.environ.get("TADA_WORKDIR", str(Path(__file__).resolve().parents[2]))
sys.path.append(WORKDIR_PATH)
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

# All 24 ACE-Step transformer blocks, named by register_vector_control convention
ALL_LAYERS: List[str] = [f"tf{i}" for i in range(24)]


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def get_prompts_pairs(
    concept: str, n_pairs: Optional[int] = None
) -> Tuple[List[str], List[str], str]:
    """
    Load pos/neg prompt pairs for *concept*; optionally truncate to *n_pairs*.

    Returns:
        prompts_pos, prompts_neg, lyrics
    """
    if concept not in CONCEPT_TO_PROMPTS:
        raise ValueError(
            f"Unknown concept: {concept!r}.  "
            f"Available: {sorted(CONCEPT_TO_PROMPTS.keys())}"
        )
    prompts_neg, prompts_pos, lyrics = CONCEPT_TO_PROMPTS[concept]()
    if n_pairs is not None:
        prompts_pos = prompts_pos[:n_pairs]
        prompts_neg = prompts_neg[:n_pairs]
    print(f"Concept '{concept}': {len(prompts_pos)} prompt pairs")
    return prompts_pos, prompts_neg, lyrics


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

def _collect_activations(
    prompt: str,
    pipe: "SimpleACEStepPipeline",
    device: str,
    num_cfg_passes: int,
    explicit_layers: List[str],
    audio_duration: float,
    num_inference_steps: int,
    seed: int,
    guidance_scale_text: float,
    guidance_scale_lyric: float,
    guidance_scale: float,
    guidance_interval: float,
    guidance_interval_decay: float,
) -> Dict[int, Dict[str, List[np.ndarray]]]:
    """
    Run a single forward pass and return the collected activation store.

    Returns:
        {denoising_step (int): {layer_name (str): [D-dim float32 np.ndarray]}}
        Each inner list has exactly one element (conditional pass only).
    """
    ctrl = VectorStore(
        device=device,
        save_only_cond=True,
        num_cfg_passes=num_cfg_passes,
    )
    ctrl.steer = False
    register_vector_control(
        pipe.ace_step_transformer, ctrl, explicit_layers=explicit_layers
    )
    pipe.generate(
        prompt=prompt,
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
    # Copy before reset so the inner dicts aren't cleared.
    # ctrl.reset() replaces self.vector_store with a new defaultdict; the old
    # mapping (captured in `store`) remains valid.
    store: Dict[int, Any] = dict(ctrl.vector_store)
    ctrl.reset()
    return store


def generate_all_extractions(
    prompts_pos: List[str],
    prompts_neg: List[str],
    seeds: List[int],
    pipe: "SimpleACEStepPipeline",
    device: str,
    explicit_layers: List[str],
    audio_duration: float,
    num_inference_steps: int,
    guidance_scale_text: float,
    guidance_scale_lyric: float,
    guidance_scale: float,
    guidance_interval: float,
    guidance_interval_decay: float,
) -> List[Dict[str, Any]]:
    """
    Run all (seed × pair) activation collections.

    Returns a flat list — one dict per extraction — in order
    seed_0/pair_0, seed_0/pair_1, …, seed_1/pair_0, …:
        {
            "pair_id": int,
            "seed":    int,
            "pos_store": {step: {layer: [D-array]}},
            "neg_store": {step: {layer: [D-array]}},
        }
    """
    num_cfg_passes = compute_num_cfg_passes(guidance_scale_text, guidance_scale_lyric)
    common_kw = dict(
        pipe=pipe,
        device=device,
        num_cfg_passes=num_cfg_passes,
        explicit_layers=explicit_layers,
        audio_duration=audio_duration,
        num_inference_steps=num_inference_steps,
        guidance_scale_text=guidance_scale_text,
        guidance_scale_lyric=guidance_scale_lyric,
        guidance_scale=guidance_scale,
        guidance_interval=guidance_interval,
        guidance_interval_decay=guidance_interval_decay,
    )

    extractions: List[Dict[str, Any]] = []
    total = len(seeds) * len(prompts_pos)
    pbar = tqdm(total=total * 2, desc="Collecting activations (pos+neg)")

    for seed in seeds:
        for pair_id, (pos_prompt, neg_prompt) in enumerate(
            zip(prompts_pos, prompts_neg)
        ):
            pos_store = _collect_activations(pos_prompt, seed=seed, **common_kw)
            pbar.update(1)
            neg_store = _collect_activations(neg_prompt, seed=seed, **common_kw)
            pbar.update(1)
            extractions.append(
                {
                    "pair_id": pair_id,
                    "seed": seed,
                    "pos_store": pos_store,
                    "neg_store": neg_store,
                }
            )

    pbar.close()
    return extractions


# ---------------------------------------------------------------------------
# Vector computations
# ---------------------------------------------------------------------------

def _mean_over_steps(
    store: Dict[int, Dict[str, List[np.ndarray]]], layer: str
) -> np.ndarray:
    """
    Return the mean activation for *layer* averaged over all denoising steps.

    Args:
        store:  output of _collect_activations
        layer:  e.g. "tf6"

    Returns:
        (D,) float32 numpy array

    Raises:
        KeyError if *layer* is absent from every step.
    """
    step_keys = sorted(store.keys())
    vecs = [
        store[s][layer][0]
        for s in step_keys
        if layer in store[s] and store[s][layer]
    ]
    if not vecs:
        raise KeyError(
            f"Layer {layer!r} not found in activation store. "
            f"Available layers: {list(store[step_keys[0]].keys()) if step_keys else '[]'}"
        )
    return np.mean(vecs, axis=0).astype(np.float32)  # (D,)


def compute_per_extraction_tensors(
    extractions: List[Dict[str, Any]],
    primary_layer: str = "tf6",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-extraction diff tensors for *primary_layer*.

    For extraction i:
        diff_i = mean_t(pos_store_i[t][primary_layer]) − mean_t(neg_store_i[t][primary_layer])

    Args:
        extractions:    output of generate_all_extractions
        primary_layer:  which transformer block (default "tf6")

    Returns:
        per_extraction: (N, D) float32 tensor — one row per extraction
        mean_vec:       (D,)   float32 tensor — mean of all rows (unnormalized)
    """
    diffs: List[np.ndarray] = []
    for ext in extractions:
        pos_mean = _mean_over_steps(ext["pos_store"], primary_layer)
        neg_mean = _mean_over_steps(ext["neg_store"], primary_layer)
        diffs.append(pos_mean - neg_mean)

    arr = np.stack(diffs, axis=0)  # (N, D)
    return (
        torch.from_numpy(arr).float(),
        torch.from_numpy(arr.mean(axis=0)).float(),
    )


def compute_all_layers_dict(
    pos_store: Dict[int, Dict[str, List[np.ndarray]]],
    neg_store: Dict[int, Dict[str, List[np.ndarray]]],
    layers: List[str] = ALL_LAYERS,
) -> Dict[str, torch.Tensor]:
    """
    For one extraction, compute the diff vector (mean over timesteps) for each layer.

    Args:
        pos_store:  from _collect_activations (positive prompt)
        neg_store:  from _collect_activations (negative prompt)
        layers:     which layers to include (default: all 24)

    Returns:
        {"tf0": (D,) tensor, "tf1": ..., ..., "tf23": (D,) tensor}
        Layers absent from the store are silently omitted.
    """
    result: Dict[str, torch.Tensor] = {}
    for layer in layers:
        try:
            diff = _mean_over_steps(pos_store, layer) - _mean_over_steps(neg_store, layer)
            result[layer] = torch.from_numpy(diff).float()
        except KeyError:
            pass  # layer not hooked; silently skip
    return result


def compute_sv_legacy(
    pos_vectors_list: List[Dict[int, Any]],
    neg_vectors_list: List[Dict[int, Any]],
) -> Dict[int, Dict[str, List[np.ndarray]]]:
    """
    Backward-compatible sv.pkl computation.

    Computes the unit-norm mean-difference vector per (step, layer) across all
    extractions.  Identical output format to the original compute_sv().

    Returns:
        {step_key: {layer_name: [D-dim normalized np.ndarray]}}
    """
    print("Computing legacy sv.pkl (unit-norm mean diffs)…")
    steering_vectors: Dict = {}
    all_step_keys = list(pos_vectors_list[0].keys())
    layer_names = list(pos_vectors_list[0][all_step_keys[0]].keys())

    for step_key in all_step_keys:
        steering_vectors[step_key] = defaultdict(list)
        for layer_name in layer_names:
            pos_vecs = [
                pos_vectors_list[i][step_key][layer_name][0]
                for i in range(len(pos_vectors_list))
            ]
            neg_vecs = [
                neg_vectors_list[i][step_key][layer_name][0]
                for i in range(len(neg_vectors_list))
            ]
            sv = np.mean(pos_vecs, axis=0) - np.mean(neg_vecs, axis=0)
            norm = np.linalg.norm(sv)
            if norm > 0:
                sv = sv / norm
            steering_vectors[step_key][layer_name].append(sv)

    return steering_vectors


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(
    concept: str,
    num_inference_steps: int = 30,
    audio_duration: float = 30.0,
    guidance_scale_text: float = 0.0,
    guidance_scale_lyric: float = 0.0,
    guidance_scale: float = 3.0,
    guidance_interval: float = 1.0,
    guidance_interval_decay: float = 0.0,
    seeds: List[int] = (42,),
    n_pairs: Optional[int] = None,
    device: str = DEFAULT_DEVICE,
    save_dir: str = DEFAULT_SAVE_DIR,
    primary_layer: str = "tf6",
    skip_all_layers: bool = False,
) -> str:
    """
    Compute CAA steering vectors for a concept and save richer outputs.

    Args:
        concept:          Concept key from CONCEPT_TO_PROMPTS
                          (piano | mood | tempo | female_vocals | drums)
        seeds:            List of random seeds; one run per seed per pair.
                          Total extractions = len(seeds) × len(prompt_pairs).
        n_pairs:          Truncate prompt list to this many pairs (None = all 50).
        save_dir:         Root output directory.  Outputs go to
                          save_dir/ace_{concept}_passes{N}/.
                          Pass /content/steer-audio/outputs/vectors (which may be
                          symlinked to Drive) and don't hardcode anything else.
        primary_layer:    Transformer block used for {concept}_per_extraction.pt
                          and {concept}_mean.pt.  Default "tf6".
        skip_all_layers:  Skip writing per-extraction *_all_layers.pt files.
                          Useful if disk space is tight.

    Returns:
        Absolute path to the output directory (str).
    """
    # Normalise seeds (Fire passes a single int if given without list brackets)
    if isinstance(seeds, int):
        seeds = [seeds]
    seeds = list(seeds)

    print(f"Loading ACE-Step pipeline on {device}…")
    pipe = SimpleACEStepPipeline(device=device)
    pipe.load()

    prompts_pos, prompts_neg, lyrics = get_prompts_pairs(concept, n_pairs=n_pairs)

    # Always hook all 24 transformer blocks for richer diagnostics.
    # explicit_layers is mandatory per CLAUDE.md to avoid size-mismatch errors.
    explicit_layers = ALL_LAYERS

    extractions = generate_all_extractions(
        prompts_pos=prompts_pos,
        prompts_neg=prompts_neg,
        seeds=seeds,
        pipe=pipe,
        device=device,
        explicit_layers=explicit_layers,
        audio_duration=audio_duration,
        num_inference_steps=num_inference_steps,
        guidance_scale_text=guidance_scale_text,
        guidance_scale_lyric=guidance_scale_lyric,
        guidance_scale=guidance_scale,
        guidance_interval=guidance_interval,
        guidance_interval_decay=guidance_interval_decay,
    )

    N = len(extractions)
    print(
        f"\nCollected {N} extractions "
        f"({len(prompts_pos)} pairs × {len(seeds)} seed(s))"
    )

    # ── Output directory ────────────────────────────────────────────────────
    num_cfg_passes = compute_num_cfg_passes(guidance_scale_text, guidance_scale_lyric)
    subdir = f"ace_{concept}_passes{num_cfg_passes}"
    save_path = (Path(save_dir) / subdir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    # ── (a) Per-extraction and mean tensors ─────────────────────────────────
    print(f"\nComputing per-extraction tensors (layer: {primary_layer})…")
    per_extraction, mean_vec = compute_per_extraction_tensors(
        extractions, primary_layer=primary_layer
    )
    torch.save(per_extraction, save_path / f"{concept}_per_extraction.pt")
    torch.save(mean_vec, save_path / f"{concept}_mean.pt")
    print(f"  {concept}_per_extraction.pt  → {tuple(per_extraction.shape)}")
    print(f"  {concept}_mean.pt            → {tuple(mean_vec.shape)}")

    # ── (b) All-layers dicts per extraction ─────────────────────────────────
    if not skip_all_layers:
        print("\nSaving per-extraction all-layers dicts…")
        for ext in tqdm(extractions, desc="all_layers"):
            layer_dict = compute_all_layers_dict(
                ext["pos_store"], ext["neg_store"], layers=explicit_layers
            )
            fname = f"{concept}_{ext['pair_id']}_{ext['seed']}_all_layers.pt"
            torch.save(layer_dict, save_path / fname)
        print(f"  Saved {N} *_all_layers.pt files")

    # ── Legacy sv.pkl ────────────────────────────────────────────────────────
    print("\nBuilding legacy sv.pkl…")
    pos_vectors_list = [ext["pos_store"] for ext in extractions]
    neg_vectors_list = [ext["neg_store"] for ext in extractions]
    steering_vectors = compute_sv_legacy(pos_vectors_list, neg_vectors_list)

    with open(save_path / "sv.pkl", "wb") as f:
        pickle.dump(steering_vectors, f)
    with open(save_path / "pos_vectors.pkl", "wb") as f:
        pickle.dump(pos_vectors_list, f)
    with open(save_path / "neg_vectors.pkl", "wb") as f:
        pickle.dump(neg_vectors_list, f)

    # ── Config ───────────────────────────────────────────────────────────────
    with open(save_path / "config.json", "w") as f:
        json.dump(
            {
                "concept": concept,
                "lyrics": lyrics,
                "num_cfg_passes": num_cfg_passes,
                "audio_duration": audio_duration,
                "num_inference_steps": num_inference_steps,
                "seeds": seeds,
                "n_pairs": n_pairs,
                "n_extractions": N,
                "primary_layer": primary_layer,
                "explicit_layers": explicit_layers,
                "skip_all_layers": skip_all_layers,
                "device": device,
                "save_dir": str(save_dir),
                "guidance_scale_text": guidance_scale_text,
                "guidance_scale_lyric": guidance_scale_lyric,
                "guidance_scale": guidance_scale,
                "guidance_interval": guidance_interval,
                "guidance_interval_decay": guidance_interval_decay,
            },
            f,
            indent=2,
        )

    print(f"\nAll outputs written to: {save_path}")
    return str(save_path)


# ---------------------------------------------------------------------------
# Colab dry-run helper
# ---------------------------------------------------------------------------

def colab_dry_run(
    concept: str = "piano",
    n_pairs: int = 2,
    n_seeds: int = 1,
    test_dir: str = "/content/drive/MyDrive/steer_audio_results/vectors/test",
    device: str = DEFAULT_DEVICE,
    num_inference_steps: int = 30,
) -> None:
    """
    Quick end-to-end sanity check for Colab.

    Runs extraction for *n_pairs* prompt pairs × *n_seeds* seeds, writes outputs
    to *test_dir*, then loads and prints all shapes.

    Valid concepts: piano | mood | tempo | female_vocals | drums

    Example (Colab cell)::

        import os; os.environ['TADA_WORKDIR'] = '/content/steer-audio'
        from steering.ace_steer.compute_steering_vectors_caa import colab_dry_run
        colab_dry_run(concept="piano", n_pairs=2, n_seeds=1)
    """
    import glob

    seeds = list(range(42, 42 + n_seeds))
    save_path_str = main(
        concept=concept,
        n_pairs=n_pairs,
        seeds=seeds,
        save_dir=test_dir,
        device=device,
        num_inference_steps=num_inference_steps,
        skip_all_layers=False,
    )
    save_path = Path(save_path_str)

    sep = "=" * 60
    print(f"\n{sep}")
    print("DRY-RUN VERIFICATION")
    print(sep)

    per_ext = torch.load(save_path / f"{concept}_per_extraction.pt")
    print(f"  {concept}_per_extraction.pt  shape: {tuple(per_ext.shape)}")
    # expected: (n_pairs * n_seeds, D)

    mean_v = torch.load(save_path / f"{concept}_mean.pt")
    print(f"  {concept}_mean.pt            shape: {tuple(mean_v.shape)}")
    # expected: (D,)

    al_files = sorted(glob.glob(str(save_path / f"{concept}_*_all_layers.pt")))
    if al_files:
        first = Path(al_files[0])
        al: Dict[str, torch.Tensor] = torch.load(first)
        print(f"\n  First all_layers file: {first.name}")
        for k in sorted(al.keys(), key=lambda x: int(x[2:])):  # tf0, tf1, …
            print(f"    {k}: {tuple(al[k].shape)}")
        print(f"\n  Total all_layers files: {len(al_files)}")
    else:
        print("  WARNING: no *_all_layers.pt files found")

    print(f"\nDry-run passed.  Outputs: {save_path}")


# ---------------------------------------------------------------------------
# Hydra entry point (HPC / RunPod)
# ---------------------------------------------------------------------------

try:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.4", config_path="../../configs", config_name="compute_caa")
    def hydra_main(cfg: DictConfig) -> None:
        """Hydra entry point — invoke via ``python compute_steering_vectors_caa.py concept=tempo``."""
        main(
            concept=cfg.concept,
            num_inference_steps=cfg.get("num_inference_steps", 30),
            audio_duration=cfg.get("audio_duration", 30.0),
            guidance_scale_text=cfg.get("guidance_scale_text", 0.0),
            guidance_scale_lyric=cfg.get("guidance_scale_lyric", 0.0),
            guidance_scale=cfg.get("guidance_scale", 3.0),
            guidance_interval=cfg.get("guidance_interval", 1.0),
            guidance_interval_decay=cfg.get("guidance_interval_decay", 0.0),
            seeds=list(cfg.get("seeds", [42])),
            n_pairs=cfg.get("n_pairs", None),
            device=cfg.get("device", DEFAULT_DEVICE),
            save_dir=cfg.get("save_dir", DEFAULT_SAVE_DIR),
            primary_layer=cfg.get("primary_layer", "tf6"),
            skip_all_layers=cfg.get("skip_all_layers", False),
        )

except ImportError:
    pass  # Hydra not installed; CLI falls back to Fire below


if __name__ == "__main__":
    # Fire is the default entry point — works in Colab and RunPod without Hydra config.
    # Example:
    #   python compute_steering_vectors_caa.py --concept piano --seeds '[42,43,44]'
    Fire(main)
