"""
Evaluate SAE-based steering for ACE-Step model.

This script generates steered audio using SAE feature interventions and saves outputs
in a directory structure compatible with eval_steering_protocol.py.

Usage:
    python eval_sae_steering.py \
        --concept piano \
        --sae_path /path/to/sae \
        --selection_method diff \
        --top_k 20

    # With specific SAE type
    python eval_sae_steering.py \
        --concept drums \
        --sae_path /path/to/sae \
        --sae_type frequency \
        --selection_method tfidf \
        --top_k 50

    # Using config from notebook analysis
    python eval_sae_steering.py \
        --concept mood \
        --sae_path /path/to/sae \
        --config_path /path/to/mood_best_config.pkl
"""

import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from fire import Fire
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Detect project root automatically (sae/scripts/ -> project root)
PATH_TO_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

sys.path.append(PATH_TO_PROJECT)
sys.path.append(PATH_TO_PROJECT + "/sae")
sys.path.append(PATH_TO_PROJECT + "/editing/AudioEditingCode")
sys.path.append(PATH_TO_PROJECT + "/steering/CASteer")
sys.path.append(PATH_TO_PROJECT + "/src/models/ace_step/ACE")

from sae_src.configs.eval import CONCEPT_TO_EVAL_PROMPTS
from sae_src.configs.steer_prompts import CONCEPT_TO_PROMPTS
from sae_src.hooked_model.acestep_hooks import ACEStepPrecomputedSteeringHook, ACEStepTimestepInterventionHook
from sae_src.hooked_model.utils import locate_block
from sae_src.sae.sae import Sae

from src.models.ace_step.ace_steering.controller import compute_num_cfg_passes
from src.models.ace_step.steering_ace import SteeredACEStepPipeline
from steering.ace_steer.prompts import LYRICS, NO_LYRICS, TEST_PROMPTS

# Generation hyperparameters (same as steering vectors config.json)
VECTORS_SEED = 42
GENERATION_SEED = 2115
GUIDANCE_SCALE = 3.0
AUDIO_LENGTH_IN_S = 30.0
NUM_INFERENCE_STEPS = 30  # Same as SV config.json
ACE_STEP_GUIDANCE_INTERVAL = 1.0
ACE_STEP_GUIDANCE_INTERVAL_DECAY = 0.0
ACE_STEP_GUIDANCE_SCALE_TEXT = 0.0
ACE_STEP_GUIDANCE_SCALE_LYRIC = 0.0

# SAE steering multipliers (from sae/scripts/steer_sae.py)
MULTIPLIERS = [
    -50.0,
    -47.5,
    -45.0,
    -42.5,
    -40.0,
    -37.5,
    -35.0,
    -32.5,
    -30.0,
    -27.5,
    -25.0,
    -22.5,
    -20.0,
    -17.5,
    -15.0,
    -12.5,
    -10.0,
    -7.5,
    -5.0,
    -2.5,
    0.0,
    2.5,
    5.0,
    7.5,
    10.0,
    12.5,
    15.0,
    17.5,
    20.0,
    22.5,
    25.0,
    27.5,
    30.0,
    32.5,
    35.0,
    37.5,
    40.0,
    42.5,
    45.0,
    47.5,
    50.0,
]


class FeatureSelector:
    """
    Compute feature importance scores using multiple methods.

    Methods:
        1. diff: Difference of mean activations (pos - neg)
        2. ratio: Ratio of mean activations (pos / neg)
        3. cohens_d: Cohen's d effect size
        4. tfidf: TF-IDF like scoring
        5. linear: Logistic regression coefficients
    """

    def __init__(
        self,
        sae,
        positive_acts,
        negative_acts,
        pool_audio=False,
        pooling="max",
        scores_cache_path=None,
    ):
        self.sae = sae
        self.positive_acts = positive_acts
        self.negative_acts = negative_acts
        self.pool_audio = pool_audio
        self.pooling = pooling
        self.num_timesteps = positive_acts.shape[1]
        self.num_latents = sae.num_latents
        self.timestep_scores = {}

        if scores_cache_path is not None and os.path.exists(scores_cache_path):
            print(f"Loading cached scores from {scores_cache_path}")
            with open(scores_cache_path, "rb") as f:
                cached = pickle.load(f)
            # Handle different score file formats:
            # Format 1 (from notebook _scores.pkl): {"tfidf": {"scores": tensor}, ...}
            # Format 2 (from notebook _all_scores.pkl): {"tfidf": tensor, "diff": tensor, ...}
            for method_name in cached:
                if method_name == "concept":
                    continue  # Skip non-score keys
                value = cached[method_name]
                if isinstance(value, dict) and "scores" in value:
                    # Format 1
                    self.timestep_scores[method_name] = value["scores"]
                elif hasattr(value, "shape"):
                    # Format 2: direct tensor
                    self.timestep_scores[method_name] = value
            print(f"  Loaded scores for methods: {list(self.timestep_scores.keys())}")

    def _get_sae_latents(self, activations):
        """Get SAE latent activations."""
        sae_input, _, _ = self.sae.preprocess_input(activations)
        pre_acts = self.sae.pre_acts(sae_input)
        latents = F.relu(pre_acts)
        return latents

    def _prepare_latents_for_timestep(self, t):
        """Prepare latents for a specific timestep."""
        pos_t = self.positive_acts[:, t]
        neg_t = self.negative_acts[:, t]

        with torch.no_grad():
            if self.pool_audio:
                pos_latents = self._get_sae_latents(pos_t)
                neg_latents = self._get_sae_latents(neg_t)

                num_prompts = pos_t.shape[0]
                audio_len = pos_t.shape[1]
                if self.pooling == "max":
                    pos_latents = pos_latents.view(num_prompts, audio_len, -1).max(dim=1)[0]
                    neg_latents = neg_latents.view(num_prompts, audio_len, -1).max(dim=1)[0]
                elif self.pooling == "mean":
                    pos_latents = pos_latents.view(num_prompts, audio_len, -1).mean(dim=1)
                    neg_latents = neg_latents.view(num_prompts, audio_len, -1).mean(dim=1)
            else:
                pos_latents = self._get_sae_latents(pos_t)
                neg_latents = self._get_sae_latents(neg_t)

        return pos_latents, neg_latents

    def method_diff(self, t):
        """Difference of mean activations (pos - neg)."""
        pos_latents, neg_latents = self._prepare_latents_for_timestep(t)
        mean_pos = pos_latents.mean(dim=0)
        mean_neg = neg_latents.mean(dim=0)
        return mean_pos - mean_neg

    def method_ratio(self, t):
        """Ratio of mean activations (pos / neg)."""
        pos_latents, neg_latents = self._prepare_latents_for_timestep(t)
        mean_pos = pos_latents.mean(dim=0)
        mean_neg = neg_latents.mean(dim=0)
        return mean_pos / (mean_neg + 1e-6)

    def method_cohens_d(self, t):
        """Cohen's d effect size."""
        pos_latents, neg_latents = self._prepare_latents_for_timestep(t)

        mean_pos = pos_latents.mean(dim=0)
        mean_neg = neg_latents.mean(dim=0)
        std_pos = pos_latents.std(dim=0)
        std_neg = neg_latents.std(dim=0)
        n_pos, n_neg = pos_latents.shape[0], neg_latents.shape[0]

        pooled_std = torch.sqrt(((n_pos - 1) * std_pos**2 + (n_neg - 1) * std_neg**2) / (n_pos + n_neg - 2))
        return (mean_pos - mean_neg) / (pooled_std + 1e-6)

    def method_tfidf(self, t):
        """TF-IDF like importance."""
        pos_latents, neg_latents = self._prepare_latents_for_timestep(t)
        mean_pos = pos_latents.mean(dim=0)
        mean_neg = neg_latents.mean(dim=0)

        tf = mean_pos
        idf = torch.log(1 + 1 / (mean_neg + 1e-6))

        return tf * idf

    def method_mean_pos(self, t):
        """Mean positive activations only."""
        pos_latents, _ = self._prepare_latents_for_timestep(t)
        return pos_latents.mean(dim=0)

    def method_linear(self, t):
        """Logistic regression coefficients."""
        pos_latents, neg_latents = self._prepare_latents_for_timestep(t)

        X_pos = pos_latents.cpu().float().numpy()
        X_neg = neg_latents.cpu().float().numpy()
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * len(X_pos) + [0] * len(X_neg))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1)
        clf.fit(X_scaled, y)

        return torch.tensor(clf.coef_[0], device=pos_latents.device, dtype=pos_latents.dtype)

    def compute_single_method(self, method_name, verbose=True):
        """Compute a single method for all timesteps."""
        methods = {
            "diff": self.method_diff,
            "ratio": self.method_ratio,
            "cohens_d": self.method_cohens_d,
            "tfidf": self.method_tfidf,
            "linear": self.method_linear,
            "mean_pos": self.method_mean_pos,
        }

        if method_name not in methods:
            raise ValueError(f"Unknown method: {method_name}. Choose from {list(methods.keys())}")

        if method_name in self.timestep_scores:
            return {"scores": self.timestep_scores[method_name]}

        method = methods[method_name]
        scores_list = []

        for t in range(self.num_timesteps):
            if verbose and t % 10 == 0:
                print(f"  Processing timestep {t}/{self.num_timesteps}...")
            scores = method(t)
            scores_list.append(scores.cpu())
        scores_list = torch.stack(scores_list)

        self.timestep_scores[method_name] = scores_list
        return {"scores": scores_list}

    def get_topk_single_method(self, method_name, top_k=50):
        if method_name not in self.timestep_scores:
            raise ValueError(f"Method {method_name} not computed yet.")
        method_scores = self.timestep_scores[method_name]
        topk_scores = [
            torch.argsort(method_scores[t_idx, :], descending=True)[:top_k] for t_idx in range(method_scores.shape[0])
        ]
        return {"top_features": torch.stack(topk_scores)}


def get_top_features_per_timestep(results, method_name, top_k=10):
    """Extract top-k features per timestep as a dict for use in hooks."""
    top_features = results[method_name]["top_features"]
    num_timesteps = top_features.shape[0]
    return {t: top_features[t, :top_k].tolist() for t in range(num_timesteps)}


def apply_weighting(scores, weighting="none"):
    """
    Apply weighting transformation to scores.

    Args:
        scores: Tensor of shape [num_features] or [num_timesteps, num_features]
        weighting: One of "none", "raw", "softmax", "sqrt", "log"

    Returns:
        Weighted scores tensor
    """
    if weighting == "none":
        return torch.ones_like(scores)
    elif weighting == "raw":
        return scores
    elif weighting == "softmax":
        if scores.dim() == 1:
            return F.softmax(scores, dim=0)
        else:
            return F.softmax(scores, dim=-1)
    elif weighting == "sqrt":
        return torch.sqrt(torch.clamp(scores, min=0))
    elif weighting == "log":
        return torch.log1p(torch.clamp(scores, min=0))
    else:
        raise ValueError(f"Unknown weighting: {weighting}")


def build_weighted_steering_vectors(
    selection_scores,
    weight_scores,
    W_dec,
    top_k,
    weighting="none",
):
    """
    Build SAE-based steering vectors using weighted feature selection.

    Args:
        selection_scores: Tensor [num_timesteps, num_features] for selecting top-k
        weight_scores: Tensor [num_timesteps, num_features] for weighting columns
        W_dec: SAE decoder weights [num_features, hidden_dim]
        top_k: Either int (fixed k for all timesteps) or list (adaptive k per timestep)
        weighting: Weighting transformation to apply

    Returns:
        steering_vectors: Dict mapping timestep -> steering vector
    """
    num_timesteps = selection_scores.shape[0]
    device = W_dec.device
    dtype = W_dec.dtype

    # Handle adaptive k
    if isinstance(top_k, int):
        k_per_t = [top_k] * num_timesteps
    else:
        k_per_t = top_k

    steering_vectors = {}

    for t in range(num_timesteps):
        k = k_per_t[t]
        # Get top-k indices using selection scores
        _, top_indices = torch.topk(selection_scores[t], k)

        # Get weights for these features
        weights = weight_scores[t, top_indices].to(device=device, dtype=dtype)
        weights = apply_weighting(weights, weighting)

        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones(k, device=device, dtype=dtype) / k

        # Build steering vector as weighted sum of decoder columns
        decoder_cols = W_dec[top_indices]  # [k, hidden_dim]
        sv = (weights.unsqueeze(1) * decoder_cols).sum(dim=0)  # [hidden_dim]

        # Normalize to unit length
        sv = sv / (sv.norm() + 1e-8)

        steering_vectors[t] = sv

    return steering_vectors


def load_notebook_config(config_path):
    """
    Load config saved by sae_feature_score_analysis.ipynb.

    Config contains:
        - concept: str
        - optimal_k: int or list (adaptive k per timestep)
        - selection_method: str (always "tfidf" for selection)
        - weight_source: str ("tfidf", "diff", or "mean_pos")
        - weighting: str ("none", "raw", "softmax", "sqrt", "log")
        - mean_cos_sim: float (similarity to CASteer vectors)
        - all_results_summary: dict (optional, full results)
    """
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    return config


def main(
    concept: str,
    sae_path: str,
    selection_method: str = "tfidf",
    top_k: int = 20,
    sae_type: str = "frequency",
    pool_audio: bool = False,
    pooling: str = "max",
    add_error: bool = False,
    renorm_enabled: bool = False,
    intervention_mode: str = "steering_vector",
    negate_for_uncond: bool = False,
    scores_cache_path: str = None,
    save_dir: str = None,
    config_path: str = None,
    weight_source: str = None,
    weighting: str = "none",
    use_weighted_vectors: bool = False,
    skip_weighting: bool = False,
    vectors_seed: int = VECTORS_SEED,
    generation_seed: int = GENERATION_SEED,
    multipliers: list = None,
):
    """Evaluate SAE-based steering for ACE-Step model.

    Generates steered audio using SAE feature interventions and saves outputs
    in a directory structure compatible with eval_steering_protocol.py.
    See TADA arXiv 2602.11910 §5 for the evaluation protocol.

    Args:
        concept: Concept to steer (piano, drums, mood, tempo, female_vocals)
        sae_path: Path to trained SAE checkpoint
        selection_method: Method for selecting features (diff, ratio, cohens_d, tfidf, linear)
        top_k: Number of top features to use per timestep (int or "adaptive")
        sae_type: SAE type (frequency or sequence)
        pool_audio: Whether to pool audio dimension
        pooling: Pooling method (max or mean)
        add_error: Whether to add reconstruction error
        renorm_enabled: Whether to renormalize activations
        intervention_mode: Intervention mode (steering_vector or weighted_vector)
        negate_for_uncond: Whether to negate for unconditional
        scores_cache_path: Path to cached scores (optional)
        save_dir: Directory to save outputs (auto-generated if None)
        config_path: Path to config file from notebook analysis (optional)
        weight_source: Score type for weighting (tfidf, diff, mean_pos). If None, uses selection_method
        weighting: Weighting transformation (none, raw, softmax, sqrt, log)
        use_weighted_vectors: If True, use weighted SAE vectors instead of feature intervention
        skip_weighting: If True, ignore weighting from config and use simple feature selection
        vectors_seed: Random seed used when collecting concept activations (default: 42)
        generation_seed: Random seed used for evaluation audio generation (default: 2115)
        multipliers: List of alpha multipliers to sweep over; defaults to the standard
            [-50, -47.5, ..., 50] range used in the paper (TADA arXiv 2602.11910 Table 4)
    """
    if multipliers is None:
        multipliers = MULTIPLIERS
    # Load config from notebook if provided
    if config_path is not None:
        print(f"Loading config from {config_path}")
        notebook_config = load_notebook_config(config_path)

        # Override parameters with config values
        if "concept" in notebook_config:
            concept = notebook_config["concept"]
        if "optimal_k" in notebook_config:
            top_k = notebook_config["optimal_k"]  # Can be int or list
        if "selection_method" in notebook_config:
            selection_method = notebook_config["selection_method"]
        if "weight_source" in notebook_config:
            weight_source = notebook_config["weight_source"]
        if "weighting" in notebook_config:
            weighting = notebook_config["weighting"]

        # Enable weighted vectors mode when config specifies weighting
        if weighting != "none" or weight_source is not None:
            use_weighted_vectors = True

        print(f"  Concept: {concept}")
        print(f"  Top-k: {top_k}")
        print(f"  Selection method: {selection_method}")
        print(f"  Weight source: {weight_source}")
        print(f"  Weighting: {weighting}")
        print(f"  Use weighted vectors: {use_weighted_vectors}")

    # Override weighting if skip_weighting is set
    if skip_weighting:
        print("Skipping weighting (--skip_weighting flag set)")
        use_weighted_vectors = False
        weighting = "none"
        weight_source = None

    # Set default weight_source if not specified
    if weight_source is None:
        weight_source = selection_method

    # Validate concept
    assert concept in CONCEPT_TO_PROMPTS, f"Concept {concept} not found. Available: {list(CONCEPT_TO_PROMPTS.keys())}"
    assert concept in CONCEPT_TO_EVAL_PROMPTS, f"Concept {concept} not in eval prompts."

    # Setup save directory (compatible with eval_steering_protocol.py)
    if save_dir is None:
        pooling_desc = f"pool_{pool_audio}"
        if pool_audio:
            pooling_desc += f"_{pooling}"
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Handle adaptive k in directory name
        if isinstance(top_k, list):
            k_desc = f"adaptive_k{min(top_k)}-{max(top_k)}"
        else:
            k_desc = f"k{top_k}"

        # Include weighting info if using weighted vectors
        if use_weighted_vectors:
            weight_desc = f"_{weight_source}_{weighting}"
        else:
            weight_desc = ""

        save_dir = f"steering/outputs/{concept}/sae_{selection_method}/{sae_type}_{k_desc}{weight_desc}_{pooling_desc}_{timestamp}"

    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving outputs to {save_dir}")

    # Compute num_cfg_passes for consistency with CASteer
    num_cfg_passes = compute_num_cfg_passes(ACE_STEP_GUIDANCE_SCALE_TEXT, ACE_STEP_GUIDANCE_SCALE_LYRIC)

    # Initialize pipeline EXACTLY like eval_steering_vectors.py
    steered_pipe = SteeredACEStepPipeline(
        device="cuda",
        steering_vectors=None,
        steer=True,  # Same as SV script
        alpha=0,
        steer_mode="cond_only",
        num_cfg_passes=num_cfg_passes,
    )
    steered_pipe.load()
    print("Pipeline loaded (SteeredACEStepPipeline)")

    # Load SAE
    sae = Sae.load_from_disk(sae_path, device="cuda").eval()
    sae = sae.to(dtype=steered_pipe.dtype)
    print(f"SAE loaded from {sae_path}")

    # Get prompts for concept
    negative_prompts, positive_prompts, concept_lyrics = CONCEPT_TO_PROMPTS[concept]()
    eval_prompts = CONCEPT_TO_EVAL_PROMPTS[concept]

    # Check if we can skip activation collection (scores already cached)
    scores_loaded = False
    if scores_cache_path is not None and os.path.exists(scores_cache_path):
        print(f"Loading precomputed scores from {scores_cache_path}")
        with open(scores_cache_path, "rb") as f:
            cached_scores = pickle.load(f)

        # Helper to extract scores from different formats
        def get_scores_from_cache(method_name, cache):
            """Handle both score file formats."""
            if method_name not in cache:
                return None
            value = cache[method_name]
            # Format 1: {"method": {"scores": tensor}}
            if isinstance(value, dict) and "scores" in value:
                return value["scores"]
            # Format 2: {"method": tensor}
            elif hasattr(value, "shape"):
                return value
            return None

        selection_scores = get_scores_from_cache(selection_method, cached_scores)
        if selection_scores is not None:
            scores_result = {"scores": selection_scores}
            scores_loaded = True
            print(f"Loaded {selection_method} scores (shape: {selection_scores.shape})")

            # Also load weight_source scores if different
            if weight_source != selection_method and use_weighted_vectors:
                weight_scores_tensor = get_scores_from_cache(weight_source, cached_scores)
                if weight_scores_tensor is not None:
                    weight_scores_result = {"scores": weight_scores_tensor}
                    print(f"Loaded {weight_source} weight scores")
                else:
                    print(f"Warning: {weight_source} not in cache, using {selection_method}")
                    weight_scores_result = scores_result
            else:
                weight_scores_result = scores_result

    if not scores_loaded:
        # Need to collect activations - import HookedACEStepModel for this
        from sae_src.hooked_model.hooked_model_acestep import HookedACEStepModel

        print("Collecting activations for feature selection...")
        hooked_model = HookedACEStepModel(
            pipeline=steered_pipe,
            device="cuda",
        )

        latents = steered_pipe.prepare_latents(
            batch_size=len(positive_prompts),
            audio_duration=AUDIO_LENGTH_IN_S,
            seed=vectors_seed,
        )

        out_pos = hooked_model.run_with_cache(
            prompt=positive_prompts,
            audio_duration=AUDIO_LENGTH_IN_S,
            lyrics=concept_lyrics,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            guidance_interval=ACE_STEP_GUIDANCE_INTERVAL,
            guidance_interval_decay=ACE_STEP_GUIDANCE_INTERVAL_DECAY,
            guidance_scale_text=ACE_STEP_GUIDANCE_SCALE_TEXT,
            guidance_scale_lyric=ACE_STEP_GUIDANCE_SCALE_LYRIC,
            manual_seed=vectors_seed,
            latents=latents,
            return_type="audio",
            positions_to_cache=["transformer_blocks.7.cross_attn"],
        )
        positive_activations = out_pos[1]["output"]["transformer_blocks.7.cross_attn"][: len(positive_prompts)]

        out_neg = hooked_model.run_with_cache(
            prompt=negative_prompts,
            audio_duration=AUDIO_LENGTH_IN_S,
            lyrics=concept_lyrics,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            guidance_interval=ACE_STEP_GUIDANCE_INTERVAL,
            guidance_interval_decay=ACE_STEP_GUIDANCE_INTERVAL_DECAY,
            guidance_scale_text=ACE_STEP_GUIDANCE_SCALE_TEXT,
            guidance_scale_lyric=ACE_STEP_GUIDANCE_SCALE_LYRIC,
            manual_seed=vectors_seed,
            latents=latents,
            return_type="audio",
            positions_to_cache=["transformer_blocks.7.cross_attn"],
        )
        negative_activations = out_neg[1]["output"]["transformer_blocks.7.cross_attn"][: len(negative_prompts)]

        # Rearrange for frequency SAE
        if sae_type == "frequency":
            positive_activations = einops.rearrange(positive_activations, "b T f d -> b T d f")
            negative_activations = einops.rearrange(negative_activations, "b T f d -> b T d f")

        print("Activations collected")

        # Compute feature importance scores
        print(f"Computing feature importance using method: {selection_method}")
        selector = FeatureSelector(
            sae=sae,
            positive_acts=positive_activations.cuda(),
            negative_acts=negative_activations.cuda(),
            pool_audio=pool_audio,
            pooling=pooling,
            scores_cache_path=scores_cache_path,
        )
        scores_result = selector.compute_single_method(selection_method, verbose=True)

        # Also compute weight_source scores if different from selection_method
        if weight_source != selection_method and use_weighted_vectors:
            print(f"Computing weight scores using method: {weight_source}")
            weight_scores_result = selector.compute_single_method(weight_source, verbose=True)
        else:
            weight_scores_result = scores_result

    # Handle adaptive k (list) vs fixed k (int)
    if isinstance(top_k, list):
        # For adaptive k, we need to get varying top-k per timestep
        # Build features_per_timestep manually
        selection_scores = scores_result["scores"]
        features_per_timestep = {}
        for t in range(selection_scores.shape[0]):
            k_t = top_k[t]
            _, top_indices = torch.topk(selection_scores[t], k_t)
            features_per_timestep[t] = top_indices.tolist()
        print(f"Selected adaptive k features per timestep: min={min(top_k)}, max={max(top_k)}")
    else:
        # Fixed k for all timesteps - compute top-k directly from scores
        selection_scores = scores_result["scores"]
        features_per_timestep = {}
        for t in range(selection_scores.shape[0]):
            _, top_indices = torch.topk(selection_scores[t], top_k)
            features_per_timestep[t] = top_indices.tolist()
        print(f"Selected top {top_k} features per timestep")

    # Build weighted steering vectors if enabled
    steering_vectors = None
    if use_weighted_vectors:
        print(f"Building weighted steering vectors with weighting={weighting}")
        steering_vectors = build_weighted_steering_vectors(
            selection_scores=scores_result["scores"].cuda(),
            weight_scores=weight_scores_result["scores"].cuda(),
            W_dec=sae.W_dec,
            top_k=top_k,
            weighting=weighting,
        )
        print(f"Built {len(steering_vectors)} steering vectors")

    # Save scores
    scores_save_path = os.path.join(save_dir, "scores.pkl")
    with open(scores_save_path, "wb") as f:
        pickle.dump({selection_method: {"scores": scores_result["scores"]}}, f)

    # Save run config
    run_config = {
        "concept": concept,
        "sae_path": sae_path,
        "sae_type": sae_type,
        "selection_method": selection_method,
        "top_k": top_k if not isinstance(top_k, list) else {"adaptive": top_k},
        "weight_source": weight_source,
        "weighting": weighting,
        "use_weighted_vectors": use_weighted_vectors,
        "pool_audio": pool_audio,
        "pooling": pooling,
        "add_error": add_error,
        "renorm_enabled": renorm_enabled,
        "intervention_mode": intervention_mode,
        "negate_for_uncond": negate_for_uncond,
        "config_path": config_path,
        "eval_prompts": eval_prompts if isinstance(eval_prompts, list) else [eval_prompts],
        "multipliers": multipliers,
        "test_prompts": TEST_PROMPTS,
        "generation_seed": generation_seed,
        "audio_duration": AUDIO_LENGTH_IN_S,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
    }
    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # Prepare test latents
    latents_test = steered_pipe.prepare_latents(
        batch_size=len(TEST_PROMPTS),
        audio_duration=AUDIO_LENGTH_IN_S,
        seed=generation_seed,
    )

    # Use lyrics for vocal concepts
    test_lyrics = NO_LYRICS
    test_prompts = TEST_PROMPTS
    if "vocal" in concept:
        test_lyrics = LYRICS
        test_prompts = [f"{p}, with vocal" for p in TEST_PROMPTS]

    # Locate the block for hook registration (same approach as steering_comparison_interface.ipynb)
    block = locate_block("transformer_blocks.7.cross_attn", steered_pipe.ace_step_transformer)

    # Generate for each multiplier (alpha)
    print(f"Generating audios for {len(multipliers)} alphas...")
    for multiplier in tqdm(multipliers, desc="Generating"):
        # Create alpha directory (compatible with eval_steering_protocol.py)
        alpha_dir = os.path.join(save_dir, f"alpha_{multiplier}")
        os.makedirs(alpha_dir, exist_ok=True)

        # Clear any CASteer hooks before registering SAE hook (like notebook)
        steered_pipe.clear_steering_hooks()

        # Use appropriate hook based on whether we have precomputed weighted vectors
        if use_weighted_vectors and steering_vectors is not None:
            hook = ACEStepPrecomputedSteeringHook(
                steering_vectors=steering_vectors,
                multiplier=multiplier,
                sae_mode=sae_type,
                uncond_preds=negate_for_uncond,
                negate_for_uncond=negate_for_uncond,
            )
        else:
            hook = ACEStepTimestepInterventionHook(
                sae=sae,
                features_per_timestep=features_per_timestep,
                multiplier=multiplier,
                sae_mode=sae_type,
                uncond_preds=negate_for_uncond,
                add_error=add_error,
                renorm=renorm_enabled,
                negate_for_uncond=negate_for_uncond,
                intervention_mode=intervention_mode,
            )
        hook.counter = -1

        # Register hook directly on the block (same as notebook)
        handle = block.register_forward_hook(hook)

        try:
            outputs = steered_pipe.generate(
                prompt=test_prompts,
                lyrics=test_lyrics,
                audio_duration=AUDIO_LENGTH_IN_S,
                infer_step=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                guidance_scale_text=ACE_STEP_GUIDANCE_SCALE_TEXT,
                guidance_scale_lyric=ACE_STEP_GUIDANCE_SCALE_LYRIC,
                guidance_interval=ACE_STEP_GUIDANCE_INTERVAL,
                guidance_interval_decay=ACE_STEP_GUIDANCE_INTERVAL_DECAY,
                manual_seed=generation_seed,
                latents=latents_test,
                return_type="audio",
            )
        finally:
            handle.remove()

        # Save with naming convention compatible with eval_steering_protocol.py
        for i, audio in enumerate(outputs):
            torchaudio.save(
                os.path.join(alpha_dir, f"p{i}.wav"),
                audio.cpu(),
                steered_pipe.sample_rate,
            )

    print(f"\nDone! Results saved to {save_dir}")
    print(f"Run eval_steering_protocol.py with:")
    print(
        f'  python steering/ace_steer/eval_steering_protocol.py --steering_dir "{save_dir}" --eval_prompt "{eval_prompts[0] if isinstance(eval_prompts, list) else eval_prompts}"'
    )


if __name__ == "__main__":
    Fire(main)
