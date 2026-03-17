"""
Concept Algebra Demo — Phase 2.3.

Demonstrates 5 SAE concept algebra expressions using the ConceptAlgebra system.
If ACE-Step model weights are available, generates audio for each expression.
Use --dry-run to evaluate expressions and inspect feature sets without audio generation.

Usage:
    # Dry run (no model needed — just shows algebra results):
    python experiments/concept_algebra_demo.py --dry-run

    # Full run with audio (requires ACE-Step weights and SAE checkpoint):
    python experiments/concept_algebra_demo.py \
        --sae-path /path/to/sae_checkpoint.pt \
        --out-dir experiments/results/concept_algebra \
        --alpha 50 \
        --duration 10.0

Reference: TADA roadmap Prompt 2.3 — arXiv 2602.11910.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAE_ROOT = _REPO_ROOT / "sae"
for _p in [str(_REPO_ROOT), str(_SAE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from steer_audio.concept_algebra import ConceptAlgebra, ConceptFeatureSet
from steer_audio.vector_bank import SteeringVectorBank

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Demo expressions
# ---------------------------------------------------------------------------

# Five concept algebra expressions as specified in the roadmap.
DEMO_EXPRESSIONS = [
    {
        "expr": "jazz + female_vocal",
        "description": "jazz with female vocals",
        "eval_prompt": "jazz music with female vocalist singing",
    },
    {
        "expr": "fast_tempo - drums",
        "description": "fast but without drums",
        "eval_prompt": "fast energetic music without drums",
    },
    {
        "expr": "0.5 * jazz + 0.5 * reggae",
        "description": "jazz-reggae hybrid",
        "eval_prompt": "jazz reggae fusion music",
    },
    {
        "expr": "energetic_mood & guitar",
        "description": "energetic guitar music",
        "eval_prompt": "energetic guitar music",
    },
    {
        "expr": "slow_tempo - sad_mood",
        "description": "slow but not sad (peaceful ballad)",
        "eval_prompt": "slow peaceful ballad music",
    },
]

# Default concept prompt to use for generation.
_DEFAULT_GEN_PROMPT = "a high quality music recording"

# ---------------------------------------------------------------------------
# Helpers: build synthetic concept feature sets for dry-run / testing
# ---------------------------------------------------------------------------


def _build_synthetic_features(
    concepts: list[str],
    hidden_dim: int = 3072,
    num_features: int = 12288,  # expansion_factor=4 × hidden_dim=3072
    tau: int = 20,
    seed: int = 42,
) -> dict[str, ConceptFeatureSet]:
    """Create random ConceptFeatureSets for dry-run / CI testing.

    In production, these would be loaded from a trained SAE checkpoint using
    cached TF-IDF activation analysis (see sae/scripts/eval_sae_steering.py).

    Args:
        concepts:     List of concept names to create feature sets for.
        hidden_dim:   SAE input dimension (ACE-Step layer 7: 3072).
        num_features: Total SAE features (hidden_dim * expansion_factor).
        tau:          Number of top features per concept.
        seed:         Random seed for reproducibility.

    Returns:
        Mapping ``concept_name → ConceptFeatureSet``.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Shared decoder matrix — in production this comes from the trained SAE.
    decoder_matrix = torch.randn(hidden_dim, num_features) * 0.02

    features: dict[str, ConceptFeatureSet] = {}
    for i, concept in enumerate(concepts):
        # Each concept gets a mostly unique set of features with some overlap.
        offset = i * (tau // 2)
        base_indices = list(range(offset, offset + tau))
        # Wrap around to keep within bounds.
        indices = [j % num_features for j in base_indices]
        scores = rng.uniform(0.1, 1.0, size=tau).astype(np.float32)

        features[concept] = ConceptFeatureSet(
            concept=concept,
            feature_indices=np.array(indices, dtype=np.int64),
            tfidf_scores=scores,
            decoder_matrix=decoder_matrix,
        )
    return features


# ---------------------------------------------------------------------------
# Helpers: load real feature sets from SAE checkpoint
# ---------------------------------------------------------------------------


def _load_real_features(
    sae_path: Path,
    concepts: list[str],
    tau: int = 20,
    layer: int = 7,
    device: str = "cpu",
) -> dict[str, ConceptFeatureSet]:
    """Load ConceptFeatureSets computed from a trained SAE checkpoint.

    This function expects a checkpoint produced by ``sae/scripts/train_ace.py``
    and pre-computed TF-IDF scores from cached activations.

    Args:
        sae_path: Path to the trained SAE ``.pt`` or ``.safetensors`` checkpoint.
        concepts: List of concept names to load.
        tau:      Number of top-τ features per concept.
        layer:    ACE-Step layer index (default: 7).
        device:   Torch device string.

    Returns:
        Mapping ``concept_name → ConceptFeatureSet``.

    Raises:
        FileNotFoundError: If *sae_path* or any per-concept activation cache
            is missing.
        ImportError: If ``sae_src`` package is not importable.
    """
    from sae_src.sae.sae import Sae
    from sae_src.sae.config import SaeConfig
    from tfidf_utils import compute_tfidf_scores, top_tau_features

    log.info("Loading SAE checkpoint from %s", sae_path)

    # Infer hidden_dim from checkpoint.
    ckpt = torch.load(sae_path, map_location=device)
    if isinstance(ckpt, dict) and "cfg" in ckpt:
        cfg = ckpt["cfg"]
    else:
        cfg = SaeConfig()  # use paper defaults

    # Try to load as safetensors first, fall back to torch.load.
    try:
        from safetensors.torch import load_file as sf_load
        state_dict = sf_load(str(sae_path))
    except Exception:
        state_dict = torch.load(sae_path, map_location=device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

    # Infer d_in from encoder.weight.
    d_in = state_dict["encoder.weight"].shape[1]
    sae = Sae(d_in=d_in, cfg=cfg, device=device)
    sae.load_state_dict(state_dict, strict=False)
    sae.eval()

    # Decoder matrix: W_dec shape (num_features, hidden_dim) → transpose to (hidden_dim, num_features)
    w_dec = sae.W_dec.detach().float().T.contiguous()  # (hidden_dim, num_features)

    features: dict[str, ConceptFeatureSet] = {}
    sae_dir = sae_path.parent

    for concept in concepts:
        # Expect cached activation files: {concept}_pos.pt and {concept}_neg.pt
        pos_path = sae_dir / f"{concept}_pos.pt"
        neg_path = sae_dir / f"{concept}_neg.pt"

        if not pos_path.exists() or not neg_path.exists():
            log.warning(
                "Activation cache for concept '%s' not found at %s / %s; "
                "skipping and using synthetic features.",
                concept, pos_path, neg_path,
            )
            rng = np.random.default_rng(hash(concept) % (2**31))
            num_features = w_dec.shape[1]
            indices = rng.choice(num_features, size=tau, replace=False).astype(np.int64)
            scores = rng.uniform(0.1, 1.0, size=tau).astype(np.float32)
            features[concept] = ConceptFeatureSet(
                concept=concept,
                feature_indices=np.sort(indices),
                tfidf_scores=scores,
                decoder_matrix=w_dec,
            )
            continue

        pos_acts = torch.load(pos_path, map_location=device)  # (n_pos, num_features)
        neg_acts = torch.load(neg_path, map_location=device)  # (n_neg, num_features)

        scores_tensor = compute_tfidf_scores(pos_acts, neg_acts)
        indices_tensor = top_tau_features(scores_tensor, tau)

        feat_indices = indices_tensor.cpu().numpy().astype(np.int64)
        feat_scores = scores_tensor[indices_tensor].cpu().numpy().astype(np.float32)

        features[concept] = ConceptFeatureSet(
            concept=concept,
            feature_indices=feat_indices,
            tfidf_scores=feat_scores,
            decoder_matrix=w_dec,
        )
        log.info("Loaded %d features for concept '%s'.", tau, concept)

    return features


# ---------------------------------------------------------------------------
# Audio generation (requires model)
# ---------------------------------------------------------------------------


def _generate_audio(
    sv_vector: torch.Tensor,
    layers: list[int],
    model: object,
    prompt: str,
    alpha: float,
    duration: float,
    seed: int,
    out_path: Path,
) -> None:
    """Apply a steering vector during ACE-Step inference and save the result.

    Args:
        sv_vector: Steering direction, shape ``(hidden_dim,)``.
        layers:    Transformer-block indices to hook.
        model:     Loaded ACE-Step pipeline.
        prompt:    Text prompt for generation.
        alpha:     Steering strength.
        duration:  Audio duration in seconds.
        seed:      Random seed for reproducibility.
        out_path:  WAV output path.
    """
    import soundfile as sf
    from steer_audio.vector_bank import SteeringVector
    from steer_audio.multi_steer import MultiConceptSteerer

    sv = SteeringVector(
        concept="algebra_result",
        method="sae",
        model_name="ace-step",
        layers=layers,
        vector=sv_vector,
        alpha_range=(-100.0, 100.0),
    )
    steerer = MultiConceptSteerer({"algebra_result": sv}, orthogonalize=False)
    audio, sr = steerer.steer(
        model=model,
        prompt=prompt,
        alphas={"algebra_result": alpha},
        duration=duration,
        seed=seed,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if audio.ndim == 1:
        sf.write(str(out_path), audio, sr)
    else:
        sf.write(str(out_path), audio.T, sr)

    log.info("Saved audio → %s (sr=%d, shape=%s)", out_path, sr, audio.shape)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate algebra expressions only; do not load the model or generate audio.",
    )
    p.add_argument(
        "--sae-path",
        type=Path,
        default=None,
        help="Path to trained SAE checkpoint (.pt or .safetensors).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("TADA_WORKDIR", "experiments/results")) / "concept_algebra",
        help="Directory where audio files and the overlap heatmap are saved.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=50.0,
        help="Steering strength α applied during generation.",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Generated audio duration in seconds.",
    )
    p.add_argument(
        "--tau",
        type=int,
        default=20,
        help="Number of top-τ SAE features per concept.",
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[6, 7],
        help="Transformer-block indices to steer (default: 6 7).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility.",
    )
    p.add_argument(
        "--gen-prompt",
        type=str,
        default=_DEFAULT_GEN_PROMPT,
        help="Text prompt used for all audio generations.",
    )
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=3072,
        help="SAE input dimension (ACE-Step layer 7 default: 3072).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()

    # All concepts referenced in the 5 demo expressions.
    all_concepts = [
        "jazz",
        "female_vocal",
        "fast_tempo",
        "drums",
        "reggae",
        "energetic_mood",
        "guitar",
        "slow_tempo",
        "sad_mood",
    ]

    # ------------------------------------------------------------------ #
    # Load or synthesise concept feature sets
    # ------------------------------------------------------------------ #
    if args.sae_path is not None and args.sae_path.exists():
        log.info("Loading real SAE features from %s", args.sae_path)
        try:
            features = _load_real_features(
                sae_path=args.sae_path,
                concepts=all_concepts,
                tau=args.tau,
                device="cpu",
            )
        except Exception as exc:
            log.warning(
                "Failed to load real SAE features (%s); falling back to synthetic.",
                exc,
            )
            features = _build_synthetic_features(
                concepts=all_concepts,
                hidden_dim=args.hidden_dim,
                tau=args.tau,
                seed=args.seed,
            )
    else:
        if not args.dry_run:
            log.warning(
                "--sae-path not provided or not found; using synthetic feature sets. "
                "Generated audio will not reflect real SAE features."
            )
        features = _build_synthetic_features(
            concepts=all_concepts,
            hidden_dim=args.hidden_dim,
            tau=args.tau,
            seed=args.seed,
        )

    algebra = ConceptAlgebra(sae_model=None, concept_features=features)

    # ------------------------------------------------------------------ #
    # Save feature overlap heatmap
    # ------------------------------------------------------------------ #
    args.out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = args.out_dir / "feature_overlap.png"
    fig = algebra.feature_overlap_heatmap()
    fig.savefig(str(heatmap_path), dpi=150, bbox_inches="tight")
    log.info("Feature overlap heatmap saved → %s", heatmap_path)

    # ------------------------------------------------------------------ #
    # Load ACE-Step model (if not dry-run)
    # ------------------------------------------------------------------ #
    model = None
    if not args.dry_run:
        try:
            log.info("Loading ACE-Step model …")
            from acestep.pipeline_ace_step import ACEStepPipeline  # type: ignore

            model_device = "cuda" if torch.cuda.is_available() else "cpu"
            model = ACEStepPipeline.from_pretrained(
                "ACE-Step/ACE-Step-v1-3.5B",
                torch_dtype=torch.float16 if model_device == "cuda" else torch.float32,
            ).to(model_device)
            log.info("Model loaded on %s.", model_device)
        except Exception as exc:
            log.warning(
                "Could not load ACE-Step model (%s). "
                "Run with --dry-run to skip audio generation.",
                exc,
            )
            model = None

    # ------------------------------------------------------------------ #
    # Evaluate each demo expression
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("  SAE Concept Algebra Demo — Phase 2.3")
    print("=" * 70)

    bank = SteeringVectorBank()

    for i, demo in enumerate(DEMO_EXPRESSIONS, start=1):
        expr_str = demo["expr"]
        description = demo["description"]
        eval_prompt = demo["eval_prompt"]

        print(f"\n[{i}/5] Expression: {expr_str!r}")
        print(f"      Description: {description}")

        # Evaluate algebra expression.
        try:
            result_cfs = algebra.expr(expr_str)
        except Exception as exc:
            log.error("Expression evaluation failed: %s", exc)
            raise

        print(
            f"      Features: {len(result_cfs.feature_indices)} "
            f"(tau per concept: {args.tau})"
        )
        print(f"      Concept string: {result_cfs.concept}")

        # Build steering vector.
        sv = algebra.to_steering_vector(
            result_cfs,
            layers=args.layers,
            model_name="ace-step",
        )
        print(f"      Steering vector norm: {sv.vector.norm().item():.4f}")

        # Save steering vector to disk.
        safe_name = expr_str.replace(" ", "_").replace("*", "x").replace("/", "_").replace("+", "plus").replace("-", "minus").replace(".", "p").replace("&", "and")
        sv_path = args.out_dir / f"sv_{i:02d}_{safe_name}.safetensors"
        try:
            bank.save(sv, sv_path)
            print(f"      SteeringVector saved → {sv_path}")
        except Exception as exc:
            log.warning("Could not save SteeringVector: %s", exc)

        # Generate audio.
        if model is not None:
            out_wav = args.out_dir / f"audio_{i:02d}_{safe_name}.wav"
            try:
                _generate_audio(
                    sv_vector=sv.vector,
                    layers=args.layers,
                    model=model,
                    prompt=args.gen_prompt,
                    alpha=args.alpha,
                    duration=args.duration,
                    seed=args.seed + i,
                    out_path=out_wav,
                )
                print(f"      Audio saved → {out_wav}")
            except Exception as exc:
                log.error("Audio generation failed for expression %r: %s", expr_str, exc)
                raise
        else:
            print(
                "      (Audio generation skipped — model not loaded. "
                "Use --sae-path and remove --dry-run to enable.)"
            )

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print(f"  Done. Outputs written to: {args.out_dir}")
    print(f"  Feature overlap heatmap:  {heatmap_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
