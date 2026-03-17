"""
Multi-concept steering experiment (TADA Roadmap Prompt 2.1).

Tests all 10 concept pairs drawn from
{tempo, mood, vocal_gender, guitar, drums, jazz, techno}.

For each pair the experiment:
  1. Generates audio using individual steering (one concept at a time).
  2. Generates audio using joint steering (both concepts simultaneously).
  3. Computes the interference matrix before and after Gram-Schmidt.
  4. Measures CLAP alignment per concept for both conditions.
  5. Saves results to ``results/multi_concept/interference_matrix.csv``.

Usage:
    python experiments/multi_concept_experiment.py \
        --vectors_dir vectors/ace-step \
        --model_dir res/ace_step \
        --output_dir results/multi_concept \
        [--n_samples 4] \
        [--alpha 50.0] \
        [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so steer_audio and steering are importable.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from steer_audio.vector_bank import SteeringVector, SteeringVectorBank
from steer_audio.multi_steer import MultiConceptSteerer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The 7 concepts used to build the 10 concept pairs (C(7,2) = 21 pairs, but
# the roadmap specifies 10 representative ones).
CONCEPTS_ALL = ["tempo", "mood", "vocal_gender", "guitar", "drums", "jazz", "techno"]

CONCEPT_PAIRS = list(itertools.combinations(CONCEPTS_ALL, 2))[:10]  # first 10 pairs

DEFAULT_ALPHA = 50.0
DEFAULT_N_SAMPLES = 4
DEFAULT_SEED = 42

# Prompt per concept for CLAP evaluation.
CONCEPT_EVAL_PROMPTS: dict[str, tuple[str, str]] = {
    "tempo":        ("fast energetic music", "slow relaxed music"),
    "mood":         ("happy upbeat music", "sad melancholic music"),
    "vocal_gender": ("music with female vocals", "music with male vocals"),
    "guitar":       ("music with prominent guitar", "music without guitar"),
    "drums":        ("music with prominent drums and percussion", "music without drums"),
    "jazz":         ("jazz music", "non-jazz music"),
    "techno":       ("techno electronic music", "acoustic non-electronic music"),
}


# ---------------------------------------------------------------------------
# CLAP alignment helper
# ---------------------------------------------------------------------------


def _clap_alignment(audio: np.ndarray, sr: int, prompt: str) -> float:
    """Compute CLAP cosine similarity between *audio* and *prompt*.

    Requires ``laion_clap`` (or a compatible CLAP library) to be installed.
    Falls back to a placeholder value (0.0) with a warning if unavailable.

    Args:
        audio:  Mono or stereo numpy array.
        sr:     Sample rate of *audio*.
        prompt: Text description to compare against.

    Returns:
        Cosine similarity in ``[-1, 1]``.
    """
    try:
        import laion_clap  # type: ignore

        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()

        # Audio embedding
        audio_mono = audio if audio.ndim == 1 else audio.mean(axis=0)
        audio_data = {
            "waveform": torch.from_numpy(audio_mono).unsqueeze(0).float(),
            "sample_rate": sr,
        }
        audio_emb = model.get_audio_embedding_from_data(audio_data, use_tensor=True)

        # Text embedding
        text_emb = model.get_text_embedding([prompt], use_tensor=True)

        sim = torch.nn.functional.cosine_similarity(audio_emb, text_emb).item()
        return float(sim)
    except Exception as exc:  # noqa: BLE001
        log.warning("CLAP alignment unavailable (%s); returning 0.0.", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Core experiment helpers
# ---------------------------------------------------------------------------


def _load_vectors(
    vectors_dir: Path,
    concepts: list[str],
) -> dict[str, SteeringVector]:
    """Load steering vectors for *concepts* from *vectors_dir*.

    Only concepts for which a ``.safetensors`` file exists are returned.
    A warning is emitted for each missing concept.

    Args:
        vectors_dir: Directory scanned by :class:`SteeringVectorBank`.
        concepts:    Concept names to load.

    Returns:
        ``{concept_name: SteeringVector}`` for available concepts.
    """
    bank = SteeringVectorBank()
    all_vecs = bank.load_all(vectors_dir)

    result: dict[str, SteeringVector] = {}
    for concept in concepts:
        # Try exact key match first, then fall-back prefix search.
        for key, sv in all_vecs.items():
            if sv.concept == concept:
                result[concept] = sv
                break
        else:
            log.warning(
                "No steering vector found for concept '%s' in %s.", concept, vectors_dir
            )
    return result


def _make_dummy_vector(concept: str, hidden_dim: int = 3072) -> SteeringVector:
    """Create a random unit-norm SteeringVector for dry-run / testing.

    Args:
        concept:    Concept name.
        hidden_dim: Dimension of the random vector.

    Returns:
        A :class:`SteeringVector` with a random unit-norm direction.
    """
    torch.manual_seed(abs(hash(concept)) % (2**31))
    v = torch.randn(hidden_dim)
    v = v / v.norm()
    return SteeringVector(
        concept=concept,
        method="caa",
        model_name="ace-step",
        layers=[6, 7],
        vector=v,
        clap_delta=float(torch.rand(1).item()),  # random quality signal for ordering
        lpaps_at_50=float(torch.rand(1).item()),
    )


def run_pair_experiment(
    concept_a: str,
    concept_b: str,
    sv_a: SteeringVector,
    sv_b: SteeringVector,
    model: object | None,
    alpha: float,
    n_samples: int,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    """Run individual vs. joint steering for one concept pair.

    Args:
        concept_a, concept_b: Names of the two concepts.
        sv_a, sv_b:           Corresponding steering vectors.
        model:                Loaded ACE-Step model or ``None`` for dry-run.
        alpha:                Steering strength applied to each concept.
        n_samples:            Number of audio clips to generate per condition.
        seed:                 Base random seed.
        output_dir:           Where to save generated audio.

    Returns:
        Dict with interference cosine similarities and CLAP scores for each
        condition and concept.
    """
    pair_label = f"{concept_a}+{concept_b}"
    log.info("Running pair: %s", pair_label)

    # Interference matrix (before orthogonalization)
    steerer_plain = MultiConceptSteerer(
        {concept_a: sv_a, concept_b: sv_b}, orthogonalize=False
    )
    imat_before = steerer_plain.interference_matrix()  # (2, 2)

    # Interference matrix (after orthogonalization)
    steerer_ortho = MultiConceptSteerer(
        {concept_a: sv_a, concept_b: sv_b}, orthogonalize=True
    )
    imat_after = steerer_ortho.interference_matrix()  # (2, 2)

    cos_before = float(imat_before[0, 1].item())  # off-diagonal = interference
    cos_after = float(imat_after[0, 1].item())

    log.info(
        "  Interference cos similarity — before GS: %.4f | after GS: %.4f",
        cos_before,
        cos_after,
    )

    result: dict[str, object] = {
        "pair": pair_label,
        "concept_a": concept_a,
        "concept_b": concept_b,
        "cos_before_gs": cos_before,
        "cos_after_gs": cos_after,
    }

    if model is None:
        log.info("  No model provided — skipping audio generation (dry-run).")
        result.update(
            {
                "clap_a_individual": np.nan,
                "clap_b_individual": np.nan,
                "clap_a_joint": np.nan,
                "clap_b_joint": np.nan,
            }
        )
        return result

    # Audio generation — individual steering.
    prompt = "a piece of music"
    audio_dir_pair = output_dir / pair_label
    audio_dir_pair.mkdir(parents=True, exist_ok=True)

    clap_a_individual: list[float] = []
    clap_b_individual: list[float] = []
    clap_a_joint: list[float] = []
    clap_b_joint: list[float] = []

    eval_prompt_a = CONCEPT_EVAL_PROMPTS.get(concept_a, ("music with " + concept_a, "music"))[0]
    eval_prompt_b = CONCEPT_EVAL_PROMPTS.get(concept_b, ("music with " + concept_b, "music"))[0]

    for i in range(n_samples):
        sample_seed = seed + i

        # --- Individual A ---
        steerer_a = MultiConceptSteerer({concept_a: sv_a})
        audio_a, sr = steerer_a.steer(
            model, prompt, {concept_a: alpha}, seed=sample_seed
        )
        clap_a_individual.append(_clap_alignment(audio_a, sr, eval_prompt_a))
        _save_wav(audio_dir_pair / f"individual_{concept_a}_{i}.wav", audio_a, sr)

        # --- Individual B ---
        steerer_b = MultiConceptSteerer({concept_b: sv_b})
        audio_b, sr = steerer_b.steer(
            model, prompt, {concept_b: alpha}, seed=sample_seed
        )
        clap_b_individual.append(_clap_alignment(audio_b, sr, eval_prompt_b))
        _save_wav(audio_dir_pair / f"individual_{concept_b}_{i}.wav", audio_b, sr)

        # --- Joint (both concepts) ---
        audio_joint, sr = steerer_plain.steer(
            model,
            prompt,
            {concept_a: alpha, concept_b: alpha},
            seed=sample_seed,
        )
        clap_a_joint.append(_clap_alignment(audio_joint, sr, eval_prompt_a))
        clap_b_joint.append(_clap_alignment(audio_joint, sr, eval_prompt_b))
        _save_wav(audio_dir_pair / f"joint_{pair_label}_{i}.wav", audio_joint, sr)

    result.update(
        {
            "clap_a_individual": float(np.mean(clap_a_individual)),
            "clap_b_individual": float(np.mean(clap_b_individual)),
            "clap_a_joint": float(np.mean(clap_a_joint)),
            "clap_b_joint": float(np.mean(clap_b_joint)),
        }
    )
    log.info(
        "  CLAP %s: individual=%.4f | joint=%.4f",
        concept_a,
        result["clap_a_individual"],
        result["clap_a_joint"],
    )
    log.info(
        "  CLAP %s: individual=%.4f | joint=%.4f",
        concept_b,
        result["clap_b_individual"],
        result["clap_b_joint"],
    )
    return result


def _save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """Save *audio* as a WAV file using soundfile or scipy.

    Args:
        path:  Destination file path.
        audio: Numpy audio array (mono or stereo).
        sr:    Sample rate.
    """
    try:
        import soundfile as sf  # type: ignore
        sf.write(str(path), audio.T if audio.ndim == 2 else audio, sr)
    except ImportError:
        try:
            from scipy.io import wavfile  # type: ignore
            wav_data = (audio * 32767).astype(np.int16)
            wavfile.write(str(path), sr, wav_data.T if wav_data.ndim == 2 else wav_data)
        except ImportError:
            log.warning(
                "Neither soundfile nor scipy available; skipping WAV save for %s.", path
            )


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------


_CSV_FIELDS = [
    "pair",
    "concept_a",
    "concept_b",
    "cos_before_gs",
    "cos_after_gs",
    "clap_a_individual",
    "clap_b_individual",
    "clap_a_joint",
    "clap_b_joint",
]


def save_results(results: list[dict], output_dir: Path) -> Path:
    """Write *results* to ``interference_matrix.csv`` in *output_dir*.

    Args:
        results:    List of per-pair result dicts from :func:`run_pair_experiment`.
        output_dir: Directory for the CSV file (created if needed).

    Returns:
        Path to the written CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "interference_matrix.csv"

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    log.info("Results written to %s", csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-concept steering experiment (TADA Prompt 2.1)."
    )
    parser.add_argument(
        "--vectors_dir",
        type=Path,
        default=Path(os.environ.get("TADA_WORKDIR", str(Path.home() / "tada_outputs")))
        / "vectors" / "ace-step",
        help="Directory containing pre-computed .safetensors steering vectors.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("res/ace_step"),
        help="Path to ACE-Step checkpoint directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(os.environ.get("TADA_WORKDIR", str(Path.home() / "tada_outputs")))
        / "results" / "multi_concept",
        help="Directory for generated audio and CSV output.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Number of audio samples per condition.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Steering alpha applied to each concept.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base random seed.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip audio generation (compute interference matrices only).",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    """Run the multi-concept steering experiment end-to-end."""
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("=== Multi-Concept Steering Experiment ===")
    log.info("Concepts: %s", CONCEPTS_ALL)
    log.info("Pairs to test: %d", len(CONCEPT_PAIRS))

    # --- Load steering vectors (or create dummies for dry-run) ---
    vectors: dict[str, SteeringVector] = {}
    if args.vectors_dir.exists():
        vectors = _load_vectors(args.vectors_dir, CONCEPTS_ALL)
        log.info("Loaded %d steering vectors from %s", len(vectors), args.vectors_dir)
    else:
        log.warning(
            "Vectors directory %s not found — using random dummy vectors.",
            args.vectors_dir,
        )

    # Fill in dummies for any missing concepts.
    for concept in CONCEPTS_ALL:
        if concept not in vectors:
            log.info("Creating dummy vector for '%s'.", concept)
            vectors[concept] = _make_dummy_vector(concept)

    # --- Optionally load the model ---
    model = None
    if not args.dry_run:
        try:
            from src.models.ace_step.steering_ace import SteeredACEStepPipeline  # noqa: PLC0415

            log.info("Loading ACE-Step model from %s …", args.model_dir)
            model = SteeredACEStepPipeline(
                persistent_storage_path=str(args.model_dir),
                device="cuda" if torch.cuda.is_available() else "cpu",
                steer=False,  # hooks injected by MultiConceptSteerer
            )
            model.load()
            log.info("Model loaded.")
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failed to load ACE-Step model (%s). Falling back to dry-run.", exc
            )
            model = None

    # --- Run experiments ---
    results: list[dict] = []
    for concept_a, concept_b in CONCEPT_PAIRS:
        sv_a = vectors[concept_a]
        sv_b = vectors[concept_b]
        row = run_pair_experiment(
            concept_a=concept_a,
            concept_b=concept_b,
            sv_a=sv_a,
            sv_b=sv_b,
            model=model,
            alpha=args.alpha,
            n_samples=args.n_samples,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        results.append(row)

    # --- Print interference summary ---
    print("\n=== Interference Matrix Summary ===")
    print(f"{'Pair':<28}  {'cos_before_GS':>15}  {'cos_after_GS':>13}")
    print("-" * 60)
    for row in results:
        print(
            f"{row['pair']:<28}  {row['cos_before_gs']:>15.4f}  "
            f"{row['cos_after_gs']:>13.4f}"
        )

    # --- Save CSV ---
    csv_path = save_results(results, args.output_dir)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
