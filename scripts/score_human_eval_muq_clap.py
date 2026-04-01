#!/usr/bin/env python3
"""
score_human_eval_muq_clap.py — Compute MuQ-MuLan text-audio similarity for
all existing human eval WAVs without regenerating any audio.

This script uses the correct MuQ-MuLan API:
  mulan.get_audio_embeddings(wav_tensor)  — audio embeddings (B, T float32 at 24kHz)
  mulan.get_text_embeddings([text])       — text embeddings (list of strings)
  mulan.calc_similarity()                 — cosine similarity

Uses soundfile + scipy for audio loading to avoid the TorchCodec / FFmpeg
crash that occurs when torchaudio routes through libtorchcodec on some pods.

Usage:
    python scripts/score_human_eval_muq_clap.py \
        [--root results/paper/human_eval] \
        [--output_csv results/paper/human_eval_muq_mulan.csv]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from math import gcd
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

# ---------------------------------------------------------------------------
# Concept → text mapping
# ---------------------------------------------------------------------------

CONCEPT_TEXT: dict[str, str] = {
    "piano": "piano",
    "tempo": "fast tempo",
    "mood": "happy mood",
    "drums": "drums",
    "jazz": "jazz",
}

# ---------------------------------------------------------------------------
# Audio helper (soundfile + scipy, no torchaudio)
# ---------------------------------------------------------------------------


def load_audio_mono(path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    """Load a WAV with soundfile, mix to mono, resample to *target_sr*.

    Uses scipy.signal.resample_poly for high-quality rational resampling
    without requiring torchaudio or FFmpeg.

    Returns:
        waveform  : float32 tensor shaped (1, n_samples) at *target_sr*
        target_sr : the sample rate of the returned waveform
    """
    data, sr = sf.read(str(path), always_2d=True)  # shape: (T, C)

    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]

    if sr != target_sr:
        g = gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        data = resample_poly(data, up, down, axis=0)

    waveform = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
    return waveform, target_sr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score human eval WAVs with MuQ-MuLan text-audio similarity."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/paper/human_eval"),
        help="Root directory containing concept subdirectories of WAV files.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("results/paper/human_eval_muq_mulan.csv"),
        help="Destination CSV file.",
    )
    args = parser.parse_args()

    root: Path = args.root
    output_csv: Path = args.output_csv

    if not root.exists():
        print(f"ERROR: root directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("Loading MuQ-MuLan …")
    from muq import MuQMuLan  # type: ignore

    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mulan = mulan.to(device).eval()
    print(f"  Model on {device}\n")

    target_sr: int = int(getattr(mulan, "sr", 24000))
    print(f"  Target sample rate: {target_sr} Hz\n")

    # ------------------------------------------------------------------
    # Iterate concept directories (skip algebra)
    # ------------------------------------------------------------------
    wav_pat = re.compile(r"^pair_(\d+)_(unsteered|steered)\.wav$")

    rows: list[dict] = []

    concept_dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name != "algebra"
    )

    if not concept_dirs:
        print(f"No concept subdirectories found under {root}", file=sys.stderr)
        sys.exit(1)

    for concept_dir in concept_dirs:
        concept = concept_dir.name
        text = CONCEPT_TEXT.get(concept, concept)

        print(f"Concept: {concept!r}  →  text: {text!r}")

        # Compute text embedding once per concept
        with torch.no_grad():
            text_embeds = mulan.get_text_embeddings([text])

        wav_files = sorted(concept_dir.glob("*.wav"))
        scored = 0

        for wav_path in wav_files:
            m = wav_pat.match(wav_path.name)
            if m is None:
                continue

            pair_id = m.group(1)
            condition = m.group(2)

            waveform, _ = load_audio_mono(wav_path, target_sr)
            waveform = waveform.to(device)

            with torch.no_grad():
                audio_embeds = mulan.get_audio_embeddings(waveform)
                sim = mulan.calc_similarity(audio_embeds, text_embeds)

            sim_val = float(sim.squeeze().cpu())

            rows.append(
                {
                    "concept": concept,
                    "pair": pair_id,
                    "condition": condition,
                    "path": str(wav_path),
                    "muq_mulan_sim": f"{sim_val:.6f}",
                }
            )
            scored += 1

        print(f"  Scored {scored} files\n")

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["concept", "pair", "condition", "path", "muq_mulan_sim"]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")

    # Print first 5 rows
    print("\nFirst 5 rows:")
    print(",".join(fieldnames))
    for row in rows[:5]:
        print(",".join(str(row[k]) for k in fieldnames))


if __name__ == "__main__":
    main()
