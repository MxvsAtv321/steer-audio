#!/usr/bin/env python3
"""
score_human_eval_muq_mulan.py — Compute MuQ-MuLan text-audio similarity for
all existing human eval WAVs without regenerating any audio.

Usage:
    python scripts/score_human_eval_muq_mulan.py \
        [--root results/paper/human_eval] \
        [--output_csv results/paper/human_eval_muq_mulan.csv]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchaudio
import torchaudio.functional as taf

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
# Audio helper
# ---------------------------------------------------------------------------


def load_audio_mono(path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    """Load a WAV, resample to *target_sr*, and mix down to mono.

    Returns:
        waveform : (1, T) float tensor at *target_sr*
        target_sr: the sample rate of the returned waveform
    """
    waveform, sr = torchaudio.load(str(path))
    if sr != target_sr:
        waveform = taf.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
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
    # Iterate concept directories
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
            text_embeds = mulan.extract_text_latents([text])

        wav_files = sorted(concept_dir.glob("*.wav"))
        scored = 0

        for wav_path in wav_files:
            m = wav_pat.match(wav_path.name)
            if m is None:
                continue  # skip files that don't match the naming convention

            pair_id = m.group(1)       # e.g. "00"
            condition = m.group(2)     # "unsteered" or "steered"

            waveform, _ = load_audio_mono(wav_path, target_sr)
            waveform = waveform.to(device)

            with torch.no_grad():
                audio_embeds = mulan.extract_audio_latents(waveform)
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


if __name__ == "__main__":
    main()
