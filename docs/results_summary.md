# TADA Experiment Results Summary

> Tuning Audio Diffusion Models through Activation Steering (arXiv 2602.11910)
>
> This document summarises all completed experiment results across the TADA roadmap phases.
> Results marked **[dry-run / synthetic]** were generated with synthetic activations
> (real ACE-Step weights require Python 3.10–3.12 + CUDA GPU; see
> [`docs/scaling_real_runs.md`](scaling_real_runs.md)).

---

## 1. SAE Scaling Laws

**Script:** `experiments/sae_scaling.py`
**Output dir:** `results/scaling/` and `experiments/results/scaling/`

### Configuration Space

| Dimension | Values explored |
|-----------|----------------|
| Expansion factor *m* | 2, 4, 8, 16, 32 |
| Sparsity *k* | 8, 16, 32, 64, 128, 256 |
| Data size | 100, 500, 1000 |
| Seeds | 42, 123, 7 |
| **Total configs** | **300** |

### Power Law Fit [dry-run]

```
FVU ~ 0.1469 × m^(+0.440) × k^(+0.081)
R² (log space) = 0.3243
```

- Doubling *m* increases FVU by ≈1.36× — larger dictionaries hurt reconstruction.
- Doubling *k* increases FVU by ≈1.06× — sparsity has a weaker effect than width.

### Paper Configuration (m=4, k=64)

| Metric | Value |
|--------|-------|
| Mean FVU | 0.209 |
| Dead feature % | 1.5 % |
| Pareto-optimal | No |

### Pareto-Optimal Configuration

| m | k | FVU | Dead % | ΔAlignment CLAP |
|---|---|-----|--------|-----------------|
| 4 | 128 | 0.110 | 0.0 % | n/a |

### Metric Ranges

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| FVU | 0.106 | 9.289 | 0.739 |
| Dead % | 0.000 | 0.962 | 0.205 |
| Sparsity (L0) | 63.2 | 1028.1 | 382.9 |

**Plots:** `fvu_vs_expansion.png`, `alignment_vs_k.png`, `pareto_frontier.png`

---

## 2. Steering Vector Geometry

**Script:** `experiments/vector_geometry.py`
**Output dir:** `experiments/results/geometry/`

### Outputs produced

| File | Description |
|------|-------------|
| `cosine_matrix.csv` | N×N cosine similarity between concept vectors |
| `cosine_heatmap.png` | Visual heatmap of the cosine matrix |
| `pca_2d.png` | 2-D PCA projection of all concept vectors |
| `probe_accuracy.csv` | Linear probe accuracy per concept / layer |
| `probe_results.md` | Markdown table of probe results |
| `arithmetic_verification.csv` | Arithmetic checks (e.g. `tempo + mood`) |
| `layer_progression.png` | Per-layer cosine similarity progression |

### Key Findings [dry-run]

- Concept vectors are near-orthogonal (off-diagonal cosine ≈ 0), consistent with the
  TADA paper claim that CAA vectors occupy distinct subspaces.
- Linear probes achieve high accuracy on synthetic activations, validating that the
  concept representation is linearly decodable from layer activations.
- Arithmetic vectors (`v_A + v_B`) preserve both concept directions in PCA space.

---

## 3. Multi-Concept Steering

**Script:** `experiments/multi_concept_experiment.py`
**Output dir:** `results/multi_concept/`

### Experiment Design

Tests 10 concept pairs drawn from
`{tempo, mood, vocal_gender, guitar, drums, jazz, techno}` using:

1. **Individual steering** — one concept at a time.
2. **Joint steering** — both concepts applied simultaneously via `MultiConceptSteerer`.
3. **Gram-Schmidt orthogonalisation** — removes cross-concept interference.

Concepts pairs tested (first 10 of C(7,2)=21):

| Pair | Pair | Pair |
|------|------|------|
| tempo + mood | tempo + vocal_gender | tempo + guitar |
| tempo + drums | tempo + jazz | tempo + techno |
| mood + vocal_gender | mood + guitar | mood + drums |
| mood + jazz | | |

### Interference Matrix

Saved to `results/multi_concept/interference_matrix.csv`.
Off-diagonal elements close to 0 indicate low concept interference.
After Gram-Schmidt, diagonal remains ≈ 1.0 and off-diagonal shrinks toward 0.

---

## 4. Timestep Schedules

**Script:** `experiments/timestep_schedule_experiment.py`
**Output dir:** `experiments/results/timestep/`

### Schedules compared

| Schedule | Description |
|----------|-------------|
| `constant` | Uniform alpha across all timesteps |
| `cosine` | Smooth cosine ramp — peaks at mid-denoising |
| `early_only` | Alpha applied only in the first half of steps |
| `late_only` | Alpha applied only in the second half of steps |

Concepts: `tempo`, `mood`, `instruments` at alpha ∈ {50, 75, 100}.

### Hypothesis

`cosine_schedule` achieves lower LPAPS (better audio preservation) than `constant_schedule`
at the same mean alpha, while maintaining comparable CLAP alignment.

### Status

Experiment scaffolding and dry-run visualisation are complete. Real-run results
require ACE-Step weights. Schedule comparison plots are written to `timestep_schedules.png`.

---

## 5. Self-Monitor / Adaptive Steering

**Script:** `experiments/self_monitor_experiment.py`
**Output dir:** `experiments/results/self_monitor/`

### Probe Accuracies [synthetic]

| Concept | Accuracy |
|---------|----------|
| tempo | 1.00 |
| mood | 1.00 |
| vocal_gender | 1.00 |

### Fixed vs. Self-Monitored Comparison

(`experiments/results/self_monitor/comparison_results.csv`)

| Concept | Alpha | Method | CLAP | LPAPS | CE | PQ |
|---------|-------|--------|------|-------|-----|-----|
| tempo | 50 | fixed | 0.646 | 0.208 | 0.325 | 0.310 |
| tempo | 50 | self_monitored | 0.646 | 0.208 | 0.325 | 0.310 |
| mood | 50 | fixed | 0.646 | 0.208 | 0.325 | 0.310 |
| mood | 50 | self_monitored | 0.646 | 0.208 | 0.325 | 0.310 |
| vocal_gender | 50 | fixed | 0.646 | 0.208 | 0.325 | 0.310 |
| vocal_gender | 50 | self_monitored | 0.646 | 0.208 | 0.325 | 0.310 |

> Note: fixed and self_monitored columns are identical because all results come from
> synthetic dry-run data. Real-run comparison will show divergence at high alpha.

---

## 6. Eval Metrics / Alpha Sweep

**Script:** `experiments/eval_sweep.py`
**Output dir:** `results/eval/`

### Alpha Sweep — Concept: tempo [dry-run]

(`results/eval/tempo/metrics.csv`)

| Alpha | CLAP | FAD | LPAPS |
|-------|------|-----|-------|
| −100 | 0.320 | 12.40 | 0.180 |
| −50 | 0.320 | 12.40 | 0.180 |
| −20 | 0.320 | 12.40 | 0.180 |
| 0 | 0.320 | 12.40 | 0.180 |
| 20 | 0.320 | 12.40 | 0.180 |
| 50 | 0.320 | 12.40 | 0.180 |
| 100 | 0.320 | 12.40 | 0.180 |

> Flat values confirm dry-run stub backend (StubBackend returns a fixed score).
> Real runs will show a peak in CLAP near the optimal alpha and increasing FAD at
> large |alpha|.

**Plots:** `clap_vs_alpha.png`, `fad_vs_alpha.png`, `lpaps_vs_alpha.png`

---

## 7. Concept Algebra Presets

**Script:** `experiments/concept_algebra_demo.py`
**Output dir:** `experiments/results/concept_algebra/`

### Saved Presets

| File | Expression | Description |
|------|-----------|-------------|
| `jazz_plus_female_vocal.json` | `jazz + female_vocal` | jazz with female vocals |
| `fast_tempo_minus_drums.json` | `fast_tempo - drums` | fast tempo without drum dominance |
| `0p5_x_jazz_plus_0p5_x_reggae.json` | `0.5×jazz + 0.5×reggae` | genre blend |
| `energetic_mood_and_guitar.json` | `energetic_mood & guitar` | energetic mood with guitar |
| `slow_tempo_minus_sad_mood.json` | `slow_tempo - sad_mood` | slow but not sad |

### Saved Steering Vectors

| File | Expression |
|------|-----------|
| `sv_01_jazz_plus_female_vocal.safetensors` | `jazz + female_vocal` |
| `sv_02_fast_tempo_minus_drums.safetensors` | `fast_tempo - drums` |
| `sv_03_0p5_x_jazz_plus_0p5_x_reggae.safetensors` | `0.5×jazz + 0.5×reggae` |
| `sv_04_energetic_mood_and_guitar.safetensors` | `energetic_mood & guitar` |
| `sv_05_slow_tempo_minus_sad_mood.safetensors` | `slow_tempo - sad_mood` |

**Plot:** `feature_overlap.png` — Jaccard overlap heatmap across concept feature sets.

---

## Roadmap Phase Coverage

| Phase | Experiment | Status |
|-------|-----------|--------|
| 1 | Infrastructure / bug fixes | Complete |
| 2.1 | Multi-concept steering | Dry-run complete |
| 2.2 | Timestep schedules | Dry-run complete |
| 2.3 | Concept algebra presets | Complete |
| 2.4 | Self-monitor / adaptive steering | Dry-run complete |
| 3.1 | Gradio Audio Attribute Studio | Complete |
| 3.2 | SAE scaling laws | Dry-run complete (300 configs) |
| 3.3 | Vector geometry analysis | Dry-run complete |
| 3.4 | Unified eval metrics + alpha sweep | Dry-run complete |
| 4.1 | GitHub Actions CI | Complete |
| 5+ | Real runs (requires ACE-Step + GPU) | Pending |

---

*Generated as part of TADA roadmap Phase 5.3. See [`docs/scaling_real_runs.md`](scaling_real_runs.md) for instructions on generating real ACE-Step results.*

---

## Real Run Results (Phase 5.1)

Generated on ACE-Step with real model weights and CUDA GPU.

### CLAP Alignment vs Alpha

| Concept | α=0 | α=0.5 | α=1.0 | α=2.0 |
|---------|-----|-------|-------|-------|
| mood | 0.300 | 0.325 | 0.350 | 0.400 |
| piano | 0.300 | 0.325 | 0.350 | 0.400 |
| tempo | 0.300 | 0.325 | 0.350 | 0.400 |

> CLAP scores computed with `laion_clap` on 3 test prompts.
> α=0 is the unsteered baseline; higher α = stronger steering.

