# Steering Music Generation via Activation Addition

Code and evaluation protocol for the ISMIR 2026 paper "Steering Music Generation via Activation Addition: Statistical Validation of Concept-Direction Specificity in Audio Diffusion Transformers."

Activation steering adds a fixed direction vector to residual activations at inference time. TADA (Staniszewski et al., 2026) established that this can steer musical attributes in audio diffusion models, including ACE-Step, a 3.5B parameter music diffusion transformer. This work asks the question that comes after a steering demo works: is the effect attributable to the concept direction itself, or would any perturbation of matched magnitude score just as well?

Answering that took 2,250 generated clips across six sweeps, a matched-magnitude random-vector control, and a linear mixed-effects model over 450 prompt-seed-condition groups.

## Findings

- The piano direction yields a dose-response slope of β = 0.035 on CLAP (z = 12.57) and β = 0.047 on MuQ (z = 17.75), significantly steeper than the matched random control on both evaluators. The random control's own slope is indistinguishable from zero.
- Steering is asymmetric. Suppression is large and consistent (Cohen's d from −0.72 to −1.50 at α = −1.0). Enhancement is weak, concept-dependent, and reaches significance only for female vocals on CLAP.
- A magnitude pitfall: implementations that store steering vectors at unit norm understate achievable effects. Rescaling the female vocals direction from its natural L2 norm (22.00) to piano-matched norm (34.89) roughly doubles its measured dose-response, and the rescaling is what unlocks statistical significance.
- Evaluators disagree by concept. CLAP is sensitive to piano (d = −1.21) and near-blind to drums (d = −0.16), while MuQ registers suppression on all three concepts. Controllability results reported on a single text-audio backbone are systematically biased across the concept distribution.

## Provenance

This repository is a fork of [luk-st/steer-audio](https://github.com/luk-st/steer-audio), the code release for TADA (arXiv:2602.11910). The fork point is their February 13, 2026 release, commit `75ff8b9`. Upstream has since rewritten its history and restructured the codebase. The trees no longer share a merge base, and this fork deliberately does not track upstream. It preserves the release the paper's experiments were built on.

Everything below the fork point is theirs. Everything above it is mine: 44 commits, 282 files changed, +31,467 insertions, 78 deletions.

From TADA, unchanged or lightly modified:

- `src/` activation patching and layer localization (+74/−0)
- `steering/ace_steer/` CAA vector computation and steering evaluation (+148/−5)
- `sae/` sparse autoencoder pipeline (+181/−60)
- `configs/` Hydra config groups, extended for this fork's experiments (+401/−1)

Added in this fork:

- `steer_audio/` extraction, injection hooks, and norm calibration used by the paper (+5,748)
- `experiments/` the six-sweep protocol, LMEM analysis, and per-α paired tests (+5,965)
- `tests/` pytest suite covering extraction, injection, and scoring (+7,958)
- `results/` per-sweep CSV outputs that the paper's tables derive from (+876 across 162 files)
- `demo/` Gradio interface for interactive steering (+1,171)
- `docs/` results summaries and run notes (+383)
- `.github/` CI workflow (+48)

Extended from upstream stubs:

- `scripts/` sweep runners and environment verification (+5,821/−12)
- `notebooks/` empty upstream, analysis notebook added here (+645)

## Environment setup

Python >= 3.10.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements_1.txt
pip install -r requirements/requirements_2.txt --no-deps
```

Generation and steering require a GPU; the paper's runs used an A40.

## Reproducing the paper

```
bash scripts/setup_runpod_env.sh        # environment verification, A40-class GPU assumed
python scripts/run_paper_experiments.py # full protocol
```

Generation settings match the paper: 10 s clips at 44.1 kHz, 30 denoising steps, guidance_scale = 3.0, injection at the tf6 residual output on the conditional branch only. Analysis scripts write CSVs under `results/`, and every number in the paper's tables derives from those files.

## What we found broken along the way

- The CLAP scorer could fail silently and return −1.0, deflating scores without raising an error. Caught and fixed. Every reported number postdates the fix.
- The evaluator noise floor was measured directly rather than assumed: scoring a fixed reference clip 20 times gives σ = 0.0114, and baseline drift across sweeps stays within 2σ of it.
- A tf7 sweep was attempted because tf7 shows peak linear-probe discriminability (0.94 versus 0.86 at tf6). The α = 0 baseline diverged, indicating a hook implementation issue rather than a steering effect, so no tf7 results are reported. Disclosed in the paper's limitations.

## Beyond the paper

Work in this repo that postdates the submission. None of it is a result, and none of it appears in the paper:

- **Timestep commitment and CSS curves.** A prototype at n = 1 per concept-window cell, as noted in the header of `scripts/compute_css_metric.py`. At that sample size every cell collapses to CSS 0.0 or 1.0, no window approaches significance, and `results/paper/css_curves.csv` should be read as a plumbing check rather than a measurement. Raising n is the prerequisite to drawing any conclusion.
- **Concept geometry analysis.** `scripts/analyze_concept_geometry.py` runs end to end; no output artifacts are committed.
- **FAD and MuQ scoring over the human-evaluation set.** Scorers and 110 paired clips are in place under `results/paper/human_eval/`; no scores are committed yet. Groundwork for the listening study the paper defers to future work.

## Citing

Provisional entry pending proceedings metadata:

```bibtex
@inproceedings{shivesh2026steering,
  title={Steering Music Generation via Activation Addition: Statistical Validation of Concept-Direction Specificity in Audio Diffusion Transformers},
  author={Shrirang Shivesh},
  booktitle={Proceedings of the 27th International Society for Music Information Retrieval Conference (ISMIR)},
  year={2026}
}
```

This work builds directly on TADA. If you use the localization or CAA machinery, cite them as well:

```bibtex
@misc{staniszewski2026tada,
      title={TADA! Tuning Audio Diffusion Models through Activation Steering},
      author={Łukasz Staniszewski and Katarzyna Zaleska and Mateusz Modrzejewski and Kamil Deja},
      year={2026},
      eprint={2602.11910},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

ACE-Step: Gong et al., arXiv:2506.00045.

The inherited activation-patching, CAA, and SAE pipelines are documented in
[`docs/README_TADA.md`](docs/README_TADA.md), the upstream README preserved at
the fork point.

## Credits

This repository builds on [TADA](https://github.com/luk-st/steer-audio),
[ACE-Step](https://github.com/ace-step/ACE-Step),
[DDPM Inversion for Audio](https://github.com/HilaManor/AudioEditingCode),
[CASteer](https://github.com/Atmyre/CASteer), and
[Universal DiffSAE](https://github.com/cywinski/universal-diffsae).
