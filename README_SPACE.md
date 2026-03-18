---
title: Audio Attribute Studio
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.24.0
app_file: demo/app.py
pinned: true
license: mit
---

# Audio Attribute Studio

**TADA** — Tuning Audio Diffusion Models through Activation Steering
([arXiv 2602.11910](https://arxiv.org/abs/2602.11910))

Steer any aspect of generated music — tempo, mood, instruments, genre — without
retraining, by applying pre-computed activation-space steering vectors during
diffusion inference.

## Tabs

| Tab | Description |
|-----|-------------|
| **Generate & Steer** | Enter a text prompt, drag concept sliders, and generate audio with waveform + spectrogram visualisation and A/B baseline comparison. |
| **SAE Feature Explorer** | Browse per-concept TF-IDF feature importance charts, a concept-overlap heatmap, and evaluate concept algebra expressions (`jazz + female_vocal - piano`). |
| **Batch Experiment** | Upload a CSV of prompts and alpha values; download a results CSV with all generation metadata. |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TADA_VECTORS_DIR` | `vectors/` | Directory of pre-computed `.safetensors` steering vectors |
| `TADA_SERVER_PORT` | `7860` | Gradio server port |
| `TADA_SHARE` | `0` | Set to `1` to create a public Gradio link |

## Local Launch

```bash
python demo/app.py
```

The app renders in demo mode (placeholder audio) when model weights or
steering vectors are not present, so the UI always loads.
