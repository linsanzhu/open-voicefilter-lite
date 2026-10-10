# Examples

Self-contained, runnable Python scripts showing how to use
`open-voicefilter-lite` from a downstream application. All examples use
the **turnkey public API** (`from voicefilter_lite import
StreamingVoiceFilter`) — the bundled ONNX, the d-vector pipeline, the
log-mel, the frame stacking, and the waveform reconstruction are all
internal.

| # | File | What it shows |
|---|------|---------------|
| 01 | [`01_offline_separation.py`](01_offline_separation.py) | Separate a mixed WAV file, write the enhanced WAV. The simplest end-to-end path. |
| 02 | [`02_streaming_inference.py`](02_streaming_inference.py) | Frame-by-frame inference over a file (offline) or live microphone. The `StreamingVoiceFilter` wrapper keeps LSTM hidden state across calls. |
| 03 | [`03_onnx_inference.py`](03_onnx_inference.py) | Same as 01 but with a custom mask ONNX (e.g. the INT8 quantised export). Pass `--mask-onnx` / `--embedder-onnx` to override the bundled models. |
| 04 | [`04_integration_mix_and_enhance.py`](04_integration_mix_and_enhance.py) | Integration test: synthesize N mixed clips from the actual training sources (LibriSpeech target + LibriSpeech / MUSAN interference), run the bundled model, and dump per-sample `target / enrollment / interference / mixed / enhanced` WAVs you can audition directly. |
| 05 | [`05_onnx_integration_mix_and_enhance.py`](05_onnx_integration_mix_and_enhance.py) | Same integration test as 04, but inference goes through a custom ONNX. Deterministic; identical mixed.wav to 04 under the same seed. |

## Common prerequisites

```bash
# 1. Install the package (editable)
pip install -e packages/open-voicefilter-lite

# 2. (optional, for the mic mode of example 02)
pip install sounddevice
```

No checkpoint, no encoder weights, no YAML config required — the
bundled ONNX graphs (`voicefilter_lite.onnx`, `embedder.int8.onnx`)
carry their own weights and hyperparameters.

## Running the examples

### 01 — Offline separation

```bash
python examples/01_offline_separation.py \
    --enrollment  data/enroll.wav \
    --mixed       data/mixed.wav \
    --out         data/enhanced.wav
```

Reads the mixed WAV, runs the bundled model frame-by-frame via
`StreamingVoiceFilter.step`, and writes the enhanced WAV. On CPU a
few-second clip runs in well under a second.

### 02 — Streaming inference

Two modes:

```bash
# File mode (deterministic; no extra deps)
python examples/02_streaming_inference.py \
    --enrollment  data/enroll.wav \
    --mixed       data/mixed.wav \
    --out         data/enhanced_streaming.wav \
    --mode file

# Mic mode (live; requires `pip install sounddevice`)
python examples/02_streaming_inference.py \
    --enrollment  data/enroll.wav \
    --mode mic --duration 30 --play-back
```

The model consumes 10 ms blocks (`hop_length`) and the stateful
wrapper makes each `step()` call O(1) past context. Audio comes out as
PCM (one `hop_length`-shaped tensor per call) — feed it directly to
your speaker / ASR frontend.

### 03 — Custom ONNX model

```bash
python examples/03_onnx_inference.py \
    --mask-onnx   voicefilter_lite.int8.onnx \
    --enrollment  data/enroll.wav \
    --mixed       data/mixed.wav \
    --out         data/enhanced_int8.wav
```

Pass any ONNX file with the same `voicefilter_lite_config` metadata
schema (see `_onnx_metadata.py` in the CLI) to swap the model. The
constructor reads the hyperparameters straight from
`metadata_props`, so no YAML / checkpoint alignment is required.

### 04 — Integration mix & enhance

```bash
python examples/04_integration_mix_and_enhance.py \
    --out-dir    data/integration_test \
    --num-samples 5
```

This is the closest thing to a "press play and judge the model"
workflow in the repo. For each of N samples the script:

1. Samples a random target speaker from LibriSpeech and a separate
   same-speaker utterance as the enrollment clip.
2. Samples interference from LibriSpeech `other` split, MUSAN music,
   MUSAN noise, or MUSAN speech overlaid as babble (paper §3.4).
3. Mixes target + interference at random SNR in
   `[snr_min_db, snr_max_db]`. Both sides are always active so the
   enhanced clip has audible content.
4. Streams the mix through the bundled `StreamingVoiceFilter` and
   writes the enhanced PCM.

The synthesis pipeline reads a HParams YAML (default
`configs/paper.yaml`) to match the training distribution. The model
itself uses the bundled ONNX.

### 05 — ONNX integration mix & enhance

```bash
python examples/05_onnx_integration_mix_and_enhance.py \
    --mask-onnx   voicefilter_lite.int8.onnx \
    --out-dir     data/integration_test_int8 \
    --num-samples 5
```

Same as 04 but the mask predictor is the user-supplied ONNX. With the
same seed the mixed audio is bit-identical to 04's, so the only
difference between `xx_enhanced.wav` files is the mask model itself —
useful for A/B listening (FP32 vs INT8).
