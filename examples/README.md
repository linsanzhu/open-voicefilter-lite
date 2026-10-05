# Examples

Self-contained, runnable Python scripts showing how to use `open-voicefilter-lite`
from a downstream application. These complement the operational scripts in
`../scripts/` (training, evaluation, ONNX export), which are oriented around
running the training/eval pipelines against LibriSpeech.

Each example:
- Lives in a single file (no shared utilities).
- Adds `../src/` to `sys.path` so it runs without `pip install -e .`.
- Uses `argparse` — all paths / device overrides come from the CLI.

You will need a trained checkpoint from `scripts/train.py` (e.g.
`runs/exp1/ckpt_final.pt`), an enrollment clip of the target speaker, and a
mixed audio file you want to separate. All three are 16 kHz mono WAVs.

| # | File | What it shows |
|---|------|---------------|
| 01 | [`01_offline_separation.py`](01_offline_separation.py) | Load a checkpoint, separate a mixed WAV file, write the enhanced WAV. The simplest end-to-end path; matches the README's snippet but as a proper script. |
| 02 | [`02_streaming_inference.py`](02_streaming_inference.py) | Frame-by-frame inference over a file or live microphone. The `StreamingVoiceFilter` wrapper keeps LSTM hidden state across calls. |
| 03 | [`03_onnx_inference.py`](03_onnx_inference.py) | Run the exported ONNX model with `onnxruntime`. The SpeakerEncoder produces the 256-d d-vector directly; the graph consumes it as-is (no in-graph projection). Optional `--verify` flag cross-checks ONNX against PyTorch. |
| 04 | [`04_integration_mix_and_enhance.py`](04_integration_mix_and_enhance.py) | Integration test: synthesize N mixed clips from the actual training sources (LibriSpeech target + LibriSpeech / Musan interference), run the model, and dump per-sample `target / enrollment / interference / mixed / enhanced` WAVs you can audition directly. |

## Common prerequisites

```bash
# 1. Install the package (editable)
pip install -e .

# 2. (optional, for the mic mode of example 02)
pip install sounddevice

# 3. Train a checkpoint (or use a pre-trained one)
python scripts/train.py --output-dir runs/exp1
```

## Running the examples

### 01 — Offline separation

```bash
python examples/01_offline_separation.py \
    --checkpoint runs/exp1/ckpt_final.pt \
    --enrollment  data/enroll.wav \
    --mixed       data/mixed.wav \
    --out         data/enhanced.wav
```

Reads the mixed WAV, runs the model frame-by-frame (the same `StreamingVoiceFilter`
used in production), and reconstructs a waveform via Griffin-Lim. On CPU a
few-second clip runs in well under a second. On MPS / CUDA it is essentially
instant.

### 02 — Streaming inference

Two modes:

```bash
# File mode (deterministic; no extra deps)
python examples/02_streaming_inference.py \
    --checkpoint runs/exp1/ckpt_final.pt \
    --enrollment  data/enroll.wav \
    --mixed       data/mixed.wav \
    --out         data/enhanced_streaming.wav \
    --mode file

# Mic mode (live; requires `pip install sounddevice`)
python examples/02_streaming_inference.py \
    --checkpoint runs/exp1/ckpt_final.pt \
    --enrollment  data/enroll.wav \
    --mode mic --duration 30 --play-back
```

The model consumes 10 ms blocks (`hp.hop_length`) and the stateful wrapper
makes each `step()` call O(1) past context. In a real phone-call pipeline you
would not reconstruct waveform at all — feed the enhanced log-mel straight
into your streaming ASR frontend.

### 03 — ONNX runtime inference

```bash
python examples/03_onnx_inference.py \
    --onnx       voicefilter_lite.onnx \
    --checkpoint runs/exp1/ckpt_final.pt \
    --enrollment data/enroll.wav \
    --mixed      data/mixed.wav \
    --out        data/enhanced_onnx.wav
```

The ONNX graph (built by `scripts/export_onnx.py`) takes the 256-d
d-vector produced directly by `SpeakerEncoder` plus the persistent LSTM
(h, c). The graph input is named `"dvec"` (the speaker encoder's raw
output). Pass `--verify` to also run PyTorch on the same prefix
and assert the parity is below `--max-abs-err` (default 1e-3) — useful
right after re-exporting.

## Notes on waveform reconstruction

None of the examples ship a neural vocoder. The `_logmel_to_wav` helper uses
torchaudio's `InverseMelScale` + `GriffinLim` (default 16 iterations). That
audio is for audition only — it is not what production callers should use.
The paper's whole point is that you do not need to reconstruct waveform;
feed enhanced log-mel to your downstream ASR directly.

### 04 — Integration mix & enhance

```bash
python examples/04_integration_mix_and_enhance.py \
    --checkpoint runs/exp1/ckpt_final.pt \
    --out-dir    data/integration_test \
    --num-samples 5
```

This is the closest thing to a "press play and judge the model" workflow
in the repo. For each of N samples the script:

1. Samples a random target speaker from LibriSpeech (`config=clean`) and a
   separate same-speaker utterance as the enrollment clip (d-vector
   source via `SpeakerEncoder`).
2. Samples interference from one of: LibriSpeech `other` split, Musan
   music, Musan noise, or Musan speech overlaid as babble (paper §3.4).
3. Mixes target + interference at random SNR in
   `[hp.snr_min_db, hp.snr_max_db]`. Both sides are always active so the
   enhanced clip has audible content.
4. Streams the mix through `StreamingVoiceFilter` and reconstructs a
   waveform with Griffin-Lim.

Output layout (`data/integration_test/`):

```
00_target.wav          # clean target speaker (reference)
00_enrollment.wav      # same speaker, different utterance (dvec source)
00_interference.wav    # the "other" sound
00_mixed.wav           # what the model hears
00_enhanced.wav        # what the model outputs  ← listen to this
...
summary.txt            # human-readable per-sample metadata
summary.json           # same data, machine-readable
```

Listen to `*_mixed.wav` vs `*_enhanced.wav` to A/B the model; cross-check
against `*_target.wav` to see how close it gets to the clean reference.

Flags worth knowing:
- `--no-musan`: skip Musan entirely; interference is LibriSpeech `other` only.
- `--num-samples N`: how many mixes to generate (default 5).
- `--gl-iter N`: Griffin-Lim iterations (default 16; more = cleaner audio
  but slower).

## Adding new examples

Keep the convention:

- One file per example, numbered `NN_short_name.py`.
- Self-contained `sys.path.insert` bootstrap so the example runs without an
  editable install.
- A module docstring that includes a usage block with realistic paths.
- `argparse` for any path/device parameters; sensible defaults otherwise.