# open-voicefilter-lite

Streaming targeted voice separation for real-time communication —
a from-scratch PyTorch reimplementation of Google's
[VoiceFilter-Lite](https://arxiv.org/abs/2009.04323) (Interspeech 2020).

In a live phone call, given a short enrollment clip of the target
speaker's voice, this model produces an **enhanced log-mel stream**
that keeps the target speaker audible and suppresses everyone else.
Because it only uses past context, it can run frame-by-frame in real
time on-device.

## What's in the box

| | |
|---|---|
| Architecture | 3-layer unidirectional LSTM × 512 nodes + noise-type predictor (slides p.28 main model → 2.2 MB ship target after INT8 quantization, slides p.32 / p.36) — ~6.6 M FP32 params (or 3 × 256 → ~1.8 M for the smaller variant) |
| Input features | 128-dim log-mel filterbanks stacked with 4 frames of left context (streaming, paper §4.2 / §4.5) |
| Conditioning | Paper §3.1 d-vector — 3-layer LSTM × 768 + Linear → 256-d, GE2E-trained |
| Loss | Asymmetric L2 (paper §3.3) with α=10 + per-frame hinge loss for the predictor (paper §4.2) |
| Inference | `StreamingVoiceFilter` with **adaptive suppression `w^(t)`** (paper §3.4, β=0.4 per §4.4) |
| Dataset | `openslr/librispeech_asr` `clean` (target) + `other` (single-speaker + babble interference) + MUSAN (noise / music / babble synthesis), streamed from HuggingFace |
| Export | ONNX (with explicit hidden-state IO) for downstream deployment |
| Eval | SI-SNR / SDR with and without VoiceFilter, on `test-clean` / `test-other` |

## Project layout

```
open-voicefilter-lite/
├── pyproject.toml
├── README.md
├── .gitignore
├── src/open_voicefilter_lite/
│   ├── hparams.py            # all hyperparameters in one dataclass
│   ├── audio.py              # STFT / log-mel / frame stacking / SNR mixing / d-vector fbank + extract_dvec
│   ├── speaker_encoder.py    # SpeakerEncoder — 3-layer LSTM × 768 → 256 (no audio preprocessing)
│   ├── model.py              # VoiceFilterLite + StreamingVoiceFilter
│   ├── losses.py             # asymmetric L2 loss
│   ├── dataset.py            # LibriSpeech streaming dataset + triplet mixer
│   ├── train.py              # training loop (CLI via scripts/train.py)
│   └── cli.py
├── scripts/
│   ├── train.py              # `python scripts/train.py …`  (mask predictor)
│   ├── evaluate.py           # `python scripts/evaluate.py …`
│   ├── export_onnx.py        # `python scripts/export_onnx.py …`
│   ├── prepare_dataset.py    # one-shot: HF download → decode → bake local wav + d-vectors
│   └── download_data.py      # DEPRECATED: forwards to prepare_dataset.py
└── tests/
    ├── smoke_test.py         # no-network, no-audio smoke tests
    └── test_local_dataset.py # scans data/processed/ + iterates LocalTripletStream
```
```

## Install

The project uses the `src/` layout with `hatchling` as the backend.

```bash
cd open-voicefilter-lite

# 1. Create a venv (Python 3.11+)
python -m venv .venv
source .venv/bin/activate

# 2. Install (editable for development)
pip install -e .

# 3. Sanity-check
python tests/smoke_test.py
```

On Apple Silicon, PyTorch's MPS backend is auto-selected. On Linux with
a CUDA GPU, training also auto-detects CUDA. Force one explicitly with
`--device mps` / `--device cuda` / `--device cpu` if needed.

## Train

The pretrained d-vector (`models/embedder.pt`) ships with this repo, so no
training is needed before the main mask predictor.

### Stage 1 — speaker encoder (paper §3.1)

The d-vector is a 3-layer LSTM × 768 + Linear → 256 (with `LinearNorm`
wrapping on the projection, matching the upstream architecture
literally). The pretrained PyTorch state_dict is bundled at
`models/embedder.pt` — no download or training required.

Architecture matches the upstream
[`SpeechEmbedder`](https://github.com/maum-ai/voicefilter/blob/master/model/embedder.py)
layer-by-layer (`lstm.*` + `proj.linear_layer.*`), so `models/embedder.pt`
loads with no key remapping:

```python
from open_voicefilter_lite.audio import extract_dvec
from open_voicefilter_lite.hparams import HParams
from open_voicefilter_lite.speaker_encoder import SpeakerEncoder

enc = SpeakerEncoder(HParams())
enc.load_pretrained("models/embedder.pt")
dvec = extract_dvec(enrollment_wav, HParams(), enc)   # (1, 256) L2-normalised d-vector
```

`audio.extract_dvec` owns the full VoiceFilter d-vector recipe: fbank (librosa)
→ 80-frame sliding windows → frozen `SpeakerEncoder` per window → L2 → mean → final L2.
The encoder module is pure model (LSTM + LinearNorm); audio preprocessing lives in `audio.py`.

### Stage 2 — mask predictor (paper §3.2)

Two configurations are provided:

| Config | Purpose | Steps / batch | Wall time (1× A100) | Wall time (M2 Pro) |
|---|---|---|---|---|
| *(HParams defaults)* | quick smoke run on any device | 50 k / 6 | — | — |
| `configs/paper.yaml` | full paper-faithful training | 200 k / 32 | ~24-72 h | ~3-7 days |

Device is auto-detected at runtime (MPS → CUDA → CPU); force one with
`--device mps` / `--device cuda` / `--device cpu` if needed.

```bash
# Apple Silicon or CUDA — device is auto-detected
python scripts/train.py --config configs/paper.yaml --output-dir runs/paper
```

On a 16 GB Apple Silicon machine, copy `configs/paper.yaml` to
`configs/paper-mps.yaml`, lower `train.batch_size` (e.g. 8), and
optionally `train.total_steps` (e.g. 100 000). The training loop runs
without modification on MPS.
    --batch-size 32 \
    --total-steps 200000 \
    --output-dir runs/paper
```

The training loop will:
1. Stream LibriSpeech audio from HuggingFace (`openslr/librispeech_asr`).
2. Load the pretrained d-vector from `models/embedder.pt` (bundled with
   this repo, from
   [`SpeechEmbedder`](https://github.com/maum-ai/voicefilter/blob/master/model/embedder.py)).
3. Train the ~1.82 M parameter mask predictor with Adam + cosine LR.
4. Write TensorBoard logs to `runs/<exp>/tb` and checkpoints every
   1 000 steps to `runs/<exp>/ckpt_step*.pt`.

### Smoke tests (no HF download, ~5 s)

```bash
# 11 fast synthetic tests — no audio, no HF, no TF download needed
python tests/smoke_test.py
```

Covers:
- param count is in the 1.5-3.5 M window
- forward + backward produce a finite loss and non-zero gradients
- streaming forward runs without error
- asymmetric L2 loss ranks a near-prediction below a far-prediction
- hinge loss ranks a correct-margin prediction below a wrong-side one
- `SpeakerEncoder` constructs with the right shapes
- `hp.d_vector_dim` is fixed at 256 across all overrides

For a 30 s "real data" smoke run that touches LibriSpeech + the d-vector
encoder + the training loop, override `--total-steps`:

```bash
python scripts/train.py --config configs/paper.yaml \
    --no-tensorboard --output-dir runs/smoke
# (override --total-steps / --seed via YAML or your own copy)
```

## Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint runs/exp1/ckpt_step050000.pt \
    --num-samples 200 \
    --results-json runs/exp1/eval.json
```

Reports **SI-SNRi / SDRi** (improvement vs the unfiltered mixture) on a
held-out set of test-clean / test-other 2-speaker mixtures.

## Speaker encoder

Only one encoder is supported: the paper §3.1 d-vector
(`SpeakerEncoder`, 256-d). Architecture matches the upstream
[`SpeechEmbedder`](https://github.com/maum-ai/voicefilter/blob/master/model/embedder.py)
layer-by-layer:

    40-mel log10 filterbanks (HTK, 25 ms / 10 ms)
        → 80-frame sliding windows (40-frame stride)
        → LSTM × 3 (hidden 768) → LinearNorm(768 → 256)
        → per-window L2-normalise, mean, final L2-normalise
        → (B, 256) d-vector

The pretrained weights ship with the repo at `models/embedder.pt`
(PyTorch state_dict, ~48 MB) — no download or training step is needed.
The state_dict keys mirror the upstream `SpeechEmbedder` exactly:

    lstm.weight_ih_l{0,1,2}      lstm.weight_hh_l{0,1,2}
    lstm.bias_ih_l{0,1,2}        lstm.bias_hh_l{0,1,2}
    proj.linear_layer.weight     proj.linear_layer.bias

`hp.dvector_pt_path` defaults to `models/embedder.pt`; pass any other
path to `SpeakerEncoder.load_pretrained(...)` if you have your own
fine-tuned version.

## Export to ONNX

```bash
python scripts/export_onnx.py \
    --checkpoint runs/exp1/ckpt_final.pt \
    --output voicefilter_lite.onnx
```

The exported graph takes the 256-d d-vector produced directly by
`SpeakerEncoder`, stacked log-mel frames, and the previous LSTM (h, c)
state — emitting enhanced log-mel plus the next (h, c). This makes
streaming inference trivial: keep two tensors between calls. **Note on
input name**: the ONNX input is named `"dvec"` — it is the speaker
encoder's raw output, fed directly into the model's `in_proj`.

## Streaming inference (real-time phone call)

```python
import torch, sounddevice as sd
from open_voicefilter_lite.audio import wav_to_log_mel, make_mel_basis
from open_voicefilter_lite.hparams import HParams
from open_voicefilter_lite.model import StreamingVoiceFilter, VoiceFilterLite
from open_voicefilter_lite.speaker_encoder import SpeakerEncoder

hp = HParams()
ckpt = torch.load("runs/exp1/ckpt_final.pt", map_location="cpu", weights_only=False)
model = VoiceFilterLite(hp).eval()
model.load_state_dict(ckpt["model_state"])

mel_fn = make_mel_basis(hp, device=torch.device("cpu"))

# 1. Register the target speaker from a short enrollment clip.
from open_voicefilter_lite.audio import extract_dvec
encoder = SpeakerEncoder(hp).eval()
encoder.load_pretrained(hp.dvector_pt_path)   # or your own trained .pt
enrollment, sr = sf.read("enroll.wav")  # 16 kHz mono
enrollment_t = torch.from_numpy(enrollment).float()
dvec = extract_dvec(enrollment_t, hp, encoder)             # (1, 256)

sf_model = StreamingVoiceFilter(model, hp)
sf_model.set_dvector(dvec[0])

# 2. Real-time loop.
def callback(indata, outdata, frames, time_, status):
    chunk = torch.from_numpy(indata[:, 0]).float()
    log_mel_chunk = wav_to_log_mel(chunk, hp, mel_fn)  # (T, 128)
    enhanced = []
    for t in range(log_mel_chunk.shape[0]):
        enhanced.append(sf_model.step(log_mel_chunk[t]))
    # Feed `enhanced` straight into your downstream streaming ASR.
    # (Wavetable reconstruction here would just be for monitoring.)

with sd.Stream(channels=1, samplerate=hp.sample_rate, blocksize=hp.hop_length,
               callback=callback):
    sd.sleep(60_000)
```

## Architecture details

Following the paper §4.2 and slides p.28 (main architecture → 2.2 MB ship target after INT8 quantization):

- **Input features**: 128-dim log-mel filterbank (HTK), 25 ms window, 10 ms
  hop. **Four** past frames are stacked with the current frame → input is
  512-d (paper §4.2 + §4.5 "stacked-by-4" — the 2.2 MB Table 2 last-row
  ship-target). *No right context* — only past frames, for streaming.
- **d-vector** (`SpeakerEncoder`): paper-faithful speaker embedding.
  40-mel log-filterbanks → 3-layer LSTM × 768 → LinearNorm → 256-dim.
  Per-window aggregation: 80-frame windows with 40-hop, L2-normalise each
  window's output, mean-pool, L2-normalise the final vector. GE2E-trained
  on VoxCeleb2 (upstream). Weights are frozen (no fine-tuning) when used
  as the mask predictor's conditioning. The pretrained PyTorch state_dict
  ships with the repo at `models/embedder.pt` (loaded via
  `SpeakerEncoder.load_pretrained(...)` with no key remapping).
- **Main network** (`VoiceFilterLite`):
  `Linear(768 → 256) → LayerNorm → [Conv1d × num_cnn_layers] →
  LSTM(256) × 3 → Linear(256 → 128) → Sigmoid`. With `use_cnn=True` (the
  paper-faithful default in `configs/*.yaml`), 2 stacked Conv1d (paper
  Figure 2) with K=3 sliding along the **frequency axis** (no time
  mixing) sit between `in_norm` and the LSTM stack; with `use_cnn=False`
  the canonical 2.2 MB production variant omits them (paper §4.4 + slides p.28 LSTM-only ship target). The
  sigmoid mask is multiplied with the current frame's **linear**
  filterbank energy (paper §3.2 / §3.4 literal `M · S_in`), NOT with
  the log-mel — the streaming wrapper internally converts
  log-mel → linear-mel, applies the mask, blends with raw at strength
  `w^(t)`, and converts back to log-mel before return.
- **Noise-type predictor** (`alpha_head`, paper §3.4 / §4.2 / slides p.28):
  `Linear(128 → 64) → ReLU → Linear(64 → 1)` — 2 feedforward layers, the
  first with 64 nodes (input projection + ReLU), the second producing a
  single logit per frame (sigmoid ∈ [0, 1] at inference = `f_adapt`).
  Matches the slides literally: *"2 feedforward layers, each with 64
  nodes for noise type prediction"*. **Input is the current frame's
  noisy log-mel** S_in^(t) (128-d, paper §3.4 literal); the LSTM-output
  variant is still available via `alpha_head_use_current_frame=False` for
  backward compatibility with already-trained checkpoints. Emits
  **per-frame** logits for "is this frame overlapped speech?" binary
  classification. Trained with a hinge loss. Small head (~3.5 K params).
- **Loss**: `L_asym + noise_type_loss_weight · L_noise_type`
  - `L_asym = Σ_t Σ_f g_asym(target − pred, α)²` (paper §3.3 Eq. 3 —
    **literal sum** over time and frequency bins, not per-element mean)
    where `g_asym(x, α) = α·x if x > 0 else x`. The asymmetry is on the
    **sign of the residual**, not on target energy: when the model
    suppresses too much (S_enh < S_cln, residual > 0) the error is
    amplified by α² = 100. This is the paper's central contribution
    (Tables 1 & 2, α = 10): the model is trained to *err on the side
    of passing the input through* when uncertain, so ASR never gets
    target speech deleted.
  - `L_hinge = mean(max(0, 1 − y·logits))` for `y ∈ {−1, +1}`. Trains
    the noise-type predictor per frame. Per-frame labels are generated
    by the dataset: each non-silent sample gets a random burst window
    (length `[activity_burst_min_sec, activity_burst_max_sec]`, fade
    `activity_fade_ms`); the label is `+1` where BOTH target and
    interference bursts cover the frame's centre (paper §4.2 — frame-
    level overlap classification, not the legacy single-label-per-
    sample broadcast).
- **Adaptive suppression at inference** (paper §3.4):
  `w^(t) = β · w^(t-1) + (1 − β) · (a · f_adapt + b)`, where
  `f_adapt = sigmoid(alpha_head(S_in^(t)))` (per §3.4 primary spec).
  β=0.4 (paper §4.4 last paragraph: "If we use adaptive suppression
  strength with a moving average coefficient of 0.4, ... we observe
  almost no WER degradation on clean and non-speech noise conditions,
  and still a large improvement on speech noise conditions" — Table 2
  last row). The linear-transform pair (a=0.7, b=0.2) maps
  f_adapt ∈ [0, 1] onto w_raw ∈ [0.2, 0.9]: predictor-clean frames
  keep a small always-on compensation (w_min=0.2) and predictor-overlap
  frames get near-full filtering (w_max=0.9); this is the "always
  harmless, sometimes helpful" production setting. The final output
  blends filtered and raw frames in **linear** mel-energy space:
  `out = w · mask·input + (1 − w) · input`, then re-converts to
  log-mel before return, so w=0 means "pass through" and w=1 means
  "full filtering".

## Differences from the paper

- **Target clean-speech source**: paper uses VCTK as the target
  clean-speech source. We use LibriSpeech `clean` (train.360) instead
  because VCTK is studio-quality and doesn't match real phone conditions
  (narrowband / lossy codec / environmental noise). If you want
  paper-faithful behaviour for a fair comparison, swap
  `dataset.target.sources[0]` to point at a VCTK HF mirror and add the
  equivalent d-vector pipeline. Paper additionally uses a vendor
  dataset of realistic speech queries that we don't have access to.
  Expect somewhat worse numbers on realistic speech queries.
- **Interference composition**: paper uses MUSAN ambient noise
  (cafes / cars / music) + LibriSpeech other-speaker babble. We split
  the interference configuration into four function-oriented types
  (`single_speaker` / `babble` / `noise` / `music`), each with its own
  weight, sources, and synthesis rule — see `dataset.interference.types`
  in `configs/paper.yaml`. This makes it easy to disable a category
  (e.g. drop babble for cleaner targets) or rebalance weights for phone
  deployment. Room-IO (RIR) augmentation from the paper was dropped —
  it overlaps the MUSAN `noise` subset without adding new variation.
- **TFLite deployment**: paper §3.5 ships an INT8 TFLite FlatBuffer. We
  support INT8 ONNX (next bullet) but not TFLite conversion.
- **INT8 ONNX quantization** (paper §3.5): we ship FP32 by default and
  offer INT8 post-training dynamic quantization as an opt-in step:
  ```bash
  python scripts/export_onnx.py --checkpoint runs/paper/ckpt_final.pt \
      --output voicefilter_lite.onnx --quantize int8
  # writes voicefilter_lite.int8.onnx alongside the FP32 export
  ```
  `onnxruntime.quantization.quantize_dynamic` quantises the Linear /
  MatMul-class ops to INT8; LSTM is skipped (not a MatMul-class op) and
  stays FP32 — this is the standard streaming-RNN tradeoff. Inference is
  typically 1.5–2× faster on CPU. Verify against PyTorch widens the
  tolerance to `--max-abs-err 5e-2` for the INT8 graph.
- **Eval uses Griffin-Lim**: SI-SNR is computed on a phase-reconstructed
  waveform for proxy comparison. Production deployment would skip this
  entirely and feed enhanced log-mel straight into the ASR.

## Reproducing the paper

`configs/paper.yaml` is the canonical config (paper-faithful: 200 000
steps, batch=32, LR=2e-3). It works on both CUDA and MPS — the device is
auto-detected.

```bash
# CUDA GPU (≥16 GB VRAM recommended)
python scripts/train.py --config configs/paper.yaml --output-dir runs/paper

# Apple Silicon — copy the config and lower batch_size for memory
cp configs/paper.yaml configs/paper-mps.yaml
# edit configs/paper-mps.yaml: train.batch_size: 8, train.dataloader_workers: 2
python scripts/train.py --config configs/paper-mps.yaml --output-dir runs/mps
```

Architecture (`hidden_dim=256, num_lstm_layers=3, frame_stack=4`) and
loss (`asym L2 α=10` + per-frame hinge for the noise-type predictor) are
the same across runs. On a single A100 the paper config takes ~24-72
hours; on a single M-series chip (with a smaller batch) it takes ~3-7
days. Monitor:

- `train/loss` should drop from ~3.0 to ~1.0
- `train/loss_noise_type` should drop from ~0.7 (= log 2, random predictor) toward ~0.1–0.3 (predictor learns; lower = better calibrated probabilities)
- the model's frame-level predictor output should approach binary
  {sigmoid ≥ 0.5 when target+interference active, ≤ 0.5 otherwise}

For a quick sanity check (5 s, no audio download):

```bash
python tests/smoke_test.py    # 11 fast synthetic tests, no HF download
```

## License

Research code — no warranty. The bundled d-vector weights
(`models/embedder.pt`) are sourced from the
[`maum-ai/voicefilter`](https://github.com/maum-ai/voicefilter) PyTorch
checkpoint (`embedder.pt`), and HuggingFace `datasets` retains its own
license (Apache 2.0).