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
| Architecture | 3-layer unidirectional LSTM × 256 nodes + noise-type predictor (paper §4.4, 2.2 MB) — ~1.82 M FP32 params |
| Input features | 80-dim log-mel filterbanks stacked with 4 frames of left context (streaming, paper §4.5) |
| Conditioning | Paper §3.1 d-vector — 3-layer LSTM × 768 + Linear → 256-d, GE2E-trained |
| Loss | Asymmetric L2 (paper §3.3) with α=10 + per-frame hinge loss for the predictor (paper §4.2) |
| Inference | `StreamingVoiceFilter` with **adaptive suppression `w^(t)`** (paper §3.4, β=0.8) |
| Dataset | `openslr/librispeech_asr` `clean` (target) + `other` (interference), streamed from HuggingFace |
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
│   └── download_data.py      # optional: pre-cache LibriSpeech + Musan + RIR + VCTK
└── tests/
    └── smoke_test.py         # no-network, no-audio smoke tests
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
a CUDA GPU, pass `--device cuda`.

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

Three configurations are provided, ordered from cheapest to most faithful:

| Config | Purpose | Steps / batch | Wall time (1× A100) | Wall time (M2 Pro) |
|---|---|---|---|---|
| *(HParams defaults)* | quick smoke run on any device | 50 k / 6 | — | — |
| `configs/mps.yaml` | Apple-Silicon training (paper-faithful architecture, reduced budget) | 100 k / 16 | n/a | ~3-7 days |
| `configs/paper.yaml` | full paper-faithful training | 200 k / 32 | ~24-72 h | n/a (too slow) |

```bash
# Apple Silicon
python scripts/train.py --config configs/mps.yaml --output-dir runs/mps

# CUDA GPU
python scripts/train.py --config configs/paper.yaml --output-dir runs/paper
```

Optional overrides (any field in the YAML can be beaten by a CLI flag):

```bash
python scripts/train.py \
    --config configs/paper.yaml \
    --device cuda \
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
python scripts/train.py --config configs/mps.yaml --total-steps 50 \
    --batch-size 2 --no-tensorboard --output-dir runs/smoke
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
    log_mel_chunk = wav_to_log_mel(chunk, hp, mel_fn)  # (T, 80)
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

Following the paper §4.2 / §4.4 (best 2.2 MB variant):

- **Input features**: 80-dim log-mel filterbank (HTK), 25 ms window, 10 ms
  hop. **Four** past frames are stacked with the current frame → input is
  320-d (paper §4.5 "stacked-by-4"). *No right context* — only past
  frames, for streaming.
- **d-vector** (`SpeakerEncoder`): paper-faithful speaker embedding.
  40-mel log-filterbanks → 3-layer LSTM × 768 → LinearNorm → 256-dim.
  Per-window aggregation: 80-frame windows with 40-hop, L2-normalise each
  window's output, mean-pool, L2-normalise the final vector. GE2E-trained
  on VoxCeleb2 (upstream). Weights are frozen (no fine-tuning) when used
  as the mask predictor's conditioning. The pretrained PyTorch state_dict
  ships with the repo at `models/embedder.pt` (loaded via
  `SpeakerEncoder.load_pretrained(...)` with no key remapping).
- **Main network** (`VoiceFilterLite`):
  `Linear(576 → 256) → LayerNorm → [Conv1d × num_cnn_layers] →
  LSTM(256) × 3 → Linear(256 → 80) → Sigmoid`. With `use_cnn=True` (the
  paper-faithful default in `configs/*.yaml`), 2 stacked Conv1d (paper
  Figure 2) with K=3 sliding along the **frequency axis** (no time
  mixing) sit between `in_norm` and the LSTM stack; with `use_cnn=False`
  the canonical 2.2 MB production variant omits them (paper §4.4). The
  sigmoid mask is multiplied with the current frame's log-mel. Output
  is the per-frame mask; the elementwise product with the current frame
  gives the enhanced log-mel.
- **Noise-type predictor** (`alpha_head`, paper §3.4 / §4.2 / slides p.28):
  `Linear(80 → 64) → ReLU → Linear(64 → 1)` — 2 feedforward layers, the
  first with 64 nodes (input projection + ReLU), the second producing a
  single logit per frame (sigmoid ∈ [0, 1] at inference = `f_adapt`).
  Matches the slides literally: *"2 feedforward layers, each with 64
  nodes for noise type prediction"*. **Input is the current frame's
  noisy log-mel** S_in^(t) (80-d, paper §3.4 literal); the LSTM-output
  variant is still available via `alpha_head_use_current_frame=False` for
  backward compatibility with already-trained checkpoints. Emits
  **per-frame** logits for "is this frame overlapped speech?" binary
  classification. Trained with a hinge loss. Small head (~3.3 K params).
- **Loss**: `L_asym + hinge_loss_weight · L_hinge`
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
  `f_adapt = sigmoid(alpha_head(S_in^(t)))` (per §3.4 primary spec)
  and β=0.8 (paper §4.4). The final output blends filtered and raw
  frames: `out = w · mask·input + (1 − w) · input`, so w=0 means
  "pass through" and w=1 means "full filtering". This is what makes the
  model "always harmless and sometimes helpful" in clean / non-speech
  conditions.

## Differences from the paper

- **Vendor dataset**: paper uses VCTK for the target clean-speech source
  and LibriSpeech for interference. `configs/paper.yaml` honours that
  (`target_source: vctk`; `scripts/download_data.py --assets vctk`
  pre-populates the cache); `configs/mps.yaml` keeps `target_source:
  librispeech` for the smaller LibriSpeech subset. Paper additionally
  uses a vendor-collected dataset of realistic speech queries that we
  don't have access to. Expect somewhat worse numbers on realistic
  speech queries.
- **Musan / RIR augmentation**: paper uses ambient noise (cafes / cars /
  music) and 3 M simulated RIRs. Both `configs/*.yaml` keep these OFF by
  default; flip `use_musan: true` / `use_real_rir: true` to opt in. The
  dataset loader gracefully falls back to procedural RIRs and all-LibriSpeech
  interference when the source datasets are unavailable.
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

Two paper-faithful configs are provided:

```bash
# CUDA GPU (≥16 GB VRAM recommended)
python scripts/train.py --config configs/paper.yaml --output-dir runs/paper

# Apple Silicon — same architecture, reduced budget (batch=16, 100 k steps)
python scripts/train.py --config configs/mps.yaml --output-dir runs/mps
```

Both configs share the same architecture (`hidden_dim=256, num_lstm_layers=3,
frame_stack=4`) and loss (`asym L2 α=10` + per-frame hinge for the noise-type
predictor). `paper.yaml` runs for 200 000 steps at batch=32 + LR=2e-3;
`mps.yaml` runs for 100 000 steps at batch=16 + LR=1e-3 (linear-scaling
rule). On a single A100 the paper config takes ~24-72 hours; on a single
M-series chip the MPS config takes ~3-7 days. Monitor:

- `train/loss` should drop from ~3.0 to ~1.0
- `train/loss_hinge` should drop from ~1.0 to ~0.1 (predictor learns)
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