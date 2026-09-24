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
| Conditioning | d-vector from `speechbrain/spkrec-ecapa-voxceleb` (192-d → 256-d projection) |
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
│   ├── audio.py              # STFT / log-mel / frame stacking / SNR mixing
│   ├── speaker_encoder.py    # frozen ECAPA-TDNN wrapper
│   ├── model.py              # VoiceFilterLite + StreamingVoiceFilter
│   ├── losses.py             # asymmetric L2 loss
│   ├── dataset.py            # LibriSpeech streaming dataset + triplet mixer
│   ├── train.py              # training loop (CLI via scripts/train.py)
│   └── cli.py
├── scripts/
│   ├── train.py              # `python scripts/train.py …`
│   ├── evaluate.py           # `python scripts/evaluate.py …`
│   ├── export_onnx.py        # `python scripts/export_onnx.py …`
│   └── download_data.py      # optional: pre-cache LibriSpeech to disk
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
2. Lazy-download `speechbrain/spkrec-ecapa-voxceleb` ECAPA weights into
   `models/` (which is `.gitignore`'d).
3. Train the ~1.82 M parameter mask predictor with Adam + cosine LR.
4. Write TensorBoard logs to `runs/<exp>/tb` and checkpoints every
   1 000 steps to `runs/<exp>/ckpt_step*.pt`.

### Smoke tests (no HF download, ~5 s)

Two complementary ways to verify the pipeline before launching a real run:

```bash
# 5 fast synthetic tests — no audio, no HF, no speechbrain needed
python tests/smoke_test.py
```

The output above (with `frame_stack=4`, `hidden_dim=256`, `num_lstm_layers=3`,
`alpha_head` as 256→64→64→1 MLP, plus the new `StreamingVoiceFilter`
adaptive-`w^(t)` path) covers:
- param count is in the 1.5-3.5 M window
- forward + backward produce a finite loss and non-zero gradients
- streaming forward runs without error (max-diff vs batched is larger than
  pure-batch because adaptive `w^(t) < 1` blends the output — this is
  expected)
- asymmetric L2 loss ranks a near-prediction below a far-prediction
- hinge loss ranks a correct-margin prediction below a wrong-side one

For a 30 s "real data" smoke run that touches LibriSpeech + ECAPA + the
training loop, override `--total-steps`:

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

## Export to ONNX

```bash
python scripts/export_onnx.py \
    --checkpoint runs/exp1/ckpt_final.pt \
    --output voicefilter_lite.onnx
```

The exported graph takes the pre-projected 256-d d-vector (callers run
ECAPA on the enrollment clip first and then `dvec_proj`), stacked log-mel
frames, and the previous LSTM (h, c) state — emitting enhanced log-mel
plus the next (h, c). This makes streaming inference trivial: keep two
tensors between calls.

## Streaming inference (real-time phone call)

```python
import torch, sounddevice as sd
from open_voicefilter_lite.audio import wav_to_log_mel, make_mel_basis
from open_voicefilter_lite.hparams import HParams
from open_voicefilter_lite.model import StreamingVoiceFilter, VoiceFilterLite

hp = HParams()
ckpt = torch.load("runs/exp1/ckpt_final.pt", map_location="cpu", weights_only=False)
model = VoiceFilterLite(hp).eval()
model.load_state_dict(ckpt["model_state"])

mel_fn = make_mel_basis(hp, device=torch.device("cpu"))

# 1. Register the target speaker from a short enrollment clip.
enrollment, sr = sf.read("enroll.wav")  # 16 kHz mono
enrollment_t = torch.from_numpy(enrollment).float()
mel = make_mel_basis(hp, device=torch.device("cpu"))
dvec = encoder.encode_batch(enrollment_t.unsqueeze(0))  # raw 192-d ECAPA

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
- **d-vector**: 192-dim ECAPA-TDNN embedding projected to 256-dim inside
  the model. ECAPA weights are frozen.
- **Main network** (`VoiceFilterLite`):
  `Linear(576 → 256) → LayerNorm → LSTM(256) × 3 → Linear(256 → 80) →
  Sigmoid`. The sigmoid mask is multiplied with the current frame's
  log-mel. Output is the per-frame mask; the elementwise product with
  the current frame gives the enhanced log-mel.
- **Noise-type predictor** (`alpha_lstm` + `alpha_fc`, paper §3.4):
  2-layer unidirectional LSTM × 128 nodes over the main LSTM's per-frame
  features, followed by a `Linear(128 → 1)` head. Emits **per-frame**
  logits for "is this frame overlapped speech?" binary classification.
  Trained with a hinge loss. ~330 K params (LSTM dominates over the
  original paper's smaller variant).
- **Loss**: `L_asym + hinge_loss_weight · L_hinge`
  - `L_asym = mean( g_asym(target − pred, α)² )` where
    `g_asym(x, α) = α·x if x > 0 else x`. The asymmetry is on the
    **sign of the residual**, not on target energy: when the model
    suppresses too much (S_enh < S_cln, residual > 0) the error is
    amplified by α² = 100. This is the paper's central contribution
    (Tables 1 & 2, α = 10): the model is trained to *err on the side
    of passing the input through* when uncertain, so ASR never gets
    target speech deleted.
  - `L_hinge = mean(max(0, 1 − y·logits))` for `y ∈ {−1, +1}`. Trains
    the noise-type predictor per frame.
- **Adaptive suppression at inference** (paper §3.4):
  `w^(t) = β · w^(t-1) + (1 − β) · (a · f_adapt + b)`, where
  `f_adapt = sigmoid(alpha_fc(alpha_lstm(features)))` and β=0.8
  (paper §4.4). The final output blends filtered and raw frames:
  `out = w · mask·input + (1 − w) · input`, so w=0 means "pass through"
  and w=1 means "full filtering". This is what makes the model "always
  harmless and sometimes helpful" in clean / non-speech conditions.

## Differences from the paper

- **Vendor dataset**: paper uses LibriSpeech + a vendor-collected dataset
  of realistic speech queries. We only use LibriSpeech (train.360 +
  train.100 in `configs/paper.yaml`; train.100 only in `configs/mps.yaml`
  to halve data load on Apple Silicon). Expect somewhat worse numbers on
  realistic speech queries.
- **Musan / RIR augmentation**: paper uses ambient noise (cafes / cars /
  music) and 3 M simulated RIRs. Both `configs/*.yaml` keep these OFF by
  default; flip `use_musan: true` / `use_real_rir: true` to opt in. The
  dataset loader gracefully falls back to procedural RIRs and all-LibriSpeech
  interference when the source datasets are unavailable.
- **INT8 quantization** (paper §3.5): we ship FP32 checkpoints; the TFLite
  FlatBuffer pipeline is not implemented here.
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
python tests/smoke_test.py    # 5 fast synthetic tests, no HF download
```

## License

Research code — no warranty. SpeechBrain and HuggingFace `datasets` retain
their own respective licenses (Apache 2.0).
