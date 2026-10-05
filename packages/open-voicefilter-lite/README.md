# open-voicefilter-lite (lib)

Streaming targeted voice separation — **library only** (model + audio
primitives). This is the PyPI-published half of the
`open-voicefilter-lite` project; the training / data-prep / ONNX-export
CLI lives in the sibling `open-voicefilter-lite-cli` package and stays
in the source repository.

## What's in the box

- `VoiceFilterLite` — the mask-predictor model (3 × 256 LSTM + 2-FC×64
  noise-type predictor, paper §3.2/§4.2).
- `StreamingVoiceFilter` — streaming inference wrapper.
- `SpeakerEncoder` — d-vector speaker encoder (paper §3.1).
- `audio.py` — wav → log-mel, fbank, sliding-window d-vector extraction.
- `hparams.HParams` — strongly-typed hyperparameter dataclass.
- `config` — YAML config loader (`load_config`, `apply_overrides`,
  `load_and_apply`).

## What's *not* in the box

Training-side concerns live in `open-voicefilter-lite-cli` and are
deliberately **not** installed by this package:

- Losses (`losses.py`)
- Training loop (`train.py`, `train_dvector.py`)
- Dataset preparation (`prepare_dataset.py`, `dataset_clear.py`)
- HF endpoint resolution (`hf.py`) — pulls in `huggingface_hub`.
- SpeechBrain dependency for the SpeechEmbedder pretrained weights.

This keeps the install footprint small for downstream consumers that
just want to load a `.pt` checkpoint and use the audio primitives.

## Install

```bash
pip install open-voicefilter-lite
```

## Quick use

```python
import torch
from voicefilter_lite import HParams, VoiceFilterLite

hp = HParams()  # paper-faithful defaults
model = VoiceFilterLite(hp).eval()

ckpt = torch.load("voicefilter_lite.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"])
```

## Source / docs

Full repo: <https://github.com/linsan/open-voicefilter-lite>