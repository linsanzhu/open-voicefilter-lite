# open-voicefilter-lite

Streaming targeted voice separation — **turnkey PCM-in / PCM-out** library.
The published package ships two ONNX models (a mask predictor and a
d-vector speaker encoder) plus a single class that handles
preprocessing, frame stacking, LSTM state, and waveform reconstruction
internally. The caller hands it PCM audio and an enrollment recording;
it hands back enhanced PCM, one chunk at a time.

This is the PyPI-published half of the `open-voicefilter-lite` project;
the training / data-prep / ONNX-export CLI lives in the sibling
`open-voicefilter-lite-cli` package and stays in the source repository.

## Install

```bash
pip install open-voicefilter-lite
```

The install pulls in PyTorch, torchaudio, onnxruntime, librosa, and
soundfile. No checkpoint download is required — the ONNX graphs are
bundled inside the wheel.

## Quick use

```python
import numpy as np
import soundfile as sf
import torch
import torchaudio
from voicefilter_lite import StreamingVoiceFilter

# Load the target speaker yourself — the library takes PCM only, not paths.
wav, sr = sf.read("enroll.wav", dtype="float32")
if sr != 16000:
    wav = torchaudio.functional.resample(
        torch.from_numpy(wav), sr, 16000
    ).numpy()

sf_vfl = StreamingVoiceFilter()              # uses bundled ONNX
sf_vfl.set_enrollment(wav)                   # numpy / torch PCM, 1-D

# Feed it PCM chunks of `hop_length` samples (= 160 at 16 kHz, 10 ms);
# the model returns the same number of enhanced samples.
hop = sf_vfl.hop_length
while True:
    pcm = microphone.read(hop)               # numpy float32, 1-D
    if pcm is None:
        break
    enhanced = sf_vfl.step(pcm)              # torch float32, 1-D
    speaker.play(enhanced.numpy())
```

`set_enrollment(audio)` accepts a numpy array or a torch tensor of PCM
samples at the model's sample rate (16 kHz by default). Multi-channel
audio is mean-pooled to mono. File I/O is the caller's responsibility
— load your wav with `soundfile` / `torchaudio` / `librosa` and
resample externally if the sample rate doesn't match.

`set_enrollment()` (no argument) or `clear_enrollment()` drops the
registered d-vector. After a clear, `step()` becomes a passthrough —
the input hop is returned unchanged with no mask / STFT / LSTM work.
This lets one model instance be used as a "speaker-detected?" gate
or a "only enhance when the right person is talking" switch: call
`set_enrollment(pcm)` when a target speaker is known, `set_enrollment(None)`
when they walk away. Mid-stream re-enrollment (switching from speaker A
to speaker B) is supported — every `set_enrollment` call resets the
streaming state so old enhanced frames never bleed into the new mode.

`step(pcm_frame)` accepts a numpy array or torch tensor of length
`hop_length` (160 samples at 16 kHz). Output is always a CPU
`torch.float32` tensor of the same length.

## What's in the box

- `StreamingVoiceFilter` — the entire user-facing surface. PCM in,
  PCM out, one hop at a time.
- Bundled ONNX models (`voicefilter_lite.onnx`, `embedder.int8.onnx`)
  with all hyperparameters embedded as `metadata_props` — no YAML,
  no checkpoint, no constructor arguments required.
- `voicefilter_lite.hparams.HParams.from_onnx(path)` /
  `HParams.apply_onnx_dvector_overrides(path)` — escape hatches for
  callers that want to read the hyperparameter metadata themselves.

## What's *not* in the box

The training-side concerns (losses, training loop, dataset preparation,
ONNX export) live in the sibling `open-voicefilter-lite-cli` package
and are deliberately **not** installed by this package. The internal
modules (`audio.py`, `model.py`, `speaker_encoder.py`, `config.py`) are
importable by full path for advanced / fine-tuning use, but they're
not part of the public surface.

## Custom models

Pass paths to your own ONNX files (or pre-built
`onnxruntime.InferenceSession`s) to override the bundled models:

```python
import numpy as np
import soundfile as sf
from voicefilter_lite import StreamingVoiceFilter

wav, sr = sf.read("enroll.wav", dtype="float32")

sf_vfl = StreamingVoiceFilter(
    mask_model="path/to/my_mask_predictor.onnx",
    dvector_model="path/to/my_dvector.onnx",
)
sf_vfl.set_enrollment(wav)
```

## Source / docs

Full repo: <https://github.com/linsan/open-voicefilter-lite>
