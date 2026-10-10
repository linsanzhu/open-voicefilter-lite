#!/usr/bin/env python3
"""Example 01 — Offline separation of a pre-mixed WAV file.

The turnkey API: hand the model an enrollment recording and a mixed
audio file, get an enhanced WAV back. The mask ONNX, embedder ONNX,
log-mel, frame stacking, and waveform reconstruction are all internal.

Usage
-----
    python examples/01_offline_separation.py \\
        --enrollment data/enroll.wav \\
        --mixed      data/mixed.wav \\
        --out        data/enhanced.wav
"""

from __future__ import annotations

import argparse
import os
import sys

import librosa
import numpy as np
import soundfile as sf

# Self-contained: add the lib's `src/` layout to sys.path so this works
# without `pip install -e .`.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..",
    "packages", "open-voicefilter-lite", "src")))

from voicefilter_lite import StreamingVoiceFilter


def _load_audio_mono_16k(path: str, target_sr: int = 16000) -> np.ndarray:
    """Read a wav file → 1-D float32 numpy array at `target_sr` mono."""
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return wav.astype(np.float32, copy=False)


def _stream_pcm_through(model: StreamingVoiceFilter, pcm: np.ndarray) -> np.ndarray:
    """Slice `pcm` into `hop`-sized chunks, run `step`, concat the outputs.

    Drops the trailing partial chunk (a partial hop can't be fully
    overlap-added by the model's OLA).
    """
    hop = model.hop_length
    n_full = pcm.shape[-1] // hop
    usable = n_full * hop
    if n_full == 0:
        return np.zeros(0, dtype=np.float32)
    out_chunks = [model.step(pcm[i * hop:(i + 1) * hop]) for i in range(n_full)]
    return np.concatenate(out_chunks)[:usable]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--enrollment", required=True,
                   help="Path to a wav of the target speaker (any SR; resampled to 16 kHz)")
    p.add_argument("--mixed", required=True,
                   help="Path to a 16 kHz mono WAV to be separated")
    p.add_argument("--out", required=True, help="Where to write the enhanced WAV")
    args = p.parse_args()

    enroll = _load_audio_mono_16k(args.enrollment)
    mixed = _load_audio_mono_16k(args.mixed)
    print(f"[offline] enroll={enroll.shape[-1] / 16000:.2f}s, "
          f"mixed={mixed.shape[-1] / 16000:.2f}s")

    model = StreamingVoiceFilter()          # bundled ONNX, no paths needed
    model.set_enrollment(enroll)
    print(f"[offline] d-vector dim={model.hp.d_vector_dim}, "
          f"hop={model.hop_length} samples")

    enhanced = _stream_pcm_through(model, mixed)

    peak = float(np.abs(enhanced).max())
    if peak > 1e-8 and peak > 1.0:
        enhanced = enhanced / peak

    sf.write(args.out, enhanced.astype(np.float32), 16000)
    print(f"[offline] wrote {args.out} ({enhanced.shape[-1] / 16000:.2f}s)")


if __name__ == "__main__":
    main()
