#!/usr/bin/env python3
"""Example 03 — Inference with a custom mask ONNX (e.g. INT8 quantised).

The bundled FP32 ONNX (`voicefilter_lite.onnx`) is used by default.
For deployment with smaller models / faster inference, pass your own
mask ONNX via `mask_model=...` — the constructor reads the
hyperparameters straight from `metadata_props.voicefilter_lite_config`.

The new public API hides log-mel extraction, frame stacking, the
d-vector pipeline, and waveform reconstruction — the example is
deliberately the same shape as 01_offline_separation.py.

Usage
-----
    python examples/03_onnx_inference.py \\
        --mask_onnx  models/voicefilter_lite.int8.onnx \\
        --enrollment data/enroll.wav \\
        --mixed      data/mixed.wav \\
        --out        data/enhanced_int8.wav
"""

from __future__ import annotations

import argparse
import os
import sys

import librosa
import numpy as np
import soundfile as sf

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mask_onnx",
                   help="Path to a custom mask ONNX. "
                        "Default: use the bundled FP32 voicefilter_lite.onnx. "
                        "Use the INT8 export (voicefilter_lite.int8.onnx) for faster inference.")
    p.add_argument("--embedder_onnx",
                   help="Path to a custom embedder ONNX. "
                        "Default: use the bundled INT8 embedder.int8.onnx.")
    p.add_argument("--enrollment", required=True,
                   help="Path to a wav of the target speaker (any SR)")
    p.add_argument("--mixed", required=True,
                   help="Path to a 16 kHz mono WAV to be separated")
    p.add_argument("--out", required=True, help="Where to write the enhanced WAV")
    args = p.parse_args()

    enroll = _load_audio_mono_16k(args.enrollment)
    mixed = _load_audio_mono_16k(args.mixed)
    print(f"[onnx] enroll={enroll.shape[-1] / 16000:.2f}s, "
          f"mixed={mixed.shape[-1] / 16000:.2f}s")

    # Pass through any user-supplied ONNX paths; the constructor reads
    # hyperparameters from `metadata_props.voicefilter_lite_config` on
    # the mask ONNX, then overlays the d-vector fields from the embedder
    # ONNX metadata.
    kwargs: dict = {}
    if args.mask_onnx:
        kwargs["mask_model"] = args.mask_onnx
    if args.embedder_onnx:
        kwargs["dvector_model"] = args.embedder_onnx
    model = StreamingVoiceFilter(**kwargs)
    model.set_enrollment(enroll)
    print(f"[onnx] mask_onnx={args.mask_onnx or '(bundled)'}, "
          f"embedder_onnx={args.embedder_onnx or '(bundled)'}, "
          f"n_fft={model.hp.n_fft}, win_length={model.hp.win_length}")

    hop = model.hop_length
    n_full = mixed.shape[-1] // hop
    if n_full == 0:
        raise SystemExit(f"input too short: {mixed.shape[-1]} samples < hop {hop}")
    out_chunks = [model.step(mixed[i * hop:(i + 1) * hop]) for i in range(n_full)]
    enhanced = np.concatenate(out_chunks)[: n_full * hop]

    peak = float(np.abs(enhanced).max())
    if peak > 1e-8 and peak > 1.0:
        enhanced = enhanced / peak
    sf.write(args.out, enhanced.astype(np.float32), 16000)
    print(f"[onnx] wrote {args.out} ({enhanced.shape[-1] / 16000:.2f}s)")


if __name__ == "__main__":
    main()
