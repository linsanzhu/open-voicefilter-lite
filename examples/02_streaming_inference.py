#!/usr/bin/env python3
"""Example 02 — Real-time streaming inference (file or microphone).

The streaming wrapper keeps LSTM hidden state between frames, so each
`step` is O(1) past context. The d-vector is extracted once from the
enrollment recording; there is no per-frame speaker-embedding cost.

  • `--mode file`   read a WAV chunk-by-chunk and run step on each
                    10 ms block. Writes an enhanced WAV.
  • `--mode mic`    capture the default input device via sounddevice,
                    push 10 ms blocks through the model, and (optionally)
                    play the enhanced audio back. Requires `sounddevice`.

Usage
-----
    # File mode (offline, deterministic — no extra deps):
    python examples/02_streaming_inference.py \\
        --enrollment data/enroll.wav \\
        --mixed      data/mixed.wav \\
        --out        data/enhanced_streaming.wav \\
        --mode file

    # Mic mode (live, requires `pip install sounddevice`):
    python examples/02_streaming_inference.py \\
        --enrollment data/enroll.wav \\
        --mode mic \\
        --duration 30
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..",
    "packages", "open-voicefilter-lite", "src")))

from voicefilter_lite import StreamingVoiceFilter


def _load_audio_mono_16k(path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import torchaudio
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), sr, target_sr
        ).numpy()
    return torch.from_numpy(wav.astype("float32"))


# ---- File mode: simulate streaming over a WAV ----


def run_file_mode(
    model: StreamingVoiceFilter,
    mixed_path: str,
    out_path: str,
) -> None:
    mixed = _load_audio_mono_16k(mixed_path)
    hop = model.hop_length
    n_full = mixed.shape[-1] // hop
    usable = n_full * hop
    if n_full == 0:
        raise SystemExit(
            f"input too short: {mixed.shape[-1]} samples < hop {hop}"
        )

    out_chunks = [
        model.step(mixed[i * hop:(i + 1) * hop].numpy())
        for i in range(n_full)
    ]
    enhanced = torch.cat(out_chunks)[:usable]

    peak = enhanced.abs().max().clamp(min=1e-8)
    if peak > 1.0:
        enhanced = enhanced / peak
    sf.write(out_path, enhanced.numpy(), 16000)
    print(f"[streaming] wrote {out_path} ({enhanced.shape[-1] / 16000:.2f}s)")


# ---- Mic mode: real-time capture via sounddevice ----


def run_mic_mode(
    model: StreamingVoiceFilter,
    duration: float,
    play_back: bool,
) -> None:
    """Capture from the default mic and stream through VoiceFilter-Lite.

    Requires `sounddevice`. Real-time, blocksize = `model.hop_length`.
    """
    try:
        import sounddevice as sd
    except ImportError as e:
        raise SystemExit(
            "mic mode requires sounddevice: `pip install sounddevice`"
        ) from e

    hop = model.hop_length
    samplerate = 16000
    print(f"[mic] samplerate={samplerate} blocksize={hop} "
          f"({1000 * hop / samplerate:.1f} ms)")

    out_chunks: list[torch.Tensor] = []

    def callback(indata, _frames, _time, _status):
        chunk = indata[:, 0].copy().astype(np.float32)
        out_chunks.append(model.step(chunk))

    print(f"[mic] capturing {duration:.1f}s — speak now")
    with sd.InputStream(
        channels=1,
        samplerate=samplerate,
        blocksize=hop,
        dtype="float32",
        callback=callback,
    ):
        sd.sleep(int(duration * 1000))

    if not out_chunks:
        print("[mic] no frames captured")
        return
    enhanced = torch.cat(out_chunks)
    print(f"[mic] processed {len(out_chunks)} hops "
          f"({enhanced.shape[-1] / samplerate:.2f}s)")

    if play_back:
        peak = enhanced.abs().max().clamp(min=1e-8)
        if peak > 1.0:
            enhanced = enhanced / peak
        sd.play(enhanced.numpy(), samplerate=samplerate)
        sd.wait()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--enrollment", required=True,
                   help="Path to a wav of the target speaker (any SR)")
    p.add_argument("--mode", choices=["file", "mic"], default="file")
    # File-mode options:
    p.add_argument("--mixed", help="Path to a 16 kHz mono WAV (required in --mode file)")
    p.add_argument("--out", help="Where to write the enhanced WAV (required in --mode file)")
    # Mic-mode options:
    p.add_argument("--duration", type=float, default=30.0,
                   help="seconds to capture (mic mode)")
    p.add_argument("--play-back", action="store_true",
                   help="mic mode: play back enhanced audio after capture")
    args = p.parse_args()

    enroll = _load_audio_mono_16k(args.enrollment)
    print(f"[streaming] enroll={enroll.shape[-1] / 16000:.2f}s")

    model = StreamingVoiceFilter()
    model.set_enrollment(enroll)
    print(f"[streaming] d-vector dim={model.hp.d_vector_dim}")

    if args.mode == "file":
        if not args.mixed or not args.out:
            raise SystemExit("--mode file requires --mixed and --out")
        run_file_mode(model, args.mixed, args.out)
    else:
        run_mic_mode(model, args.duration, args.play_back)


if __name__ == "__main__":
    main()
