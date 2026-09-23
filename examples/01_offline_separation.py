#!/usr/bin/env python3
"""Example 01 — Offline separation of a pre-mixed WAV file.

Given a trained checkpoint, an enrollment clip of the target speaker, and a
mixed audio file (target speaker + other speakers / noise), this script runs
VoiceFilter-Lite frame-by-frame and writes the enhanced log-mel back out as a
16 kHz WAV using Griffin-Lim phase reconstruction.

This is the simplest end-to-end path and the one most people will copy first.
For real production deployment you would feed the enhanced log-mel straight
into a streaming ASR instead of reconstructing waveform — see
`02_streaming_inference.py` and `03_onnx_inference.py`.

Usage
-----
    python examples/01_offline_separation.py \\
        --checkpoint runs/exp1/ckpt_final.pt \\
        --enrollment  data/enroll.wav \\
        --mixed       data/mixed.wav \\
        --out         data/enhanced.wav

The script auto-selects MPS / CUDA / CPU. No GPU is required — a few-second
clip runs in well under a second on CPU.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

# Self-contained: add the `src/` layout to sys.path so this works without
# `pip install -e .`.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from open_voicefilter_lite.audio import (  # noqa: E402
    make_mel_basis,
    wav_to_log_mel,
)
from open_voicefilter_lite.hparams import HParams  # noqa: E402
from open_voicefilter_lite.model import (  # noqa: E402
    StreamingVoiceFilter,
    VoiceFilterLite,
)
from open_voicefilter_lite.speaker_encoder import (  # noqa: E402
    FrozenSpeakerEncoder,
)


def _select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_enrollment_dvec(
    encoder: FrozenSpeakerEncoder,
    wav: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """ECAPA returns (1, 512); squeeze to (512,) for StreamingVoiceFilter."""
    with torch.no_grad():
        return encoder(wav.unsqueeze(0).to(device)).squeeze(0).cpu()


def _logmel_to_wav(log_mel: torch.Tensor, hp: HParams, n_iter: int = 16) -> torch.Tensor:
    """Inverse log-mel → waveform via Griffin-Lim.

    NOTE: This is for offline audition only. Production streaming pipelines
    feed `enhanced_logmel` directly into an ASR frontend.
    """
    import torchaudio

    mel = torch.exp(log_mel) - hp.log_mel_floor        # (T, F)
    mel = mel.transpose(0, 1).unsqueeze(0)             # (1, F, T)

    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=hp.n_fft // 2 + 1,
        n_mels=hp.n_mels,
        sample_rate=hp.sample_rate,
        max_iter=128,
    )
    linear = inv_mel(mel)                              # (1, n_freq, T)

    griffin = torchaudio.transforms.GriffinLim(
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        power=1.0,
        n_iter=n_iter,
    )
    return griffin(linear).squeeze(0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True, help="Path to ckpt_*.pt")
    p.add_argument("--enrollment", required=True, help="16 kHz mono WAV of the target speaker")
    p.add_argument("--mixed", required=True, help="16 kHz mono WAV to be separated")
    p.add_argument("--out", required=True, help="Where to write the enhanced WAV")
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--gl-iter", type=int, default=16, help="Griffin-Lim iterations (more = better)")
    args = p.parse_args()

    device = _select_device(args.device)
    print(f"[offline] device={device}")

    # ---- Load checkpoint + restore hparams ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hp = HParams.from_dict(ckpt["hparams"])
    hp.device = str(device)

    model = VoiceFilterLite(hp).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    print(f"[offline] loaded {args.checkpoint} (step={ckpt.get('step')})")

    # ---- Speaker encoder (lazy-loads ECAPA weights on first call) ----
    encoder = FrozenSpeakerEncoder(hp).to(device)

    # ---- Audio IO ----
    enroll_wav, sr_enroll = sf.read(args.enrollment, dtype="float32")
    if sr_enroll != hp.sample_rate:
        import torchaudio
        enroll_wav = torchaudio.functional.resample(
            torch.from_numpy(enroll_wav), sr_enroll, hp.sample_rate
        ).numpy()
    if enroll_wav.ndim > 1:
        enroll_wav = enroll_wav.mean(axis=1)
    enroll_wav_t = torch.from_numpy(enroll_wav)

    mixed_wav, sr_mix = sf.read(args.mixed, dtype="float32")
    if sr_mix != hp.sample_rate:
        import torchaudio
        mixed_wav = torchaudio.functional.resample(
            torch.from_numpy(mixed_wav), sr_mix, hp.sample_rate
        ).numpy()
    if mixed_wav.ndim > 1:
        mixed_wav = mixed_wav.mean(axis=1)
    mixed_wav_t = torch.from_numpy(mixed_wav)
    print(f"[offline] mixed={mixed_wav_t.shape[-1] / hp.sample_rate:.2f}s")

    # ---- Register speaker ----
    dvec = _load_enrollment_dvec(encoder, enroll_wav_t, device)
    sf_model = StreamingVoiceFilter(model, hp)
    sf_model.set_dvector(dvec)

    # ---- Frame-by-frame streaming inference ----
    mel_fn = make_mel_basis(hp, device=torch.device("cpu"))
    mixed_logmel = wav_to_log_mel(mixed_wav_t, hp, mel_fn)         # (T, 80)

    enhanced_frames = []
    for t in range(mixed_logmel.shape[0]):
        enhanced_frames.append(sf_model.step(mixed_logmel[t]).cpu())
    enhanced_logmel = torch.stack(enhanced_frames, dim=0)           # (T, 80)

    # ---- Reconstruct waveform + save ----
    enhanced_wav = _logmel_to_wav(enhanced_logmel, hp, n_iter=args.gl_iter)
    peak = enhanced_wav.abs().max().clamp(min=1e-8)
    if peak > 1.0:
        enhanced_wav = enhanced_wav / peak

    sf.write(args.out, enhanced_wav.numpy(), hp.sample_rate)
    print(f"[offline] wrote {args.out} ({enhanced_wav.shape[-1] / hp.sample_rate:.2f}s)")


if __name__ == "__main__":
    main()