#!/usr/bin/env python3
"""Example 02 — Real-time streaming inference (frame-by-frame).

VoiceFilter-Lite's defining property is *left-context-only* masking — it never
peeks at future frames, so the same code path works for both an offline file
and a live microphone. This script demonstrates both:

  • `--mode file`   read a WAV chunk-by-chunk and run StreamingVoiceFilter
                    on each frame. Writes an enhanced WAV.
  • `--mode mic`    capture the default input device via sounddevice, push
                    10 ms blocks through the model, and (optionally) play
                    the enhanced audio back. Requires `sounddevice`.

The streaming wrapper keeps LSTM hidden state between frames, so each step is
O(1) past context. The d-vector is set ONCE per enrollment (or per call to
`set_dvector`) — there is no per-frame speaker embedding cost.

Usage
-----
    # File mode (offline, deterministic — no extra deps):
    python examples/02_streaming_inference.py \\
        --checkpoint runs/exp1/ckpt_final.pt \\
        --enrollment  data/enroll.wav \\
        --mixed       data/mixed.wav \\
        --out         data/enhanced_streaming.wav \\
        --mode file

    # Mic mode (live, requires `pip install sounddevice`):
    python examples/02_streaming_inference.py \\
        --checkpoint runs/exp1/ckpt_final.pt \\
        --enrollment  data/enroll.wav \\
        --mode mic \\
        --duration 30
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from open_voicefilter_lite.audio import (  # noqa: E402
    extract_dvec,
    make_mel_basis,
    wav_to_log_mel,
)
from open_voicefilter_lite.hparams import HParams  # noqa: E402
from open_voicefilter_lite.model import (  # noqa: E402
    StreamingVoiceFilter,
    VoiceFilterLite,
)
from open_voicefilter_lite.speaker_encoder import SpeakerEncoder  # noqa: E402


def _select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_audio_mono_16k(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import torchaudio
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), sr, target_sr
        ).numpy()
    return torch.from_numpy(wav)


def _logmel_to_wav(log_mel: torch.Tensor, hp: HParams, n_iter: int = 16) -> torch.Tensor:
    """Inverse log-mel → waveform via Griffin-Lim (audition only, NOT a
    production path — real pipelines feed enhanced log-mel straight to ASR)."""
    import torchaudio

    mel = torch.exp(log_mel) - hp.log_mel_floor
    mel = mel.transpose(0, 1).unsqueeze(0)

    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=hp.n_fft // 2 + 1,
        n_mels=hp.n_mels,
        sample_rate=hp.sample_rate,
        max_iter=128,
    )
    linear = inv_mel(mel)

    griffin = torchaudio.transforms.GriffinLim(
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        power=1.0,
        n_iter=n_iter,
    )
    return griffin(linear).squeeze(0)


def _register_speaker(
    encoder,
    enroll_path: str,
    hp: HParams,
    device: torch.device,
) -> torch.Tensor:
    wav = _load_audio_mono_16k(enroll_path, hp.sample_rate)
    with torch.no_grad():
        return extract_dvec(wav, hp, encoder, device=device).squeeze(0).cpu()


# ---- File mode: simulate streaming over a WAV ----


def run_file_mode(
    model: VoiceFilterLite,
    sf_model: StreamingVoiceFilter,
    hp: HParams,
    mel_fn,
    mixed_path: str,
    out_path: str,
    gl_iter: int,
) -> None:
    mixed = _load_audio_mono_16k(mixed_path, hp.sample_rate)
    log_mel = wav_to_log_mel(mixed, hp, mel_fn)        # (T, 80)

    enhanced = []
    for t in range(log_mel.shape[0]):
        enhanced.append(sf_model.step(log_mel[t]).cpu())
    enhanced_logmel = torch.stack(enhanced, dim=0)     # (T, 80)

    wav = _logmel_to_wav(enhanced_logmel, hp, n_iter=gl_iter)
    peak = wav.abs().max().clamp(min=1e-8)
    if peak > 1.0:
        wav = wav / peak
    sf.write(out_path, wav.numpy(), hp.sample_rate)
    print(f"[streaming] wrote {out_path} ({wav.shape[-1] / hp.sample_rate:.2f}s)")


# ---- Mic mode: real-time capture via sounddevice ----


def run_mic_mode(
    sf_model: StreamingVoiceFilter,
    hp: HParams,
    mel_fn,
    duration: float,
    play_back: bool,
) -> None:
    """Capture from the default mic and stream through VoiceFilter-Lite.

    Requires `sounddevice`. Real-time, blocksize = hp.hop_length (10 ms).
    """
    try:
        import sounddevice as sd
    except ImportError as e:
        raise SystemExit(
            "mic mode requires sounddevice: `pip install sounddevice`"
        ) from e

    blocksize = hp.hop_length
    samplerate = hp.sample_rate

    print(f"[mic] samplerate={samplerate} blocksize={blocksize} ({1000 * blocksize / samplerate:.1f} ms)")
    if play_back:
        print("[mic] WARNING: live Griffin-Lim is too slow; enhanced audio is "
              "logged per second, not played back in real time.")

    enhanced_logmel_chunks = []

    def callback(indata, _frames, _time, _status):
        # indata shape: (blocksize, n_channels). Use the first channel.
        chunk = torch.from_numpy(indata[:, 0].copy())
        lm = wav_to_log_mel(chunk, hp, mel_fn)
        for t in range(lm.shape[0]):
            enhanced_logmel_chunks.append(sf_model.step(lm[t]).cpu())

    print(f"[mic] capturing {duration:.1f}s — speak now")
    with sd.InputStream(
        channels=1,
        samplerate=samplerate,
        blocksize=blocksize,
        dtype="float32",
        callback=callback,
    ):
        sd.sleep(int(duration * 1000))

    print(f"[mic] processed {len(enhanced_logmel_chunks)} frames")
    if play_back and enhanced_logmel_chunks:
        enhanced = torch.stack(enhanced_logmel_chunks, dim=0)
        wav = _logmel_to_wav(enhanced, hp, n_iter=8)
        peak = wav.abs().max().clamp(min=1e-8)
        if peak > 1.0:
            wav = wav / peak
        # Use sounddevice to play back the captured-then-enhanced waveform.
        sd.play(wav.numpy(), samplerate=samplerate)
        sd.wait()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True, help="Path to ckpt_*.pt")
    p.add_argument("--enrollment", required=True, help="16 kHz mono WAV of the target speaker")
    p.add_argument("--mode", choices=["file", "mic"], default="file")
    # File-mode options:
    p.add_argument("--mixed", help="16 kHz mono WAV (required in --mode file)")
    p.add_argument("--out", help="Where to write the enhanced WAV (required in --mode file)")
    p.add_argument("--gl-iter", type=int, default=16)
    # Mic-mode options:
    p.add_argument("--duration", type=float, default=30.0, help="seconds to capture (mic mode)")
    p.add_argument("--play-back", action="store_true", help="mic mode: play back enhanced audio after capture")
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    args = p.parse_args()

    device = _select_device(args.device)
    print(f"[streaming] device={device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hp = HParams.from_dict(ckpt["hparams"])
    hp.device = str(device)

    model = VoiceFilterLite(hp).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    print(f"[streaming] loaded {args.checkpoint} (step={ckpt.get('step')})")

    encoder = SpeakerEncoder(hp).to(device)
    if Path(hp.dvector_pt_path).is_file():
        encoder.load_pretrained(hp.dvector_pt_path)
        print(f"[streaming] SpeakerEncoder loaded from {hp.dvector_pt_path}")
    else:
        print(
            f"[streaming] WARNING: dvector weights not found at {hp.dvector_pt_path}; "
            f"encoder will use RANDOM init. The pretrained embedder "
            f"(`models/embedder.pt`) ships with this repo (from "
            f"https://github.com/maum-ai/voicefilter)."
        )
        encoder.freeze()
    print(f"[streaming] d_vector_dim={hp.d_vector_dim}")
    mel_fn = make_mel_basis(hp, device=torch.device("cpu"))

    dvec = _register_speaker(encoder, args.enrollment, hp, device)
    sf_model = StreamingVoiceFilter(model, hp)
    sf_model.set_dvector(dvec)

    if args.mode == "file":
        if not args.mixed or not args.out:
            raise SystemExit("--mode file requires --mixed and --out")
        run_file_mode(model, sf_model, hp, mel_fn, args.mixed, args.out, args.gl_iter)
    else:
        run_mic_mode(sf_model, hp, mel_fn, args.duration, args.play_back)


if __name__ == "__main__":
    main()