"""Audio feature extraction: STFT → magnitude → log-mel → frame stacking."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torchaudio

from open_voicefilter_lite.hparams import HParams


# ---- Cached transforms (constructed once, reused across calls) ----

def make_mel_basis(hp: HParams, device: torch.device) -> torchaudio.transforms.MelSpectrogram:
    """Build a torchaudio MelSpectrogram configured per HParams.

    Returns log-mel filterbank energies. Always runs on CPU first then moves;
    MelSpectrogram's mel basis is data-independent so device is irrelevant.
    """
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=hp.sample_rate,
        n_fft=hp.n_fft,
        win_length=hp.win_length,
        hop_length=hp.hop_length,
        n_mels=hp.n_mels,
        power=2.0,                      # power spectrogram → |S|^2
        center=False,                   # streaming: no future padding
        window_fn=torch.hann_window,
        mel_scale="htk",
        normalized=False,
    )
    return mel


def wav_to_log_mel(
    wav: torch.Tensor,
    hp: HParams,
    mel_transform: torchaudio.transforms.MelSpectrogram,
) -> torch.Tensor:
    """Convert waveform to log-mel filterbank energies.

    Args:
        wav: (B, T) or (T,) waveform in [-1, 1].
        hp: hyperparameters.
        mel_transform: shared MelSpectrogram module.

    Returns:
        (B, T_frames, n_mels) log-mel features.
    """
    squeeze = False
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
        squeeze = True

    # torchaudio expects (B, T); output is (B, n_mels, T_frames)
    mel = mel_transform(wav)
    mel = mel.transpose(1, 2)  # → (B, T_frames, n_mels)

    # log compression with floor
    log_mel = torch.log(mel + hp.log_mel_floor)

    if squeeze:
        log_mel = log_mel.squeeze(0)
    return log_mel


def frame_stack_left(
    log_mel: torch.Tensor,
    stack: int,
) -> torch.Tensor:
    """Stack `stack` past frames with the current frame for streaming.

    Args:
        log_mel: (B, T, F).
        stack: number of left-context frames including the current one
               (e.g. 3 means: [t-2, t-1, t] stacked).

    Returns:
        (B, T, F * stack). The first (stack-1) frames are zero-padded (no past).
    """
    if stack <= 0:
        raise ValueError("stack must be >= 1")

    if log_mel.dim() == 2:
        log_mel = log_mel.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B, T, F = log_mel.shape
    out = torch.zeros(B, T, F * stack, dtype=log_mel.dtype, device=log_mel.device)

    for i in range(stack):
        # i-th lag: how many frames back, 0 = current
        lag = stack - 1 - i
        if lag == 0:
            out[..., i * F:(i + 1) * F] = log_mel
        else:
            # Shift forward: out[t, i*F:(i+1)*F] = log_mel[t-lag, :], 0 for first `lag` frames
            if T > lag:
                out[:, lag:, i * F:(i + 1) * F] = log_mel[:, :-lag, :]

    if squeeze:
        out = out.squeeze(0)
    return out


def unstack_frame(
    stacked: torch.Tensor,
    stack: int,
    target_index: int = -1,
) -> torch.Tensor:
    """Extract one frame from a stacked tensor (inverse of frame_stack_left).

    Args:
        stacked: (B, T, F * stack).
        stack: stacking factor used to build `stacked`.
        target_index: which slot to pull out. -1 = current frame.

    Returns:
        (B, T, F).
    """
    if stacked.dim() == 2:
        stacked = stacked.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    if target_index < 0:
        target_index = stack + target_index

    B, T, _ = stacked.shape
    F = stacked.shape[-1] // stack
    out = stacked[..., target_index * F:(target_index + 1) * F].contiguous()

    if squeeze:
        out = out.squeeze(0)
    return out


def pad_or_trim(wav: torch.Tensor, length: int) -> torch.Tensor:
    """Center-crop or zero-pad to a fixed length."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    T = wav.shape[-1]
    if T == length:
        out = wav
    elif T > length:
        # Random offset crop is the caller's job; here we center-crop.
        start = (T - length) // 2
        out = wav[..., start:start + length]
    else:
        # Zero-pad at the end
        pad = torch.zeros(*wav.shape[:-1], length - T, dtype=wav.dtype, device=wav.device)
        out = torch.cat([wav, pad], dim=-1)

    if squeeze:
        out = out.squeeze(0)
    return out


def snr_mix(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Mix two signals at a target SNR (dB), scaling the noise.

    SNR = 20 * log10(||clean|| / ||scaled_noise||).
    """
    if clean.dim() == 1:
        clean = clean.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    clean_power = (clean ** 2).mean(dim=-1, keepdim=True).clamp(min=1e-12)
    noise_power = (noise ** 2).mean(dim=-1, keepdim=True).clamp(min=1e-12)
    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    scale = (target_noise_power / noise_power).sqrt()
    mixed = clean + noise * scale

    if squeeze:
        mixed = mixed.squeeze(0)
    return mixed


def random_crop(wav: torch.Tensor, length: int) -> torch.Tensor:
    """Random-crop a waveform to a fixed length (zero-pad if shorter)."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    T = wav.shape[-1]
    if T == length:
        out = wav
    elif T > length:
        start = torch.randint(0, T - length + 1, (1,)).item()
        out = wav[..., start:start + length]
    else:
        pad = torch.zeros(*wav.shape[:-1], length - T, dtype=wav.dtype, device=wav.device)
        out = torch.cat([wav, pad], dim=-1)

    if squeeze:
        out = out.squeeze(0)
    return out
