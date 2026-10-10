"""Inference-side mel basis construction.

This is the only audio helper kept inside the published
`open-voicefilter-lite` wheel — the rest of the DSP pipeline (STFT, OLA,
d-vector extraction, ...) lives in the training-only `model.audio`
package. `StreamingVoiceFilter.__init__` calls `make_mel_basis(self.hp)`
once at construction to derive the (n_freq, n_mels) mel filterbank
matrix and the Hann window the forward STFT and inverse OLA both share.

No torch / torchaudio: the only thing the inference side ever needed
from `torchaudio.MelSpectrogram` was the mel basis + the Hann window.
`librosa.filters.mel` (HTK scale) + `np.hanning` reproduce both
exactly, so the forward-STFT alignment with the training pipeline
that produced the ONNX weights is preserved bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from voicefilter_lite.hparams import HParams


@dataclass(frozen=True)
class MelBasis:
    """Cached mel basis + Hann window for the streaming forward STFT.

    Pre-computed once in `StreamingVoiceFilter.__init__` and reused for
    the lifetime of the instance. The forward STFT and the inverse OLA
    loop both consume the same Hann window, so the time alignment
    between the model's mask output and the reconstructed waveform
    matches the training pipeline exactly.

    Attributes:
        fb: (n_freq, n_mels) float32 mel filterbank (transposed relative
            to `librosa.filters.mel`'s output so that `fb.T @ mag2`
            computes the linear-mel projection in one matmul).
        hann: (win_length,) float32 periodic Hann window. Matches
            `torch.hann_window(win_length, periodic=True)` — the
            periodic variant (cosine has period N, not N-1) that
            STFTs expect, so the inverse OLA lines up with the
            forward STFT the model trained against.
    """
    fb: np.ndarray
    hann: np.ndarray


def make_mel_basis(hp: HParams) -> MelBasis:
    """Build the (n_freq, n_mels) mel basis + Hann window used by `step`.

    `librosa.filters.mel(..., n_mels=...)` defaults to `htk=True`, which
    matches the training pipeline's `torchaudio.MelSpectrogram(mel_scale="htk")`.
    `np.hanning(N)` matches `torch.hann_window(N, periodic=True)` — the
    periodic Hann, which is what STFTs expect (window length N, cosine
    has period N, not N-1). Both choices preserve the exact forward-STFT
    alignment with the training pipeline that produced the ONNX weights.
    """
    fb = librosa.filters.mel(
        sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.n_mels,
    ).T.astype(np.float32)                                  # (n_freq, n_mels)
    hann = np.hanning(hp.win_length).astype(np.float32)     # (win_length,)
    return MelBasis(fb=fb, hann=hann)
