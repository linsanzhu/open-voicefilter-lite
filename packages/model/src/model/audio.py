"""Audio feature extraction: STFT → magnitude → log-mel → frame stacking.

Two pipelines live here:

* **Main-model mel** (`wav_to_log_mel`) — 80-mel log features for the
  VoiceFilter mask predictor. Uses `torchaudio.MelSpectrogram`; runs once
  per training batch on CPU.
* **d-vector mel** (`wav_to_dvec_mel`, `extract_dvec`) — 40-mel log10
  features feeding the `SpeakerEncoder`. Uses `librosa` STFT + mel basis,
  matching the upstream VoiceFilter `utils/audio.py` recipe exactly so
  the pretrained `models/embedder.pt` weights line up without any
  remapping. `extract_dvec` is the high-level helper that does
  fbank → 80-frame sliding windows → `SpeakerEncoder` per window → L2 →
  final L2, replacing the audio-logic that used to be inside `SpeakerEncoder.forward`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from voicefilter_lite.hparams import HParams


# ---- Cached transforms (constructed once, reused across calls) ----

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


def log_mel_to_linear(log_mel: torch.Tensor, floor: float = 1e-6) -> torch.Tensor:
    """Convert log-mel → linear filterbank energies, undoing `log(x + floor)`.

    Inverse of `wav_to_log_mel`'s log compression. Used to recover
    linear-mel values from the model's log-mel input so the mask
    application (paper §3.2, paper §3.4) can run in the linear energy
    domain — the paper's mask is `M · S_in` on linear filterbank
    energies, NOT on log-mel.

    Args:
        log_mel: (..., F) tensor that came out of `log(x + floor)`.
        floor:  the same `floor` used in `wav_to_log_mel` so the
                round-trip is exact for `log_mel > log(floor)`. Below
                that, the result is clamped to 0 (silence floor).

    Returns:
        (..., F) non-negative linear filterbank energies.
    """
    return (torch.exp(log_mel) - floor).clamp(min=0.0)


def linear_to_log_mel(linear: torch.Tensor, floor: float = 1e-6) -> torch.Tensor:
    """Convert linear → log-mel filterbank energies (forward compression).

    Inverse of `log_mel_to_linear`. Used by the streaming wrapper to
    hand the ASR pipeline the same domain it would have produced from
    the un-filtered input (`log(x + floor)`), so downstream ASR sees no
    feature-distribution shift even though the mask application itself
    runs in linear-mel.

    Args:
        linear: (..., F) non-negative linear filterbank energies.
        floor:  the same `floor` used in `wav_to_log_mel`.

    Returns:
        (..., F) log-mel features with the same compression convention
        as `wav_to_log_mel`.
    """
    return torch.log(linear + floor)


# ---- Mel basis pseudo-inverse (cached, used by log-mel → wav inversion) ----
#
# Reusing the cached mel basis (n_mels, n_freq) across all calls — the basis
# depends only on (sample_rate, n_fft, n_mels, mel_scale). The pseudo-inverse
# itself is also cacheable on the same key. Used by `reconstruct_wav_from_logmel`
# to lift linear-mel magnitudes back to linear-spectrum magnitudes.
#
# Why truncated SVD (`rcond=0.1`) instead of the default `rcond=1e-15`? With
# `n_mels=128` and `n_fft=400` the mel basis has rank ≈ 107 — the upper
# frequency bins are linearly dependent (their mel filter weights overlap
# heavily). A full pseudo-inverse divides by all 128 singular values, the
# last 21 of which are at numerical noise level; the resulting inverse
# amplifies those small components by 50-100x and the reconstructed
# spectrum spikes wildly. Truncating at `rcond=0.1` keeps only the well-
# conditioned subspace and produces a much more conservative
# reconstruction (relative error ~2% vs 50x spikes for the default).
_MEL_PINV_CACHE: dict[Tuple[int, int, int, float], torch.Tensor] = {}


def get_mel_pinv(
    hp: HParams,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    device: torch.device | None = None,
    *,
    rcond: float = 0.1,
) -> torch.Tensor:
    """Return the cached (n_mels, n_freq) Moore-Penrose inverse of the mel basis.

    `fb` inside the torchaudio MelSpectrogram has shape (n_freq, n_mels), so
    its pseudo-inverse (n_mels, n_freq) is the right operator for

        linear_mel (n_mels, T)  @  mel_pinv (n_mels, n_freq)
            →  linear_spec (n_freq, T).

    `rcond` is the singular-value cutoff passed to `torch.linalg.pinv` —
    any singular value below `rcond * max(sv)` is dropped, which tames
    the rank-deficient parts of the mel basis (see module comment).

    Cached on (sample_rate, n_fft, n_mels, rcond). Falls back to CPU
    computation then moves to `device` if requested.
    """
    key = (hp.sample_rate, hp.n_fft, hp.n_mels, rcond)
    cached = _MEL_PINV_CACHE.get(key)
    if cached is None:
        fb = mel_transform.mel_scale.fb                     # (n_freq, n_mels)
        cached = torch.linalg.pinv(fb.transpose(-1, -2), rcond)   # (n_mels, n_freq)
        _MEL_PINV_CACHE[key] = cached
    if device is not None and cached.device != device:
        cached = cached.to(device)
        _MEL_PINV_CACHE[key] = cached
    return cached


def reconstruct_wav_from_logmel(
    enhanced_logmel: torch.Tensor,
    hp: HParams,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    *,
    ref_wav: Optional[torch.Tensor] = None,
    method: str = "phase",
    gl_iter: int = 16,
) -> torch.Tensor:
    """Inverse log-mel → waveform.

    Two paths are supported, picked via `method`:

    * ``"phase"`` (default, one-shot, no iteration). Borrows the STFT phase
      from `ref_wav` (the same mixed wav the model's enhanced log-mel was
      derived from) and combines it with the magnitude implied by
      `enhanced_logmel`. This avoids Griffin-Lim's 16+ iteration loop and
      — in practice — sounds noticeably better: human-grade phase + a
      magnitude that has only lost the masker is already a good listen,
      and STT-grade intelligibility is essentially free.

      The mel basis is a (n_mels, n_freq) projection, so the model has
      already collapsed `n_freq` channels of phase into `n_mels` channels of
      magnitude. We *recover* the linear magnitude by pseudo-inverting
      that projection, then paint the *original* phase back on. The
      recovered phase is consistent (not random) and matches the time
      alignment of the model's frame stack, so the iSTFT produces a
      time-coherent waveform at `ref_wav`'s length.

      `ref_wav` is REQUIRED for this path; missing it raises
      ``ValueError``.

    * ``"griffin_lim"`` (legacy, iterative). Estimates phase via
      Griffin-Lim with `gl_iter` iterations. Use only when `ref_wav` is
      unavailable (e.g. raw `enhanced_logmel` synthesised by hand). Sounds
      metallic and degrades sharply with fewer iterations.

    Args:
        enhanced_logmel: (T, n_mels) post-blend log-mel (the streaming
            wrapper's `step()` return value, stacked over time).
        hp: HParams (n_mels, n_fft, hop_length, win_length, sample_rate,
            log_mel_floor).
        mel_transform: shared torchaudio `MelSpectrogram`
            (constructed in the training loop — see
            `model.dataset.HfTripletStream.__iter__` for the
            canonical construction). Carries the (n_freq, n_mels) mel
            basis + the Hann window used in the forward STFT.
        ref_wav: (T_samples,) reference waveform, REQUIRED for
            ``method="phase"``. Same wav that was fed to `wav_to_log_mel`
            to produce the model's input frames.
        method: ``"phase"`` or ``"griffin_lim"``.
        gl_iter: Griffin-Lim iterations, ignored unless
            ``method="griffin_lim"``.

    Returns:
        (T_samples,) reconstructed waveform on CPU.
    """
    if method not in ("phase", "griffin_lim"):
        raise ValueError(
            f"method must be 'phase' or 'griffin_lim', got {method!r}"
        )
    if method == "phase" and ref_wav is None:
        raise ValueError(
            "method='phase' requires ref_wav (the mixed wav used to derive "
            "the model's input log-mel)."
        )

    # 1) log-mel → linear-mel, (T, n_mels) → (n_mels, T)
    linear = log_mel_to_linear(enhanced_logmel, floor=hp.log_mel_floor).T
    # (n_mels, T) is the form we need to project up to (n_freq, T).

    # Grab the same Hann window the MelSpectrogram used in its STFT so the
    # forward (ref_wav) and inverse (iSTFT) match exactly.
    window = mel_transform.spectrogram.window

    if method == "phase":
        assert ref_wav is not None
        # 2a) Ref-wav STFT → phase AND a magnitude cap baseline.
        target_device = linear.device
        if ref_wav.dim() != 1:
            ref_wav = ref_wav.squeeze(0)
        ref_spec = torch.stft(
            ref_wav.to(target_device),
            n_fft=hp.n_fft,
            hop_length=hp.hop_length,
            win_length=hp.win_length,
            window=window.to(target_device),
            center=False,
            return_complex=True,
        )                                              # (n_freq, T_ref)
        ref_mag = ref_spec.abs()

        # 3a) linear-mel → linear-spectrum magnitude via the cached
        # truncated-SVD pseudo-inverse, then `relu` (the magnitude is
        # non-negative by definition) and a soft cap at 2× the ref
        # magnitude. The cap is the bit that tames the rank-deficient
        # parts of the mel basis: well-conditioned low-frequency bins
        # pass through unchanged, and the wildest ill-conditioned bins
        # (which can otherwise be 50x the true |S|) are clipped to
        # something physical — the model can't create energy from
        # nothing in a single frame, so 2x is a generous bound.
        mel_pinv = get_mel_pinv(hp, mel_transform, device=target_device)
        linear_spec = torch.relu(mel_pinv.to(linear.dtype) @ linear)   # (n_freq, T)

        # Trim both to the common frame count (they should match exactly
        # when ref_wav is the same wav that produced enhanced_logmel, but
        # a `center=False` rounding can cost one frame in edge cases).
        T = min(linear_spec.shape[-1], ref_mag.shape[-1])
        linear_spec = linear_spec[..., :T]
        ref_mag = ref_mag[..., :T]
        # Cap: per-bin, never exceed 2x the ref. Where ref is silent
        # (ref_mag < 1e-6), cap at a small constant so the cap itself
        # doesn't blow up at a near-zero denominator.
        cap = torch.where(ref_mag > 1e-6, ref_mag * 2.0, torch.full_like(ref_mag, 1e-3))
        linear_spec = torch.minimum(linear_spec, cap)

        # Combine: enhanced magnitude + ref phase. This is the
        # VoiceFilter-Lite "re-skin" trick: borrow the original
        # microphone's phase (which the model never sees), keep only
        # the magnitude the model changed. Time-domain coherence is
        # preserved by the ref phase, so iSTFT overlaps the windows
        # without audible glitches.
        phase = torch.angle(ref_spec[..., :T])
        enhanced_spec = linear_spec * (torch.cos(phase) + 1j * torch.sin(phase))

        # 4a) iSTFT back to waveform via a hand-rolled OLA loop.
        #
        # Why not `torch.istft(..., center=False)`? It checks
        # `window_sq_sum >= WINDOW_OVERLAP_ADD_MIN` per sample and raises
        # "window overlap add min: 1" when the Hann window has zero
        # endpoints (so the first and last hops are only covered by a
        # single window whose amplitude is `hann(0) = 0`). This bites any
        # `n_fft == win_length` config — including the HParams default
        # (`n_fft=400, win_length=400`) and the paper config (`n_fft=1200,
        # win_length=400` — but with the larger n_fft, the window is
        # padded internally to 1200 and COLA is fine; still, we want
        # this helper to work for arbitrary configs without a magic
        # threshold). The same configuration with `center=True` does work
        # but shifts every frame by `n_fft // 2` samples, breaking the
        # ref-phase alignment we carefully set up above.
        #
        # Hand-rolled OLA in 8 lines, no boundary check, exactly the
        # frame alignment the model trained with.
        win = window.to(target_device)
        T_frames = enhanced_spec.shape[-1]
        ola_len = (T_frames - 1) * hp.hop_length + hp.win_length
        ola = torch.zeros(ola_len, dtype=enhanced_spec.real.dtype, device=target_device)
        wss = torch.zeros_like(ola)
        for t in range(T_frames):
            # enhanced_spec[..., t] is (n_freq,); IRFFT along the only axis.
            frame = torch.fft.irfft(enhanced_spec[..., t], n=hp.n_fft, dim=-1)
            if hp.n_fft != hp.win_length:
                frame = frame[..., : hp.win_length]
            start = t * hp.hop_length
            ola[start:start + hp.win_length] += frame * win
            wss[start:start + hp.win_length] += win ** 2
        # Normalise by per-sample window^2 sum. The Hann window has
        # zero endpoints, so a handful of samples at the very start /
        # end stay at zero — that's the same "edge silence" the
        # forward STFT discarded when going wav → log-mel, so the
        # alignment matches.
        nz = wss > 1e-10
        ola[nz] = ola[nz] / wss[nz]
        # Edge trimming: the OLA only has one window of "coverage" in
        # the first `win_length - hop_length` and the last
        # `win_length - hop_length` samples, and that one window has
        # small Hann amplitudes at its edges → dividing by tiny
        # `wss` blows the value up by 10-100x. The forward
        # `MelSpectrogram(center=False)` STFT silently drops exactly
        # these same edge samples (it never even sees them), so the
        # *model's* log-mel is "as if" they were zero. We mirror that
        # here: zero out the unreliable edges before padding to
        # `ref_len`. The result is the same length as the input wav
        # with a clean interior.
        edge = hp.win_length - hp.hop_length
        if ola.shape[-1] > 2 * edge:
            ola[:edge] = 0.0
            ola[-edge:] = 0.0
        # Pad/trim to the ref wav length so the caller gets a (T_samples,)
        # tensor at the same length they handed in.
        ref_len = ref_wav.shape[-1]
        if ola.shape[-1] < ref_len:
            ola = torch.nn.functional.pad(ola, (0, ref_len - ola.shape[-1]))
        elif ola.shape[-1] > ref_len:
            ola = ola[..., :ref_len]
        return ola.detach().cpu()

    # method == "griffin_lim"
    # Same projection as the phase path. If `ref_wav` is available we
    # apply the same `2x ref_mag` cap to tame the rank-deficient parts
    # of the mel basis — without it, GL's iterative refinement can
    # diverge on the spurious high-magnitude bins.
    mel_pinv = get_mel_pinv(hp, mel_transform, device=linear.device)
    linear_spec = torch.relu(mel_pinv.to(linear.dtype) @ linear)   # (n_freq, T)
    if ref_wav is not None:
        if ref_wav.dim() != 1:
            ref_wav = ref_wav.squeeze(0)
        ref_spec_gl = torch.stft(
            ref_wav.to(linear.device),
            n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length,
            window=mel_transform.spectrogram.window.to(linear.device),
            center=False, return_complex=True,
        )
        ref_mag_gl = ref_spec_gl.abs()
        T_gl = min(linear_spec.shape[-1], ref_mag_gl.shape[-1])
        linear_spec = linear_spec[..., :T_gl]
        ref_mag_gl = ref_mag_gl[..., :T_gl]
        cap_gl = torch.where(
            ref_mag_gl > 1e-6,
            ref_mag_gl * 2.0,
            torch.full_like(ref_mag_gl, 1e-3),
        )
        linear_spec = torch.minimum(linear_spec, cap_gl)
    target_len = ref_wav.shape[-1] if ref_wav is not None else (
        hp.sample_length_sec * hp.sample_rate
    )
    # Pad to the frame count that `GriffinLim(length=...)` will expect
    # (`target_len // hop + 1`). Without this it complains about
    # `specgram` vs the internally-derived frame count being 48 vs 51.
    needed_frames = int(target_len) // hp.hop_length + 1
    if linear_spec.shape[-1] < needed_frames:
        pad = torch.zeros(
            linear_spec.shape[0],
            needed_frames - linear_spec.shape[-1],
            dtype=linear_spec.dtype, device=linear_spec.device,
        )
        linear_spec = torch.cat([linear_spec, pad], dim=-1)
    griffin = torchaudio.transforms.GriffinLim(
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        power=1.0,
        n_iter=gl_iter,
        length=int(target_len),
    )
    return griffin(linear_spec.unsqueeze(0)).squeeze(0)


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


# ---- d-vector pipeline (paper §3.1) ----
#
# Mirrors the upstream VoiceFilter `utils/audio.py` recipe exactly:
#   librosa.filters.mel(...) → (n_mels, 1 + n_fft//2)
#   librosa.feature.melspectrogram(y, n_fft, hop, win, n_mels, window='hamming', center=False)
#   np.log10(S + 1e-10)
# The pretrained `models/embedder.pt` (paper §3.1 weights) was trained on
# hamming-windowed, log10-mel features; switching to hann or log would
# silently change every output. Keep this exactly as upstream.

# Cached mel basis (depend only on (sample_rate, n_fft, n_mels)). Building
# it once avoids recomputing for every batch.
_DVEC_MEL_BASIS_CACHE: dict[Tuple[int, int, int], np.ndarray] = {}


def _get_dvec_mel_basis(hp: HParams) -> np.ndarray:
    """Return the cached librosa mel basis for the d-vector pipeline.

    Shape: (hp.dvector_n_mels, 1 + hp.dvector_n_fft // 2). Matches the
    upstream VoiceFilter `self.mel_basis = librosa.filters.mel(...)`.
    """
    key = (hp.sample_rate, hp.dvector_n_fft, hp.dvector_n_mels)
    basis = _DVEC_MEL_BASIS_CACHE.get(key)
    if basis is None:
        basis = librosa.filters.mel(
            sr=key[0], n_fft=key[1], n_mels=key[2],
        )
        _DVEC_MEL_BASIS_CACHE[key] = basis
    return basis


def wav_to_dvec_mel(
    wav: torch.Tensor,
    hp: HParams,
) -> torch.Tensor:
    """(B, T) or (T,) waveform @ hp.sample_rate → (B, T_frames, 40) log10 mel.

    Mirrors the upstream VoiceFilter `utils/audio.py` `get_mel` exactly:
    hamming-windowed STFT, power=2, no center padding, log10 with `1e-10`
    floor. librosa operates per-1D-sample so we loop over the batch
    (B is small in this project — B=6 for training, B=1 for inference).

    Different-length batch entries are zero-padded to the longest entry's
    T_frames so the output is a single rectangular tensor — matches the
    training pipeline (refs are already collate-padded to batch max).

    Parallelism: librosa's STFT/mel runs in C under numpy/scipy and
    releases the GIL, so a `ThreadPoolExecutor` gives real wall-clock
    speedup on Apple Silicon for B >= 2. The executor is created lazily
    once per process and reused — spawning a new pool per call would pay
    thread-startup overhead (~5-10 ms) that dwarfs the per-call cost.
    """
    squeeze = wav.dim() == 1
    if squeeze:
        wav = wav.unsqueeze(0)

    wav_np = wav.detach().cpu().numpy().astype(np.float32)
    B = wav_np.shape[0]

    n_fft = hp.dvector_n_fft
    win_length = hp.dvector_win_length
    hop_length = hp.dvector_hop_length
    n_mels = hp.dvector_n_mels

    def _mel_one(i: int) -> np.ndarray:
        S = librosa.feature.melspectrogram(
            y=wav_np[i],
            sr=hp.sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            window="hamming",
            center=False,
            power=2.0,
        )  # (n_mels, T_frames_i)
        return np.log10(S + 1e-10).T  # (T_frames_i, n_mels)

    if B == 1:
        feats = [_mel_one(0)]
    else:
        # Cap workers at 4: each librosa STFT uses ~1 perf core on Apple
        # Silicon, and the main training thread + dataset decode pool are
        # already competing for the 6 perf cores on M-series chips. 4
        # leaves room for the rest of the pipeline without saturating.
        workers = min(B, 4)
        pool = _get_dvec_mel_pool(workers)
        feats = list(pool.map(_mel_one, range(B)))

    T_max = max(f.shape[0] for f in feats)
    out = np.zeros((B, T_max, n_mels), dtype=np.float32)
    for i, f in enumerate(feats):
        out[i, : f.shape[0]] = f

    feat = torch.from_numpy(out).float()
    if squeeze:
        feat = feat.squeeze(0)
    return feat


# ---- Lazy module-level dvec-mel thread pool ----
# Created on first use (rather than at import) so module import stays free
# of side effects and so callers that fork before training don't inherit
# an idle pool. Grow-only: never shrink, so we don't churn threads when
# callers alternate between batch sizes.
_DVEC_MEL_POOL: Optional[ThreadPoolExecutor] = None
_DVEC_MEL_POOL_SIZE: int = 0


def _get_dvec_mel_pool(size: int) -> ThreadPoolExecutor:
    global _DVEC_MEL_POOL, _DVEC_MEL_POOL_SIZE
    if _DVEC_MEL_POOL is None or _DVEC_MEL_POOL_SIZE < size:
        eff = max(size, _DVEC_MEL_POOL_SIZE)
        eff = min(eff, 4)
        _DVEC_MEL_POOL = ThreadPoolExecutor(
            max_workers=eff, thread_name_prefix="dvec-mel",
        )
        _DVEC_MEL_POOL_SIZE = eff
    return _DVEC_MEL_POOL


def extract_dvec(
    wav: torch.Tensor,
    hp: HParams,
    encoder,  # `SpeakerEncoder` — duck-typed to avoid circular import
    device: torch.device | None = None,
    ref_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """(B, T) reference waveform → (B, hp.d_vector_dim) L2-normalised d-vector.

    The full VoiceFilter d-vector recipe (paper §3.1, matching upstream
    `inference.py`):

        1) 40-mel log10 fbank       (librosa, see `wav_to_dvec_mel`)
        2) 80-frame sliding windows (40-frame stride)
        3) per window: encoder → L2-normalise
        4) mean over windows
        5) final L2-normalise

    Args:
        wav: (B, T) padded reference batch or (T,) single waveform
            @ hp.sample_rate.
        hp: HParams.
        encoder: `SpeakerEncoder` instance (already loaded + frozen for
            production). Duck-typed as `forward(mel_window) → (B, 256)`
            to avoid an `audio → speaker_encoder` circular import.
        device: target device for the encoder forward + output (defaults
            to `next(encoder.parameters()).device`).
        ref_lens: optional (B,) **relative** lengths in [0, 1] for batched
            input, parallel to rows of `wav`. Used to crop each entry to
            its actual length before mel extraction so zero-padding from
            `collate_triplets` doesn't pollute the d-vector. If `None`,
            `wav` is treated as fully packed (no padding) — typical for
            single-wav inference (`(T,)` input → `ref_lens=None`).

    Returns:
        (B, hp.d_vector_dim) float32 d-vectors, L2-normalised.

    Performance: all `n_win * B` sliding windows are stacked into one
    `(n_win*B, win, F_dim)` tensor and pushed through the encoder in a
    single call, instead of the legacy per-window Python loop. For the
    default ref_max_sec=6.0 (T_frames≈598, n_win=13) and B=16, that
    collapses 208 LSTM forwards into 1 — CPU launch-overhead drops from
    ~2 s/batch to ~400 ms on Apple Silicon. See `wav_to_dvec_mel` for the
    parallel librosa half of the same speedup.
    """
    if device is None:
        device = next(encoder.parameters()).device
    encoder.eval()

    # ---- Crop zero padding from `collate_triplets` if lengths given ----
    # Without this, librosa STFT would compute silence-floor mels for
    # the padded tail and the per-window L2 aggregation would average
    # them in (collapsing the d-vector toward the silence degenerate
    # point). For single-wav (T,) input this branch is skipped — the
    # caller already trims/pads to `hp.ref_max_sec * sample_rate`.
    if ref_lens is not None and wav.dim() == 2:
        T_max = wav.shape[1]
        actual_lens = (ref_lens * T_max).long()           # (B,) sample counts
        target_len = int(actual_lens.max().item())
        target_len = max(target_len, int(0.5 * hp.sample_rate))  # ≥ 0.5 s floor
        target_len = min(target_len, int(hp.ref_max_sec * hp.sample_rate))
        B = wav.shape[0]
        cropped = torch.zeros(B, target_len, dtype=wav.dtype)
        for i in range(B):
            L = min(int(actual_lens[i].item()), target_len)
            cropped[i, :L] = wav[i, :L]
        wav = cropped

    feat = wav_to_dvec_mel(wav, hp)        # (B, T_frames, 40)
    squeeze_out = feat.dim() == 2
    if squeeze_out:
        feat = feat.unsqueeze(0)
    B, T_frames, F_dim = feat.shape
    win = hp.dvector_window                  # 80
    hop = hp.dvector_stride                  # 40

    # Pad short refs up to one window so the LSTM still produces a vec.
    if T_frames < win:
        pad = torch.zeros(B, win - T_frames, F_dim, dtype=feat.dtype)
        feat = torch.cat([feat, pad], dim=1)
        T_frames = win

    feat = feat.to(device)
    n_win = 1 + (T_frames - win) // hop

    # ---- Batched encoder forward over all (ref × window) slices ----
    # The legacy loop calls `encoder(chunk)` once per sliding window — for a
    # 16-ref batch with 13 windows each that's 208 separate LSTM forwards
    # on CPU. CPU LSTM has measurable per-call launch overhead
    # (~3-5 ms each: Python + operator dispatch + MKL), so the launch cost
    # alone dominates. Stacking all windows into one (n_win*B, win, F_dim)
    # tensor collapses the 208 calls into a single forward — same total
    # FLOPs, ~13x less overhead. Wall-time on Apple Silicon drops from
    # ~2 s/batch to ~400 ms.
    with torch.no_grad():
        if n_win == 1:
            embs = encoder(feat)                                # (B, 256)
            embs = F.normalize(embs, dim=-1, eps=1e-12)         # per-window L2
            mean = embs                                          # (B, 256)
        else:
            # `unfold` returns a view with all `win`-sized slices along
            # dim 1 with stride `hop`; permute+reshape flattens the
            # (B, n_win, F_dim, win) view to (n_win*B, win, F_dim) for
            # a single encoder call.
            windows = feat.unfold(1, win, hop)                   # (B, n_win, F_dim, win)
            windows = windows.permute(1, 0, 3, 2).reshape(
                n_win * B, win, F_dim,
            )
            embs = encoder(windows)                              # (n_win*B, 256)
            embs = embs.reshape(n_win, B, -1)                   # (n_win, B, 256)
            embs = F.normalize(embs, dim=-1, eps=1e-12)         # per-window L2
            mean = embs.mean(dim=0)                             # (B, 256)
    out = F.normalize(mean, dim=-1, eps=1e-12)
    if squeeze_out:
        out = out.squeeze(0)
    return out