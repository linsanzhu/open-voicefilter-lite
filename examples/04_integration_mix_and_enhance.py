#!/usr/bin/env python3
"""Example 04 — Integration test: synthesize mixed clips from the dataset and
run the model on them so you can audition the result directly.

For each of `--num-samples` synthetic "real-world" mixes this script:

  1. Picks a random target speaker from LibriSpeech (config=clean).
  2. Picks a separate utterance from the same speaker as the enrollment
     clip (used to compute the d-vector at run time).
  3. Picks an interference clip from a configurable pool:
        - LibriSpeech "other" split (a different speaker)
        - Musan music subset
        - Musan noise subset
        - Musan speech subset overlaid as "babble" (paper §3.4 babble recipe)
  4. Optionally convolves the target with a real (or procedural fallback)
     room impulse response — the same RIR augmentation the training
     pipeline uses.
  5. Mixes target + interference at a random SNR ∈
     [hp.snr_min_db, hp.snr_max_db].
  6. Runs VoiceFilter-Lite frame-by-frame (`StreamingVoiceFilter.step`)
     exactly the way a production ASR frontend would consume it, then
     reconstructs a waveform with Griffin-Lim.

Output layout (one folder per sample, plus a human-readable summary):

    out-dir/
      00_target.wav          # clean target speaker
      00_enrollment.wav      # same speaker, different utterance (dvec source)
      00_interference.wav    # the "other" sound
      00_mixed.wav           # target + interference at the picked SNR
      00_enhanced.wav        # model output — listen to this
      ...
      NN_target.wav
      ...
      summary.txt            # per-sample SNR / sources / paths

The construction mirrors `LibriSpeechTripletStream` (paper §3.4). Unlike
training samples, we *always* keep both target and interference active —
silent-target / silent-interference "do nothing" branches are skipped, so
every output has something for the model to enhance (otherwise the
enhanced clip would just be the un-mixed silence / interference).

This script is for audition, not for SI-SDR; for numerical evaluation use
`scripts/evaluate.py`. For ONNX parity checks use
`examples/03_onnx_inference.py --verify`.

Usage
-----
    python examples/04_integration_mix_and_enhance.py \\
        --checkpoint runs/exp1/ckpt_final.pt \\
        --out-dir    data/integration_test \\
        --num-samples 5

    # Force-disable Musan (use only LibriSpeech "other" speakers as interference):
    python examples/04_integration_mix_and_enhance.py \\
        --checkpoint runs/exp1/ckpt_final.pt \\
        --out-dir    data/integration_test \\
        --num-samples 5 \\
        --no-musan

    # Behind a HF mirror:
    python examples/04_integration_mix_and_enhance.py \\
        --checkpoint runs/exp1/ckpt_final.pt \\
        --out-dir    data/integration_test \\
        --hf-endpoint https://hf-mirror.com
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

# Self-contained: add the `src/` layout to sys.path so this works without
# `pip install -e .`.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from open_voicefilter_lite.audio import (  # noqa: E402
    extract_dvec,
    make_mel_basis,
    snr_mix,
    wav_to_log_mel,
)
from open_voicefilter_lite.cli import apply_hf_endpoint  # noqa: E402
from open_voicefilter_lite.dataset import (  # noqa: E402
    _cast_audio_no_decode,
    _convolve,
    _decode_audio,
    _disable_hf_audio_decode,
    _synthetic_rir,
)
from open_voicefilter_lite.hparams import HParams  # noqa: E402
from open_voicefilter_lite.model import VoiceFilterLite  # noqa: E402
from open_voicefilter_lite.speaker_encoder import (  # noqa: E402
    SpeakerEncoder,
)
from open_voicefilter_lite.streaming import (  # noqa: E402
    StreamingVoiceFilter,
)


# ---- Constants mirroring LibriSpeechTripletStream ----
INTERFERENCE_TAG_LIBRI = "librispeech"
INTERFERENCE_TAG_MUSIC = "music"
INTERFERENCE_TAG_NOISE = "noise"
INTERFERENCE_TAG_BABBLE = "babble"
# Multi-talker mode: N distinct LibriSpeech "other" speakers overlaid with
# random gains. Forced when `--libri-babble N >= 2` is set; uses LibriSpeech
# speech (same distribution the model was trained on) rather than Musan
# speech (which is non-Librispeech recordings).
INTERFERENCE_TAG_LIBRI_BABBLE = "libri_babble"


# --------------------------------------------------------------------------- #
# Device + checkpoint helpers (kept identical to examples 01–03)              #
# --------------------------------------------------------------------------- #


def _select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _logmel_to_wav(log_mel: torch.Tensor, hp: HParams, n_iter: int = 16) -> torch.Tensor:
    """Inverse log-mel → waveform via Griffin-Lim.

    NOTE: This is for offline audition only. Production streaming pipelines
    feed `enhanced_logmel` straight into an ASR frontend — see
    `examples/01_offline_separation.py`'s `_logmel_to_wav` for the same
    helper. Kept here so example 04 stands alone.

    Two torchaudio-version hazards handled here:

    1. `InverseMelScale` accepted `max_iter=` on torchaudio < 2.0 but
       dropped the kwarg in 2.0+. Pass it conditionally.

    2. The 80-bin HTK mel basis has rank 75 (overlapping triangular
       filters are linearly dependent), so `InverseMelScale`'s default
       `lstsq(gels)` driver raises `LinAlgError: input matrix does not
       have full rank` on every call. We compute the inverse with
       `torch.linalg.pinv` (Moore-Penrose pseudo-inverse) directly,
       which is well-defined for rank-deficient matrices.
    """
    import torchaudio

    mel = torch.exp(log_mel) - hp.log_mel_floor
    mel = mel.transpose(0, 1).unsqueeze(0)              # (1, F, T)

    # ---- Inverse mel via Moore-Penrose pseudo-inverse ----
    # We want linear_spec (n_freq, T) s.t. fb^T @ linear_spec ≈ mel
    # (fb is (n_freq, n_mels)). Least-squares: linear_spec = pinv(fb^T) @ mel.
    # `MelSpectrogram.mel_scale.fb` is the canonical accessor across
    # torchaudio versions.
    mel_transform = make_mel_basis(hp, device=torch.device("cpu"))
    fb = mel_transform.mel_scale.fb                    # (n_freq, n_mels)
    fb_pinv = torch.linalg.pinv(fb.transpose(-1, -2))  # (n_freq, n_mels)
    # Batched matmul: (1, n_freq, n_mels) @ (1, n_mels, T) → (1, n_freq, T).
    # `unsqueeze(0)` lets us broadcast fb_pinv across the batch dim.
    linear = torch.relu(fb_pinv.to(mel.dtype).to(mel.device) @ mel)  # (1, n_freq, T)

    # torchaudio's GriffinLim rebuilds via STFT(center=True), which expects
    # T_frames = target_len // hop + 1 (e.g. 48000/160 + 1 = 301) for the
    # target output length. Our STFT was center=False and yields 298 frames
    # for 48000 samples — the trailing half-window is dropped. Right-pad
    # `linear` with silence so the internal `specgram * angles` matmul shape
    # matches and the wav round-trips to the full `target_len`.
    target_len = int(hp.sample_length_sec * hp.sample_rate)
    needed_frames = target_len // hp.hop_length + 1
    if linear.shape[-1] < needed_frames:
        pad = torch.zeros(
            1, linear.shape[1], needed_frames - linear.shape[-1],
            dtype=linear.dtype, device=linear.device,
        )
        linear = torch.cat([linear, pad], dim=-1)

    griffin = torchaudio.transforms.GriffinLim(
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        power=1.0,
        n_iter=n_iter,
        length=target_len,
    )
    return griffin(linear).squeeze(0)


def _save_wav(path: str, wav: torch.Tensor, sample_rate: int) -> None:
    """Peak-normalise then write a 16-bit WAV.

    Peak normalisation keeps audibility consistent across the per-sample
    WAVs — mixed.wav might come in quieter than target.wav because of the
    SNR scaling, but the listener's volume knob shouldn't have to chase
    the per-file energy. We clamp to ±1 only if a clip would clip.
    """
    peak = wav.abs().max().clamp(min=1e-8)
    if peak > 1.0:
        wav = wav / peak
    sf.write(path, wav.numpy().astype(np.float32), sample_rate)


# --------------------------------------------------------------------------- #
# Dataset source wrappers                                                     #
# --------------------------------------------------------------------------- #


class AudioSource:
    """Lazy, on-disk LibriSpeech wrapper.

    Builds an in-memory speaker index (`speaker_id` → `[row_idx]`) without
    touching the FLAC bytes. Audio is fetched and decoded on demand via
    `ds[row_idx]` — the arrow cache reads only that row's bytes (O(1)
    memory, not O(N × clip_size)).
    """

    def __init__(self, hp: HParams, config: str, split: str, label: str):
        from datasets import load_dataset

        os.makedirs(hp.librispeech_cache, exist_ok=True)
        ds = load_dataset(
            "openslr/librispeech_asr",
            config,
            split=split,
            streaming=hp.streaming,
            cache_dir=hp.librispeech_cache,
        )
        # IMPORTANT: do NOT call `_cast_audio_no_decode(ds)` here — it would
        # walk the schema; we want to keep `ds` as a normal arrow-backed
        # dataset with `__getitem__` so we can do `ds[row_idx]` on demand
        # without ever materialising every row's FLAC bytes in RAM.
        self._ds = ds
        self._label = label
        self._len = len(ds)

        # Build the per-speaker index. The arrow table exposes
        # `ds.column_names` (including `speaker_id`) and supports fast
        # column reads without touching audio bytes.
        spk_col = ds.column_names.index("speaker_id")
        # `ds.data.column(spk_col).to_pylist()` reads the column once and
        # returns plain Python strings — the cheapest way to enumerate
        # ~104k speaker IDs without paying for any FLAC decoding.
        speaker_ids = ds.data.column(spk_col).to_pylist()
        self._by_spk: dict[str, list[int]] = {}
        for idx, spk in enumerate(speaker_ids):
            self._by_spk.setdefault(str(spk), []).append(idx)
        print(f"[{label}] {self._len} rows, {len(self._by_spk)} distinct speakers")

    @property
    def label(self) -> str:
        return self._label

    def __len__(self) -> int:
        return self._len

    def random_item(self, rng: random.Random, exclude: Optional[set[int]] = None) -> tuple[int, torch.Tensor]:
        """Pick a random row, decode it, return (row_idx, wav_1d_f32)."""
        n = self._len
        for _ in range(20):
            idx = rng.randrange(n)
            if exclude is None or idx not in exclude:
                break
        else:
            idx = rng.randrange(n)
        return idx, _decode_audio(self._ds[idx]["audio"], 16_000)

    def item_by_speaker(
        self,
        rng: random.Random,
        spk: str,
        exclude: Optional[set[int]] = None,
    ) -> Optional[tuple[int, torch.Tensor]]:
        """Pick a different row for the same speaker (enrollment pick).

        Returns None if the speaker has only one utterance (rare in
        LibriSpeech train.360 — train.100 single-utterance speakers do
        exist and the caller should fall back to reusing the target).
        """
        cands = self._by_spk.get(spk, [])
        if not cands:
            return None
        for _ in range(20):
            idx = rng.choice(cands)
            if exclude is None or idx not in exclude:
                return idx, _decode_audio(self._ds[idx]["audio"], 16_000)
        return None

    def babble(
        self,
        length: int,
        rng: random.Random,
        n_speakers: int,
        gain_range: tuple[float, float] = (0.3, 1.0),
        max_tries: int = 40,
    ) -> torch.Tensor:
        """Overlay N random LibriSpeech "other" clips with random gains.

        Cocktail-party interference: N different speakers from the
        LibriSpeech `other` split, each with a uniform gain in
        `gain_range`, concatenated and cropped to `length` samples. Uses
        LibriSpeech speech (same distribution as training) so the test
        matches what the model actually learned — vs Musan babble which
        pulls from non-Librispeech recordings.

        Args:
            length: target wav length in samples.
            rng: random source.
            n_speakers: minimum number of speakers to draw. We pull more
                if needed to ensure `length` samples of audio.
            gain_range: per-clip gain range (uniform).
            max_tries: max random-row attempts before giving up.

        Returns:
            (length,) waveform, dtype=float32.
        """
        chunks: list[torch.Tensor] = []
        gains: list[float] = []
        for _ in range(max_tries):
            _, wav = self.random_item(rng)
            chunks.append(wav)
            gains.append(rng.uniform(*gain_range))
            if len(chunks) >= n_speakers and sum(c.shape[-1] for c in chunks) >= length:
                break
        full = torch.cat(chunks).to(torch.float32)
        g = torch.cat(
            [torch.full((c.shape[-1],), gi) for c, gi in zip(chunks, gains)]
        )
        # `_crop_to` lives at module scope; it zero-pads if the concatenated
        # babble came up short, which is rare (we over-pull until length).
        return _crop_to(full * g, length, rng)


class MusanSource:
    """On-disk MUSAN wrapper supporting music / noise / speech subsets.

    `babble` is virtual: overlay N random `speech` clips with per-clip
    gains ∈ [gain_min, gain_max), then crop — same recipe as
    `LibriSpeechTripletStream._synthesise_babble`.
    """

    def __init__(self, hp: HParams, subsets: list[str], babble_n: int,
                 babble_gain: tuple[float, float]):
        from datasets import load_dataset

        os.makedirs(hp.musan_cache, exist_ok=True)
        self._iters: dict[str, list[torch.Tensor]] = {}
        for subset in subsets:
            ds = load_dataset(
                hp.musan_source,
                subset,
                split="train",
                streaming=hp.streaming,
                cache_dir=hp.musan_cache,
            )
            ds = _cast_audio_no_decode(ds)
            # Pre-decode all rows into RAM. Music/noise subsets are
            # ~10–60 s clips at 16 kHz → a few hundred MB tops; well
            # within budget for an audition script. We keep them as
            # decoded 1-D tensors so random_item() never has to touch
            # disk or soundfile again.
            wavs: list[torch.Tensor] = []
            for ex in ds:
                try:
                    wavs.append(_decode_audio(ex["audio"], 16_000))
                except Exception:
                    continue
            self._iters[subset] = wavs

        self._babble_n = max(2, babble_n)
        self._babble_gain = babble_gain
        self._enabled_subsets = list(subsets)

    def random_item(self, rng: random.Random, length: int) -> tuple[torch.Tensor, str]:
        """Pick a random clip (or synthesised babble) cropped to `length`.

        Returns (wav, subset_label).
        """
        subset = rng.choice(self._enabled_subsets)
        if subset == "speech" and rng.random() < 0.5:
            # Half the speech picks are returned as synthesised babble so
            # the listener hears what the model does to babble, not just
            # single-talker interference.
            return self._babble(length, rng), INTERFERENCE_TAG_BABBLE
        wavs = self._iters[subset]
        if not wavs:
            raise RuntimeError(f"[musan] subset '{subset}' has no decodable rows")
        wav = rng.choice(wavs)
        return self._crop_or_pad(wav, length, rng), subset

    def _babble(self, length: int, rng: random.Random) -> torch.Tensor:
        speech = self._iters.get("speech", [])
        if not speech:
            raise RuntimeError("[musan] babble requested but 'speech' subset is empty")
        n = self._babble_n
        chunks: list[torch.Tensor] = []
        gains: list[float] = []
        for _ in range(n * 4):
            wav = rng.choice(speech)
            chunks.append(wav)
            gains.append(rng.uniform(*self._babble_gain))
            if sum(c.shape[-1] for c in chunks) >= length:
                break
        full = torch.cat(chunks).to(torch.float32)
        g = torch.cat([torch.full((c.shape[-1],), gi) for c, gi in zip(chunks, gains)])
        return self._crop_or_pad(full * g, length, rng)

    @staticmethod
    def _crop_or_pad(wav: torch.Tensor, length: int, rng: random.Random) -> torch.Tensor:
        T = wav.shape[-1]
        if T == length:
            return wav
        if T > length:
            start = rng.randrange(0, T - length + 1)
            return wav[..., start:start + length]
        pad = torch.zeros(length - T, dtype=wav.dtype)
        return torch.cat([wav, pad])


def _load_rir_pool(hp: HParams, rng: random.Random, max_n: int = 256) -> list[torch.Tensor]:
    """Try real RIRs first; fall back to procedural exponential-decay RIRs."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("[example04] `datasets` is required: pip install datasets") from e

    sources: list[str] = [hp.rir_source, *hp.rir_source_fallbacks]
    config_candidates = [
        ({"name": "simulated_rirs"}, "simulated_rirs"),
        ({}, "default"),
    ]
    for source in sources:
        for kwargs, label in config_candidates:
            try:
                ds = load_dataset(
                    source, split="train", streaming=False,
                    cache_dir=hp.rir_cache, **kwargs,
                )
                rirs: list[torch.Tensor] = []
                for ex in ds:
                    h = _decode_audio(ex["audio"], hp.sample_rate)
                    if h.shape[-1] >= int(0.05 * hp.sample_rate):
                        rirs.append(h)
                    if len(rirs) >= max_n:
                        break
                if rirs:
                    print(f"[rir] loaded {len(rirs)} real RIRs from {source} ({label})")
                    return rirs
            except Exception as e:
                print(f"[rir] {source} ({label}) failed: {e}")
                continue
    print(f"[rir] no real RIRs available; using {max_n} procedural fallbacks")
    return [_synthetic_rir(hp.sample_rate, length_s=0.5, rng=rng) for _ in range(max_n)]


# --------------------------------------------------------------------------- #
# Per-sample construction                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class SampleSummary:
    """Per-sample metadata persisted to `summary.txt` and `summary.json`."""
    index: int
    target_speaker: str
    target_path: str
    enrollment_path: str
    interference_kind: str
    interference_detail: str
    interference_path: str
    snr_db: float
    rir_applied: bool
    mixed_path: str
    enhanced_path: str
    notes: list[str] = field(default_factory=list)


def _resolve_interference(
    rng: random.Random,
    hp: HParams,
    sample_len: int,
    use_musan: bool,
    musan: Optional[MusanSource],
    inter: AudioSource,
    libri_babble_n: int = 0,
) -> tuple[torch.Tensor, str, str]:
    """Pick an interference source and crop it to `sample_len`.

    Returns (interference_wav, kind, detail). detail is the row idx (for
    LibriSpeech) or subset name (for Musan / babble) — written into the
    summary so the user can look up what they heard.

    If `libri_babble_n >= 2`, every sample uses N-speaker LibriSpeech
    babble as interference (forces multi-talker mode; ignores the musan
    pool).
    """
    if libri_babble_n >= 2:
        wav = inter.babble(sample_len, rng, libri_babble_n)
        return (
            _crop_to(wav, sample_len, rng),
            INTERFERENCE_TAG_LIBRI_BABBLE,
            f"librispeech_babble:n={libri_babble_n}",
        )

    pool: list[str] = [INTERFERENCE_TAG_LIBRI]
    if use_musan and musan is not None:
        for kind in (INTERFERENCE_TAG_MUSIC, INTERFERENCE_TAG_NOISE, INTERFERENCE_TAG_BABBLE):
            if kind in musan._enabled_subsets or kind == INTERFERENCE_TAG_BABBLE:
                pool.append(kind)
    kind = rng.choice(pool)

    if kind == INTERFERENCE_TAG_LIBRI:
        idx, wav = inter.random_item(rng)
        return _crop_to(wav, sample_len, rng), kind, f"librispeech_other:row={idx}"
    if kind in (INTERFERENCE_TAG_MUSIC, INTERFERENCE_TAG_NOISE, INTERFERENCE_TAG_BABBLE):
        wav, label = musan.random_item(rng, sample_len)
        # `musan.random_item` already returned a `length`-cropped wav.
        return _crop_to(wav, sample_len, rng), kind, f"musan:{label}"
    raise RuntimeError(f"[example04] unknown interference kind {kind!r}")


def _crop_to(wav: torch.Tensor, length: int, rng: random.Random) -> torch.Tensor:
    """Random-crop (or zero-pad) a 1-D wav to exactly `length` samples."""
    T = wav.shape[-1]
    if T == length:
        return wav
    if T > length:
        start = rng.randrange(0, T - length + 1)
        return wav[..., start:start + length]
    pad = torch.zeros(length - T, dtype=wav.dtype)
    return torch.cat([wav, pad])


def _pick_enrollment(
    rng: random.Random,
    target_source: AudioSource,
    target_idx: int,
    spk: str,
    hp: HParams,
) -> Optional[torch.Tensor]:
    """Pick a different same-speaker utterance (capped to ref_max_sec).

    Returns None if the speaker has only one utterance — caller falls
    back to using the target wav itself (the dataset does the same in
    that case; the d-vector is then noisy, but the test still runs).
    """
    # Try same-speaker first.
    pick = target_source.item_by_speaker(rng, spk, exclude={target_idx})
    if pick is None:
        # Single-utterance speaker: fall back to reusing the target.
        return None
    _, wav = pick
    cap = int(hp.ref_max_sec * hp.sample_rate)
    if wav.shape[-1] > cap:
        start = rng.randrange(0, wav.shape[-1] - cap + 1)
        wav = wav[..., start:start + cap]
    return wav


def _apply_rir(
    rng: random.Random,
    hp: HParams,
    target_wav: torch.Tensor,
    rirs: list[torch.Tensor],
) -> tuple[torch.Tensor, bool]:
    """Optionally convolve target with a random RIR (mirrors training prob)."""
    if rirs and rng.random() < hp.rir_prob:
        h = rng.choice(rirs)
        y = _convolve(target_wav, h)
        # Rescale convolved energy to match pre-reverb target energy so the
        # SNR math downstream still has the same reference level.
        ref = target_wav.pow(2).mean().sqrt().clamp(min=1e-8)
        cur = y.pow(2).mean().sqrt().clamp(min=1e-8)
        y = y * (ref / cur)
        return y, True
    return target_wav, False


def build_one_sample(
    idx: int,
    hp: HParams,
    rng: random.Random,
    target_source: AudioSource,
    inter: AudioSource,
    musan: Optional[MusanSource],
    rirs: list[torch.Tensor],
    use_musan: bool,
    out_dir: Path,
    model: VoiceFilterLite,
    encoder,
    mel_fn,
    gl_iter: int,
    libri_babble_n: int = 0,
) -> SampleSummary:
    """Construct, mix, and enhance one sample end-to-end. Returns a summary."""
    notes: list[str] = []
    sample_len = int(hp.sample_length_sec * hp.sample_rate)

    # 1) Target utterance
    target_idx, target_wav_full = target_source.random_item(rng)
    spk = str(target_source._ds[target_idx].get("speaker_id", "?"))

    # 2) Enrollment — different utterance, same speaker if possible
    enroll_wav = _pick_enrollment(rng, target_source, target_idx, spk, hp)
    used_fallback_enroll = enroll_wav is None
    if used_fallback_enroll:
        # Fall back to a sub-clip of the target itself. Worse dvec quality
        # but the test still runs.
        cap = int(hp.ref_max_sec * hp.sample_rate)
        if target_wav_full.shape[-1] > cap:
            start = rng.randrange(0, target_wav_full.shape[-1] - cap + 1)
            enroll_wav = target_wav_full[..., start:start + cap]
        else:
            enroll_wav = target_wav_full.clone()
        notes.append("enrollment fallback: speaker has only one utterance")

    # 3) Interference
    interference_wav, kind, detail = _resolve_interference(
        rng, hp, sample_len, use_musan, musan, inter,
        libri_babble_n=libri_babble_n,
    )

    # 4) RIR + crop target
    target_wav = _crop_to(target_wav_full, sample_len, rng)
    target_wav, rir_applied = _apply_rir(rng, hp, target_wav, rirs)
    if rir_applied:
        notes.append("target convolved with random RIR")

    # 5) Mix at random SNR (always both active — see module docstring)
    snr = rng.uniform(hp.snr_min_db, hp.snr_max_db)
    mixed_wav = snr_mix(target_wav, interference_wav, snr)
    notes.append(f"SNR={snr:+.1f} dB; interference={detail}")

    # 6) Compute d-vector from the enrollment clip
    with torch.no_grad():
        enroll_dev = enroll_wav.to(_model_device(model))
        dvec = extract_dvec(enroll_dev, hp, encoder, device=enroll_dev.device).squeeze(0).cpu()
    notes.append(f"d-vector dim={dvec.shape[-1]} (extract_dvec → in_proj directly)")

    # 7) Streaming inference frame-by-frame
    sf_model = StreamingVoiceFilter(model, hp)
    sf_model.set_dvector(dvec)

    mixed_logmel = wav_to_log_mel(mixed_wav, hp, mel_fn)            # (T, 80)
    enhanced_frames: list[torch.Tensor] = []
    for t in range(mixed_logmel.shape[0]):
        enhanced_frames.append(sf_model.step(mixed_logmel[t]).cpu())
    enhanced_logmel = torch.stack(enhanced_frames, dim=0)            # (T, 80)

    # 8) Reconstruct waveform via Griffin-Lim (audition only)
    enhanced_wav = _logmel_to_wav(enhanced_logmel, hp, n_iter=gl_iter)

    # 9) Persist WAVs
    tag = f"{idx:02d}"
    target_path = out_dir / f"{tag}_target.wav"
    enroll_path = out_dir / f"{tag}_enrollment.wav"
    interference_path = out_dir / f"{tag}_interference.wav"
    mixed_path = out_dir / f"{tag}_mixed.wav"
    enhanced_path = out_dir / f"{tag}_enhanced.wav"

    _save_wav(str(target_path), target_wav, hp.sample_rate)
    _save_wav(str(enroll_path), enroll_wav, hp.sample_rate)
    _save_wav(str(interference_path), interference_wav, hp.sample_rate)
    _save_wav(str(mixed_path), mixed_wav, hp.sample_rate)
    _save_wav(str(enhanced_path), enhanced_wav, hp.sample_rate)

    return SampleSummary(
        index=idx,
        target_speaker=spk,
        target_path=str(target_path.name),
        enrollment_path=str(enroll_path.name),
        interference_kind=kind,
        interference_detail=detail,
        interference_path=str(interference_path.name),
        snr_db=float(snr),
        rir_applied=rir_applied,
        mixed_path=str(mixed_path.name),
        enhanced_path=str(enhanced_path.name),
        notes=notes,
    )


def _model_device(model: VoiceFilterLite) -> torch.device:
    return next(model.parameters()).device


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True, help="Path to ckpt_*.pt")
    p.add_argument("--out-dir", required=True,
                   help="Directory for per-sample WAVs + summary")
    p.add_argument("--num-samples", type=int, default=5,
                   help="Number of mixed clips to generate")
    p.add_argument("--sample-length-sec", type=float, default=None,
                   help="Override hp.sample_length_sec (default: use checkpoint's)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--gl-iter", type=int, default=16,
                   help="Griffin-Lim iterations for waveform reconstruction")
    # Force-pulls things one way or the other regardless of what the
    # checkpoint says; useful for fast iteration.
    p.add_argument("--no-musan", dest="musan", action="store_false", default=True,
                   help="Don't load Musan; interference = LibriSpeech other only")
    p.add_argument("--no-rir", dest="rir", action="store_false", default=True,
                   help="Don't load real or procedural RIRs")
    p.add_argument("--libri-babble", type=int, default=0,
                   help="Multi-talker mode: every sample uses N-speaker LibriSpeech "
                        "babble as interference (>=2; default 0=off, i.e. mixed pool). "
                        "Overrides --no-musan choice — no Musan / single-other interference.")
    # HF Hub endpoint override (same precedence chain as train.py).
    p.add_argument("--hf-endpoint", type=str, default=None,
                   help="Override HF Hub base URL (e.g. https://hf-mirror.com)")
    p.add_argument("--hf-transfer", dest="hf_transfer", action="store_true", default=None)
    p.add_argument("--no-hf-transfer", dest="hf_transfer", action="store_false", default=None)
    args = p.parse_args()

    # Resolve HF endpoint up front, before any datasets.load_dataset call.
    apply_hf_endpoint(cli_override=args.hf_endpoint, cli_transfer=args.hf_transfer)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _select_device(args.device)
    print(f"[example04] device={device}")
    if args.libri_babble >= 2:
        print(f"[example04] multi-talker mode: every sample uses "
              f"{args.libri_babble}-speaker LibriSpeech babble as interference")

    # ---- Load checkpoint + restore hparams ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hp = HParams.from_dict(ckpt["hparams"])
    hp.device = str(device)
    if args.sample_length_sec is not None:
        hp.sample_length_sec = float(args.sample_length_sec)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = VoiceFilterLite(hp).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    print(f"[example04] loaded {args.checkpoint} (step={ckpt.get('step')})")

    encoder = SpeakerEncoder(hp).to(device)
    if Path(hp.dvector_pt_path).is_file():
        encoder.load_pretrained(hp.dvector_pt_path)
        print(f"[example04] SpeakerEncoder loaded from {hp.dvector_pt_path}")
    else:
        print(
            f"[example04] WARNING: dvector weights not found at {hp.dvector_pt_path}; "
            f"encoder will use RANDOM init. The pretrained embedder "
            f"(`models/embedder.pt`) ships with this repo (from "
            f"https://github.com/maum-ai/voicefilter)."
        )
        encoder.freeze()
    print(f"[example04] d_vector_dim={hp.d_vector_dim}")
    mel_fn = make_mel_basis(hp, device=torch.device("cpu"))

    # Avoid FFmpeg-backed AudioDecoder crashes for malformed LibriSpeech FLAC
    # packets — same workaround the dataset uses.
    _disable_hf_audio_decode()

    # ---- Open sources ----
    t_open = time.time()
    print(f"[example04] opening LibriSpeech clean={hp.target_config} (splits={hp.target_splits})")
    target_source = AudioSource(hp, hp.target_config, ",".join(hp.target_splits), "target")
    print(f"[example04]   target rows: {len(target_source)} (loaded in {time.time() - t_open:.1f}s)")

    t_open = time.time()
    print(f"[example04] opening LibriSpeech other={hp.interference_config} (split={hp.interference_split})")
    inter = AudioSource(hp, hp.interference_config, hp.interference_split, "interference")
    print(f"[example04]   interference rows: {len(inter)} (loaded in {time.time() - t_open:.1f}s)")

    musan: Optional[MusanSource] = None
    use_musan = bool(args.musan) and bool(hp.use_musan) and args.libri_babble < 2
    if use_musan:
        try:
            requested = [s.strip() for s in hp.musan_subset.split(",") if s.strip()]
            # Always load `speech` if babble is in the list (babble is virtual).
            if hp.musan_babble and "babble" in requested and "speech" not in requested:
                requested.append("speech")
            real = [s for s in requested if s != "babble"]
            print(f"[example04] opening Musan subsets: {real}")
            t_open = time.time()
            musan = MusanSource(hp, real,
                                hp.musan_babble_n_speakers,
                                (hp.musan_babble_gain_min, hp.musan_babble_gain_max))
            print(f"[example04]   musan loaded in {time.time() - t_open:.1f}s "
                  f"(rows/subset: {', '.join(f'{s}={len(v)}' for s, v in musan._iters.items())})")
        except Exception as e:
            print(f"[example04] WARNING: failed to open Musan ({type(e).__name__}: {e}); "
                  f"falling back to LibriSpeech-only interference")
            use_musan = False
            musan = None
    if not use_musan:
        print("[example04] interference pool: LibriSpeech 'other' only")

    rirs: list[torch.Tensor] = []
    if args.rir:
        try:
            rirs = _load_rir_pool(hp, rng)
        except Exception as e:
            print(f"[example04] WARNING: failed to load RIRs ({type(e).__name__}: {e}); "
                  f"target will be mixed without reverberation")
            rirs = []

    # ---- Build N samples ----
    summaries: list[SampleSummary] = []
    t0 = time.time()
    for i in range(args.num_samples):
        ts = time.time()
        try:
            summary = build_one_sample(
                idx=i, hp=hp, rng=rng,
                target_source=target_source, inter=inter, musan=musan,
                rirs=rirs, use_musan=use_musan, out_dir=out_dir,
                model=model, encoder=encoder, mel_fn=mel_fn, gl_iter=args.gl_iter,
                libri_babble_n=args.libri_babble,
            )
        except Exception as e:
            print(f"[example04] sample {i:02d} FAILED: {type(e).__name__}: {e}")
            continue
        summaries.append(summary)
        print(f"[example04] sample {i:02d} done in {time.time() - ts:.1f}s "
              f"(target_spk={summary.target_speaker}, "
              f"interference={summary.interference_kind}, "
              f"SNR={summary.snr_db:+.1f} dB, "
              f"rir={summary.rir_applied})")

    # ---- Write summaries ----
    json_path = out_dir / "summary.json"
    txt_path = out_dir / "summary.txt"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in summaries], f, indent=2, ensure_ascii=False)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"# Integration test summary\n")
        f.write(f"# generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# checkpoint: {args.checkpoint}\n")
        f.write(f"# num_samples: {len(summaries)} (requested {args.num_samples})\n")
        f.write(f"# sample_length_sec: {hp.sample_length_sec}\n")
        f.write(f"# snr_range_db: [{hp.snr_min_db}, {hp.snr_max_db}]\n")
        f.write(f"# use_musan: {use_musan}; real_rir: {bool(rirs)}\n")
        f.write(f"# libri_babble_n: {args.libri_babble} (0 = mixed pool, >1 = multi-talker forced)\n")
        f.write(f"# total_runtime_sec: {time.time() - t0:.1f}\n\n")
        for s in summaries:
            f.write(f"--- sample {s.index:02d} ---\n")
            f.write(f"  target_speaker   : {s.target_speaker}\n")
            f.write(f"  interference      : {s.interference_kind}  ({s.interference_detail})\n")
            f.write(f"  snr_db            : {s.snr_db:+.1f}\n")
            f.write(f"  rir_applied       : {s.rir_applied}\n")
            f.write(f"  target wav        : {s.target_path}\n")
            f.write(f"  enrollment wav    : {s.enrollment_path}\n")
            f.write(f"  interference wav  : {s.interference_path}\n")
            f.write(f"  mixed wav         : {s.mixed_path}\n")
            f.write(f"  enhanced wav      : {s.enhanced_path}\n")
            for note in s.notes:
                f.write(f"  note              : {note}\n")
            f.write("\n")

    print(f"\n[example04] wrote {len(summaries)} samples to {out_dir}")
    print(f"[example04] summary: {txt_path}")
    print(f"[example04] summary: {json_path}")
    print(f"[example04] total runtime: {time.time() - t0:.1f}s")
    print("[example04] Listen to *_mixed.wav vs *_enhanced.wav to compare "
          "input vs model output.")


if __name__ == "__main__":
    main()