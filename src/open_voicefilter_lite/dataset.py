"""LibriSpeech dataset for VoiceFilter-Lite training, with optional
Musan noise and real RIR reverberation augmentation (paper §3.4).

Streams from `openslr/librispeech_asr` on HuggingFace:
    - Target speakers:  `clean` config (train.100 + train.360 = 500 h, ~1,172 spk)
    - Interference:     `other` config (train.500, ~1,166 spk)

Optional interference augmentation (gated by hp.use_musan / hp.use_real_rir):
    - Musan noise from `corypaik/musan` (music / noise / speech subsets;
      `babble` is synthesised on the fly by overlaying N `speech` clips with
      random per-clip gains — see Kaldi/ESPnet spec-aug recipe)
    - Real room RIRs (convolution reverberation of target)

Each sample is a triplet:
    (stacked_mixed_logmel, dvec, target_logmel, noise_kind)
"""

from __future__ import annotations

import io
import os
import random
import time
from collections import deque
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import chain
from typing import Iterator, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import IterableDataset

from open_voicefilter_lite.audio import (
    frame_stack_left,
    random_crop,
    snr_mix,
    wav_to_log_mel,
)
from open_voicefilter_lite.hparams import HParams


# Note: the d-vector is computed on the main training thread by
# `audio.extract_dvec(batch["ref_wavs"], hp, speaker_encoder)` in
# `train.py`. The dataset yields the raw reference waveform; the encoder
# lives in `train.py` and is held by the main process only. Earlier
# versions ran `SpeakerEncoder` inside this iterator's background thread
# — the cross-thread MPS use crashed the process with `BUG IN CLIENT OF
# LIBMALLOC` (DiagnosticReports/python3.13-*.ips). The encoder has since
# moved out of the dataset entirely.


NOISE_KIND_BABBLE = "babble"
NOISE_KIND_MUSIC = "music"
NOISE_KIND_NOISE = "noise"
NOISE_KIND_LIBRISPEECH = "librispeech"
NOISE_KIND_RIR = "rir"   # tag for the *target* (informational)


def _sample_activity_burst(
    sample_len: int,
    hp: HParams,
    rng: random.Random,
) -> torch.Tensor:
    """Build a sample-level 0/1 activity mask with one active burst (deviation 5).

    The burst is a single contiguous window of
    `[activity_burst_min_sec, activity_burst_max_sec]` seconds, drawn
    uniformly within `[0, sample_len]`. Edges fade linearly over
    `activity_fade_ms` (avoids click artefacts at burst boundaries).
    The mask is returned at sample rate (length `sample_len`).

    Paper §4.2 doesn't prescribe burst sizes; the activity-mask framing
    is the deviation-5 alignment — see plan section "偏差 5".
    """
    burst_min_samples = max(1, int(hp.activity_burst_min_sec * hp.sample_rate))
    burst_max_samples = max(
        burst_min_samples,
        int(hp.activity_burst_max_sec * hp.sample_rate),
    )
    # `rng.randint(a, b)` is inclusive on both ends, so the upper bound is
    # already `burst_max_samples` — adding `+ 1` here would let `burst_len`
    # reach `sample_len + 1`, after which `burst_start` (in
    # `[0, sample_len - burst_len + 1]`) can push `burst_end` past
    # `sample_len` and the trailing fade slice clamps to a length shorter
    # than the ramp (the 799-vs-800 mismatch the training log crashed on).
    burst_max_samples = min(burst_max_samples, sample_len)
    burst_len = rng.randint(burst_min_samples, burst_max_samples)
    # Same inclusive-randint fix as above: `burst_start + burst_len` must
    # not exceed `sample_len`, so the upper bound here is `sample_len -
    # burst_len` (already inclusive on both ends).
    burst_start = rng.randint(0, sample_len - burst_len)
    burst_end = burst_start + burst_len

    fade = max(1, int(hp.activity_fade_ms * 1e-3 * hp.sample_rate))
    fade = min(fade, burst_len // 2)

    act = torch.zeros(sample_len, dtype=torch.float32)
    # Pre-fade region stays at 0 (no ramp from 0 to 1 at burst start).
    if fade > 1 and burst_len > 2 * fade:
        ramp = torch.linspace(0.0, 1.0, steps=fade + 1, dtype=torch.float32)[1:]
        act[burst_start : burst_start + fade] = ramp
        act[burst_end - fade : burst_end] = ramp.flip(0)
        act[burst_start + fade : burst_end - fade] = 1.0
    else:
        # Burst too short for a fade — just hard on/off (no click risk
        # when fade ≥ burst_len / 2 would already collapse to 0).
        act[burst_start:burst_end] = 1.0
    return act


def _per_frame_overlap_label(
    target_act: torch.Tensor,
    interference_act: torch.Tensor,
    T_frames: int,
    sample_len: int,
    hp: HParams,
) -> torch.Tensor:
    """Compute per-frame noise-type labels (paper §4.2 / deviation 5).

    For each frame `t`, look at the sample index at the frame's centre:
        center = t * hp.hop_length + hp.n_fft // 2
    The label is `+1` iff both target and interference activity masks
    are > 0 at that centre (i.e. the two bursts overlap at the frame's
    temporal centre), else `-1`. Frames whose centre is past
    `sample_len` clamp to the last sample so we always return a
    `(T_frames,)` tensor.
    """
    labels = torch.full((T_frames,), -1, dtype=torch.long)
    # Vectorised: build the centre index for every frame, then sample.
    centre_idx = torch.arange(T_frames, dtype=torch.long) * hp.hop_length + hp.n_fft // 2
    centre_idx = centre_idx.clamp(max=sample_len - 1)
    tgt_at = target_act[centre_idx] > 0.0
    int_at = interference_act[centre_idx] > 0.0
    both = tgt_at & int_at
    labels[both] = 1
    return labels


@dataclass
class Sample:
    stacked_mixed: torch.Tensor   # (T, n_mels * frame_stack)  — model input
    target_logmel: torch.Tensor   # (T, n_mels)                — loss target
    # Reference waveform + speaker_id for d-vector extraction. SpeakerEncoder
    # runs on these in the main training loop (via `DVecCache.get_dvecs`)
    # — the dataset itself stays MPS-free so the PrefetchIterator's
    # background thread can stream samples without touching MPS.
    # Reference waveform + speaker_id for d-vector extraction. The encoder
    # lives in `train.py` and runs on the main training thread via
    # `audio.extract_dvec(...)` — the dataset itself stays MPS-free so
    # the PrefetchIterator's background thread can stream samples
    # without touching MPS.
    ref_wav: torch.Tensor         # 1-D float32 CPU; variable length
    ref_spk_id: str               # LibriSpeech speaker_id (kept for diagnostics)
    noise_kind: str = NOISE_KIND_LIBRISPEECH  # diagnostic tag
    # Per-frame label for the noise-type predictor (paper §4.2):
    # +1 = "this frame is overlapped speech", -1 = "pass-through / clean".
    # In the triplet mixer, target and interference are either fully silent
    # or fully active for the whole sample, so every frame in a sample
    # shares the same label — we broadcast a single int to (T,).
    overlap_label: Optional[torch.Tensor] = None  # (T,) of {-1, +1}
    # Cached d-vector (256-d) for the ref wav, if this ref came from the
    # same-speaker buffer on a *revisit* — `None` means the main training
    # thread still needs to compute it via `audio.extract_dvec`. Paired
    # with `ref_entry_id` so the main thread can write the freshly
    # computed dvec back into the buffer entry (see
    # `LibriSpeechTripletStream.set_dvec_for_entry`). Cache hit rate after
    # per-speaker warmup approaches `musan_prob`-free buffer-revisit
    # probability (currently ~70%).
    ref_dvec: Optional[torch.Tensor] = None       # (256,) or None
    ref_entry_id: Optional[int] = None            # id(entry dict) for write-back


def _decode_audio(audio_field, target_sr: int) -> torch.Tensor:
    """Decode a HuggingFace audio column to a 16 kHz mono torch.Tensor.

    The dataset is loaded with `Audio(decode=False)` (see `_ensure_open`),
    so the audio column arrives as a dict `{"bytes", "path": ...}`. We
    feed the raw FLAC bytes to `soundfile` (libsndfile), which is more
    lenient than FFmpeg with corrupt FLAC frames — it skips them rather
    than SIGSEGV-ing inside `av_packet_make_refcounted`.

    Falls back to `AudioDecoder.get_all_samples()` (torchcodec → FFmpeg)
    if the field somehow lacks `bytes`/`path` — but in normal operation
    this branch is never reached because `_ensure_open` casts the audio
    feature to `decode=False`. The AudioDecoder path is kept only as a
    safety net so older `datasets` versions that don't honour
    `decode=False` still produce audio instead of crashing.
    """
    # ---- Normal path: dict with bytes/path (Audio(decode=False)) ----
    if isinstance(audio_field, dict):
        if "bytes" in audio_field and audio_field["bytes"] is not None:
            data, sr = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32")
        elif "array" in audio_field and audio_field["array"] is not None:
            data = np.asarray(audio_field["array"], dtype=np.float32)
            sr = int(audio_field.get("sampling_rate", target_sr))
        elif "path" in audio_field and audio_field["path"]:
            data, sr = sf.read(audio_field["path"], dtype="float32")
        else:
            raise ValueError(f"Unrecognised audio field: {list(audio_field.keys())}")
        if data.ndim > 1:
            data = data.mean(axis=1)
        wav = torch.from_numpy(data)
        if sr != target_sr:
            import torchaudio
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
        peak = wav.abs().max().clamp(min=1e-8)
        if peak > 5.0:
            wav = wav / peak
        return wav

    # ---- Fallback: AudioDecoder (Audio(decode=True)). Can SIGSEGV on
    # corrupt packets — try soundfile on .bytes first if exposed. ----
    if hasattr(audio_field, "get_all_samples") and callable(
        getattr(audio_field, "get_all_samples", None)
    ):
        recovered = _soundfile_fallback(audio_field, target_sr)
        if recovered is not None:
            return recovered
        try:
            samples = audio_field.get_all_samples()
            wav = samples.data            # torch.Tensor (C, T) float32 CPU
            sr = int(samples.sample_rate)
            # Downmix (C, T) → (T,). Mono files already arrive as (1, T).
            if wav.dim() == 2:
                wav = wav.mean(dim=0)
            elif wav.dim() == 1:
                pass
            else:
                wav = wav.squeeze(0)
            if sr != target_sr:
                import torchaudio
                wav = torchaudio.functional.resample(
                    wav.unsqueeze(0), sr, target_sr
                ).squeeze(0)
            peak = wav.abs().max().clamp(min=1e-8)
            if peak > 5.0:
                wav = wav / peak
            return wav
        except Exception as audio_decoder_err:
            raise audio_decoder_err

    raise ValueError(
        f"[dataset] cannot decode audio field of type "
        f"{type(audio_field).__name__}; no soundfile bytes/path and no "
        f"AudioDecoder.get_all_samples"
    )


def _soundfile_fallback(audio_field, target_sr: int) -> Optional[torch.Tensor]:
    """Best-effort soundfile decode from an AudioDecoder's raw bytes/path.

    Called only when the AudioDecoder (FFmpeg) path raises on a corrupt
    FLAC packet. AudioDecoder exposes either `.bytes` (raw FLAC bytes)
    or `.path` (on-disk file) so we can re-decode with libsndfile, which
    is more lenient with corrupt frames than FFmpeg — it skips bad
    frames instead of aborting the whole decode.

    Returns `None` if neither bytes nor path is accessible (older
    datasets versions), or if soundfile also fails — caller re-raises
    the original error and the dataset's skip path takes over.
    """
    try:
        if hasattr(audio_field, "bytes") and audio_field.bytes is not None:
            data, sr = sf.read(io.BytesIO(audio_field.bytes), dtype="float32")
        elif hasattr(audio_field, "path") and audio_field.path:
            data, sr = sf.read(audio_field.path, dtype="float32")
        else:
            return None

        if data.ndim > 1:
            data = data.mean(axis=1)
        wav = torch.from_numpy(data)
        if sr != target_sr:
            import torchaudio
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
        peak = wav.abs().max().clamp(min=1e-8)
        if peak > 5.0:
            wav = wav / peak
        return wav
    except Exception:
        return None


# ---- Procedural RIR fallback (used if real RIR dataset is unavailable) ----


def _synthetic_rir(sample_rate: int, length_s: float = 0.5, rng: random.Random | None = None) -> torch.Tensor:
    """Exponentially-decaying noise RIR — a quick procedural approximation.

    Real RIRs are tried first via the `rir_source` / `rir_source_fallbacks`
    chain in HParams (see `_ensure_rir_open`). This procedural fallback is
    used only when every HF mirror fails; it still teaches the model to
    handle reverberation.
    """
    rng = rng or random
    T = int(sample_rate * length_s)
    t = torch.linspace(0, length_s, T)
    # RT60 ~ 0.3 s (typical small room).
    decay = torch.exp(-6.91 * t / 0.3)
    noise = torch.randn(T) * decay
    # Pre-delay (0-20 ms of zeros).
    pre = int(rng.uniform(0, 0.02) * sample_rate)
    h = torch.cat([torch.zeros(pre), noise[pre:]])
    # Normalise to unit energy.
    h = h / (h.pow(2).sum().sqrt() + 1e-8)
    return h


def _convolve(wav: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Single-channel convolution via FFT, with output trimmed to input length."""
    n = wav.shape[-1]
    # FFT length must be >= n + h - 1.
    n_fft = 1
    while n_fft < n + h.shape[-1]:
        n_fft *= 2
    W = torch.fft.rfft(wav, n=n_fft)
    H = torch.fft.rfft(h, n=n_fft)
    y = torch.fft.irfft(W * H, n=n_fft)
    return y[..., :n]


# ---- HF datasets audio-decode workaround ----
#
# HF datasets 5.x eagerly constructs a torchcodec `AudioDecoder`
# (FFmpeg-backed) on every audio row during iteration. When FFmpeg
# isn't installed, that `dlopen` raises OSError that propagates all
# the way out and kills the process. When FFmpeg IS installed, certain
# malformed FLAC packets trigger `SIGSEGV` inside
# `av_packet_make_refcounted` — a native crash Python can't catch.
#
# Workaround (apply once per process):
#   1) `_disable_hf_audio_decode()` monkey-patches
#      `Audio.decode_example` so it returns the raw
#      `{"bytes", "path": ...}` dict without instantiating AudioDecoder.
#   2) `_cast_audio_no_decode(ds)` rewrites the dataset's audio feature
#      schema to `Audio(decode=False)` so HF datasets keeps raw FLAC
#      bytes in the column instead of trying to instantiate an
#      `AudioDecoder` per row.
# Callers then feed the bytes through `_decode_audio` (libsndfile via
# `soundfile`), which is more lenient than FFmpeg with corrupt FLAC
# frames.
_HF_AUDIO_DECODE_PATCHED: bool = False


def _disable_hf_audio_decode() -> None:
    """Idempotently monkey-patch HF `Audio.decode_example` to never decode."""
    global _HF_AUDIO_DECODE_PATCHED
    if _HF_AUDIO_DECODE_PATCHED:
        return
    from datasets.features import audio as _hf_audio

    _orig_decode_example = _hf_audio.Audio.decode_example

    def _passthrough_decode_example(self, value, token_per_repo_id=None):
        # Always return the raw dict — never instantiate AudioDecoder.
        # HF datasets' format_batch calls this for every row, so without
        # this patch we don't even get a chance to decode with libsndfile
        # first.
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        # Last-resort fallback to upstream behavior for non-dict inputs
        # (e.g. AudioDecoder objects from older datasets).
        return _orig_decode_example(self, value, token_per_repo_id)

    _hf_audio.Audio.decode_example = _passthrough_decode_example
    _HF_AUDIO_DECODE_PATCHED = True


def _cast_audio_no_decode(ds):
    """Cast every `Audio` feature in `ds` to `decode=False` (raw FLAC bytes)."""
    from datasets import Audio, Features

    new_features = Features(
        {k: (Audio(decode=False) if k == "audio" else v)
         for k, v in ds.features.items()}
    )
    try:
        ds = ds.cast_features(new_features)
    except Exception:
        # Fallback for old datasets without `cast_features`.
        ds.features.update(new_features)
    # Belt-and-suspenders: flip `decode=False` in place too, in case
    # `cast_features` on IterableDataset doesn't propagate the change
    # into the `python_features_decoder` that `format_batch` consults.
    for k, v in ds.features.items():
        if isinstance(v, Audio):
            v.decode = False
    return ds


# ---- Dataset ----


class LibriSpeechTripletStream(IterableDataset):
    """Streams (target, interference, reference) triplets from LibriSpeech,
    with optional Musan noise (interference) and real-RIR convolution (target)."""

    def __init__(
        self,
        hp: HParams,
        worker_id: int = 0,
        num_workers: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.hp = hp
        self.worker_id = worker_id
        self.num_workers = num_workers
        self._seed = seed if seed is not None else hp.seed
        self._rng = random.Random(self._seed + worker_id)

        self._target_ds = None
        self._interference_ds = None
        self._target_iter = None
        self._interference_iter = None
        self._mel = None

        # Musan streams (one iterator per real subset: music / noise / speech).
        # `babble` is virtual — synthesised from `speech` on the fly when sampled.
        self._musan_iters: dict[str, Iterator[dict]] = {}
        self._musan_synthetic: set[str] = set()
        self._musan_open = False
        self._musan_failed: Optional[str] = None  # if loading failed once

        # Real-RIR cache (lazy-loaded once; small, fits in memory).
        self._rirs: list[torch.Tensor] = []
        self._rir_open = False
        self._rir_failed: Optional[str] = None

        # Speaker → list of cached ref audios for picking same-speaker ref.
        # Bounded by `hp.ref_buffer_max_per_spk` (per-speaker) and
        # `hp.ref_buffer_max_speakers` (total distinct speakers, LRU) —
        # see `hparams.py` for the leak that motivated both caps.
        # OrderedDict so we can LRU-evict speakers we haven't touched
        # recently (a hot talker stays cached; cold ones get dropped first).
        # Split by worker_id in __iter__ so workers don't race on appends.
        self._target_buffer: "OrderedDict[str, list[dict]]" = OrderedDict()

        # Thread pool for parallel audio decoding (set up lazily in __iter__
        # so the dataset is still picklable if anyone forks it later).
        self._decode_pool: Optional[ThreadPoolExecutor] = None

    # ---- Lazy dataset open ----

    def _ensure_open(self) -> None:
        if self._target_ds is not None:
            return

        from datasets import load_dataset

        # `cache_dir` is ignored when `streaming=True` (HF datasets streams
        # directly from the Hub cache); when streaming=False, point HF at
        # the on-disk arrow cache populated by `scripts/download_data.py`.
        load_kwargs: dict = {"streaming": self.hp.streaming}
        if not self.hp.streaming:
            load_kwargs["cache_dir"] = self.hp.librispeech_cache

        # Disable HF datasets' eager torchcodec/FFmpeg audio decode once
        # per process — see the module-level helper docstring for the why.
        _disable_hf_audio_decode()

        def _load_no_decode(config: str, split: str):
            ds = load_dataset(
                "openslr/librispeech_asr",
                config,
                split=split,
                **load_kwargs,
            )
            return _cast_audio_no_decode(ds)

        def _load_vctk_no_decode(split: str):
            # Paper §3.2: target speakers come from VCTK. CSTR-Edinburgh/vctk
            # exposes one `train` split (and `test`); each row has the usual
            # `audio` dict plus a `speaker_id` string (e.g. "p225"). We keep
            # audio in `Audio(decode=False)` form (bytes in memory) and let
            # `_decode_audio` decode via soundfile like the LibriSpeech path,
            # so the dvec pipeline is identical for both sources.
            ds = load_dataset(
                self.hp.vctk_repo,
                split=split,
                **load_kwargs,
            )
            return _cast_audio_no_decode(ds)

        target_splits = list(self.hp.target_splits)
        if self.hp.target_source == "vctk":
            # Paper-faithful: target = VCTK. CSTR-Edinburgh/vctk exposes a
            # single `train` split; we map our `target_splits` tuple
            # (kept as a config knob for future VCTK subsets) to that.
            vctk_splits = target_splits or ["train"]
            self._target_ds_list = [
                _load_vctk_no_decode(s) for s in vctk_splits
            ]
        elif self.hp.target_source == "librispeech":
            # Legacy: target = LibriSpeech `clean` split.
            self._target_ds_list = [
                _load_no_decode(self.hp.target_config, s)
                for s in target_splits
            ]
        else:
            raise ValueError(
                f"[dataset] unknown hp.target_source={self.hp.target_source!r}; "
                "expected 'librispeech' or 'vctk'."
            )
        # Store the list rather than a one-shot `chain.from_iterable(...)`.
        # The chain is stateful (its `__iter__` returns self and `__next__`
        # tracks a cursor) — once exhausted, re-calling `iter(chain)` gives
        # the same exhausted object, and `_infinite_iter`'s `while True: for`
        # busy-loops with no yields. We rebuild the chain inside `_iter_target`
        # so the dataset can cycle back to the start.
        self._target_ds = chain.from_iterable(self._target_ds_list)

        self._interference_ds = _load_no_decode(
            self.hp.interference_config, self.hp.interference_split,
        )

        from open_voicefilter_lite.audio import make_mel_basis
        self._mel = make_mel_basis(self.hp, device=torch.device("cpu"))

    def _ensure_musan_open(self) -> None:
        if self._musan_open or self._musan_failed is not None:
            return
        from datasets import load_dataset

        os.makedirs(self.hp.musan_cache, exist_ok=True)

        requested = [s.strip() for s in self.hp.musan_subset.split(",") if s.strip()]

        # `babble` is a virtual subset — synthesised from `speech`. We need a
        # `speech` iterator whenever babble is requested, even if the user
        # didn't list `speech` directly.
        if self.hp.musan_babble and "babble" in requested:
            self._musan_synthetic.add("babble")
            if "speech" not in requested:
                requested.append("speech")

        # Only try to load real (HF-hosted) subsets from `musan_source`.
        # `babble` has no HF config; the synthesis branch handles it later.
        real_subsets = [s for s in requested if s not in self._musan_synthetic]
        for subset in real_subsets:
            try:
                ds = load_dataset(
                    self.hp.musan_source,
                    subset,
                    split="train",
                    streaming=self.hp.streaming,
                )
                self._musan_iters[subset] = self._infinite_iter(ds)
                print(f"[dataset] loaded musan subset '{subset}' from {self.hp.musan_source}")
            except Exception as e:
                self._musan_failed = f"musan:{subset}: {e}"
                print(f"[dataset] WARNING musan subset '{subset}' failed: {e}")
                break
        if "babble" in self._musan_synthetic and "speech" in self._musan_iters:
            print(
                f"[dataset] musan 'babble' will be synthesised "
                f"(n={self.hp.musan_babble_n_speakers} speech clips overlaid, "
                f"gain∈[{self.hp.musan_babble_gain_min}, {self.hp.musan_babble_gain_max}])"
            )
        self._musan_open = True

    def _ensure_rir_open(self) -> None:
        if self._rir_open or self._rir_failed is not None:
            return

        # Iterate (source × config). `rir_source` first, then its fallbacks.
        # Each source is tried with `simulated_rirs` then default config. If
        # every (source, config) pair raises, fall back to procedural RIRs.
        from datasets import load_dataset

        sources: list[str] = [self.hp.rir_source, *self.hp.rir_source_fallbacks]
        config_candidates: list[tuple[dict, str]] = [
            ({"name": "simulated_rirs"}, "config=simulated_rirs"),
            ({}, "default config"),
        ]

        last_err: Exception | None = None
        for source in sources:
            for load_kwargs, label in config_candidates:
                try:
                    ds = load_dataset(
                        source,
                        split="train",
                        streaming=False,
                        cache_dir=self.hp.rir_cache,
                        **load_kwargs,
                    )
                    for ex in ds:
                        h = _decode_audio(ex["audio"], self.hp.sample_rate)
                        if h.shape[-1] >= int(0.05 * self.hp.sample_rate):
                            self._rirs.append(h)
                        if len(self._rirs) >= 512:
                            break
                    if self._rirs:
                        print(
                            f"[dataset] loaded {len(self._rirs)} real RIRs "
                            f"from {source} ({label})"
                        )
                        self._rir_open = True
                        return
                except Exception as e:
                    last_err = e
                    continue
            print(f"[dataset] WARNING rir source '{source}' returned nothing usable")

        # All real-RIR attempts failed.
        self._rir_failed = f"rir-source:{last_err}"
        print(f"[dataset] WARNING real RIR dataset failed ({last_err}); falling back to procedural RIR")

        # Procedural fallback (always works).
        self._rirs = [
            _synthetic_rir(self.hp.sample_rate, length_s=0.5, rng=self._rng)
            for _ in range(256)
        ]
        print(f"[dataset] generated {len(self._rirs)} synthetic RIRs")
        self._rir_open = True

    @staticmethod
    def _infinite_iter(ds) -> Iterator[dict]:
        # Yield from `ds` forever. **Must not** be a `while True: for ex in ds`
        # loop over a one-shot iterator: when `ds` is a `chain.from_iterable`
        # object, `for ex in ds` returns the chain itself on `iter()`, and
        # once the chain is exhausted `for ex in ds` becomes an empty `for`
        # loop that runs forever with no yields — `next()` on the caller side
        # would then hang forever (LibriSpeech train.360 is ~104 000
        # utterances, so the hang lands exactly there). The fix is to
        # rebuild the chain on every pass with `chain.from_iterable(...)`,
        # which iter()'s the underlying HF streams afresh; streaming-mode
        # HF `IterableDataset`s support this (`__iter__` returns a fresh
        # `ArrowExamplesIterable` each call).
        while True:
            yield from chain.from_iterable(ds)

    def _iter_target(self) -> Iterator[dict]:
        assert self._target_ds_list is not None
        return self._infinite_iter(self._target_ds_list)

    def _iter_interference(self) -> Iterator[dict]:
        assert self._interference_ds is not None
        return self._infinite_iter([self._interference_ds])

    # ---- Sampling helpers ----

    def _sample_musan(self, length: int) -> Optional[tuple[torch.Tensor, str]]:
        if not self._musan_iters:
            return None
        # Build the option pool. Babble is virtual but is a legal pick whenever
        # its backing `speech` iterator is available.
        pool = list(self._musan_iters.keys())
        if "babble" in self._musan_synthetic and "speech" in self._musan_iters:
            pool.append("babble")
        subset = self._rng.choice(pool)
        if subset == "babble":
            wav = self._synthesise_babble(length)
            if wav is None:
                return None
            return wav, "babble"
        try:
            ex = next(self._musan_iters[subset])
        except Exception:
            return None
        try:
            wav = _decode_audio(ex["audio"], self.hp.sample_rate)
        except Exception:
            return None
        return random_crop(wav, length), subset

    def _synthesise_babble(self, length: int) -> Optional[torch.Tensor]:
        """Overlay N MUSAN speech clips with random per-clip gains, then crop.

        Mirrors the spec-augment `babble` recipe used in Kaldi
        (Snyder, Chen, Povey 2015, Appendix A): pick N speech files at random,
        multiply each by a uniform gain in [gain_min, gain_max), sum them.
        """
        if "speech" not in self._musan_iters:
            return None
        n = max(2, int(self.hp.musan_babble_n_speakers))
        chunks: list[torch.Tensor] = []
        gains: list[float] = []
        total = 0
        # Pull enough clips that their concatenation covers `length`. Each clip
        # from MUSAN speech is 10–30 s, so 3–5 draws comfortably exceed the
        # 3 s training segment; we still safeguard against iterator faults.
        for _ in range(n * 4):
            if total >= length and chunks:
                break
            try:
                ex = next(self._musan_iters["speech"])
                wav = _decode_audio(ex["audio"], self.hp.sample_rate)
            except Exception:
                continue
            if wav.numel() == 0:
                continue
            chunks.append(wav)
            gains.append(self._rng.uniform(
                self.hp.musan_babble_gain_min, self.hp.musan_babble_gain_max
            ))
            total += wav.shape[-1]
        if not chunks:
            return None
        full = torch.cat(chunks).to(torch.float32)
        g = torch.cat([torch.full((c.shape[-1],), gi) for c, gi in zip(chunks, gains)])
        mixed = full * g
        return random_crop(mixed, length)

    def _sample_rir(self) -> Optional[torch.Tensor]:
        if not self._rirs:
            return None
        return self._rng.choice(self._rirs)

    # ---- Dvec cache write-back (called from main training thread) ----

    def set_dvec_for_entry(self, entry_id: int, dvec: torch.Tensor) -> bool:
        """Cache a freshly-computed d-vector back into a buffer entry.

        `train.py` calls this after `audio.extract_dvec` returns, with the
        `entry_id` it read from `batch["ref_entry_ids"]`. On a buffer-hit
        cache, the next iteration that picks the same entry as a ref will
        find `entry["dvec"]` populated and skip the recompute.

        Returns `True` if the entry was found and updated, `False` if the
        entry has been evicted (which is a benign race: a buffer overflow
        or LRU cap drop is the only way this can happen, and the next
        batch will simply compute the dvec again).

        Iterates every speaker's list — O(num_speakers × max_buf), which is
        ~2k iterations on the default 512-speaker cap. Cheap enough that
        a side index isn't worth its complexity.
        """
        for spk, buf in self._target_buffer.items():
            for entry in buf:
                if id(entry) == entry_id:
                    entry["dvec"] = dvec.detach().cpu()
                    return True
        return False

    # ---- Main iterator ----

    def __iter__(self) -> Iterator[Sample]:
        self._ensure_open()
        if self.hp.use_musan:
            self._ensure_musan_open()
        if self.hp.use_real_rir:
            self._ensure_rir_open()

        # When this dataset is consumed by a multi-worker DataLoader, each
        # worker process runs its own copy of `__iter__` concurrently.
        # Without per-worker RNG seeding, every worker would draw the same
        # sequence from `self._rng` and emit identical samples — wasting
        # half the parallelism on duplicate work. Re-seed by worker id so
        # the N workers produce N disjoint sample streams.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self._rng = random.Random(self._seed + worker_info.id)
            # Also split the per-speaker utterance buffer so two workers
            # don't race on `_target_buffer[spk].append(...)` and stomp
            # each other's ref-pick history. Must stay an OrderedDict so
            # the global LRU cap (`ref_buffer_max_speakers`) still applies.
            self._target_buffer = OrderedDict()

        # Heartbeat: the first batch through audio decode + mel is
        # multi-second on Apple Silicon. Print progress so the user can see
        # the dataset is alive. SpeakerEncoder is held by the main process
        # and runs on the main thread (see `train.py`'s
        # `extract_dvec(...)` call) so the dataset stays MPS-free.
        _t0 = time.time()
        _last_beat = _t0
        _samples = 0
        _HEARTBEAT_S = 5.0

        rng = self._rng
        hp = self.hp
        sample_len = hp.sample_length_samples

        target_iter = self._iter_target()
        interference_iter = self._iter_interference()

        # ---- Same-speaker ref cache caps ----
        # `max_buf` caps each speaker's local cache (entries ≤ ref_max_len).
        # `max_speakers` caps the total number of distinct speaker keys in
        # the dict — without this, every speaker LibriSpeech has ever fed
        # us stays in the dict holding up to `max_buf` decoded audios, and
        # on a 1172-speaker corpus × 64 × 384 KB the buffer can top 28 GB.
        # We LRU-evict from the dict so hot speakers stay warm.
        max_buf = hp.ref_buffer_max_per_spk
        max_speakers = hp.ref_buffer_max_speakers

        # ---- Parallel decode via thread pool ----
        # `soundfile.read` (libsndfile under C) and `torchcodec` (FFmpeg
        # under C) both release the GIL during the heavy decode work, so
        # running them in a ThreadPoolExecutor gives real parallel speedup
        # on the per-sample audio I/O — the dominant cost on MPS. Each
        # training sample needs 3 audio decodes (target, interference,
        # reference), so with `decode_workers=4` the pool keeps 4 decodes
        # in flight while the main thread continues preparing mel/mix.
        # SpeakerEncoder used to also run here (the source of the MPS race
        # that crashed the process); it now runs on the main thread
        # instead.
        decode_workers = max(1, hp.decode_workers)
        decode_pool: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(
                max_workers=decode_workers,
                thread_name_prefix="audio-decode",
            )
            if decode_workers > 1
            else None
        )

        # Pipeline of decode jobs (each entry: dict with tgt_ex/int_ex/ref_ex/
        # ref_spk + futures). We keep `decode_prefetch` jobs in flight ahead
        # of consumption so the main thread's `fut.result()` rarely blocks.
        pipeline: deque = deque()

        # Cap ref_wav to this many samples (default ~6 s @ 16 kHz).
        # SpeakerEncoder only needs ~3-6 s for a stable embedding; capping
        # here keeps the padded ref_wavs tensor small (≈2 MB vs ≈30 MB at
        # batch=16), which is the single biggest memory win on
        # memory-constrained MPS machines. The dvec forward in
        # `audio.extract_dvec` enforces the same cap, so refs longer
        # than `ref_max_sec` get cleanly truncated to the cap before
        # the encoder forward.
        ref_max_len = int(hp.ref_max_sec * hp.sample_rate)

        def _submit_next_decode() -> Optional[dict]:
            """Pull next (target, interference), pick ref, submit decode jobs.

            Returns a dict with the example refs + futures, or None on
            StopIteration. Decoding of `ref_ex` is skipped if it's the same
            Python object as `tgt_ex` (reuse target's already-decoded wav)
            **or** if `ref_ex` was picked from the same-speaker buffer
            (already-decoded, ref-capped tensor — see notes on
            `_target_buffer` in the class docstring).
            """
            try:
                tgt_ex = next(target_iter)
                int_ex = next(interference_iter)
            except StopIteration:
                return None

            spk = str(tgt_ex.get("speaker_id", rng.randint(0, 10_000)))

            # Same-speaker reference for d-vector (70% of the time).
            # `buf` (after refactor) holds dicts shaped like
            # `{"audio": <1D float32 tensor, ref_max_len-capped>, "spk": <str>}`
            # — already-decoded, ready to use as ref_wav without re-running
            # soundfile. See the buffer-write step at the bottom of the
            # main iterator loop for how those entries get populated.
            buf = self._target_buffer.get(spk, [])
            ref_ex = tgt_ex
            ref_cached: Optional[torch.Tensor] = None
            # `ref_cached_dvec` + `ref_entry_id` capture the buffer hit's
            # already-computed d-vector (if any) and a write-back handle so
            # the main training thread can populate the entry's `dvec`
            # field after it runs `extract_dvec`. Both are None when the
            # ref is the just-decoded target (no buffer hit).
            ref_cached_dvec: Optional[torch.Tensor] = None
            ref_entry_id: Optional[int] = None
            if len(buf) >= 2 and rng.random() < 0.7:
                cached = rng.choice(buf[:-1])
                ref_ex = cached
                ref_cached = cached["audio"]
                ref_cached_dvec = cached.get("dvec")
                # Python `id()` is stable for the lifetime of the dict,
                # and `_target_buffer[spk]` keeps the dict alive by
                # reference — so this id stays valid for the main
                # thread's write-back even after list-slicing on overflow.
                ref_entry_id = id(cached)

            if decode_pool is not None:
                fut_t = decode_pool.submit(
                    _decode_audio, tgt_ex["audio"], hp.sample_rate
                )
                fut_i = decode_pool.submit(
                    _decode_audio, int_ex["audio"], hp.sample_rate
                )
                # `ref_ex` is either `tgt_ex` (covered by fut_t) or a
                # buffer entry with a cached tensor (no decode needed).
                # In neither case do we submit a third decode.
                fut_r = None
            else:
                fut_t = fut_i = fut_r = None  # sentinel: decode inline

            ref_spk = spk  # buffer is keyed by speaker, so any ref pick is same-spk
            return {
                "tgt_ex": tgt_ex,
                "tgt_spk": spk,
                "int_ex": int_ex,
                "ref_ex": ref_ex,
                "ref_spk": ref_spk,
                "ref_cached": ref_cached,
                "ref_cached_dvec": ref_cached_dvec,
                "ref_entry_id": ref_entry_id,
                "fut_t": fut_t,
                "fut_i": fut_i,
                "fut_r": fut_r,
            }

        def _await_decodes(job: dict) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            """Block on the decode futures for `job`; return (target, interference, ref).

            Returns None on decode failure (caller skips this sample and
            increments `_decode_fail_count` for diagnostics).
            """
            tgt_ex = job["tgt_ex"]
            int_ex = job["int_ex"]
            try:
                if job["fut_t"] is not None:
                    target_wav = job["fut_t"].result()
                    interference_wav = job["fut_i"].result()
                else:
                    target_wav = _decode_audio(tgt_ex["audio"], hp.sample_rate)
                    interference_wav = _decode_audio(int_ex["audio"], hp.sample_rate)
                # ref is either a pre-decoded cached tensor (`ref_cached`)
                # or `tgt_ex` itself (reuse `target_wav`). With the buffer
                # refactor there is no third decode path here.
                ref_cached = job.get("ref_cached")
                if ref_cached is not None:
                    ref_wav = ref_cached
                else:
                    ref_wav = target_wav
                # Defensive cap: cached tensors are already capped at append
                # time, but if `ref_max_sec` ever shrinks between runs we
                # still want the safer crop.
                if ref_wav.shape[-1] > ref_max_len:
                    ref_wav = ref_wav[..., :ref_max_len]
                return target_wav, interference_wav, ref_wav
            except Exception as e:
                _skipped = getattr(self, "_decode_fail_count", 0) + 1
                self._decode_fail_count = _skipped
                if _skipped <= 3 or _skipped % 1000 == 0:
                    print(
                        f"[dataset] _decode_audio failed ({_skipped}× so far): "
                        f"audio type={type(tgt_ex.get('audio')).__name__}, "
                        f"err={type(e).__name__}: {e}",
                        flush=True,
                    )
                return None

        # ---- Prime the pipeline ----
        for _ in range(max(1, hp.decode_prefetch)):
            job = _submit_next_decode()
            if job is None:
                break
            pipeline.append(job)

        try:
            while pipeline:
                job = pipeline.popleft()

                decoded = _await_decodes(job)
                if decoded is None:
                    # Refill pipeline with a fresh job so we don't stall.
                    new_job = _submit_next_decode()
                    if new_job is not None:
                        pipeline.append(new_job)
                    continue
                target_wav, interference_wav, ref_wav = decoded
                ref_spk = job["ref_spk"]
                # Buffer-hit dvec (None on cold ref). The fresh-target path
                # produces ref_wav == target_wav, but `job["ref_entry_id"]`
                # is still None there — main thread computes + writes back.
                ref_dvec_cached = job.get("ref_cached_dvec")
                ref_entry_id = job.get("ref_entry_id")

                # ---- Interference augmentation: Musan ----
                noise_kind = NOISE_KIND_LIBRISPEECH
                if self._musan_iters and rng.random() < hp.musan_prob:
                    sample = self._sample_musan(sample_len)
                    if sample is not None:
                        interference_wav, noise_kind = sample

                # Both target and interference must be the same length (sample_len)
                # before snr_mix. The LibriSpeech utterances are ~10–25 s long, so
                # a 3 s random crop is required — without this, snr_mix raises
                # "size mismatch" because interference stays at full utterance
                # length while target is cropped.
                # Capture a *pre-crop* ref-cap of `target_wav` BEFORE the in-place
                # mutations below (RIR conv, silent zeroing) so we cache clean
                # audio for future ref picks. Truncating to `ref_max_len` here is
                # the same cap `_await_decodes` applies for dvec extraction,
                # so cached entries are ready to use as-is on a cache hit.
                target_spk_for_buf = job["tgt_spk"]
                tgt_buf = self._target_buffer.setdefault(target_spk_for_buf, [])
                # `clone()` because the in-place RIR conv / silence-zero below
                # would otherwise mutate the cached tensor in place.
                cached_audio = target_wav[..., :min(target_wav.shape[-1], ref_max_len)].clone()
                tgt_buf.append({
                    "audio": cached_audio,
                    "spk": target_spk_for_buf,
                    "dvec": None,    # lazily populated by main-thread write-back
                })
                # If this sample's ref is the just-decoded target (the
                # "fresh ref" path), the main thread will need the entry's
                # id to write the computed dvec back. Buffer-hit refs
                # already carry their `ref_entry_id` from `_submit_next_decode`.
                if ref_entry_id is None:
                    ref_entry_id = id(tgt_buf[-1])
                # Per-speaker FIFO: drop oldest entries beyond `max_buf`.
                # Slicing rebinds the list to a fresh one so the dropped
                # `cached_audio` tensors are released for GC.
                if len(tgt_buf) > max_buf:
                    self._target_buffer[target_spk_for_buf] = tgt_buf[-max_buf:]
                # Global LRU cap on distinct speakers. `move_to_end` makes
                # this speaker the most-recently-used; if we just crossed
                # the cap, popitem(last=False) drops the LRU speaker (and
                # its entire per-speaker list of decoded audios).
                self._target_buffer.move_to_end(target_spk_for_buf)
                while len(self._target_buffer) > max_speakers:
                    evicted_spk, evicted_list = self._target_buffer.popitem(last=False)
                    del evicted_list

                target_wav = random_crop(target_wav, sample_len)
                interference_wav = random_crop(interference_wav, sample_len)

                # ---- Target augmentation: real RIR convolution ----
                if self._rirs and rng.random() < hp.rir_prob:
                    h = self._sample_rir()
                    if h is not None:
                        target_wav = _convolve(target_wav, h)
                        # Convolved signal can be louder; rescale to roughly the
                        # same energy as the pre-reverb target.
                        target_wav = target_wav * (random_crop(ref_wav, sample_len).pow(2).mean().sqrt()
                                                   / (target_wav.pow(2).mean().sqrt() + 1e-8))

                # ---- "do nothing" training ----
                # Sample silent flags first (so we know the label for the noise-
                # type predictor before mutating the waveforms).
                target_silent = rng.random() < hp.silent_target_prob
                interference_silent = rng.random() < hp.silent_interference_prob

                # ---- Per-frame activity bursts (paper §4.2 / deviation 5) ----
                # For non-silent target/interference, gate the waveform with a
                # sample-level 0/1 activity mask that has a single active burst
                # of `[activity_burst_min_sec, activity_burst_max_sec]` seconds,
                # with `activity_fade_ms` fade-in / fade-out at the edges
                # (avoids click artefacts at burst boundaries). The noise-type
                # predictor's per-frame label is then `+1` exactly where BOTH
                # target and interference bursts cover the frame's centre
                # (i.e. frame centres in the *intersection* of the two burst
                # windows). This matches paper §4.2's frame-level "is this
                # frame overlapped speech?" labelling — replacing the legacy
                # single-label-per-sample broadcast.
                if not target_silent:
                    target_act = _sample_activity_burst(
                        sample_len, hp, rng,
                    )
                    target_wav = target_wav * target_act
                else:
                    target_act = torch.zeros(sample_len, dtype=target_wav.dtype)
                if not interference_silent:
                    interference_act = _sample_activity_burst(
                        sample_len, hp, rng,
                    )
                    interference_wav = interference_wav * interference_act
                else:
                    interference_act = torch.zeros(
                        sample_len, dtype=interference_wav.dtype,
                    )

                if target_silent:
                    target_wav = torch.zeros_like(target_wav)
                if interference_silent:
                    interference_wav = torch.zeros_like(interference_wav)

                # Bypass `snr_mix` when either side is silent. `snr_mix` clamps
                # `clean_power` to 1e-12 when clean=0, which collapses the SNR
                # scaling to ~0 and silently turns `mixed_wav` into silence
                # even when interference is fully active. That removed the
                # training signal for "target absent → suppress": the model
                # saw silent inputs paired with silent targets and learned
                # mask≈1 unconditionally. By bypassing `snr_mix` for the
                # silent branches:
                #   * both silent → mixed = 0
                #   * target silent → mixed = interference (model must learn
                #     to suppress the active interference to hit target=-13.8)
                #   * interference silent → mixed = target (clean pass-through,
                #     same end result as snr_mix but explicit + cheap)
                #   * both active → snr_mix as before (using the
                #     activity-gated waveforms; the burst window's mean
                #     power is still used for the SNR rescale, which is a
                #     deliberate design point — see plan deviation-5).
                snr = rng.uniform(hp.snr_min_db, hp.snr_max_db)
                if target_silent and interference_silent:
                    mixed_wav = torch.zeros_like(target_wav)
                elif target_silent:
                    mixed_wav = interference_wav.clone()
                elif interference_silent:
                    mixed_wav = target_wav.clone()
                else:
                    mixed_wav = snr_mix(target_wav, interference_wav, snr)

                # ---- Features (CPU-side) ----
                # SpeakerEncoder lives in `train.py` and runs on the main
                # training thread via `audio.extract_dvec(...)` — the
                # dataset stays MPS-free so the PrefetchIterator's
                # background thread never touches MPS.
                mixed_logmel = wav_to_log_mel(mixed_wav, hp, self._mel)
                target_logmel = wav_to_log_mel(target_wav, hp, self._mel)
                stacked = frame_stack_left(mixed_logmel, hp.frame_stack)
                T_frames = stacked.shape[0]

                # ---- Noise-type predictor label (paper §4.2) ----
                # +1 if this frame is overlapped speech (both target and
                # interference activity masks > 0 at the frame's centre
                # sample), -1 otherwise. Replaces the legacy whole-sample
                # broadcast label.
                overlap_label = _per_frame_overlap_label(
                    target_act=target_act,
                    interference_act=interference_act,
                    T_frames=T_frames,
                    sample_len=sample_len,
                    hp=hp,
                )

                # ---- Heartbeat ----
                _samples += 1
                _now = time.time()
                if (
                    _samples <= hp.batch_size
                    or _samples % (hp.batch_size * 4) == 0
                    or (_now - _last_beat) > _HEARTBEAT_S
                ):
                    _elapsed = _now - _t0
                    _rate = _samples / max(0.001, _elapsed)
                    # Buffer diagnostics: how many speakers and how many
                    # cached ref audios the iterator is currently holding.
                    # With the LRU cap (`ref_buffer_max_speakers`) the
                    # first number saturates; the second stays bounded by
                    # `ref_buffer_max_per_spk × ref_buffer_max_speakers`.
                    _buf = self._target_buffer
                    _n_speakers = len(_buf)
                    _n_entries = sum(len(v) for v in _buf.values())
                    print(
                        f"[dataset] {_samples} samples ready "
                        f"(t={_elapsed:.1f}s, {_rate:.1f} samples/s, "
                        f"buf={_n_speakers}/{hp.ref_buffer_max_speakers} "
                        f"entries={_n_entries})",
                        flush=True,
                    )
                    _last_beat = _now

                # Yield the CPU-only Sample. SpeakerEncoder / dvec extraction
                # happens on the main thread via `audio.extract_dvec(...)`
                # in `train.py`. Batching is done by DataLoader(batch_size=).
                yield Sample(
                    stacked_mixed=stacked,
                    target_logmel=target_logmel,
                    ref_wav=ref_wav,
                    ref_spk_id=ref_spk,
                    noise_kind=noise_kind,
                    overlap_label=overlap_label,
                    ref_dvec=ref_dvec_cached,
                    ref_entry_id=ref_entry_id,
                )

                # Refill pipeline with the next decode job. This keeps
                # `decode_prefetch` jobs in flight ahead of consumption,
                # so the main thread's `fut.result()` almost never blocks
                # on a single decode — the pool has the next sample's
                # audio already decoded by the time we're ready for it.
                new_job = _submit_next_decode()
                if new_job is not None:
                    pipeline.append(new_job)
        finally:
            if decode_pool is not None:
                # Don't block on pending decode jobs — the iterator is
                # done and we're shutting down.
                decode_pool.shutdown(wait=False)


def collate_triplets(batch: list[Sample]) -> dict:
    """Collate a list of `Sample` into CPU tensors.

    The training loop moves tensors to the training device itself, so that
    the DataLoader (and any background prefetch thread) only has to ship
    plain CPU tensors through its iterator / queue. This avoids pickling
    MPS tensors when the queue transfers batches across threads.

    Variable-length `ref_wav` entries are zero-padded to the longest in
    the batch and reported as relative lengths (`ref_lens`) so
    `audio.extract_dvec(...)` in `train.py` can crop the padding before
    librosa mel extraction. `ref_spk_ids` is kept parallel to rows for
    diagnostics.
    """
    stacked = torch.stack([s.stacked_mixed for s in batch])
    target = torch.stack([s.target_logmel for s in batch])
    overlap_label = torch.stack([s.overlap_label for s in batch])

    # Variable-length padded ref batch (CPU float32). `ref_max_sec` was
    # already enforced on the raw utterance in `_await_decodes`; the
    # batch-max pad here only adds trailing zeros for shorter refs in
    # the same batch — `extract_dvec` then trims them via `ref_lens`.
    ref_wavs_list = [s.ref_wav for s in batch]
    max_len = max(r.shape[-1] for r in ref_wavs_list)
    ref_wavs = torch.zeros(len(batch), max_len, dtype=torch.float32)
    ref_lens = torch.zeros(len(batch), dtype=torch.float32)
    for i, r in enumerate(ref_wavs_list):
        L = r.shape[-1]
        ref_wavs[i, :L] = r
        ref_lens[i] = L / max_len
    ref_spk_ids = [s.ref_spk_id for s in batch]

    # ---- Dvec cache (per-buffer-entry write-back) ----
    # Stack any pre-computed d-vectors into (B, 256). Missing entries
    # (None) keep their row as a zero placeholder — `train.py` then
    # recomputes the full (B, 256) batch via `extract_dvec` and writes
    # the result back into the buffer through `ref_entry_ids` so the
    # next time the same entry is picked from the buffer, this field
    # comes back non-None and we skip the recompute.
    ref_dvecs_list = [s.ref_dvec for s in batch]
    have_all_dvecs = all(d is not None for d in ref_dvecs_list)
    if have_all_dvecs:
        ref_dvecs = torch.stack(ref_dvecs_list)        # (B, 256)
    else:
        ref_dvecs = None                               # train.py will compute
    ref_entry_ids = [s.ref_entry_id for s in batch]

    return {
        "stacked_mixed": stacked,
        "target_logmel": target,
        "overlap_label": overlap_label,
        "noise_kind": [s.noise_kind for s in batch],
        "ref_wavs": ref_wavs,
        "ref_lens": ref_lens,
        "ref_spk_ids": ref_spk_ids,
        "ref_dvecs": ref_dvecs,
        "ref_entry_ids": ref_entry_ids,
    }