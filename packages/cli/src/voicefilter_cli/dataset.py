"""HF-backed triplet iterator for VoiceFilter-Lite training.

Reads from the HuggingFace arrow cache that `voicefilter-cli dataset
prepare` populates — **no wav files are written to disk**. Each row of
each (source, config, split) becomes an `HfEntry` (a `(dataset, row_idx,
spk_id, utt_id, source_sr)` pointer). At iteration time the
`wav_loader` reads `entry.dataset[entry.row_idx]['audio']` (a
`{"bytes": ...}` dict with FLAC bytes when HF was downloaded as audio,
or `{"array": ..., "sampling_rate": ...}` when upstream pre-decoded),
then decodes via `soundfile.read(io.BytesIO(...))` and resamples to
`hp.sample_rate` if needed. LibriSpeech is 16 kHz natively; MUSAN
`speech` is 16 kHz, `noise` / `music` are 44.1 kHz → the loader
handles both transparently.

`prepare_dataset` now does **only** HF download + per-source index
writing. Disk savings vs. the old wav-baking pipeline is ~80-200 GB on
a full LibriSpeech-500 + MUSAN bake (HF arrow cache is ~10-20 GB;
decoded wavs at 16 kHz float32 were 200+ GB).

D-vector extraction happens lazily inside this module via
`DVectorCache`. Per-speaker `<spk>.npy` files live under
`<cache_dir>/voicefilter-dvectors/<spec_id>/<spk>.npy` (≈256 KB per
speaker, ~256 KB total for LibriSpeech clean-360). Cache invalidation:
wipe `<cache_dir>/voicefilter-dvectors/` to force recomputation after
retraining the d-vector encoder.

Per-sample stochastic work that must vary every iteration:

  1. Sample (target_speaker, target_utt)            # random
  2. Sample ref wav (70% same-speaker, 30% self)   # random
  3. Read 2 HF entries (target + ref)             # ~10 ms/sample
  4. Random crop target to `sample_length_sec`     # stochastic
  5. Compute `target_power = mean(target²)`        # used for budget
  6. Roll silent flags (target, interference)      # random
  7. Draw `snr` ∈ [snr_min_db, snr_max_db] uniform  # random
  9. Synthesise interference with ENERGY BUDGET   # stochastic
       For each TYPE (`babble` / `noise` / `music`) whose `prob > 0`,
       pool is non-empty, and `enabled=True`: roll `rng.random() <
       prob` independently (multiple TYPES can roll true → any
       combination: pure noise, pure music, noise+music, babble+music,
       all three, or none). For each included TYPE, synthesize at
       natural energy, then allocate the **total interference energy
       budget** = `target_power / 10^(snr/10)` across the K
       included TYPES via Dirichlet(α=1) sampling on the simplex
       (uniform on it). Scale each component so its RMS² = `w_i ×
       budget`. Sum the K scaled components — SUM RMS² = budget, so
       post-mix SIR = `snr_db` exactly regardless of K. Each
       component is peak-clipped to ±1.0 BEFORE summing; the
       aggregate is soft-clipped via `tanh` to ±1.0 to prevent
       downstream `snr_mix` from clipping into wideband distortion.
 10. Activity burst × 2                             # stochastic
 11. Apply silent flags (defensive zero-out)        # conditional
 12. SNR mix                                        # stochastic
 13. log-mel × 2 + frame-stack                      # per-sample
 14. Per-frame overlap label                        # per-sample
 15. Get-or-compute d-vector for ref speaker        # DVectorCache
     (in-process hit: ~µs; disk hit: one .npy read;
      miss: one SpeakerEncoder forward + atomic save)

The energy-budget allocation in step 9 is the BIGGEST fix vs. naive
summing. Without it, RMS SUM of K components inflates by ~K, dropping
the effective per-TYPE SIR by `10·log10(K)` dB and making the task
artificially harder (drops have nothing to do with "added noise" — you
made the problem harder). The Dirichlet split also keeps babble (N
speakers, natural RMS² is large) from dominating the aggregate just
because its natural energy is higher than a raw music clip's.

Interference TYPEs (`babble` / `noise` / `music`) are configured as
direct fields on `hp.dataset.interference` (no `types` wrapper).
`prepare_dataset` writes a JSON index per source at
`<cache_dir>/voicefilter-index/<spec_id>__<subset>__<split>.json`; this
iterator reads those JSONs at scan time to build the in-memory
`{spk_id: [HfEntry, ...]}` map. Per-sample, each TYPE's `prob` is an
INDEPENDENT Bernoulli inclusion probability; types with `prob <= 0` or
empty backing pools are skipped. The babble pool is **config-driven**
— `_collect_babble_pool_entries` walks
`hp.interference.babble.sources[*]` at scan time and builds
`_spk_to_paths` from the union of every babble-eligible source
(currently musan(speech) + librispeech(other)). The librispeech
`other` pool is NOT used for babble unless `babble.sources` declares
it; the canonical paper.yaml keeps librispeech(other) under
`noise.sources` for `prepare_dataset` prefetch +
`first_librispeech_other_source` discovery (eval / audition harnesses),
not as runtime babble input.

Each yielded `Sample` carries `(stacked_mixed, target_logmel, ref_dvec,
overlap_label, noise_kind)`. `noise_kind` is a `tuple[str, ...]` of the
TYPE names actually included in this sample's interference (e.g.
`()`, `("babble",)`, `("noise", "music")`, `("babble", "noise",
"music")`) — used for diagnostics only. The order is deterministic
(babble, noise, music) so test assertions and summaries are stable.
"""

from __future__ import annotations

import io
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import IterableDataset

from voicefilter_lite.audio import (
    frame_stack_left,
    random_crop,
    snr_mix,
    wav_to_log_mel,
)
from voicefilter_lite.hparams import HParams, ResolvedSource, resolve_source


NOISE_KIND_BABBLE = "babble"
NOISE_KIND_MUSIC = "music"
NOISE_KIND_NOISE = "noise"


# Per-source subdirectory names that live next to the HF arrow cache.
# These are the ONLY on-disk artefacts outside the HF cache itself.
# Per-source so each cache is self-describing — no central processed_dir.
DVEC_DIRNAME = "voicefilter-dvectors"
INDEX_DIRNAME = "voicefilter-index"
MANIFEST_FILENAME = "voicefilter-manifest.json"


@dataclass(frozen=True)
class HfEntry:
    """Pointer into one row of one HF `datasets.Dataset`.

    The audio is NOT loaded here — `wav_loader(entry, target_sr)` is
    called at iteration. Forking a `HfTripletStream` is safe because the
    payload is just ints + refcount-stable dataset objects (the
    `datasets.Dataset` itself is mmap-backed by arrow, so copy-on-write
    is fine for read-only access).

    Fields:
        dataset_id: human-readable id, e.g. `"librispeech/train.clean.360"`.
        dataset: the mmap'd `datasets.Dataset` for this (source, config, split).
        row_idx: row index into `dataset`.
        spk_id: speaker id (`speaker_id` field for LibriSpeech; clip stem
            for MUSAN). Empty string only when truly anonymous.
        utt_id: utterance id (HF `file`/`id` field, sanitised).
        source_sr: native sample rate of this row's audio (16k for
            LibriSpeech, 16k for MUSAN `speech`, 44.1k for MUSAN
            `noise` / `music`). The wav_loader uses this to decide
            whether to resample to `target_sr`.
    """
    dataset_id: str
    dataset: Any
    row_idx: int
    spk_id: str
    utt_id: str
    source_sr: int


WavLoader = Callable[[HfEntry, int], np.ndarray]
# `(entry, target_sr) → mono float32 (T,) at target_sr`.
# Implementation: read `entry.dataset[entry.row_idx]['audio']` (the HF
# row with `Audio(decode=False)`), pick `bytes` / `array` / `path`,
# decode with `soundfile`, mixdown to mono, resample if sr != target_sr,
# peak-clip if |x|>5.0. See `default_hf_wav_loader` below.


def first_librispeech_other_source(hp: HParams) -> ResolvedSource:
    """Return the first enabled LibriSpeech `other` source.

    Walks `hp.interference.{babble,noise,music}.sources[*]` for the
    first enabled `SourceRef` whose resolved splits carry the `other`
    role tag (parsed from `train.other.<size>` split names). Returns
    the `ResolvedSource` (not the raw `SourceRef` — callers want the
    full spec, splits, cache_dir).

    Eval / diag scripts use this to find the canonical "other"-role
    pool (paper.yaml declares exactly one). Raises if no enabled
    LibriSpeech `other` source is configured.
    """
    for type_cfg in hp.interference_types.values():
        if not type_cfg.enabled or type_cfg.prob <= 0:
            continue
        for ref in type_cfg.sources:
            resolved = resolve_source(hp, ref)
            if resolved.librispeech_role() == "other":
                return resolved
    raise RuntimeError(
        "[first_librispeech_other_source] no enabled LibriSpeech `other` "
        f"source under hp.interference; got keys="
        f"{list(hp.interference_types)}"
    )


def first_musan_repo(hp: HParams, subset: str) -> str:
    """Return the MUSAN HF repo for `subset` ("speech" / "noise" / "music").

    Walks `hp.interference.{babble,noise,music}.sources[*]` for the
    first enabled `SourceRef` whose `subset == <subset>` and that
    resolves to a MUSAN spec. Returns the spec's `repo` (the MUSAN
    repo lives on the spec, not on the SourceRef). Falls back to the
    conventional default `corypaik/musan` if no MUSAN source matches.
    """
    for type_cfg in hp.interference_types.values():
        if not type_cfg.enabled or type_cfg.prob <= 0:
            continue
        for ref in type_cfg.sources:
            if ref.subset != subset:
                continue
            resolved = resolve_source(hp, ref)
            if resolved.is_musan():
                return resolved.repo
    return "corypaik/musan"


# ============================================================ #
# HF dataset open + iter helpers (per-source HfEntry builders)    #
# ============================================================ #


# `_disable_hf_audio_decode` monkey-patches HF's `Audio.decode_example`
# and guards against re-patching on repeated calls via this module-level
# flag. Initialised here so the first call from any module-import path
# sees `False` rather than `NameError`.
_HF_AUDIO_DECODE_PATCHED = False


def _disable_hf_audio_decode() -> None:
    """Idempotently monkey-patch HF `Audio.decode_example` to never decode.

    Idempotent: the `_HF_AUDIO_DECODE_PATCHED` flag (module-level, set
    above) prevents re-patching on repeated calls.
    """
    global _HF_AUDIO_DECODE_PATCHED
    if _HF_AUDIO_DECODE_PATCHED:
        return
    from datasets.features import audio as _hf_audio
    _orig_decode_example = _hf_audio.Audio.decode_example

    def _passthrough_decode_example(self, value, token_per_repo_id=None):
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        return _orig_decode_example(self, value, token_per_repo_id)

    _hf_audio.Audio.decode_example = _passthrough_decode_example
    _HF_AUDIO_DECODE_PATCHED = True


def _cast_audio_no_decode(ds):
    """Return a copy of `ds` with every `Audio` column set to `decode=False`.

    Same module-level helper exposed for callers that build a single
    dataset directly. Combined with `_disable_hf_audio_decode` above,
    row access returns the raw dict payload (`{"bytes": ..., "path": ...}`)
    so `default_hf_wav_loader` decodes via `soundfile` only when needed.
    """
    from datasets import Audio, Features
    new_features = Features(
        {k: (Audio(decode=False) if k == "audio" else v) for k, v in ds.features.items()}
    )
    try:
        ds = ds.cast_features(new_features)
    except Exception:
        ds.features.update(new_features)
    for k, v in ds.features.items():
        if isinstance(v, Audio):
            v.decode = False
    return ds


def _open_hf_dataset(
    resolved: ResolvedSource,
    split: str,
) -> "datasets.Dataset":
    """Open one HF `datasets.Dataset` for a single (spec, subset, split).

    Uses `load_dataset(..., split=..., cache_dir=...)` so the result is
    a single concrete split — easier to index by row than a
    `DatasetDict`. Applies the no-decode patch so row access returns
    `{"bytes": ...}` (or `{"array": ...}`) without forcing a full
    FLAC decode.

    Cached per-process by `HfTripletStream` (the iterator doesn't
    re-call `load_dataset` every iteration — only at scan time).
    """
    import datasets as _ds

    # `load_dataset` takes the repo id as positional `path` (NOT a
    # `repo=` kwarg — passing `repo=...` silently drops it and HF raises
    # `TypeError("missing 1 required positional argument: 'path')`,
    # which is exactly the prepare crash we hit on 2026-10-08).
    kwargs: dict[str, Any] = {
        "path": resolved.repo,
        "split": split,
        "cache_dir": resolved.cache_dir,
    }
    if resolved.is_librispeech():
        # LibriSpeech uses `config="all"` (the only config that exposes
        # both clean and other); `subset` defaults to that single config
        # when the SourceRef didn't override it.
        kwargs["name"] = resolved.subset or "all"
    elif resolved.is_musan():
        kwargs["name"] = resolved.subset

    ds = _ds.load_dataset(**kwargs)
    _disable_hf_audio_decode()
    return _cast_audio_no_decode(ds)


def _dataset_id(resolved: ResolvedSource, split: str) -> str:
    """Stable id used in `HfEntry.dataset_id` (e.g. "librispeech/train.clean.360").

    Human-readable; helps in error messages + d-vector path lookup.
    """
    if resolved.is_librispeech():
        return f"{resolved.id}/{split}"
    if resolved.is_musan():
        return f"{resolved.id}/{resolved.subset}/{split}"
    return f"{resolved.id}/{split}"


def _load_index(
    resolved: ResolvedSource,
    split: str,
    cache_dir: Path,
) -> Optional[list[dict]]:
    """Load `<cache_dir>/voicefilter-index/<dataset_id>.json` if present.

    `prepare_dataset` writes one JSON per (spec, subset, split) so
    training can rebuild the in-memory `{spk_id → [HfEntry, …]}` map
    without re-walking the HF arrow cache at startup (~1-2 s saved for
    LibriSpeech clean-360 on cold cache; ~30 s saved when the cache
    is on cold spinning disk).

    Returns the `entries` list (each dict: `{row_idx, spk_id, utt_id,
    source_sr}`) or `None` when the JSON is missing (the iterator then
    falls back to walking the dataset). Missing file is OK — older
    prepare outputs didn't write this JSON.
    """
    ds_id = _dataset_id(resolved, split)
    safe_ds = ds_id.replace("/", "__")
    path = Path(cache_dir) / INDEX_DIRNAME / f"{safe_ds}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        return None
    return entries


def _row_metadata(
    resolved: ResolvedSource,
    row: Any,
    source_sr: int,
) -> tuple[str, str]:
    """Extract `(spk_id, utt_id)` from a single HF row.

    LibriSpeech: `speaker_id` (int → str) + `file` (path → utt id).
    MUSAN: clip stem from `audio.path` (or `id` / `file`). MUSAN has
    no real speaker; we use the clip stem as the identity.
    """
    if resolved.is_librispeech():
        spk = row.get("speaker_id") if hasattr(row, "get") else row["speaker_id"]
        utt = row.get("id") or row.get("file") if hasattr(row, "get") else (
            row.get("id") or row.get("file")
        )
        spk_id = str(spk).strip() if spk is not None else ""
        utt_id = Path(str(utt).strip()).stem if utt else ""
        if not utt_id:
            audio = row.get("audio") if hasattr(row, "get") else row.get("audio")
            if isinstance(audio, dict) and audio.get("path"):
                utt_id = Path(audio["path"]).stem
        if not spk_id:
            spk_id = "unknown"
        if not utt_id:
            utt_id = "unknown"
        return spk_id, utt_id
    if resolved.is_musan():
        audio = row.get("audio") if hasattr(row, "get") else row.get("audio")
        path_str = ""
        if isinstance(audio, dict):
            path_str = audio.get("path") or ""
        if not path_str:
            for k in ("id", "file", "utt_id"):
                v = row.get(k) if hasattr(row, "get") else None
                if v:
                    path_str = str(v)
                    break
        stem = Path(path_str).stem if path_str else "(unknown)"
        spk_id = f"musan_{resolved.subset}_{stem}" if resolved.subset else stem
        return spk_id, stem
    # Generic: pull spk_id from `speaker_id`, `spk_id`, or fall back.
    spk = None
    for k in ("speaker_id", "spk_id", "speaker"):
        if hasattr(row, "get"):
            spk = row.get(k)
        else:
            spk = row.get(k) if k in row else None
        if spk:
            break
    utt = None
    for k in ("id", "utt_id", "file"):
        if hasattr(row, "get"):
            utt = row.get(k)
        else:
            utt = row.get(k) if k in row else None
        if utt:
            break
    return (str(spk).strip() if spk else "unknown"), (
        str(utt).strip() if utt else "unknown"
    )


def _iter_musan_speech_entries(
    resolved: ResolvedSource,
    dataset_cache: dict[str, "datasets.Dataset"],
) -> list[HfEntry]:
    """Build `[HfEntry]` for a MUSAN `speech` resolved source.

    Cache key = `_dataset_id(resolved, split)`. The same cache key is
    reused by `_iter_musan_subset_entries` (called by `_collect_passthrough_pool_entries`
    when the noise config also lists a MUSAN source) — each unique
    (spec, subset, split) opens exactly ONE `datasets.Dataset`.
    """
    out: list[HfEntry] = []
    source_sr = 16000
    for split in resolved.splits:
        ds_id = _dataset_id(resolved, split)
        ds = dataset_cache.get(ds_id)
        if ds is None:
            ds = _open_hf_dataset(resolved, split)
            dataset_cache[ds_id] = ds
        for row_idx in range(len(ds)):
            row = ds[row_idx]
            spk_id, utt_id = _row_metadata(resolved, row, source_sr)
            out.append(
                HfEntry(
                    dataset_id=ds_id,
                    dataset=ds,
                    row_idx=row_idx,
                    spk_id=spk_id,
                    utt_id=utt_id,
                    source_sr=source_sr,
                )
            )
    return out


def _iter_musan_subset_entries(
    resolved: ResolvedSource,
    dataset_cache: dict[str, "datasets.Dataset"],
    type_name: str,
) -> list[HfEntry]:
    """Build `[HfEntry]` for a MUSAN `noise` or `music` resolved source.

    `type_name` is `noise` / `music` — used to pick the source_sr (both
    `noise` and `music` are 44.1 kHz on MUSAN). The caller only feeds
    musan sources here (the librispeech(other) refs are skipped before
    we get called).
    """
    out: list[HfEntry] = []
    # MUSAN noise / music subsets are natively 44.1 kHz (sampling rate
    # carried on each row's `audio.sampling_rate` too; we hardcode here
    # for speed — see plan).
    source_sr = 44100
    for split in resolved.splits:
        ds_id = _dataset_id(resolved, split)
        ds = dataset_cache.get(ds_id)
        if ds is None:
            ds = _open_hf_dataset(resolved, split)
            dataset_cache[ds_id] = ds
        for row_idx in range(len(ds)):
            row = ds[row_idx]
            spk_id, utt_id = _row_metadata(resolved, row, source_sr)
            out.append(
                HfEntry(
                    dataset_id=ds_id,
                    dataset=ds,
                    row_idx=row_idx,
                    spk_id=spk_id,
                    utt_id=utt_id,
                    source_sr=source_sr,
                )
            )
    return out


def _iter_librispeech_entries(
    resolved: ResolvedSource,
    dataset_cache: dict[str, "datasets.Dataset"],
) -> list[HfEntry]:
    """Build `[HfEntry]` for a LibriSpeech resolved source (target or `other` pool).

    Walks each split in `resolved.splits` (e.g. `train.clean.360`,
    `train.other.500`). Source_sr is uniformly 16 kHz on LibriSpeech.
    """
    out: list[HfEntry] = []
    source_sr = 16000
    for split in resolved.splits:
        ds_id = _dataset_id(resolved, split)
        ds = dataset_cache.get(ds_id)
        if ds is None:
            ds = _open_hf_dataset(resolved, split)
            dataset_cache[ds_id] = ds
        for row_idx in range(len(ds)):
            row = ds[row_idx]
            spk_id, utt_id = _row_metadata(resolved, row, source_sr)
            out.append(
                HfEntry(
                    dataset_id=ds_id,
                    dataset=ds,
                    row_idx=row_idx,
                    spk_id=spk_id,
                    utt_id=utt_id,
                    source_sr=source_sr,
                )
            )
    return out


def _build_target_index(
    hp: HParams,
    dataset_cache: dict[str, "datasets.Dataset"],
) -> tuple[
    dict[str, list[HfEntry]],
    dict[str, str],
]:
    """Build target `{spk_id → [HfEntry, …]}` + `{spk_id → spec_id}`.

    Walks `hp.target_sources[*]`, resolves each via `resolve_source`,
    and unions every HfEntry. The spec_id partition is used by
    `DVectorCache` so multi-source targets don't collide on speaker ids.

    Returns `(target_index, spk_to_spec_id)`.
    """
    target_index: dict[str, list[HfEntry]] = {}
    spk_to_spec_id: dict[str, str] = {}

    for ref in hp.target_sources:
        try:
            resolved = resolve_source(hp, ref)
        except KeyError:
            continue
        if resolved.is_librispeech():
            entries = _iter_librispeech_entries(resolved, dataset_cache)
        elif resolved.is_musan():
            # MUSAN is not a target source (no real speakers) — skip.
            continue
        else:
            entries = _iter_librispeech_entries(resolved, dataset_cache)
        for entry in entries:
            target_index.setdefault(entry.spk_id, []).append(entry)
            spk_to_spec_id[entry.spk_id] = resolved.id

    # Sort each speaker's entry list by `(dataset_id, row_idx)` so the
    # "first available utterance" reference choice is deterministic
    # across runs (the d-vector cache uses `paths[0]` for ref-wav
    # resolution).
    for spk in target_index:
        target_index[spk].sort(key=lambda e: (e.dataset_id, e.row_idx))

    return target_index, spk_to_spec_id


def _collect_babble_pool_entries(
    hp: HParams,
    dataset_cache: dict[str, "datasets.Dataset"],
) -> list[tuple[str, HfEntry]]:
    """Walk `hp.interference.babble.sources[*]` and return babble-pool `HfEntry`s.

    Returns a list of `(spk_id, entry)` pairs. `spk_id` is a stable
    identity key for babble dedup (the runtime picks N DISTINCT
    `spk_id`s — see `_pick_babble_speakers`):

      * musan(speech)        : `musan_speech__<clip_stem>` —
                                each clip is one long-monologue
                                identity, exactly one entry each.
      * librispeech(other)   : `librispeech_other__<spk>` —
                                `parts[0]` under the split dir is the
                                speaker id (multiple utts per spk).

    Source kinds not listed above (noise / music subsets, target
    librispeech `clean`) are skipped — they are passthrough or
    target pools, not babble-eligible.

    Empty list when babble is disabled, or when no configured source
    has a non-empty pool in the HF cache. An empty return is a config /
    pipeline error (the user ran training without `prepare`) — surfaced
    elsewhere.
    """
    babble_cfg = hp.interference_types.get("babble")
    if babble_cfg is None or not babble_cfg.enabled or babble_cfg.prob <= 0:
        return []

    out: list[tuple[str, HfEntry]] = []
    seen_keys: set[tuple[str, int]] = set()  # (dataset_id, row_idx)

    for ref in babble_cfg.sources:
        try:
            resolved = resolve_source(hp, ref)
        except KeyError:
            # Unknown source id — skip silently (config bug; surface
            # elsewhere if it matters).
            continue

        if resolved.is_musan() and resolved.subset == "speech":
            for entry in _iter_musan_speech_entries(resolved, dataset_cache):
                key = (entry.dataset_id, entry.row_idx)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                out.append((f"musan_speech__{entry.utt_id}", entry))
            continue

        if resolved.is_librispeech() and resolved.librispeech_role() == "other":
            for entry in _iter_librispeech_entries(resolved, dataset_cache):
                key = (entry.dataset_id, entry.row_idx)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                out.append((f"librispeech_other__{entry.spk_id}", entry))
            continue

        # Other kinds (noise / music subsets, librispeech clean role):
        # not babble-eligible — skip.
    return out


def _collect_passthrough_pool_entries(
    hp: HParams,
    dataset_cache: dict[str, "datasets.Dataset"],
    type_name: str,
) -> list[HfEntry]:
    """Collect passthrough-pool clips for the `noise` / `music` TYPEs.

    Walks `hp.interference.<type_name>.sources[*]` and unions every
    `HfEntry` from each MUSAN-configured source. Only MUSAN sources are
    consumed at runtime (the noise / music TYPE mixer is a "1 random
    clip + random_crop" passthrough, and we keep the speech-source
    concern out of it). Non-musan refs (e.g. librispeech(other) under
    `noise.sources` for eval discoverability) are skipped silently —
    `prepare_dataset` still downloads them so `first_librispeech_other_source`
    can find them.

    Empty when the TYPE is disabled, has `prob <= 0`, or has no enabled
    musan source with a non-empty on-disk cache.
    """
    type_cfg = hp.interference_types.get(type_name)
    if type_cfg is None or not type_cfg.enabled or type_cfg.prob <= 0:
        return []

    out: list[HfEntry] = []
    seen_keys: set[tuple[str, int]] = set()
    for ref in type_cfg.sources:
        try:
            resolved = resolve_source(hp, ref)
        except KeyError:
            continue
        if not resolved.is_musan() or not resolved.subset:
            # Runtime noise/music mixer only consumes musan pools;
            # librispeech(other) etc. are downloaded-only for eval
            # discovery.
            continue
        for entry in _iter_musan_subset_entries(resolved, dataset_cache, type_name):
            key = (entry.dataset_id, entry.row_idx)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.append(entry)
    return out


@dataclass
class Sample:
    stacked_mixed: torch.Tensor   # (T, n_mels * frame_stack)  — model input
    target_logmel: torch.Tensor   # (T, n_mels)                — loss target
    # D-vector for the reference speaker (256-d float32). Sourced via
    # `DVectorCache.get(tgt_spk)`: in-process hit (~µs), disk hit
    # (one .npy read from `dvectors/<source>/<spk>.npy`), or compute
    # (one SpeakerEncoder forward + atomic save on first sight).
    ref_dvec: torch.Tensor        # (256,) float32 CPU
    ref_spk_id: str               # speaker_id (diagnostics)
    # Diagnostic tag = the TYPES actually mixed into this sample's
    # interference. `tuple[str, ...]` of TYPE names in canonical order
    # (`babble`, `noise`, `music`); `()` when all TYPES rolled false.
    # E.g. `("babble",)`, `("noise", "music")`, `("babble", "noise",
    # "music")`. Used for diagnostics / per-bucket SDR reports only.
    noise_kind: tuple[str, ...] = (NOISE_KIND_BABBLE,)
    # Per-frame label for the noise-type predictor (paper §4.2):
    # +1 = "this frame is overlapped speech", -1 = "pass-through / clean".
    overlap_label: Optional[torch.Tensor] = None  # (T,) of {-1, +1}


@dataclass
class RawSample:
    """Per-sample waveforms produced by `synthesize_raw_sample`.

    Carries the WAV-level outputs of the training pipeline (target /
    interference / mixed wavs, plus the activity-burst masks and silent
    flags) so that callers other than the training iterator — e.g.
    offline audition scripts (`examples/04_*`) or eval harnesses — can
    invoke the SAME per-sample synthesis logic without re-implementing
    the silent-flag / activity-burst / Dirichlet-budget dance.

    The training iterator (`HfTripletStream.__iter__`) calls
    `synthesize_raw_sample` and then converts these wavs into log-mel
    features for `Sample` (see `_sample_to_feature`). The `RawSample`
    fields are exactly the WAV-side state at the moment log-mel would
    be computed.
    """
    # (sample_len,) float32 — random-cropped target AFTER the target
    # burst mask and the silent-target zero-out. May be all zeros if
    # `target_silent=True`.
    target_wav: torch.Tensor
    # (sample_len,) float32 — synthesised interference AFTER the
    # interference burst mask and the silent-interference zero-out. May
    # be all zeros if `interference_silent=True`.
    interference_wav: torch.Tensor
    # (sample_len,) float32 — `mixed_wav` after the silent-side branch
    # in `snr_mix` (target_silent → interference alone, interference_silent
    # → target alone, both silent → zeros, else `snr_mix` at the drawn SNR).
    mixed_wav: torch.Tensor
    # (sample_len,) float32 — raw burst mask BEFORE the silent-target
    # zero-out. 0 where the burst was inactive, 1 (with linear fades)
    # where active. `target_act` is the same value used to derive the
    # per-frame overlap label downstream.
    target_act: torch.Tensor
    # (sample_len,) float32 — interference burst mask (analogous).
    interference_act: torch.Tensor
    # (≤ref_max_sec*sr,) float32 — the reference utterance used for the
    # d-vector, capped at `hp.ref_max_sec`. Returns the full ref wav when
    # it was already short enough (no random-window pick — matches
    # training's `[:, :cap]` truncation).
    ref_wav: torch.Tensor
    # SNR draw (uniform in [snr_min_db, snr_max_db]). Used by the
    # energy-budget allocation in `_sample_interference`.
    snr_db: float
    # Canonical-order tuple of TYPE names actually mixed in
    # (`babble`/`noise`/`music`). `()` when every Bernoulli roll came
    # up false.
    included_kinds: tuple[str, ...]
    # Whether the silent-target / silent-interference branch fired. The
    # training iterator exposes both because the loss / metrics care
    # which branch was taken (the `noise_kind` tuple still reports the
    # un-muted TYPE rolls even when `interference_silent=True`).
    target_silent: bool
    interference_silent: bool


# Module-level default wav loaders. Two related APIs:
#
#   `_default_wav_loader(path)`        — legacy, takes a `Path` to a
#                                         baked `.wav` on disk. Retained
#                                         for unit tests that still build
#                                         tiny tempdir wavs; NOT used by
#                                         `HfTripletStream` at runtime.
#
#   `default_hf_wav_loader(entry, sr)` — the canonical loader for the
#                                         HF arrow cache pipeline. Reads
#                                         `entry.dataset[entry.row_idx]['audio']`
#                                         and decodes via `soundfile`
#                                         (or returns the pre-decoded
#                                         array). Resamples via
#                                         librosa if `entry.source_sr !=
#                                         target_sr`. Peak-clips `|x|>5.0`
#                                         defensively (don't let a single
#                                         spike dominate the d-vector or
#                                         log-mel).


def _default_wav_loader(path: Path) -> np.ndarray:
    """Legacy `Path → np.ndarray` loader.

    Reads a baked `.wav` from disk via `soundfile.read(...)`, downmixes to
    mono, returns float32. Used by unit tests that still build a tempdir
    of small wavs; the runtime iterator (`HfTripletStream`) does NOT
    use this loader — it uses `default_hf_wav_loader`.
    """
    wav, _ = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32, copy=False)


def default_hf_wav_loader(entry: HfEntry, target_sr: int) -> np.ndarray:
    """Load one `HfEntry` as a mono float32 `(T,)` array at `target_sr`.

    Reads `entry.dataset[entry.row_idx]['audio']` — HF `Audio(decode=False)`
    exposes three payloads depending on upstream:
      * `{"bytes": ..., "path": ...}` — FLAC bytes on disk; `soundfile`
        decodes via `BytesIO`. **The common path** for LibriSpeech /
        MUSAN after `_disable_hf_audio_decode` + `_cast_audio_no_decode`.
      * `{"array": np.ndarray, "sampling_rate": int}` — pre-decoded
        array; we copy + resample if SR differs.
      * `{"path": str}` only (no `array`, no `bytes`) — fallback path
        decode via the on-disk file.

    Always mono (mean over channels). Resamples via `librosa.resample`
    when `entry.source_sr != target_sr` (MUSAN `noise` / `music` are
    44.1 kHz; LibriSpeech + MUSAN `speech` are 16 kHz).

    Defensive peak clip `|x|>5.0`: a single transient can dominate
    `extract_dvec` (large magnitude → softmax saturation → uniform
    embedding) and the log-mel pipeline. Clip anything clearly
    pathological; leave normal speech / noise untouched.
    """
    row = entry.dataset[entry.row_idx]
    audio = row.get("audio") if hasattr(row, "get") else row["audio"]
    if audio is None:
        raise RuntimeError(
            f"[default_hf_wav_loader] entry has no 'audio' field; "
            f"row keys={list(row.keys()) if hasattr(row, 'keys') else 'n/a'}"
        )

    sr = entry.source_sr
    wav: np.ndarray

    # Case 1: pre-decoded array (HF sometimes ships one).
    if isinstance(audio, dict) and audio.get("array") is not None:
        wav = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate", sr))
    # Case 2: bytes payload — `soundfile.read` handles FLAC via BytesIO.
    elif isinstance(audio, dict) and audio.get("bytes") is not None:
        with io.BytesIO(audio["bytes"]) as buf:
            wav, file_sr = sf.read(buf, dtype="float32")
        sr = int(file_sr)
    # Case 3: bare on-disk path.
    elif isinstance(audio, dict) and audio.get("path"):
        wav, file_sr = sf.read(audio["path"], dtype="float32")
        sr = int(file_sr)
    else:
        raise RuntimeError(
            f"[default_hf_wav_loader] audio payload has no "
            f"`array`/`bytes`/`path`; got keys="
            f"{list(audio.keys()) if isinstance(audio, dict) else type(audio)}"
        )

    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != target_sr:
        import librosa  # local import; only needed for resample path

        wav = librosa.resample(
            wav, orig_sr=int(sr), target_sr=int(target_sr)
        ).astype(np.float32)

    peak = float(np.abs(wav).max()) if wav.size else 0.0
    if peak > 5.0:
        wav = (wav / peak).astype(np.float32)

    return wav.astype(np.float32, copy=False)


def _sample_activity_burst(
    sample_len: int,
    hp: HParams,
    rng: random.Random,
) -> torch.Tensor:
    """Sample-level 0/1 activity mask with one active burst (deviation 5).

    Burst length uniform in `[burst_min_sec, burst_max_sec]`, drawn
    uniformly within `[0, sample_len]`. Edges fade linearly over
    `activity_fade_ms` (avoids click artefacts).
    """
    burst_min_samples = max(1, int(hp.activity_burst_min_sec * hp.sample_rate))
    burst_max_samples = max(
        burst_min_samples,
        int(hp.activity_burst_max_sec * hp.sample_rate),
    )
    burst_max_samples = min(burst_max_samples, sample_len)
    # Defensive guard: if `sample_len < burst_min_samples` (very short
    # sources), `rng.randint(burst_min_samples, burst_max_samples)` would
    # raise ValueError because the lower bound exceeds the upper bound
    # after the previous `min(..., sample_len)` clamp. Clamp `burst_min`
    # down too so the burst covers the whole source.
    burst_min_samples = min(burst_min_samples, burst_max_samples)
    burst_len = rng.randint(burst_min_samples, burst_max_samples)
    burst_start = rng.randint(0, sample_len - burst_len)
    burst_end = burst_start + burst_len

    fade = max(1, int(hp.activity_fade_ms * 1e-3 * hp.sample_rate))
    fade = min(fade, burst_len // 2)

    act = torch.zeros(sample_len, dtype=torch.float32)
    if fade > 1 and burst_len > 2 * fade:
        ramp = torch.linspace(0.0, 1.0, steps=fade + 1, dtype=torch.float32)[1:]
        act[burst_start : burst_start + fade] = ramp
        act[burst_end - fade : burst_end] = ramp.flip(0)
        act[burst_start + fade : burst_end - fade] = 1.0
    else:
        act[burst_start:burst_end] = 1.0
    return act


def _per_frame_overlap_label(
    target_act: torch.Tensor,
    interference_act: torch.Tensor,
    T_frames: int,
    sample_len: int,
    hp: HParams,
) -> torch.Tensor:
    """Per-frame +1/-1 labels for the noise-type predictor (paper §4.2).

    Frame `t` centre sample = `t * hp.hop_length + hp.n_fft // 2`.
    Label is `+1` iff both target and interference activity masks > 0
    at the centre, else `-1`. Centre indices past `sample_len` clamp
    to the last sample so we always return `(T_frames,)`.
    """
    labels = torch.full((T_frames,), -1, dtype=torch.long)
    centre_idx = (
        torch.arange(T_frames, dtype=torch.long) * hp.hop_length + hp.n_fft // 2
    )
    centre_idx = centre_idx.clamp(max=sample_len - 1)
    tgt_at = target_act[centre_idx] > 0.0
    int_at = interference_act[centre_idx] > 0.0
    both = tgt_at & int_at
    labels[both] = 1
    return labels


# ============================================================ #
# Babble synthesis (module-level for testability)                #
# ============================================================ #


def _pick_babble_speakers(
    rng: random.Random,
    spk_to_entries: dict[str, list[HfEntry]],
    n_speakers_min: int,
    n_speakers_max: int,
) -> list[tuple[str, HfEntry]]:
    """Pick N DISTINCT speakers for one babble mix (no replacement).

    Returns `[(spk_id, entry), ...]` of length N ∈ `[n_speakers_min,
    n_speakers_max]` (clamped to `len(spk_to_entries)`). For each chosen
    speaker one random `HfEntry` from `spk_to_entries[spk]` is selected.

    Splitting the picker from the mixer lets the unit test verify
    "exactly N distinct speakers" without stubbing `sf.read`; the
    iterator calls this then hands the picks to `_mix_babble_from_picks`.
    """
    if not spk_to_entries:
        return []
    n_avail = len(spk_to_entries)
    n_min = max(1, min(int(n_speakers_min), n_avail))
    n_max = max(n_min, min(int(n_speakers_max), n_avail))
    n = rng.randint(n_min, n_max)
    chosen_spks = rng.sample(list(spk_to_entries.keys()), k=n)
    return [(spk, rng.choice(spk_to_entries[spk])) for spk in chosen_spks]


def _mix_babble_from_picks(
    rng: random.Random,
    sample_len: int,
    sr: int,
    picks: list[tuple[str, HfEntry]],
    seg_min_sec: float,
    seg_max_sec: float,
    gain_min: float,
    gain_max: float,
    wav_loader=default_hf_wav_loader,
) -> torch.Tensor:
    """Mix N speakers (already picked) into one `(sample_len,)` wav.

    For each `(spk, entry)`:
      1. Load the wav via `wav_loader(entry, sr)` (default =
         `default_hf_wav_loader` — reads FLAC bytes from HF arrow
         cache, decodes via soundfile, resamples to `sr` if needed).
      2. Pick a random segment length ∈ `[seg_min_sec, seg_max_sec]`
         (clamped to the wav length).
      3. Pick a random start within the wav for that segment.
      4. Pick a random start within `[0, sample_len - seg_len]` for the
         output placement.
      5. Multiply by a uniform-random gain in `[gain_min, gain_max]`.
      6. Sum into the output buffer (overlaps add).

    Clips shorter than `seg_min_sec * sr` samples are skipped — should
    only matter for tiny fixture wavs in tests. Empty input → silence.

    The `wav_loader` hook lets callers with in-memory wavs (e.g.
    unit-test mocks that hand back canned arrays) supply their own
    loader without touching the HF arrow cache.
    """
    out = np.zeros(sample_len, dtype=np.float32)
    if not picks:
        return torch.from_numpy(out)
    seg_min = max(1, int(seg_min_sec * sr))
    seg_max = max(seg_min, int(seg_max_sec * sr))
    for _spk, entry in picks:
        try:
            wav = wav_loader(entry, sr)
        except Exception:
            continue
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = np.asarray(wav, dtype=np.float32)
        if wav.size < seg_min:
            continue
        seg_len = min(rng.randint(seg_min, seg_max), wav.size, sample_len)
        if seg_len < 1:
            continue
        # Use exclusive upper bound (`wav.size - seg_len`) for seg_start
        # and `sample_len - seg_len` for out_start so the slices fit
        # exactly with no numpy clipping — `rng.randint` is inclusive
        # on both ends, so passing `+ 1` here would risk
        # `seg`/`out_slice` getting silently truncated and the +2
        # broadcasting against an off-by-one `seg_len` (regression:
        # shapes (21192,) (21193,) (21192,) at sample_len=21192,
        # wav.size=21193).
        seg_start = rng.randint(0, wav.size - seg_len)
        seg = wav[seg_start : seg_start + seg_len]
        out_start = rng.randint(0, sample_len - seg_len)
        gain = float(rng.uniform(gain_min, gain_max))
        out[out_start : out_start + seg_len] += seg * gain
    return torch.from_numpy(out)


# ============================================================ #
# Energy-budget allocation (module-level for testability)        #
# ============================================================ #


def _allocate_interference_budget(
    components: list[torch.Tensor],
    target_power: float,
    snr_db: float,
    rng: random.Random,
    apply_budget: bool = True,
) -> torch.Tensor:
    """Allocate `target_power / 10^(snr_db/10)` energy across K components.

    Pure helper used by `_sample_interference` and `examples/04_*` for
    the energy-budget fix. Splits the total interference RMS² budget
    across K components via Dirichlet(α=1) sampling on the simplex,
    scales each component so its RMS² = `w_i × budget`, sums them,
    peak-clips each component to ±1.0 before scaling (so the scale
    factor isn't extreme), and tanh-soft-clips the aggregate to ±1.0
    to block downstream `snr_mix` from clipping into wideband
    distortion.

    Post-condition (when `apply_budget=True`):
    `(aggregate ** 2).mean() ≈ target_power / 10^(snr/10)` (within a
    few % of tolerance, modulo the tanh soft-clip when the un-clipped
    SUM's peak exceeds 1.0).

    Args:
        components: K `(sample_len,)` float32 wavs at "natural" energy
            (peak may exceed 1.0; RMS² can span orders of magnitude
            across TYPES — the per-component peak-clip normalises
            this). Empty list → returns zeros.
        target_power: per-time-step target RMS² (positive scalar;
            eps-clamped by the caller).
        snr_db: target SIR in dB. The total interference budget is
            `target_power / 10^(snr_db/10)`.
        rng: deterministic RNG for the Dirichlet draw (K uniforms).
            Caller must pass the same RNG that drove component
            synthesis so budget draws land on the same stream as
            Bernoulli inclusion rolls + component picks (otherwise
            the budget would be re-seeded and replay wouldn't match
            across runs).
        apply_budget: True (default) → allocate per-component budget
            as described. False → skip budget allocation entirely:
            just peak-clip per component, sum, soft-clip. Used when
            the target is silent (no SIR to match; we keep interference
            at audible natural energy so the model sees "interference
            alone, output silence" examples rather than train-as-one).

    Returns:
        `(sample_len,)` float32 aggregate wav, soft-clipped to ±1.0,
        with SUM RMS² ≈ budget (or natural-energy RMS² when
        `apply_budget=False`).
    """
    eps = 1e-12
    K = len(components)
    if K == 0:
        # Defensive: callers (`_sample_interference` in this module and
        # `_resolve_interference` in `examples/04_*`) already guard K=0
        # before calling.  We return a length-0 zero tensor so the
        # return type is consistent; we don't try to peek at
        # `components[0]` (it doesn't exist).
        return torch.zeros(0, dtype=torch.float32)

    # Per-component peak-clip to ±1.0 BEFORE any further work.  Without
    # this, a single transient peak would dominate RMS² and the budget
    # scale would push the rest of the component into the weeds.  Same
    # loop serves both the `apply_budget=False` (natural-energy) branch
    # and the Dirichlet-scaled branch.
    clipped: list[torch.Tensor] = []
    for w in components:
        peak = float(w.abs().max().item())
        if peak > 1.0:
            clipped.append(w / peak)
        else:
            clipped.append(w)

    if not apply_budget:
        # Natural-energy sum: no scaling, no Dirichlet.  Used when the
        # target is silent and there's no SIR to match — the model
        # needs to see real interference to learn "output silence when
        # the target is absent".  Soft-clip the aggregate as usual.
        aggregate = sum(clipped)
        agg_peak = float(aggregate.abs().max().item())
        if agg_peak > 1.0:
            # `tanh(aggregate)` is bounded by [-1, 1] — that's the
            # actual soft-clip to ±1 the docstring promises.  The
            # earlier `tanh(x/peak) * peak` formula only brought peak
            # down to `0.76 × peak`, NOT bounded by ±1.
            aggregate = torch.tanh(aggregate)
        return aggregate

    total_budget = float(target_power) / (10.0 ** (snr_db / 10.0))

    # Per-component RMS² (post peak-clip; eps floor for true-zero components).
    rms2 = [max(eps, float((w ** 2).mean().item())) for w in clipped]

    # Dirichlet(α=1) weights summing to 1.  K=1 collapses to w=1.
    if K == 1:
        weights = [1.0]
    else:
        e = [max(eps, rng.random()) for _ in range(K)]
        s = sum(e)
        weights = [v / s for v in e]

    # Scale each component so its RMS² = w_i × total_budget.
    scaled = [
        w * ((weights[i] * total_budget / rms2[i]) ** 0.5)
        for i, w in enumerate(clipped)
    ]
    aggregate = sum(scaled)

    # Anti-clipping: soft-clip the aggregate to ±1.0 via tanh so
    # `snr_mix` doesn't clip into wideband distortion in the labels.
    agg_peak = float(aggregate.abs().max().item())
    if agg_peak > 1.0:
        # See note above — `tanh(aggregate)` (NOT `tanh(x/peak)*peak`)
        # is the formula that actually bounds the output by ±1.
        aggregate = torch.tanh(aggregate)
    return aggregate


# ============================================================ #
# Module-level interference synthesis (pool-agnostic)              #
# ============================================================ #
#
# These are the same algorithms `HfTripletStream._sample_interference`
# runs, but lifted to module scope so callers that hold their own
# pool structures (e.g. `examples/04_*` after backing in-memory wavs to
# a tmp `pool/` directory) can invoke them without instantiating the
# full `HfTripletStream` dataset. Training and example scripts share
# the same code path → identical sampling statistics.


def _mix_passthrough_v2(
    rng: random.Random,
    sample_len: int,
    pool: list[HfEntry],
    target_sr: int,
    wav_loader=default_hf_wav_loader,
) -> torch.Tensor:
    """`noise` / `music` TYPEs — 1 random clip from `pool`, cropped/padded to `sample_len`.

    Module-level twin of `HfTripletStream._mix_passthrough`. Same
    algorithm, same RNG contract (the `random_crop` call inside
    `random_crop` consumes one RNG draw); the only difference is the
    `wav_loader` hook so callers with in-memory wavs can supply their
    own. Returns zeros when `pool` is empty (defensive).

    `target_sr` is passed to `wav_loader(entry, target_sr)` so MUSAN
    `noise` / `music` clips (44.1 kHz natively) get resampled to the
    model sample rate on read.
    """
    if not pool:
        return torch.zeros(sample_len, dtype=torch.float32)
    entry = rng.choice(pool)
    wav_np = wav_loader(entry, target_sr)
    wav_t = torch.from_numpy(wav_np)
    return random_crop(wav_t, sample_len)


# ============================================================ #
# Simplified mix helpers (v3)                                     #
# ============================================================ #
# These replace the old `_mix_babble_from_picks` /
# `_mix_passthrough_v2` for the categorical-distribution code path
# (`_sample_interference_v3`). Both helpers:
#   * produce a wav of EXACTLY `sample_len` samples;
#   * start at the first sample (no random offset within the target);
#   * keep a random offset WITHIN the source wav for per-batch diversity.
# No burst windows, no segment placement, no activity fades.


def _mix_babble_simple(
    rng: random.Random,
    sample_len: int,
    sr: int,
    picks: list[tuple[str, HfEntry]],
    gain_min: float,
    gain_max: float,
    wav_loader=default_hf_wav_loader,
) -> torch.Tensor:
    """Babble mixer for the v3 path: N picked speakers, full-length output.

    For each `(spk, entry)`:
      1. Load the wav via `wav_loader(entry, sr)`.
      2. Pick a random start within the source wav; take `sample_len`
         samples (zero-pad if shorter).
      3. Multiply by a uniform-random gain in `[gain_min, gain_max]`.
      4. Sum into the output buffer (the babble "speech-like" content
         of every speaker fills the whole window from sample 0).

    Output is always `(sample_len,)` float32 — no random offset inside
    the target window, no segment-length variation. The dataset is
    still diverse because each pick's source-wav offset varies.

    Empty `picks` → silence (matches `_mix_babble_from_picks`).
    """
    out = np.zeros(sample_len, dtype=np.float32)
    if not picks:
        return torch.from_numpy(out)
    for _spk, entry in picks:
        try:
            wav = wav_loader(entry, sr)
        except Exception:
            continue
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if wav.size == 0:
            continue
        if wav.size >= sample_len:
            start = rng.randint(0, wav.size - sample_len)
            seg = wav[start : start + sample_len]
        else:
            seg = np.pad(wav, (0, sample_len - wav.size))
        gain = float(rng.uniform(gain_min, gain_max))
        out += seg * gain
    return torch.from_numpy(out)


def _mix_passthrough_simple(
    rng: random.Random,
    sample_len: int,
    pool: list[HfEntry],
    target_sr: int,
    wav_loader=default_hf_wav_loader,
) -> torch.Tensor:
    """`noise` / `music` mixer for the v3 path: 1 random clip, full-length.

    Picks a random entry from `pool`, then a random start within the
    source wav, takes `sample_len` samples (zero-pad if shorter), and
    returns. Output always occupies `[0, sample_len]` of the target
    window — no `random_crop` placement choice (the helper simplified away
    from the v2 random_crop helper).

    Empty pool → silence (defensive; the v3 sampler guards against
    this before calling).
    """
    if not pool:
        return torch.zeros(sample_len, dtype=torch.float32)
    entry = rng.choice(pool)
    wav_np = wav_loader(entry, target_sr)
    wav = np.asarray(wav_np, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if wav.size == 0:
        return torch.zeros(sample_len, dtype=torch.float32)
    if wav.size >= sample_len:
        start = rng.randint(0, wav.size - sample_len)
        seg = wav[start : start + sample_len]
    else:
        seg = np.pad(wav, (0, sample_len - wav.size))
    return torch.from_numpy(seg)


# ============================================================ #
# Categorical interference sampler (v3)                            #
# ============================================================ #


def _draw_interference_category(
    none_weight: float,
    single_weights: dict[str, float],
    mixed_weight: float,
    rng: random.Random,
) -> tuple[str, tuple[str, ...]]:
    """Draw one bucket from the categorical interference distribution.

    Returns `(bucket_label, included_kinds)`:
      * `("none", ())`                       — no interference (silent).
      * `("single", ("babble",))`            — pure single TYPE.
      * `("mixed", ("babble", "noise"))`     — 2+ TYPES stacked under
                                               the energy-budget allocator.

    The `included_kinds` tuple is in canonical order (`babble`, `noise`,
    `music`) so test assertions and per-bucket summaries stay stable.

    Distribution contract (caller's responsibility): the four weight
    arguments must sum to 1.0 (we renormalise defensively to keep the
    behaviour well-defined if a config drifts). Empty single_weights
    with a non-zero `none_weight + mixed_weight` is allowed (silently
    routes the freed weight to the others).
    """
    total = float(none_weight) + float(sum(single_weights.values())) + float(mixed_weight)
    if total <= 0:
        # Degenerate config: collapse to "no interference".
        return ("none", ())
    # Renormalise (defensive — see contract above).
    p_none = max(0.0, float(none_weight)) / total
    p_single_total = max(0.0, float(sum(single_weights.values()))) / total
    p_mixed = max(0.0, float(mixed_weight)) / total

    u = rng.random()
    if u < p_none:
        return ("none", ())
    if u < p_none + p_mixed:
        # Mixed bucket — defer TYPE picking to the caller (it needs the
        # pool sizes + the multi-type_prob curriculum knob to decide how
        # many TYPES to draw). Here we only mark the bucket.
        return ("mixed", ())
    # Single bucket — pick one TYPE proportional to its weight.
    bucket: list[str] = []
    weights: list[float] = []
    for name, weight in single_weights.items():
        if weight <= 0:
            continue
        bucket.append(name)
        weights.append(float(weight))
    if not bucket:
        # No singles eligible → fall back to "none". Caller can detect
        # via the ("none", ()) return.
        return ("none", ())
    s = sum(weights)
    pick = rng.choices(bucket, weights=[w / s for w in weights], k=1)[0]
    return ("single", (pick,))


def _draw_mixed_kinds(
    single_weights: dict[str, float],
    rng: random.Random,
    *,
    eligible: dict[str, bool],
) -> tuple[str, ...]:
    """Draw a fixed-size composite bucket from `eligible` TYPEs.

    Always returns 2 or 3 TYPES (never 1 — single-type is a separate
    bucket). `eligible[name]` says whether TYPE `name` has a non-empty
    pool (skip silently otherwise). TYPEs with `single_weights[name] <= 0`
    are also skipped.

    IMPORTANT: this uses **weighted sampling WITHOUT replacement**, so
    the returned TYPEs are always distinct. (`rng.choices(..., k=n)`
    samples with replacement and produces duplicates for high-weight
    TYPEs — e.g. babble weighted 4× noise/music gets picked twice in a
    2-draw with prob ~44%, and `set()`-dedup would silently downgrade
    a "mixed" pick to a single. The bug this left — babble single rate
    inflated, mixed rate deflated — was visible in the v3 smoke test
    empirical distribution; we now do weighted reservoir sampling
    inline.)

    Returns kinds in canonical order (`babble`, `noise`, `music`) so the
    tuple is directly usable as the `noise_kind` tag.
    """
    bucket: list[str] = []
    weights: list[float] = []
    for name in (NOISE_KIND_BABBLE, NOISE_KIND_NOISE, NOISE_KIND_MUSIC):
        if not eligible.get(name, False):
            continue
        w = float(single_weights.get(name, 0.0))
        if w <= 0:
            continue
        bucket.append(name)
        weights.append(w)
    if len(bucket) < 2:
        # Fallback: not enough eligible TYPES for a composite. Caller
        # should re-roll — returning the singleton here lets the caller
        # detect this without a separate return channel.
        return tuple(bucket)

    # Weighted sampling without replacement via sequential inverse-CDF.
    # For each pick, draw u ~ Uniform(0, sum_w) and find the bucket whose
    # cumulative weight crosses u; remove it and continue. Costs O(n * K)
    # for n picks over K bucket entries — fine for n ∈ {2, 3}, K ≤ 3.
    n = rng.randint(2, min(3, len(bucket)))
    remaining_names = list(bucket)
    remaining_weights = list(weights)
    picks: list[str] = []
    for _ in range(n):
        total = sum(remaining_weights)
        if total <= 0:
            break  # Defensive — shouldn't happen given the pre-filter.
        target = rng.random() * total
        cumsum = 0.0
        chosen_idx = len(remaining_weights) - 1  # default to last on FP edge
        for i, w in enumerate(remaining_weights):
            cumsum += w
            if cumsum >= target:
                chosen_idx = i
                break
        picks.append(remaining_names[chosen_idx])
        del remaining_names[chosen_idx]
        del remaining_weights[chosen_idx]

    canonical = {NOISE_KIND_BABBLE: 0, "noise": 1, "music": 2}
    return tuple(sorted(set(picks), key=lambda k: canonical[k]))


def _sample_interference_v3(
    hp: HParams,
    rng: random.Random,
    sample_len: int,
    target_power: float,
    snr_db: float,
    *,
    babble_spk_to_entries: dict[str, list[HfEntry]],
    noise_pool: list[HfEntry],
    music_pool: list[HfEntry],
    babble_cfg,
    noise_cfg,
    music_cfg,
    multi_type_prob: float,
    wav_loader=default_hf_wav_loader,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Categorical sampler with curriculum-controlled mixed weight.

    Replacement for `_sample_interference_v2` — instead of independent
    per-TYPE Bernoullis, the sampler draws ONE bucket from a 3-way
    categorical (`none` / `single` / `mixed`), then:

        * `none`  → zeros.
        * `single`→ draw one TYPE proportional to its single-weight;
                     mix its full-length component (no random segment
                     placement) at the energy budget.
        * `mixed` → draw 2-3 distinct TYPES proportional to their single
                     weights; stack them via `_allocate_interference_budget`
                     (peak-clip per component, Dirichlet split across the
                     K components, tanh soft-clip of the aggregate).

    `multi_type_prob` (∈ [0, 1]) is the curriculum knob for the mixed
    bucket. The sampler linearly blends:

        effective_mixed = mixed_weight * multi_type_prob
        effective_singles = single_weights × (1 - effective_mixed)
                                       / (1 - mixed_weight)

    so that at `multi_type_prob=0` the mixed bucket is fully suppressed
    (mass redistributed onto the singles, keeping the no-interference
    fraction fixed), and at `multi_type_prob=1` the distribution
    matches the config. Single-type mass is preserved across the
    curriculum so early training still shows the model a balanced
    single-type diet.

    The TYPE iteration order for both single picking and mixed picking is
    fixed (`babble`, `noise`, `music`) so test assertions / per-bucket
    summaries stay reproducible across runs given the same RNG seed.
    """
    none_w = float(hp.interference_none_weight)
    singles = {k: float(v) for k, v in hp.interference_single_weights.items()}
    mixed_w = float(hp.interference_mixed_weight)

    # Curriculum: scale the mixed bucket by `multi_type_prob`. The freed
    # mass is added to the singles proportionally (so the no-interference
    # fraction stays fixed across the curriculum).
    mtp = max(0.0, min(1.0, float(multi_type_prob)))
    effective_mixed = mixed_w * mtp
    if mixed_w > 0:
        scale = (mixed_w - effective_mixed) / mixed_w  # mass to redistribute
    else:
        scale = 0.0
    for k in singles:
        singles[k] = singles[k] + singles[k] * scale  # = singles[k] / (1 - mtp) if mixed_w > 0

    # Eligibility check (skip TYPES with empty pools in v3 — they would
    # just produce silence and waste a Bernoulli roll).
    has_babble = bool(babble_spk_to_entries) and babble_cfg is not None and babble_cfg.enabled
    has_noise = bool(noise_pool) and noise_cfg is not None and noise_cfg.enabled
    has_music = bool(music_pool) and music_cfg is not None and music_cfg.enabled

    # Suppress ineligible TYPES from the sampler (silently zero their
    # weight so the categorical falls back to the eligible alternatives).
    eligible = {
        NOISE_KIND_BABBLE: has_babble,
        NOISE_KIND_NOISE: has_noise,
        NOISE_KIND_MUSIC: has_music,
    }
    if not has_babble:
        singles[NOISE_KIND_BABBLE] = 0.0
    if not has_noise:
        singles[NOISE_KIND_NOISE] = 0.0
    if not has_music:
        singles[NOISE_KIND_MUSIC] = 0.0

    # If the curriculum fully suppresses the mixed bucket AND no singles
    # are eligible, the sampler collapses to "none" — return zeros.
    any_single = any(singles[k] > 0 for k in singles)
    if effective_mixed <= 0 and not any_single:
        return torch.zeros(sample_len, dtype=torch.float32), ()

    # Renormalise the effective weights (defensive — config drift).
    total = none_w + sum(singles.values()) + effective_mixed
    if total <= 0:
        return torch.zeros(sample_len, dtype=torch.float32), ()
    p_none = none_w / total
    p_mixed = effective_mixed / total

    u = rng.random()
    if u < p_none:
        return torch.zeros(sample_len, dtype=torch.float32), ()
    if u < p_none + p_mixed:
        # Mixed bucket — pick 2-3 TYPES.
        kinds = _draw_mixed_kinds(
            singles, rng, eligible=eligible,
        )
        if len(kinds) < 2:
            # Fallback: not enough eligible TYPES → re-roll as a single.
            kinds = (kinds[0],) if kinds else ()
    else:
        # Single bucket — pick one TYPE.
        pool = [(k, w) for k, w in singles.items() if w > 0]
        if not pool:
            return torch.zeros(sample_len, dtype=torch.float32), ()
        names = [p[0] for p in pool]
        weights = [p[1] / sum(p[1] for p in pool) for p in pool]
        kind = rng.choices(names, weights=weights, k=1)[0]
        kinds = (kind,)

    if not kinds:
        return torch.zeros(sample_len, dtype=torch.float32), ()

    # ---- Synthesise the picked TYPES ----
    components: list[torch.Tensor] = []
    for kind in kinds:
        if kind == NOISE_KIND_BABBLE:
            picks = _pick_babble_speakers(
                rng=rng,
                spk_to_entries=babble_spk_to_entries,
                n_speakers_min=babble_cfg.n_speakers_min,
                n_speakers_max=babble_cfg.n_speakers_max,
            )
            component = _mix_babble_simple(
                rng=rng,
                sample_len=sample_len,
                sr=hp.sample_rate,
                picks=picks,
                gain_min=babble_cfg.gain_min,
                gain_max=babble_cfg.gain_max,
                wav_loader=wav_loader,
            )
        elif kind == NOISE_KIND_NOISE:
            component = _mix_passthrough_simple(
                rng, sample_len, noise_pool, hp.sample_rate, wav_loader,
            )
        elif kind == NOISE_KIND_MUSIC:
            component = _mix_passthrough_simple(
                rng, sample_len, music_pool, hp.sample_rate, wav_loader,
            )
        else:
            continue
        components.append(component)

    K = len(components)
    if K == 0:
        return torch.zeros(sample_len, dtype=torch.float32), ()

    apply_budget = target_power > 1e-9
    interference_wav = _allocate_interference_budget(
        components=components,
        target_power=target_power,
        snr_db=snr_db,
        rng=rng,
        apply_budget=apply_budget,
    )
    return interference_wav, tuple(kinds)


def _sample_interference_v2(
    hp: HParams,
    rng: random.Random,
    sample_len: int,
    target_power: float,
    snr_db: float,
    *,
    babble_spk_to_entries: dict[str, list[HfEntry]],
    noise_pool: list[HfEntry],
    music_pool: list[HfEntry],
    babble_cfg,
    noise_cfg,
    music_cfg,
    wav_loader=default_hf_wav_loader,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Module-level twin of `HfTripletStream._sample_interference`.

    Rolls each TYPE's independent Bernoulli `prob`, synthesises the
    included components, then delegates energy-budget allocation to
    `_allocate_interference_budget` (peak-clip per component, Dirichlet
    split, tanh soft-clip of the aggregate). Returns `(interference_wav,
    included_kinds)` exactly like the instance method.

    The TYPE iteration order is fixed (babble → noise → music) so test
    assertions / per-bucket summaries are reproducible across runs given
    the same RNG seed.

    Args:
        babble_spk_to_entries: pooled babble pool (spk_id → list of
            `HfEntry`s). `babble_cfg.sources` decides which sources
            feed in; this map is the union. `_pick_babble_speakers`
            ensures N distinct identities.
        noise_pool / music_pool: passthrough pools (`_mix_passthrough_v2`
            picks one clip + random-crop).
        babble_cfg / noise_cfg / music_cfg: `InterferenceTypeCfg` objects
            from `hp.interference_types[<name>]` — `prob`, segment
            lengths, gain range, etc. `None` is treated as disabled.
        wav_loader: `(HfEntry, target_sr) → np.ndarray (T,)` hook.
            Default = `default_hf_wav_loader` (reads FLAC bytes from
            HF arrow cache, decodes via soundfile). Example scripts
            with in-memory wavs pass their own loader.
    """
    components: list[torch.Tensor] = []
    included: list[str] = []

    def _has_clips(type_name: str, cfg) -> bool:
        if cfg is None or not cfg.enabled or float(cfg.prob) <= 0.0:
            return False
        if type_name == NOISE_KIND_BABBLE:
            return bool(babble_spk_to_entries)
        if type_name == NOISE_KIND_NOISE:
            return bool(noise_pool)
        if type_name == NOISE_KIND_MUSIC:
            return bool(music_pool)
        return False

    for type_name in (NOISE_KIND_BABBLE, NOISE_KIND_NOISE, NOISE_KIND_MUSIC):
        cfg = {
            NOISE_KIND_BABBLE: babble_cfg,
            NOISE_KIND_NOISE: noise_cfg,
            NOISE_KIND_MUSIC: music_cfg,
        }[type_name]
        if not _has_clips(type_name, cfg):
            continue
        # Independent Bernoulli inclusion roll per TYPE.
        if rng.random() >= float(cfg.prob):
            continue
        if type_name == NOISE_KIND_BABBLE:
            picks = _pick_babble_speakers(
                rng=rng,
                spk_to_entries=babble_spk_to_entries,
                n_speakers_min=cfg.n_speakers_min,
                n_speakers_max=cfg.n_speakers_max,
            )
            component = _mix_babble_from_picks(
                rng=rng,
                sample_len=sample_len,
                sr=hp.sample_rate,
                picks=picks,
                seg_min_sec=cfg.babble_seg_min_sec,
                seg_max_sec=cfg.babble_seg_max_sec,
                gain_min=cfg.gain_min,
                gain_max=cfg.gain_max,
                wav_loader=wav_loader,
            )
        elif type_name == NOISE_KIND_NOISE:
            component = _mix_passthrough_v2(
                rng, sample_len, noise_pool, hp.sample_rate, wav_loader,
            )
        elif type_name == NOISE_KIND_MUSIC:
            component = _mix_passthrough_v2(
                rng, sample_len, music_pool, hp.sample_rate, wav_loader,
            )
        else:
            continue  # unreachable; defensive
        components.append(component)
        included.append(type_name)

    K = len(components)
    if K == 0:
        return torch.zeros(sample_len, dtype=torch.float32), ()

    # Silent-target branch (matches the instance method):
    # `target_power ≤ 1e-9` → keep interference at NATURAL energy
    # (the model needs to see real interference to learn "output silence
    # when target is absent").
    apply_budget = target_power > 1e-9
    interference_wav = _allocate_interference_budget(
        components=components,
        target_power=target_power,
        snr_db=snr_db,
        rng=rng,
        apply_budget=apply_budget,
    )
    return interference_wav, tuple(included)


# ============================================================ #
# synthesize_raw_sample — single source of truth for per-sample   #
# WAV synthesis (training + audition scripts share this).        #
# ============================================================ #


def synthesize_raw_sample(
    hp: HParams,
    rng: random.Random,
    *,
    target_wav_full: torch.Tensor,
    ref_wav_full: torch.Tensor,
    babble_spk_to_entries: dict[str, list[HfEntry]],
    noise_pool: list[HfEntry],
    music_pool: list[HfEntry],
    type_probs: dict[str, float],
    babble_cfg,
    noise_cfg,
    music_cfg,
    multi_type_prob: float = 1.0,
    wav_loader=default_hf_wav_loader,
) -> RawSample:
    """One end-to-end per-sample WAV synthesis — mirrors `HfTripletStream.__iter__`.

    v3 path. The caller picks the target speaker/utterance and the
    reference utterance (training does this from a pre-baked index;
    `examples/04_*` does this from an HF stream). This function takes
    those two pre-loaded wav tensors + the pool structures and returns
    the WAV-level outputs the training iterator would feed into the
    log-mel pipeline.

    Simplifications vs. the v2 path (kept here for backward compat in
    the docstring; v2 still exists for tests but is no longer called by
    the training iterator):
      * Interference fills `[0, sample_len]` from sample 0 — no random
        placement, no segment-length choice, no activity burst on the
        interference side.
      * Interference category is drawn from the categorical
        `interference_distribution` (none / single / mixed) by
        `_sample_interference_v3`. The `multi_type_prob` curriculum knob
        smoothly ramps the mixed-bucket weight from `multi_type_start_prob`
        to `multi_type_end_prob` over `multi_type_warmup_steps` so
        early training shows the model only single-type interference.
      * Target keeps its burst-mask (predictor labels need the
        structure). Interference has no burst (it always covers the
        whole window).
      * `type_probs` is kept for backward-compat signature symmetry
        but is no longer used (the categorical distribution replaces the
        independent-Bernoulli sampler).

    Steps (in order — same RNG semantics as training):
      1. Sample activity burst on the **FULL target audio** and apply
         it, then crop to `sample_len` at a position that **OVERLAPS
         with the burst** (deviation 5 fix — burst always lands on real
         audio energy). Burst mask stays sparse (never collapses to 1).
      2. Cap `ref_wav_full` at `hp.ref_max_sec * sample_rate` samples
         (left-truncate, no random offset — matches training).
      3. `target_power = mean(target²)` on the burst-masked crop,
         eps-clamped to 1e-12.
      4. Draw `snr_db ∈ [snr_min_db, snr_max_db]` UNCONDITIONALLY
         (silent flags are rolled AFTER synthesis so the categorical
         draw lands on the same RNG state regardless of the silent
         outcome).
      5. `_sample_interference_v3(...)` — categorical draw; for
         non-none buckets, synthesise each picked TYPE (v3 simplified
         helpers: `_mix_babble_simple` / `_mix_passthrough_simple`)
         and stack via `_allocate_interference_budget`.
      6. Roll silent flags (target_silent / interference_silent).
         `interference_silent=True` overrides the synthesised wav with
         zeros; `included_kinds` still reports the categorical draw.
      7. No interference activity burst (v3 always full-length). Set
         `interference_act = ones` so downstream `_per_frame_overlap_label`
         computes the labels purely from `target_act`.
      8. Defensive zero-out when silent.
      9. `snr_mix` branch:
            both silent         → zeros
            target_silent       → interference.clone()
            interference_silent → target.clone()
            else                → snr_mix(target, interference, snr)

    Returns a `RawSample` carrying the wavs, activity masks, ref wav,
    snr draw, included kinds, and the silent flags.
    """
    sample_len = hp.sample_length_samples

    # ---- 1. Target activity burst on FULL audio, then constrained crop ----
    # Deviation 5 fix: sample the burst on the FULL source first so the
    # burst always overlaps with real audio energy. Then sample a crop
    # start that lies inside the burst's overlap range, so the cropped
    # window still contains that voice. Burst mask stays sparse.
    T_full = int(target_wav_full.shape[-1])
    target_act_full = _sample_activity_burst(T_full, hp, rng)
    target_wav_full_masked = target_wav_full * target_act_full

    if T_full <= sample_len:
        # Source shorter than (or equal to) sample_len: keep the whole
        # source, zero-pad to sample_len.
        crop_start = 0
        if T_full < sample_len:
            pad = torch.zeros(
                sample_len - T_full,
                dtype=target_wav_full_masked.dtype,
            )
            target_wav = torch.cat([target_wav_full_masked, pad])
            target_act = torch.cat([target_act_full, pad])
        else:
            target_wav = target_wav_full_masked
            target_act = target_act_full
    else:
        burst_active = (target_act_full > 0.5).nonzero(as_tuple=True)[0]
        if burst_active.numel() == 0:
            crop_start = 0
        else:
            burst_start_full = int(burst_active[0].item())
            burst_end_full = int(burst_active[-1].item()) + 1
            crop_min = max(0, burst_end_full - sample_len)
            crop_max = min(T_full - sample_len, burst_start_full)
            crop_start = rng.randint(crop_min, max(crop_min, crop_max))
        target_wav = target_wav_full_masked[crop_start : crop_start + sample_len]
        target_act = target_act_full[crop_start : crop_start + sample_len]

    # ---- 1b. Per-sample target RMS normalisation (opt-in) ----
    # `target_rms_norm_db` is None → pass-through (legacy behaviour).
    # When enabled, scale the (burst-masked) crop so its RMS hits the
    # configured dBFS target. This makes the downstream interference
    # energy budget `target_power / 10^(snr/10)` a constant per drawn
    # SNR, eliminating the 1-2 order-of-magnitude per-batch loss
    # variance that comes from raw LibriSpeech loudness.
    #   * Skipped when the wav is silent / near-silent (current_rms ≤
    #     1e-4) so the silent-target branch (step 5) keeps working
    #     without divide-by-zero.
    #   * Defensive peak-cap at 1.0 if RMS norm pushed a transient
    #     over 1.0 (very-low-RMS LibriSpeech samples can have spikey
    #     transients; 0.1 RMS × 10 crest factor → peak ~1.0 normally).
    #   * Does NOT touch `target_act` (the activity mask drives the
    #     per-frame overlap label downstream).
    if hp.target_rms_norm_db is not None and target_wav.numel() > 0:
        cur_rms = float(target_wav.pow(2).mean().sqrt().item())
        if cur_rms > 1e-4:
            tgt_rms = float(10.0 ** (hp.target_rms_norm_db / 20.0))
            target_wav = target_wav * (tgt_rms / cur_rms)
            peak = float(target_wav.abs().max().item())
            if peak > 1.0:
                target_wav = target_wav / peak

    # ---- 2. Cap ref_wav at `ref_max_sec` samples (matches training).
    ref_max_len = int(hp.ref_max_sec * hp.sample_rate)
    if ref_wav_full.shape[-1] > ref_max_len:
        ref_wav_capped = ref_wav_full[..., :ref_max_len]
    else:
        ref_wav_capped = ref_wav_full

    # ---- 3. target_power on burst-masked crop ----
    target_power = max(1e-12, float((target_wav ** 2).mean().item()))

    # ---- 4. SNR draw + interference synthesis (v3 categorical) ----
    snr_db = float(rng.uniform(hp.snr_min_db, hp.snr_max_db))
    interference_wav, included_kinds = _sample_interference_v3(
        hp, rng, sample_len, target_power, snr_db,
        babble_spk_to_entries=babble_spk_to_entries,
        noise_pool=noise_pool,
        music_pool=music_pool,
        babble_cfg=babble_cfg,
        noise_cfg=noise_cfg,
        music_cfg=music_cfg,
        multi_type_prob=multi_type_prob,
        wav_loader=wav_loader,
    )

    # ---- 5. Silent flags (rolled AFTER synthesis) ----
    target_silent = rng.random() < hp.silent_target_prob
    interference_silent = rng.random() < hp.silent_interference_prob
    if interference_silent:
        interference_wav = torch.zeros(sample_len, dtype=torch.float32)

    # ---- 6. Activity bursts (target applied above; v3 interference has no burst) ----
    if target_silent:
        # Target was burst-masked at step 1; reset to zeros so downstream
        # `_per_frame_overlap_label` sees no activity.
        target_act = torch.zeros(sample_len, dtype=target_wav.dtype)

    # v3: interference always full-length → `interference_act = ones`.
    # (Or zeros if the silent flag fired.) The `+ 1.0 / - 1.0` tags the
    # state so `_per_frame_overlap_label` can detect "interference
    # silent" without scanning `interference_silent` separately.
    if interference_silent:
        interference_act = torch.zeros(sample_len, dtype=interference_wav.dtype)
    else:
        interference_act = torch.ones(sample_len, dtype=interference_wav.dtype)

    # ---- 7. Defensive zero-out (silent flag fired → wav still carries
    # pre-burst values; zero it explicitly). ----
    if target_silent:
        target_wav = torch.zeros_like(target_wav)
    if interference_silent:
        interference_wav = torch.zeros_like(interference_wav)

    # ---- 8. SNR mix (silent-side bypasses) ----
    if target_silent and interference_silent:
        mixed_wav = torch.zeros_like(target_wav)
    elif target_silent:
        mixed_wav = interference_wav.clone()
    elif interference_silent:
        mixed_wav = target_wav.clone()
    else:
        mixed_wav = snr_mix(target_wav, interference_wav, snr_db)

    return RawSample(
        target_wav=target_wav,
        interference_wav=interference_wav,
        mixed_wav=mixed_wav,
        target_act=target_act,
        interference_act=interference_act,
        ref_wav=ref_wav_capped,
        snr_db=snr_db,
        included_kinds=included_kinds,
        target_silent=target_silent,
        interference_silent=interference_silent,
    )


# ============================================================ #
# DVectorCache — lazy per-speaker d-vector with disk persistence   #
# ============================================================ #


# Module-level cache of `(encoder, mtime_ns, size) → SpeakerEncoder`.
# One process holds at most one SpeakerEncoder instance per unique
# (pt_path, mtime_ns, size) triple — every `DVectorCache` instance in
# the same process reuses it. Multi-worker DataLoader workers each get
# their own process and therefore their own encoder (cheap: ~16 MB).
_ENCODER_SINGLETON: dict[tuple, object] = {}


class DVectorCache:
    """Lazy per-speaker d-vector computation with on-disk cache.

    Replaces the old `voicefilter-cli dataset prepare` d-vector step:
    instead of pre-baking one `.npy` per speaker before training, we
    fill the cache on first sight of each new speaker during training
    iteration. Disk path is
    `<spec.cache_dir>/voicefilter-dvectors/<spec_id>/<spk>.npy` where
    `spec_id` is the `HfSourceSpec.id` of the source the speaker came
    from (e.g. `voicefilter-dvectors/librispeech/<spk>.npy` for the
    default LibriSpeech target). Multi-source targets get separate
    partitions (no speaker-id collision across sources in practice).
    `cache_dir` is taken from the resolved spec so the d-vector cache
    lives next to the source's HF arrow cache (no central `processed_dir`).

    Lookup order (cheapest first):
        1. In-process dict (`_mem_cache`) — O(1), µs.
        2. On-disk file — one `np.load`.
        3. Compute — read the speaker's reference `HfEntry`, run
           `extract_dvec(...)`, atomic-rename-save to disk.

    Concurrency:
        * In-process: `_mem_cache` is plain dict access — safe (CPython
          GIL). Encoder is a module-level singleton keyed by
          `(pt_path, mtime_ns, size)` so multiple `HfTripletStream`
          instances share weights.
        * Cross-process (DataLoader workers): each worker has its own
          `_mem_cache` and its own encoder singleton. The disk write
          uses `np.save(tmp.pid)` + `os.replace()` for atomicity; if two
          workers race to compute the same new speaker, both compute
          (one encoder forward is ~50 ms) and the loser's rename is a
          no-op against the winner's file. After that, every worker
          sees the same disk file and falls through to the cheap path.

    Failure mode:
        * No reference entry for the speaker → zeros + WARN.
        * Encoder weights missing at `hp.dvector_pt_path` → random init
          + WARN; the d-vector is uninformative but training still runs.
        * Any compute exception → zeros + WARN. (Training would
          otherwise crash on the first encounter.)

    Cache invalidation: keyed only by (spec_id, speaker id). If you
    retrain the d-vector encoder (`voicefilter-cli train dvector`),
    delete `<cache_dir>/voicefilter-dvectors/` to force recomputation.
    Future improvements (mtime-based invalidation) tracked separately.
    """

    def __init__(
        self,
        hp: HParams,
        target_index: dict[str, list[HfEntry]],
        spk_to_spec_id: Optional[dict[str, str]] = None,
        cache_root: Optional[Path] = None,
    ):
        self.hp = hp
        # spk → spec_id map built by `HfTripletStream._scan_hf_datasets`
        # (falls back to an empty dict when target is empty / not scanned yet).
        self._spk_to_spec_id: dict[str, str] = dict(spk_to_spec_id or {})
        # Default cache root = first target spec's cache_dir. When no
        # target exists (empty unit-test fixture) the caller can pass a
        # fallback path explicitly; otherwise we use cwd/.cache/voicefilter-dvectors.
        if cache_root is None:
            first_spec = self._first_target_spec()
            if first_spec is not None:
                cache_root = Path(first_spec.cache_dir)
            else:
                cache_root = Path(".cache") / DVEC_DIRNAME
        self.dvec_root = Path(cache_root) / DVEC_DIRNAME
        self.dvec_root.mkdir(parents=True, exist_ok=True)
        # Stable reference utterance per speaker (sorted HfEntry list → [0]).
        # Mirrors the old `_bake_dvectors` reference-wav choice so cached
        # d-vectors are byte-compatible with what the previous pipeline
        # would have produced.
        self._ref_for: dict[str, HfEntry] = {
            spk: entries[0] for spk, entries in target_index.items() if entries
        }
        # In-process LRU-by-construction (we never evict; spk count is
        # bounded by LibriSpeech). ~256 KB at 1000 speakers × 1 KB.
        self._mem_cache: dict[str, np.ndarray] = {}

    # ---- public API ----

    def get_or_compute(self, spk: str) -> np.ndarray:
        """Return `(hp.d_vector_dim,)` float32 d-vector for `spk`.

        On any failure path (no ref entry, encoder error, …) returns a
        zero vector and prints one WARN the first time per speaker.
        """
        if spk in self._mem_cache:
            return self._mem_cache[spk]
        path = self._path(spk)
        if path.is_file() and path.stat().st_size > 0:
            try:
                v = np.load(str(path))
                if v.shape == (self.hp.d_vector_dim,):
                    self._mem_cache[spk] = v
                    return v
            except Exception:
                pass  # corrupt → recompute
        v = self._compute(spk)
        self._save(spk, v)
        self._mem_cache[spk] = v
        return v

    # ---- internals ----

    def _first_target_spec(self) -> Optional["ResolvedSource"]:
        """Return the first declared target spec.

        Used to pick the d-vector cache root when no `cache_root` is
        passed explicitly. Returns `None` when `hp.target_sources` is
        empty or the first ref doesn't resolve to a known spec.
        """
        if not self.hp.target_sources:
            return None
        try:
            return resolve_source(self.hp, self.hp.target_sources[0])
        except KeyError:
            return None

    def _spec_id_for(self, spk: str) -> str:
        """Return the spec_id that owns `spk`'s d-vector partition.

        Lookup order:
          1. The spk → spec_id map built at scan time (correct for
             multi-source targets where the speaker namespace is
             partitioned by source).
          2. The first declared target's spec.id (back-compat shim —
             when only one target source exists this matches #1).
          3. The literal string `"unknown"` (defensive fallback so the
             cache keeps working even on a misconfigured run).
        """
        sid = self._spk_to_spec_id.get(spk)
        if sid is not None:
            return sid
        first = self._first_target_spec()
        if first is not None:
            return first.id
        return "unknown"

    def _path(self, spk: str) -> Path:
        """`<cache_root>/voicefilter-dvectors/<spec_id>/<spk>.npy`.

        `spec_id` is looked up per-speaker so multi-source targets
        partition cleanly. Falls back to the first target's spec_id
        when the speaker wasn't seen at scan time.
        """
        sid = self._spec_id_for(spk)
        return self.dvec_root / sid / f"{spk}.npy"

    def _encoder(self):
        """Return (process-singleton) SpeakerEncoder for `hp.dvector_pt_path`.

        Falls back to a randomly-initialised frozen encoder with a WARN
        if the pretrained weights file is missing — same behaviour as
        the old prepare step, so training can run on machines without
        `models/embedder.pt`.
        """
        from voicefilter_lite.speaker_encoder import SpeakerEncoder

        pt = Path(self.hp.dvector_pt_path)
        try:
            st = pt.stat()
            key = (str(pt.resolve()), st.st_mtime_ns, st.st_size)
        except OSError:
            key = (str(pt.resolve()), -1, -1)
        cached = _ENCODER_SINGLETON.get(key)
        if cached is not None:
            return cached
        enc = SpeakerEncoder(self.hp).to("cpu").eval()
        if pt.is_file():
            try:
                enc.load_pretrained(str(pt))
                enc.freeze()
                # (Removed: `[dvec-cache] loaded SpeakerEncoder from {pt}`)
            except Exception as e:
                # (Removed: WARN print — per training-log-only policy.
                # The fallback below is identical: load fails → random-init encoder.)
                enc.freeze()
        else:
            # (Removed: WARN print — per training-log-only policy.)
            enc.freeze()
        _ENCODER_SINGLETON[key] = enc
        return enc

    def _compute(self, spk: str) -> np.ndarray:
        """Read `spk`'s reference HfEntry → `extract_dvec` → (d_dim,) np."""
        ref = self._ref_for.get(spk)
        if ref is None:
            # (Removed: WARN print — per training-log-only policy.
            # Behaviour is identical: silently return zero d-vector.)
            return np.zeros(self.hp.d_vector_dim, dtype=np.float32)
        from voicefilter_lite.audio import extract_dvec

        wav = default_hf_wav_loader(ref, self.hp.sample_rate)
        wav_t = torch.from_numpy(wav.astype(np.float32))
        cap = int(self.hp.ref_max_sec * self.hp.sample_rate)
        if wav_t.shape[-1] > cap:
            wav_t = wav_t[..., :cap]
        enc = self._encoder()
        with torch.no_grad():
            d = extract_dvec(wav_t, self.hp, enc, device=torch.device("cpu"))
        return d.detach().cpu().numpy().reshape(-1).astype(np.float32)

    def _save(self, spk: str, vec: np.ndarray) -> None:
        """Atomic write: tmp.<pid>.npy → os.replace → final.

        Multi-worker safe: if another worker beat us to the rename,
        the loser's tmp file just becomes a dangling file we clean up.

        Implementation note: we serialise via a `BytesIO` instead of
        `np.save(path, ...)` because numpy auto-appends `.npy` to any
        path that doesn't end with one. Using an in-memory buffer + a
        hand-picked `.npy` tmp suffix avoids that footgun.
        """
        import io

        path = self._path(spk)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.tmp.{os.getpid()}.npy")
        buf = io.BytesIO()
        np.save(buf, vec)
        try:
            tmp.write_bytes(buf.getvalue())
            os.replace(str(tmp), str(path))
        except FileNotFoundError:
            # Another worker renamed first → nothing to do.
            pass
        except Exception as e:
            # (Removed: WARN print — per training-log-only policy. The tmp
            # file is still cleaned up by the `finally` block below.)
            pass
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass


# ============================================================ #
# HfTripletStream                                                 #
# ============================================================ #


class HfTripletStream(IterableDataset):
    """Reads HF arrow cache lazily + decodes FLAC on the fly.

    Reads from the HuggingFace `datasets.Dataset` objects that
    `voicefilter-cli dataset prepare` populated — there are no wav files
    on disk. Each row in each (source, subset, split) becomes an
    `HfEntry` (a `(dataset, row_idx, spk_id, utt_id, source_sr)`
    pointer). At iteration time the loader calls
    `default_hf_wav_loader(entry, hp.sample_rate)`, which reads
    `entry.dataset[entry.row_idx]['audio']` and decodes via
    `soundfile` (FLAC bytes via `BytesIO`), resampling to
    `hp.sample_rate` if `entry.source_sr` differs (MUSAN `noise` /
    `music` are 44.1 kHz).

    Multi-worker strategy: `datasets.Dataset` is mmap-backed; forking a
    worker process creates a CoW view of the arrow file and is
    fork-safe for read-only access. Each worker opens its own
    `datasets.Dataset` instance lazily (`__iter__` does a single
    `load_dataset` per (source, subset, split) and caches the handle).

    Per-speaker d-vectors land in
    `<spec.cache_dir>/voicefilter-dvectors/<spec_id>/<spk>.npy` — see
    `DVectorCache`. The d-vector partition lives next to the source's
    HF cache, not under a central `processed_dir`.

    Per-sample synthesis:

      * `babble`         : N ∈ [n_min, n_max] DISTINCT identities drawn
                            from the babble pool (union of
                            `hp.interference.babble.sources[*]`). Each
                            supported kind is keyed by a `spk_id` prefix:
                              - musan(speech)      → `musan_speech__<stem>`
                                (each clip = one monologue identity)
                              - librispeech(other) → `librispeech_other__<spk>`
                            Each speaker contributes a random-length
                            segment placed at a random offset within
                            the sample, summed with per-speaker gain.
      * `noise` / `music`: 1 random clip from the matching MUSAN
                            subset, `random_crop` to `sample_len`. The
                            loader resamples 44.1 kHz → 16 kHz.

    Per-sample, each TYPE's `prob` is an INDEPENDENT Bernoulli
    inclusion probability; types whose pool is empty or whose `prob <=
    0` are skipped. Multiple TYPES can be included in one sample —
    the energy budget is preserved across K components via Dirichlet
    sampling on the simplex (see `_allocate_interference_budget`).

    Lazy initialisation: the constructor is intentionally cheap — it
    does NOT open any HF dataset or build any indices. Each worker
    process (after fork) runs the one-shot scan on its first
    `__iter__` call via `_ensure_initialized`. With N dataloader
    workers, the N scans run in parallel after fork and the OS page
    cache for arrow mmap files is shared between them — wall-clock
    init time is close to a single-process scan rather than N× it.
    Once the scan has run, the dataset exposes the same surface
    area as before: `_target_index`, `_speech_pool`, `_noise_pool`,
    `_music_pool`, `_spk_to_entries`, `_spk_to_spec_id`, plus a
    `DVectorCache` that lazily fills per-speaker d-vectors.
    """

    # Per-source manifest filename. Re-exported for tests / CLI code
    # that wants to locate the manifest next to a specific HF cache.
    MANIFEST_FILENAME = MANIFEST_FILENAME
    DVEC_DIRNAME = DVEC_DIRNAME
    INDEX_DIRNAME = INDEX_DIRNAME

    def __init__(
        self,
        hp: HParams,
        worker_id: int = 0,
        num_workers: int = 1,
        seed: Optional[int] = None,
        multi_type_prob_value: Any = None,
    ):
        super().__init__()
        self.hp = hp
        self.worker_id = worker_id
        self.num_workers = num_workers
        self._seed = seed if seed is not None else hp.seed

        # Shared-memory handle (torch.multiprocessing.Value) for the
        # curriculum-controlled `multi_type_prob` scalar. The main
        # training loop updates this each `log_every` window via the
        # value's `.value = ...` setter; fork'd workers read it
        # transparently (the underlying smp float is shared, only the
        # Python handle is CoW). When `None`, the iterator uses the
        # config-time final value (`multi_type_end_prob`) and skips the
        # curriculum (useful for `examples/04_*` and tests).
        self._multi_type_prob_value = multi_type_prob_value

        # Mel basis built lazily on first `__iter__` (per worker).
        self._mel = None

        # The remaining state is populated lazily by `_scan_hf_datasets()`
        # on the first `__iter__` call of each worker process — see
        # `_ensure_initialized`. The previous design ran the scan
        # synchronously in `__init__`, which blocked the main thread
        # for 10-60 s on a full LibriSpeech+MUSAN bake before the
        # DataLoader could even fork its workers. Lazy init lets the
        # main thread reach the training loop almost immediately; the
        # scan then runs in parallel across N worker processes after
        # fork.
        #
        # With `dataloader_workers > 0` + fork, each worker invokes
        # `_ensure_initialized` exactly once on its first iteration,
        # and the scans run in parallel after fork. Wall-clock scan
        # time is bounded by the slowest worker rather than the sum:
        # arrow files are mmap'd, and the OS page cache is shared
        # across fork'd processes, so the first worker to scan warms
        # the cache and subsequent workers hit it (near memory-
        # bandwidth speed). The end result is "data flows continuously
        # from workers" rather than "main thread waits for a giant
        # synchronous scan before workers can spawn".
        #
        # With `dataloader_workers == 0`, the main thread scans inline
        # before yielding the first sample — same observable behaviour
        # as the old synchronous `__init__`, just deferred by one
        # `__iter__` call.
        self._datasets: dict[str, Any] = {}
        self._target_index: dict[str, list[HfEntry]] = {}
        # spk → spec_id, populated by the lazy scan. Used by
        # `DVectorCache` to partition d-vectors per source so multi-
        # source targets never collide on speaker ids.
        self._spk_to_spec_id: dict[str, str] = {}
        # Babble pool — config-driven via `_collect_babble_pool_entries`:
        # walks `hp.interference.babble.sources[*]` and unions every
        # babble-eligible kind (musan_speech__ + librispeech_other__).
        self._speech_pool: list[tuple[str, HfEntry]] = []
        # spk_id → list[HfEntry] derived from `_speech_pool` once the
        # scan runs. `babble` picks N distinct keys from this map.
        self._spk_to_entries: dict[str, list[HfEntry]] = {}
        # MUSAN pools for the `noise` / `music` TYPEs (no speaker concept).
        self._noise_pool: list[HfEntry] = []
        self._music_pool: list[HfEntry] = []
        # type_name → independent Bernoulli inclusion `prob` (0 = disabled).
        self._interference_type_probs: dict[str, float] = {}
        # Lazy per-speaker d-vector.
        self._dvec_cache: Optional[DVectorCache] = None

        # Per-worker init gate — first `__iter__` call triggers the scan.
        self._initialized: bool = False
        # Wall-clock cost of the first scan (debug aid / training-log
        # heartbeat). Recorded once per worker so the training loop can
        # print "workers ready after Xs" alongside the existing
        # "first batch built in Ys" line.
        self._init_seconds: float = 0.0

    # ---- One-time dataset scan ----

    def _ensure_initialized(self) -> None:
        """Populate per-worker state on first `__iter__` call.

        Each worker process (after fork) invokes this exactly once on
        its first iteration. With `dataloader_workers > 0`, N workers
        invoke it concurrently and their scans run in parallel — the
        wall-clock init time is bounded by the slowest single scan,
        not the sum of N scans, because:

          * Arrow files are mmap'd; the OS page cache is shared across
            fork'd processes. The first worker to scan warms the
            cache; subsequent workers (and subsequent iterations in
            the same worker) hit cache and run near memory-bandwidth
            speed.
          * Per-row work is small (`ds[row_idx]` mmap read + a dict
            insert in `_row_metadata`). On a multi-core M-series chip
            with 4-8 workers the scans overlap well enough that the
            wall-clock cost is close to a single-process scan rather
            than N×.

        With `dataloader_workers == 0`, the main thread invokes this
        inline before the first sample is yielded — same observable
        behaviour as the old synchronous `__init__`, just deferred by
        one `__iter__` call. Tests that access `_target_index` (or
        other scan-built state) after construction must call
        `_ensure_initialized()` explicitly.

        Idempotent — subsequent calls early-return in O(1). Not
        thread-safe (each worker is single-threaded; main-thread
        iteration with `num_workers=0` is also single-threaded).
        """
        if self._initialized:
            return
        t0 = time.monotonic()
        self._scan_hf_datasets()
        self._init_seconds = time.monotonic() - t0
        self._initialized = True

    def _scan_hf_datasets(self) -> None:
        """Open one HF `datasets.Dataset` per (source, subset, split).

        Builds the in-memory `{spk_id → [HfEntry, …]}` maps for the
        target pool + the per-TYPE interference pools. Idempotent —
        called once per worker process via `_ensure_initialized`. No
        wavs are decoded here; every read at iteration time uses
        `default_hf_wav_loader` so we pay FLAC decode only when an
        utterance is actually picked.
        """
        # ---- Target ----
        target_index, spk_to_spec_id = _build_target_index(
            self.hp, self._datasets,
        )
        self._target_index = target_index
        self._spk_to_spec_id = spk_to_spec_id

        # ---- Babble pool (config-driven) ----
        babble_entries = _collect_babble_pool_entries(self.hp, self._datasets)
        self._speech_pool = babble_entries
        # Build speaker-keyed index for babble dedup (used to pick N
        # distinct speakers without repetition). O(N) one-shot at scan.
        for spk, entry in babble_entries:
            self._spk_to_entries.setdefault(spk, []).append(entry)

        # ---- Noise / music pools (musan-only at runtime) ----
        self._noise_pool = _collect_passthrough_pool_entries(
            self.hp, self._datasets, "noise",
        )
        self._music_pool = _collect_passthrough_pool_entries(
            self.hp, self._datasets, "music",
        )

        # Per-type INDEPENDENT inclusion probabilities: only types that
        # are enabled + have `prob > 0` are eligible.
        for type_name, type_cfg in self.hp.interference_types.items():
            if not type_cfg.enabled or type_cfg.prob <= 0:
                continue
            self._interference_type_probs[type_name] = float(type_cfg.prob)

        # ---- D-vector cache (lazy fill) ----
        # Always constructed (even when `_target_index` is empty) so
        # the dataset exposes the same surface area across runs; the
        # cache itself refuses to compute without a reference entry
        # and falls back to zeros.
        self._dvec_cache = DVectorCache(
            self.hp,
            self._target_index,
            spk_to_spec_id=self._spk_to_spec_id,
        )

    # ---- Pool eligibility per TYPE ----

    def _type_has_clips(self, type_name: str) -> bool:
        """True iff the underlying pool for `type_name` is non-empty."""
        if type_name == "babble":
            return bool(self._spk_to_entries)
        if type_name == "noise":
            return bool(self._noise_pool)
        if type_name == "music":
            return bool(self._music_pool)
        return False

    # ---- Iteration ----

    def __iter__(self) -> Iterator[Sample]:
        # Lazy one-shot scan per worker process. With `dataloader_workers > 0`,
        # each fork'd worker runs its own scan on first `__iter__` in
        # parallel — see `_ensure_initialized` for the rationale.
        self._ensure_initialized()

        from voicefilter_lite.audio import make_mel_basis
        if self._mel is None:
            self._mel = make_mel_basis(self.hp, device=torch.device("cpu"))

        # Per-worker RNG: without re-seeding, all worker processes would
        # draw identical samples — duplicates instead of parallelism.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self._rng = random.Random(self._seed + worker_info.id)
        else:
            self._rng = random.Random(self._seed + self.worker_id)

        rng = self._rng
        hp = self.hp
        sample_len = hp.sample_length_samples

        target_speakers = list(self._target_index.keys())
        if not target_speakers:
            raise RuntimeError(
                "[HfTripletStream] no target utterances found; "
                "did `voicefilter-cli dataset prepare` finish?"
            )

        while True:
            # ---- 1. Pick target speaker + utterance ----
            tgt_spk = rng.choice(target_speakers)
            tgt_utt_entries = self._target_index[tgt_spk]
            if len(tgt_utt_entries) < 2:
                # Single-utterance speaker: ref must be self.
                tgt_entry = tgt_utt_entries[0]
                ref_entry = tgt_entry
            else:
                tgt_entry = rng.choice(tgt_utt_entries)
                # ---- 3. Pick ref: 70% same-speaker (different utt), 30% self ----
                if rng.random() < 0.7:
                    others = [e for e in tgt_utt_entries if e != tgt_entry]
                    if others:
                        ref_entry = rng.choice(others)
                    else:
                        ref_entry = tgt_entry
                else:
                    ref_entry = tgt_entry

            # ---- 4. Read target + ref wavs (BEFORE interference synthesis) ----
            # We need `target_power` (= mean of target wav²) to compute
            # the interference energy budget `target_power / 10^(snr/10)`,
            # which `_sample_interference` uses to allocate per-component
            # RMS² via Dirichlet sampling on the simplex.
            try:
                tgt_wav_np = default_hf_wav_loader(tgt_entry, hp.sample_rate)
                ref_wav_np = default_hf_wav_loader(ref_entry, hp.sample_rate)
            except Exception:
                _skipped = getattr(self, "_read_fail_count", 0) + 1
                self._read_fail_count = _skipped
                # (Per-failure WARN removed per training-log-only policy.
                # `self._read_fail_count` is still bumped so external health
                # checks can poll it; the underlying `continue` ensures the
                # bad sample is silently dropped from training.)
                continue

            tgt_wav_t = torch.from_numpy(tgt_wav_np)
            ref_wav_t = torch.from_numpy(ref_wav_np)

            # ---- 5–8. WAV-level synthesis (random crop, interference,
            # silent flags, activity bursts, snr_mix) — delegated to the
            # shared module-level `synthesize_raw_sample` so the
            # training iterator and offline audition scripts
            # (`examples/04_*`) consume the SAME per-sample logic.
            # Read the current `multi_type_prob` from the shared handle
            # (the training loop writes the ramped value once per
            # `log_every` window). Fall back to the config-time end prob
            # when no shared handle was supplied (audition scripts +
            # tests).
            if self._multi_type_prob_value is not None:
                multi_type_prob = float(self._multi_type_prob_value.value)
            else:
                multi_type_prob = float(hp.multi_type_end_prob)
            raw = synthesize_raw_sample(
                hp,
                rng,
                target_wav_full=tgt_wav_t,
                ref_wav_full=ref_wav_t,
                babble_spk_to_entries=self._spk_to_entries,
                noise_pool=self._noise_pool,
                music_pool=self._music_pool,
                type_probs=self._interference_type_probs,
                babble_cfg=self.hp.interference_types.get(NOISE_KIND_BABBLE),
                noise_cfg=self.hp.interference_types.get(NOISE_KIND_NOISE),
                music_cfg=self.hp.interference_types.get(NOISE_KIND_MUSIC),
                multi_type_prob=multi_type_prob,
            )
            target_wav = raw.target_wav
            interference_wav = raw.interference_wav
            mixed_wav = raw.mixed_wav
            target_act = raw.target_act
            interference_act = raw.interference_act
            noise_kind = raw.included_kinds

            # ---- 9. log-mel + frame-stack ----
            mixed_logmel = wav_to_log_mel(mixed_wav, hp, self._mel)
            target_logmel = wav_to_log_mel(target_wav, hp, self._mel)
            stacked = frame_stack_left(mixed_logmel, hp.frame_stack)
            T_frames = stacked.shape[0]

            # ---- 9. Per-frame overlap label ----
            overlap_label = _per_frame_overlap_label(
                target_act=target_act,
                interference_act=interference_act,
                T_frames=T_frames,
                sample_len=sample_len,
                hp=hp,
            )

            # ---- 10. Get-or-compute d-vector ----
            # `DVectorCache` returns the on-disk `.npy` if present, else
            # computes via the SpeakerEncoder + saves atomically. On any
            # failure path (no ref entry, encoder missing) it returns
            # zeros + a one-shot WARN. See `DVectorCache.get_or_compute`.
            ref_dvec_np = self._dvec_cache.get_or_compute(tgt_spk)
            ref_dvec = torch.from_numpy(ref_dvec_np).float()

            yield Sample(
                stacked_mixed=stacked,
                target_logmel=target_logmel,
                ref_dvec=ref_dvec,
                ref_spk_id=tgt_spk,
                noise_kind=noise_kind,
                overlap_label=overlap_label,
            )


def collate_triplets(batch: list[Sample]) -> dict:
    """Collate a list of `Sample` into CPU tensors.

    Training loop moves tensors to the training device itself, so the
    DataLoader (and any background prefetch thread) only ships plain
    CPU tensors through its iterator / queue.

    All `ref_dvec` entries are pre-baked and (256,) shaped, so they
    stack cleanly. `ref_spk_ids` stays as a parallel list of strings
    for diagnostics.
    """
    stacked = torch.stack([s.stacked_mixed for s in batch])
    target = torch.stack([s.target_logmel for s in batch])
    overlap_label = torch.stack([s.overlap_label for s in batch])
    ref_dvecs = torch.stack([s.ref_dvec for s in batch])        # (B, 256)
    ref_spk_ids = [s.ref_spk_id for s in batch]

    return {
        "stacked_mixed": stacked,
        "target_logmel": target,
        "overlap_label": overlap_label,
        "noise_kind": [s.noise_kind for s in batch],
        "ref_dvecs": ref_dvecs,
        "ref_spk_ids": ref_spk_ids,
    }
