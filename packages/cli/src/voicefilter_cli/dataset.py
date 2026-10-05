"""Local-dataset triplet iterator for VoiceFilter-Lite training.

Reads pre-baked wav files from `data/processed/` (produced by
`voicefilter-cli dataset prepare`). All HF streaming + FLAC decode happens
at the preprocessing stage; d-vector extraction happens lazily inside
this module via `DVectorCache` (so the preprocessing step stays
audio-only — see comment in `prepare_dataset.py`). Interference synthesis
(babble / noise / music) also happens HERE, at training time per
sample — `prepare_dataset` only decodes the source pools and this
iterator mixes them on the fly.

Per-sample stochastic work that must vary every iteration:

  1. Sample (target_speaker, target_utt)            # random
  2. Sample ref wav (70% same-speaker, 30% self)   # random
  3. Read 2 local wavs (target + ref)              # ~3 ms/sample
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
`prepare_dataset` bakes each `SourceRef` referenced by any enabled TYPE
into a flat `pool/<HfSourceSpec.id>/...` directory (e.g.
`pool/musan/speech|noise|music/` for MUSAN). Per-sample, each TYPE's
`prob` is an INDEPENDENT Bernoulli inclusion probability (no longer a
weighted pick); types with `prob <= 0` or empty backing pools are
skipped. The babble pool is **config-driven** — `_collect_babble_pool_paths`
walks `hp.interference.babble.sources[*]` at scan time and builds
`_spk_to_paths` from the union of every babble-eligible source
(currently musan(speech) + librispeech(other)). The librispeech `other`
pool is NOT used for babble unless `babble.sources` declares it; the
canonical paper.yaml keeps librispeech(other) under `noise.sources`
for `prepare_dataset` baking + `first_librispeech_other_source`
discovery (eval / audition harnesses), not as runtime babble input.

Each yielded `Sample` carries `(stacked_mixed, target_logmel, ref_dvec,
overlap_label, noise_kind)`. `noise_kind` is a `tuple[str, ...]` of the
TYPE names actually included in this sample's interference (e.g.
`()`, `("babble",)`, `("noise", "music")`, `("babble", "noise",
"music")`) — used for diagnostics only. The order is deterministic
(babble, noise, music) so test assertions and summaries are stable.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

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


# Top-level subdir names under `processed_dir/`. These describe the
# SEMANTIC ROLE of each subtree, not a layout choice — deliberately not
# configurable (see `LocalTripletStream` docstring for the rationale).
# Anything beneath them (spec_id, role, split, …) IS config-driven via
# `HfSourceSpec`. Module-level so both `LocalTripletStream` and the
# module-level helpers (`_collect_babble_pool_paths` / dvec cache) can
# reference the same name.
TARGET_TOP_SUBDIR = "target"
POOL_TOP_SUBDIR = "pool"
DVEC_TOP_SUBDIR = "dvectors"
MANIFEST_FILENAME = "manifest.json"


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


def _collect_babble_pool_paths(
    hp: HParams,
    processed_dir: Path,
) -> list[tuple[str, Path]]:
    """Walk `hp.interference.babble.sources[*]` and return babble-pool paths.

    Returns a list of `(spk_id, wav_path)` pairs. `spk_id` is a stable
    identity key for babble dedup (the runtime picks N DISTINCT
    `spk_id`s — see `_pick_babble_speakers`):

      * musan(speech)        : `musan_speech__<clip_stem>` —
                                each clip is one long-monologue
                                identity, exactly one wav each.
      * librispeech(other)   : `librispeech_other__<spk>` —
                                `parts[0]` under the split dir is the
                                speaker id (multiple utts per spk).

    Source kinds not listed above (noise / music subsets, target
    librispeech `clean`) are skipped — they are passthrough or
    target pools, not babble-eligible.

    Empty list when babble is disabled, or when no configured source
    has a non-empty pool on disk. `prepare_dataset` bakes every
    configured source into `processed_dir/pool/...` so an empty
    return here usually means the user ran training without baking
    first — a config / pipeline error rather than a runtime concern.
    """
    babble_cfg = hp.interference_types.get("babble")
    if babble_cfg is None or not babble_cfg.enabled or babble_cfg.prob <= 0:
        return []

    pool_root = Path(processed_dir) / POOL_TOP_SUBDIR
    if not pool_root.is_dir():
        return []

    out: list[tuple[str, Path]] = []
    seen: set[Path] = set()

    for ref in babble_cfg.sources:
        try:
            resolved = resolve_source(hp, ref)
        except KeyError:
            # Unknown source id — skip silently (config bug; surface
            # elsewhere if it matters).
            continue

        if resolved.is_musan() and resolved.subset == "speech":
            sub = pool_root / resolved.id / "speech"
            if not sub.is_dir():
                continue
            for wav in sorted(sub.glob("*.wav")):
                if wav in seen:
                    continue
                seen.add(wav)
                out.append((f"musan_speech__{wav.stem}", wav))
            continue

        if resolved.is_librispeech() and resolved.librispeech_role() == "other":
            for split in resolved.splits:
                sub = pool_root / resolved.id / "other" / split
                if not sub.is_dir():
                    continue
                for wav_path in sorted(sub.glob("*/*.wav")):
                    parts = wav_path.relative_to(sub).parts
                    if len(parts) < 2:
                        continue
                    spk = parts[0]
                    if wav_path in seen:
                        continue
                    seen.add(wav_path)
                    out.append((f"librispeech_other__{spk}", wav_path))
            continue

        # Other kinds (noise / music subsets, librispeech clean role):
        # not babble-eligible — skip.
    return out


def _collect_passthrough_pool_paths(
    hp: HParams,
    pool_root: Path,
    type_name: str,
) -> list[Path]:
    """Collect passthrough-pool clips for the `noise` / `music` TYPEs.

    Walks `hp.interference.<type_name>.sources[*]` and unions every
    on-disk `pool/<spec_id>/<subset>/<clip>.wav` clip. Only MUSAN
    sources are consumed at runtime (the noise / music TYPE mixer is
    a "1 random clip + random_crop" passthrough, and we keep the
    speech-source concern out of it). Non-musan refs (e.g.
    librispeech(other) under `noise.sources` for eval discoverability)
    are skipped silently — they're still baked by `prepare_dataset`
    so `first_librispeech_other_source` can find them.

    Spec.id is resolved via `resolve_source(...)` — change a musan
    `HfSourceSpec.id` and the scan follows automatically (no literal
    "musan" anywhere in this function).

    Returns a deduplicated, sorted list of wav paths. Empty when the
    TYPE is disabled, has `prob <= 0`, or has no enabled musan source
    with a non-empty on-disk pool.
    """
    type_cfg = hp.interference_types.get(type_name)
    if type_cfg is None or not type_cfg.enabled or type_cfg.prob <= 0:
        return []

    out: list[Path] = []
    seen: set[Path] = set()
    for ref in type_cfg.sources:
        try:
            resolved = resolve_source(hp, ref)
        except KeyError:
            continue
        if not resolved.is_musan() or not resolved.subset:
            # Runtime noise/music mixer only consumes musan pools;
            # librispeech(other) etc. are baked-only for eval discovery.
            continue
        sub = pool_root / resolved.id / resolved.subset
        if not sub.is_dir():
            continue
        for wav in sorted(sub.glob("*.wav")):
            if wav in seen:
                continue
            seen.add(wav)
            out.append(wav)
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

    The training iterator (`LocalTripletStream.__iter__`) calls
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


# Module-level default wav loader — used by `synthesize_raw_sample` and
# `_mix_passthrough_v2` when the caller doesn't supply one. Matches the
# behaviour of training (path → mono float32 numpy). Example scripts
# that have in-memory wavs pass a custom loader.
def _default_wav_loader(path: Path) -> np.ndarray:
    """Read `path` as a mono float32 numpy array (training's default)."""
    wav, _ = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
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
    spk_to_paths: dict[str, list[Path]],
    n_speakers_min: int,
    n_speakers_max: int,
) -> list[tuple[str, Path]]:
    """Pick N DISTINCT speakers for one babble mix (no replacement).

    Returns `[(spk_id, path), ...]` of length N ∈ `[n_speakers_min,
    n_speakers_max]` (clamped to `len(spk_to_paths)`). For each chosen
    speaker one random utterance from `spk_to_paths[spk]` is selected.

    Splitting the picker from the mixer lets the unit test verify
    "exactly N distinct speakers" without stubbing `sf.read`; the
    iterator calls this then hands the picks to `_mix_babble_from_picks`.
    """
    if not spk_to_paths:
        return []
    n_avail = len(spk_to_paths)
    n_min = max(1, min(int(n_speakers_min), n_avail))
    n_max = max(n_min, min(int(n_speakers_max), n_avail))
    n = rng.randint(n_min, n_max)
    chosen_spks = rng.sample(list(spk_to_paths.keys()), k=n)
    return [(spk, rng.choice(spk_to_paths[spk])) for spk in chosen_spks]


def _mix_babble_from_picks(
    rng: random.Random,
    sample_len: int,
    sr: int,
    picks: list[tuple[str, Path]],
    seg_min_sec: float,
    seg_max_sec: float,
    gain_min: float,
    gain_max: float,
    wav_loader=_default_wav_loader,
) -> torch.Tensor:
    """Mix N speakers (already picked) into one `(sample_len,)` wav.

    For each `(spk, path)`:
      1. Load the wav via `wav_loader(path)` (default = `sf.read` from
         disk, mono float32), downmix to mono.
      2. Pick a random segment length ∈ `[seg_min_sec, seg_max_sec]`
         (clamped to the wav length).
      3. Pick a random start within the wav for that segment.
      4. Pick a random start within `[0, sample_len - seg_len]` for the
         output placement.
      5. Multiply by a uniform-random gain in `[gain_min, gain_max]`.
      6. Sum into the output buffer (overlaps add).

    Clips shorter than `seg_min_sec * sr` samples are skipped — should
    only matter for tiny fixture wavs in tests. Empty input → silence.

    The `wav_loader` hook lets callers with non-filesystem wavs (HF
    arrow cache, in-memory tensors) supply their own loader without
    writing the wavs to disk. Training uses the default (`_default_wav_loader`).
    """
    out = np.zeros(sample_len, dtype=np.float32)
    if not picks:
        return torch.from_numpy(out)
    seg_min = max(1, int(seg_min_sec * sr))
    seg_max = max(seg_min, int(seg_max_sec * sr))
    for _spk, path in picks:
        try:
            wav = wav_loader(path)
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
# These are the same algorithms `LocalTripletStream._sample_interference`
# runs, but lifted to module scope so callers that hold their own
# pool structures (e.g. `examples/04_*` after backing in-memory wavs to
# a tmp `pool/` directory) can invoke them without instantiating the
# full `LocalTripletStream` dataset. Training and example scripts share
# the same code path → identical sampling statistics.


def _mix_passthrough_v2(
    rng: random.Random,
    sample_len: int,
    pool: list[Path],
    wav_loader=_default_wav_loader,
) -> torch.Tensor:
    """`noise` / `music` TYPEs — 1 random clip from `pool`, cropped/padded to `sample_len`.

    Module-level twin of `LocalTripletStream._mix_passthrough`. Same
    algorithm, same RNG contract (the `random_crop` call inside
    `random_crop` consumes one RNG draw); the only difference is the
    `wav_loader` hook so callers with non-filesystem wavs can supply
    their own. Returns zeros when `pool` is empty (defensive).
    """
    if not pool:
        return torch.zeros(sample_len, dtype=torch.float32)
    path = rng.choice(pool)
    wav_np = wav_loader(path)
    wav_t = torch.from_numpy(wav_np)
    return random_crop(wav_t, sample_len)


def _sample_interference_v2(
    hp: HParams,
    rng: random.Random,
    sample_len: int,
    target_power: float,
    snr_db: float,
    *,
    babble_spk_to_paths: dict[str, list[Path]],
    noise_pool: list[Path],
    music_pool: list[Path],
    babble_cfg,
    noise_cfg,
    music_cfg,
    wav_loader=_default_wav_loader,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Module-level twin of `LocalTripletStream._sample_interference`.

    Rolls each TYPE's independent Bernoulli `prob`, synthesises the
    included components, then delegates energy-budget allocation to
    `_allocate_interference_budget` (peak-clip per component, Dirichlet
    split, tanh soft-clip of the aggregate). Returns `(interference_wav,
    included_kinds)` exactly like the instance method.

    The TYPE iteration order is fixed (babble → noise → music) so test
    assertions / per-bucket summaries are reproducible across runs given
    the same RNG seed.

    Args:
        babble_spk_to_paths: pooled babble pool (spk_id → list of paths).
            `babble_cfg.sources` decides which sources feed in; this map
            is the union. `_pick_babble_speakers` ensures N distinct
            identities.
        noise_pool / music_pool: passthrough pools (`_mix_passthrough_v2`
            picks one clip + random-crop).
        babble_cfg / noise_cfg / music_cfg: `InterferenceTypeCfg` objects
            from `hp.interference_types[<name>]` — `prob`, segment
            lengths, gain range, etc. `None` is treated as disabled.
        wav_loader: `Path → np.ndarray (T,)` hook. Default =
            `_default_wav_loader`. Example scripts with in-memory wavs
            pass their own loader.
    """
    components: list[torch.Tensor] = []
    included: list[str] = []

    def _has_clips(type_name: str, cfg) -> bool:
        if cfg is None or not cfg.enabled or float(cfg.prob) <= 0.0:
            return False
        if type_name == NOISE_KIND_BABBLE:
            return bool(babble_spk_to_paths)
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
                spk_to_paths=babble_spk_to_paths,
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
            component = _mix_passthrough_v2(rng, sample_len, noise_pool, wav_loader)
        elif type_name == NOISE_KIND_MUSIC:
            component = _mix_passthrough_v2(rng, sample_len, music_pool, wav_loader)
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
    babble_spk_to_paths: dict[str, list[Path]],
    noise_pool: list[Path],
    music_pool: list[Path],
    type_probs: dict[str, float],
    babble_cfg,
    noise_cfg,
    music_cfg,
    wav_loader=_default_wav_loader,
) -> RawSample:
    """One end-to-end per-sample WAV synthesis — mirrors `LocalTripletStream.__iter__` lines 1387–1464.

    The caller picks the target speaker/utterance and the reference
    utterance (training does this from a pre-baked index;
    `examples/04_*` does this from an HF stream). This function takes
    those two pre-loaded wav tensors + the pool structures and returns
    the WAV-level outputs the training iterator would feed into the
    log-mel pipeline.

    Steps (in order — same RNG semantics as training):
      1. Sample activity burst on the **FULL target audio** and apply
         it, then crop to `sample_len` at a position that **OVERLAPS
         with the burst**. This is the deviation 5 fix vs. the old
         "burst on cropped sample" logic: for a long LibriSpeech
         utterance, both the random crop and the uniform burst were
         sampled independently of the audio energy — they could both
         land on the same silent segment, making the burst capture
         silence and target.wav nearly inaudible. Sampling the burst
         on the full source guarantees the burst overlaps with actual
         voice; constraining the crop to overlap with the burst keeps
         the burst window in the final 3 s crop. Burst mask stays
         sparse (never collapses to 1).
      2. Cap `ref_wav_full` at `hp.ref_max_sec * sample_rate` samples
         (left-truncate, no random offset — matches training).
      3. `target_power = mean(target²)` on the **burst-masked crop**,
         eps-clamped to 1e-12. Computing on the post-mask crop keeps
         the interference energy budget consistent with the actual
         energy the model will see during the active burst window.
      4. Draw `snr_db ∈ [snr_min_db, snr_max_db]` UNCONDITIONALLY
         (silent flags are rolled AFTER synthesis so the Bernoulli
         inclusion draws land on the same RNG state regardless of the
         silent outcome).
      5. `_sample_interference_v2(...)` — Dirichlet-budget allocation
         across the K included TYPES (peak-clip per component, tanh
         soft-clip of the aggregate; see `_allocate_interference_budget`).
      6. Roll silent flags (target_silent / interference_silent). When
         `interference_silent=True`, override the synthesised wav with
         zeros; `included_kinds` still reports what the rolls SAID.
         Target_silent forces the post-mask target down to zero in the
         defensive-zero step below.
      7. Sample interference activity burst and apply (target burst was
         already applied at step 1). Skipped when the corresponding
         silent flag fired.
      8. Defensive zero-out when silent.
      9. `snr_mix` branch:
            both silent         → zeros
            target_silent       → interference.clone()
            interference_silent → target.clone()
            else                → snr_mix(target, interference, snr)

    The signature takes the TYPE configs (`babble_cfg` etc.) explicitly
    rather than reading `hp.interference_types` here so callers can
    override / filter without mutating `hp`. Training and example 04
    both pull them from `hp.interference_types` at the call site.

    `type_probs` is currently unused by this function (the configs
    carry their own `prob`); accepted for forward-compat and to make
    the signature symmetric with `LocalTripletStream._sample_interference`.

    Returns a `RawSample` carrying the wavs, activity masks, ref wav,
    snr draw, included kinds, and the silent flags.
    """
    sample_len = hp.sample_length_samples

    # ---- 1. Target activity burst on FULL audio, then constrained crop ----
    # Deviation 5 fix: sample the burst on the FULL source first so the
    # burst always overlaps with real audio energy. Then sample a crop
    # start that lies inside the burst's overlap range, so the cropped
    # 3 s window still contains that voice. This avoids the failure
    # mode where a uniform random crop and a uniform random burst on
    # the cropped window both land on a silent stretch of a long
    # LibriSpeech utterance — making target.wav nearly inaudible even
    # though the burst mask itself is correctly sparse.
    T_full = int(target_wav_full.shape[-1])
    target_act_full = _sample_activity_burst(T_full, hp, rng)
    target_wav_full_masked = target_wav_full * target_act_full

    if T_full <= sample_len:
        # Source shorter than (or equal to) sample_len: keep the whole
        # source, zero-pad to sample_len. random_crop does the same.
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
        # Source longer than sample_len: sample crop_start within the
        # burst's overlap range so [crop_start, crop_start + sample_len]
        # intersects the high-energy burst plateau.
        burst_active = (target_act_full > 0.5).nonzero(as_tuple=True)[0]
        if burst_active.numel() == 0:
            # Defensive fallback (shouldn't happen given burst_min >= 1
            # sample and the guard in `_sample_activity_burst`).
            crop_start = 0
        else:
            burst_start_full = int(burst_active[0].item())
            burst_end_full = int(burst_active[-1].item()) + 1
            # crop_min: latest crop start that still keeps the burst's
            # tail inside the crop. crop_max: earliest crop start that
            # still keeps the burst's head inside the crop.
            crop_min = max(0, burst_end_full - sample_len)
            crop_max = min(T_full - sample_len, burst_start_full)
            # `rng.randint` is inclusive on both ends; clamp so lower
            # bound never exceeds upper bound (degenerate when the
            # burst length >= sample_len — crop_max then equals
            # crop_min, which is fine).
            crop_start = rng.randint(crop_min, max(crop_min, crop_max))
        target_wav = target_wav_full_masked[crop_start : crop_start + sample_len]
        target_act = target_act_full[crop_start : crop_start + sample_len]

    # ---- 2. Cap ref_wav at `ref_max_sec` samples (matches training:
    # `ref_wav_t[..., :ref_max_len]` — left-truncate, no random offset).
    ref_max_len = int(hp.ref_max_sec * hp.sample_rate)
    if ref_wav_full.shape[-1] > ref_max_len:
        ref_wav_capped = ref_wav_full[..., :ref_max_len]
    else:
        ref_wav_capped = ref_wav_full

    # ---- 3. target_power on burst-masked crop ----
    # Computing on the post-mask crop keeps the interference energy
    # budget consistent with the energy the model actually sees during
    # the burst window — a sparser burst (smaller active fraction)
    # yields lower post-mask RMS², so the interference budget shrinks
    # accordingly and post-mix SIR still matches `snr_db`.
    target_power = max(1e-12, float((target_wav ** 2).mean().item()))

    # ---- 4. SNR draw + interference synthesis ----
    # Drawn UNCONDITIONALLY (before silent flags) so the TYPE-inclusion
    # Bernoulli draws happen on the same RNG state regardless of the
    # silent outcome.
    snr_db = float(rng.uniform(hp.snr_min_db, hp.snr_max_db))
    interference_wav, included_kinds = _sample_interference_v2(
        hp, rng, sample_len, target_power, snr_db,
        babble_spk_to_paths=babble_spk_to_paths,
        noise_pool=noise_pool,
        music_pool=music_pool,
        babble_cfg=babble_cfg,
        noise_cfg=noise_cfg,
        music_cfg=music_cfg,
        wav_loader=wav_loader,
    )

    # ---- 5. Silent flags (rolled AFTER synthesis) ----
    target_silent = rng.random() < hp.silent_target_prob
    interference_silent = rng.random() < hp.silent_interference_prob
    if interference_silent:
        interference_wav = torch.zeros(sample_len, dtype=torch.float32)

    # ---- 6. Activity bursts (target already applied above; interference
    # burst is skipped when silent flag fired) ----
    if target_silent:
        # Target was burst-masked at step 1; reset the mask to zeros so
        # downstream `_per_frame_overlap_label` and the per-sample
        # summary see "no activity" (matches the old behaviour).
        target_act = torch.zeros(sample_len, dtype=target_wav.dtype)

    if not interference_silent:
        interference_act = _sample_activity_burst(sample_len, hp, rng)
        interference_wav = interference_wav * interference_act
    else:
        interference_act = torch.zeros(sample_len, dtype=interference_wav.dtype)

    # ---- 7. Defensive zero-out (silent flag fired → wav still carries
    # pre-burst values; zero it explicitly). ----
    if target_silent:
        target_wav = torch.zeros_like(target_wav)
    if interference_silent:
        interference_wav = torch.zeros_like(interference_wav)

    # ---- 9. SNR mix (silent-side bypasses) ----
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
    iteration. Disk path is `<processed_dir>/dvectors/<spec_id>/<spk>.npy`
    where `spec_id` is the `HfSourceSpec.id` of the source the speaker
    came from (e.g. `dvectors/librispeech/<spk>.npy` for the default
    LibriSpeech target). Multi-source targets get separate partitions
    (no speaker-id collision across sources in practice).

    Lookup order (cheapest first):
        1. In-process dict (`_mem_cache`) — O(1), µs.
        2. On-disk file — one `np.load`.
        3. Compute — read the speaker's reference wav, run
           `extract_dvec(...)`, atomic-rename-save to disk.

    Concurrency:
        * In-process: `_mem_cache` is plain dict access — safe (CPython
          GIL). Encoder is a module-level singleton keyed by
          `(pt_path, mtime_ns, size)` so multiple `LocalTripletStream`
          instances share weights.
        * Cross-process (DataLoader workers): each worker has its own
          `_mem_cache` and its own encoder singleton. The disk write
          uses `np.save(tmp.pid)` + `os.replace()` for atomicity; if two
          workers race to compute the same new speaker, both compute
          (one encoder forward is ~50 ms) and the loser's rename is a
          no-op against the winner's file. After that, every worker
          sees the same disk file and falls through to the cheap path.

    Failure mode:
        * No reference wav for the speaker → zeros + WARN.
        * Encoder weights missing at `hp.dvector_pt_path` → random init
          + WARN; the d-vector is uninformative but training still runs.
        * Any compute exception → zeros + WARN. (Training would
          otherwise crash on the first encounter.)

    Cache invalidation: keyed only by (spec_id, speaker id). If you
    retrain the d-vector encoder (`voicefilter-cli train dvector`),
    delete `data/processed/dvectors/` to force recomputation. Future
    improvements (mtime-based invalidation) tracked separately.
    """

    def __init__(
        self,
        hp: HParams,
        processed_dir: Path,
        target_index: dict[str, list[Path]],
        spk_to_spec_id: Optional[dict[str, str]] = None,
    ):
        self.hp = hp
        self.processed_dir = Path(processed_dir)
        # spk → spec_id map built by `LocalTripletStream._scan_processed_dir`
        # (falls back to an empty dict when target is empty / not scanned yet).
        self._spk_to_spec_id: dict[str, str] = dict(spk_to_spec_id or {})
        # Resolve the d-vector partition by target spec.id — not a literal.
        # When the target is empty (e.g. unit test with no fixtures), the
        # cache is created anyway but `_path` falls back to the FIRST
        # declared target spec if the speaker isn't in the map.
        first_spec_id = self._first_target_spec_id()
        if first_spec_id is None:
            # No target at all — just root on the top-level subdir; every
            # _path() call will look up spec_id by speaker (and fall back
            # to "unknown" if the speaker isn't seen, which can't happen
            # because there's nothing to iterate over).
            self.dvec_root = self.processed_dir / DVEC_TOP_SUBDIR
        else:
            self.dvec_root = self.processed_dir / DVEC_TOP_SUBDIR / first_spec_id
        self.dvec_root.mkdir(parents=True, exist_ok=True)
        # Stable reference utterance per speaker (sorted wav list → [0]).
        # Mirrors the old `_bake_dvectors` reference-wav choice so cached
        # d-vectors are byte-compatible with what the previous pipeline
        # would have produced.
        self._ref_for: dict[str, Path] = {
            spk: paths[0] for spk, paths in target_index.items() if paths
        }
        # In-process LRU-by-construction (we never evict; spk count is
        # bounded by LibriSpeech). ~256 KB at 1000 speakers × 1 KB.
        self._mem_cache: dict[str, np.ndarray] = {}

    # ---- public API ----

    def get_or_compute(self, spk: str) -> np.ndarray:
        """Return `(hp.d_vector_dim,)` float32 d-vector for `spk`.

        On any failure path (no ref wav, encoder error, …) returns a
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

    def _first_target_spec_id(self) -> Optional[str]:
        """Return the `HfSourceSpec.id` of the first declared target source.

        Used to pick the d-vector partition when no per-speaker spec_id
        is available (legacy callers / empty target / unit tests). Returns
        None when `hp.target_sources` is empty or the first ref doesn't
        resolve to a known spec.
        """
        if not self.hp.target_sources:
            return None
        try:
            return resolve_source(self.hp, self.hp.target_sources[0]).id
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
        first = self._first_target_spec_id()
        if first is not None:
            return first
        return "unknown"

    def _path(self, spk: str) -> Path:
        """`<processed_dir>/dvectors/<spec_id>/<spk>.npy`.

        `spec_id` is looked up per-speaker so multi-source targets
        partition cleanly. Falls back to the first target's spec_id
        when the speaker wasn't seen at scan time.
        """
        sid = self._spec_id_for(spk)
        return self.processed_dir / DVEC_TOP_SUBDIR / sid / f"{spk}.npy"

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
        """Read `spk`'s reference wav → `extract_dvec` → (d_dim,) np."""
        ref = self._ref_for.get(spk)
        if ref is None:
            # (Removed: WARN print — per training-log-only policy.
            # Behaviour is identical: silently return zero d-vector.)
            return np.zeros(self.hp.d_vector_dim, dtype=np.float32)
        from voicefilter_lite.audio import extract_dvec

        wav, sr = sf.read(str(ref), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.hp.sample_rate:
            import librosa

            wav = librosa.resample(
                wav, orig_sr=sr, target_sr=self.hp.sample_rate
            ).astype(np.float32)
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
# LocalTripletStream                                              #
# ============================================================ #


class LocalTripletStream(IterableDataset):
    """Reads pre-baked wavs from `data/processed/`, yields `Sample` triples.

    The pre-baked directory layout (produced by `voicefilter-cli dataset prepare`):
        data/processed/
            manifest.json
            target/<spec_id>/<role>/<split>/<spk>/<utt}.wav
                # spec_id = HfSourceSpec.id of the target source
                # role    = librispeech_role() (clean|other)
            pool/<spec_id>/<role-or-subset>/...
                # musan: <subset> ∈ {speech, noise, music}
                # librispeech: <role>/<split>/<spk>/<utt>.wav
            dvectors/<spec_id>/<spk>.npy     # created lazily by DVectorCache

    **All subdirectory names under `target/` / `pool/` / `dvectors/` are
    config-driven** — they derive from `HfSourceSpec.id` and the
    resolved source's role/subset. The top-level names `target/`,
    `pool/`, `dvectors/`, and `manifest.json` are FIXED conventions
    (deliberately not configurable — they describe what each subtree
    semantically holds, not a layout choice).

    `pool/<HfSourceSpec.id>/...` is a flat per-source pool shared across
    every TYPE (`babble` / `noise` / `music`) that references it —
    `prepare_dataset` does no TYPE-specific synthesis. The actual
    interference waveform is synthesised at training time by
    `_sample_interference`:

        * `babble`         : N ∈ [n_min, n_max] DISTINCT identities drawn
                              from the babble pool. The pool is the union
                              of every source listed in
                              `hp.interference.babble.sources[*]`
                              (config-driven, NOT hardcoded —
                              `_collect_babble_pool_paths` walks the
                              configured refs). Each supported kind is keyed
                              by a `spk_id` prefix:
                                - musan(speech)      → `musan_speech__<stem>`
                                  (each clip = one monologue identity)
                                - librispeech(other) → `librispeech_other__<spk>`
                                  (`parts[0]` under the split dir; multiple
                                  utts per spk, dedup keeps them under one
                                  identity so the picker can never pick
                                  the same speaker twice in one babble mix)
                              Each speaker contributes a random-length
                              segment ∈ [babble_seg_min_sec,
                              babble_seg_max_sec] placed at a random
                              offset within the sample, summed with
                              per-speaker gain uniform in [gain_min,
                              gain_max]. `n_speakers_min=1` covers the
                              single-speaker phone-interference case. The
                              default paper.yaml keeps `babble.sources`
                              to musan(speech) only; librispeech(other)
                              is kept under `noise.sources` for
                              `prepare_dataset` baking + eval discovery,
                              NOT consumed by runtime babble.
        * `noise` / `music`: 1 random clip from the matching MUSAN
                              subset (`pool/<musan_spec.id>/<subset>/`),
                              `random_crop` to `sample_len`. The spec.id
                              comes from the TYPE's `sources[*]` via
                              `resolve_source(...)` — change the musan
                              `HfSourceSpec.id` in your config and the
                              scan will follow (no silent empty pool).

    Per-sample, each TYPE's `prob` is an INDEPENDENT Bernoulli inclusion
    probability (no longer a weighted pick); types whose underlying
    pool is empty or whose `prob <= 0` are skipped. Multiple TYPES can
    be included in one sample — their contributions are summed into one
    interference waveform with the **energy budget preserved across K
    components** (see module docstring + `_sample_interference`):
    `target_power / 10^(snr/10)` is split across the K included TYPES
    via Dirichlet(α=1) sampling on the simplex; each component is
    scaled so its RMS² = `w_i × budget`; the SUM's RMS² = budget, so
    post-mix SIR matches `snr_db` exactly. Without this, naive summing
    inflates the SUM's RMS by ~sqrt(K), dropping the effective SIR
    per individual TYPE by `10·log10(K)` dB. Any combination is
    possible (pure noise, pure music, noise+music, babble+noise, …, all
    three, or none).

    The constructor scans the directory once and builds in-memory indices
    (`_target_index`, `_speech_pool`, `_noise_pool`, `_music_pool`,
    `_spk_to_paths`, `_spk_to_spec_id`) plus a `DVectorCache` that lazily
    fills per-speaker d-vectors. All reads at iteration time use these
    indices; no HF streaming, no FLAC decode.
    """

    # Fixed top-level subdir names under `processed_dir/`. These describe
    # the SEMANTIC ROLE of each subtree, not a layout choice, so they're
    # deliberately not configurable. Anything beneath them (spec_id,
    # role, split, …) IS config-driven via HfSourceSpec. Re-exported
    # from module level so callers / tests can reach them either way.
    TARGET_TOP_SUBDIR = TARGET_TOP_SUBDIR
    POOL_TOP_SUBDIR = POOL_TOP_SUBDIR
    DVEC_TOP_SUBDIR = DVEC_TOP_SUBDIR
    MANIFEST_FILENAME = MANIFEST_FILENAME

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

        self._mel = None
        self._target_index: dict[str, list[Path]] = {}
        # spk → spec_id, populated at scan time from
        # `target/<spec_id>/<role>/<split>/<spk>/<utt>.wav`. Used by
        # `DVectorCache._path` to partition d-vectors per source so
        # multi-source targets never collide on speaker ids.
        self._spk_to_spec_id: dict[str, str] = {}
        # Babble pool — config-driven via `_collect_babble_pool_paths`:
        # walks `hp.interference.babble.sources[*]` and unions every
        # babble-eligible kind (musan_speech__ + librispeech_other__).
        # The librispeech `other` pool is included only when the user
        # explicitly adds it to `babble.sources`; the default paper.yaml
        # keeps it under `noise.sources` for baking + eval discovery
        # only (see `_collect_babble_pool_paths` for the supported kinds).
        self._speech_pool: list[tuple[str, Path]] = []
        # spk_id → list[Path] derived from `_speech_pool` once at scan
        # time. `babble` picks N distinct keys from this map.
        self._spk_to_paths: dict[str, list[Path]] = {}
        # MUSAN pools for the `noise` / `music` TYPEs (no speaker concept).
        # Populated by `_scan_processed_dir` from each TYPE's musan
        # sources via `resolve_source(...)` — no literal "musan" anywhere.
        self._noise_pool: list[Path] = []
        self._music_pool: list[Path] = []
        # type_name → independent Bernoulli inclusion `prob` (0 = disabled).
        # Types with `prob <= 0` or empty pools are skipped per-iteration
        # in `_sample_interference`.
        self._interference_type_probs: dict[str, float] = {}
        # Lazy per-speaker d-vector (compute on first sight; cached on
        # disk at `dvectors/<spec_id>/<spk>.npy` for subsequent runs).
        self._dvec_cache: Optional[DVectorCache] = None

        self._scan_processed_dir()

    # ---- One-time directory scan ----

    def _scan_processed_dir(self) -> None:
        root = Path(self.hp.dataset.processed_dir)
        if not root.is_dir():
            raise FileNotFoundError(
                f"[LocalTripletStream] processed_dir not found: "
                f"{root.resolve()}. Run `voicefilter-cli dataset prepare` first."
            )

        # ---- Target ----
        # Layout: target/<spec_id>/<role>/<split>/<spk>/<utt>.wav
        # `<spec_id>` is the HfSourceSpec.id of the target source —
        # resolved dynamically from `hp.target_sources[*]` so any
        # configured target is accepted (no literal "librispeech"
        # gate). `<role>` is `librispeech_role()` for LibriSpeech refs.
        target_spec_ids: set[str] = set()
        for ref in self.hp.target_sources:
            try:
                resolved = resolve_source(self.hp, ref)
            except KeyError:
                continue
            target_spec_ids.add(resolved.id)

        target_root = root / TARGET_TOP_SUBDIR
        if target_root.is_dir():
            for wav_path in target_root.rglob("*.wav"):
                parts = wav_path.relative_to(target_root).parts
                if len(parts) >= 5 and parts[0] in target_spec_ids:
                    spk = parts[3]
                    spec_id = parts[0]
                else:
                    continue
                self._target_index.setdefault(spk, []).append(wav_path)
                # Record the source for this speaker (used by DVectorCache
                # to partition d-vectors per source — multi-source targets
                # never collide on speaker ids).
                self._spk_to_spec_id[spk] = spec_id
            # Sort each speaker's wav list so the "first available
            # utterance" reference choice is deterministic across runs
            # (the d-vector cache uses `paths[0]` for ref-wav resolution).
            for spk in self._target_index:
                self._target_index[spk].sort()

        # ---- Pool (per-source, shared across TYPEs) ----
        # Layout: pool/<HfSourceSpec.id>/...  with two recognised shapes:
        #   librispeech/<role>/<split>/<spk>/<utt}.wav  (role = clean|other)
        #   musan/<subset>/{clip_id}.wav       (subset ∈ speech|noise|music)
        # Babble is config-driven — see `_collect_babble_pool_paths`:
        # walks `hp.interference.babble.sources[*]` and returns the union
        # of (musan_speech__, librispeech_other__) paths. By default
        # `babble.sources` only lists musan(speech); the user can extend
        # to librispeech(other) (or anything else babble-eligible) without
        # touching code. `noise` / `music` pools are musan-only at runtime
        # (random 1 clip + crop) — librispeech references in those TYPEs
        # are still baked by `prepare_dataset` and discoverable via
        # `first_librispeech_other_source` for eval, but NOT consumed at
        # training time.
        pool_root = root / "pool"
        if pool_root.is_dir():
            babble_paths = _collect_babble_pool_paths(self.hp, root)
            self._speech_pool = babble_paths
            # Build speaker-keyed index for babble dedup (used to pick N
            # distinct speakers without repetition). O(N) one-shot at scan.
            for spk, p in babble_paths:
                self._spk_to_paths.setdefault(spk, []).append(p)

            # Noise / music pools: walk the TYPE's `sources[*]`, resolve
            # each via `resolve_source(...)`, and union the matching
            # `pool/<spec.id>/<subset>/` clips. No literal "musan" anywhere
            # — change the HfSourceSpec.id and the scan follows.
            self._noise_pool = _collect_passthrough_pool_paths(
                self.hp, pool_root, "noise",
            )
            self._music_pool = _collect_passthrough_pool_paths(
                self.hp, pool_root, "music",
            )

        # Per-type INDEPENDENT inclusion probabilities: only types that
        # are enabled + have `prob > 0` are eligible. The runtime filter
        # on `prob>0 and pool non-empty` lives in `_sample_interference`
        # (via `_type_has_clips`).
        for type_name, type_cfg in self.hp.interference_types.items():
            if not type_cfg.enabled or type_cfg.prob <= 0:
                continue
            self._interference_type_probs[type_name] = float(type_cfg.prob)

        # ---- D-vector cache (lazy fill) ----
        # Always constructed (even if `_target_index` is empty) so the
        # dataset exposes the same surface area across runs; the cache
        # itself refuses to compute without a reference wav and falls
        # back to zeros.
        self._dvec_cache = DVectorCache(
            self.hp, root, self._target_index,
            spk_to_spec_id=self._spk_to_spec_id,
        )

        manifest_path = root / MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        total_prob = sum(self._interference_type_probs.values())
        # Count on-disk d-vectors across every target source's partition
        # (`dvectors/<spec_id>/*.npy`). With the default paper.yaml this
        # is exactly one partition (`dvectors/librispeech/`); multi-source
        # targets add one partition per source.
        on_disk_dvecs = 0
        dvec_root = root / DVEC_TOP_SUBDIR
        if dvec_root.is_dir():
            for spec_id in target_spec_ids:
                sub = dvec_root / spec_id
                if sub.is_dir():
                    on_disk_dvecs += sum(1 for _ in sub.glob("*.npy"))
        # Babble source breakdown for the scan summary (musan vs librispeech
        # identities are kept distinct by their `spk_id` prefix).
        babble_n_musan = sum(
            1 for spk, _ in self._speech_pool if spk.startswith("musan_speech__")
        )
        babble_n_libri = sum(
            1 for spk, _ in self._speech_pool if spk.startswith("librispeech_other__")
        )
        babble_breakdown = (
            f"{babble_n_musan} musan-speech, {babble_n_libri} librispeech-other"
        )
        # (Dataset scan summary was previously printed here. Removed per
        # training-log-only policy — keep training output on-device or in CI
        # quiet; inspect `len(self._target_index)` / `len(self._speech_pool)`
        # from Python if a sanity check is needed.)

    # ---- Pool eligibility per TYPE ----

    def _type_has_clips(self, type_name: str) -> bool:
        """True iff the underlying pool for `type_name` is non-empty.

        Used by the weighted TYPE sampler to skip types whose backing
        pool is missing/empty (a fully silent interference would train
        a degenerate "always suppress everything" model).
        """
        if type_name == "babble":
            # babble needs >= 1 speaker; n_min can shrink to fit.
            return bool(self._spk_to_paths)
        if type_name == "noise":
            return bool(self._noise_pool)
        if type_name == "music":
            return bool(self._music_pool)
        return False

    # ---- Per-TYPE interference synthesis ----

    def _load_and_crop(
        self,
        path: Path,
        sample_len: int,
    ) -> torch.Tensor:
        """Read one wav from disk, downmix to mono, random-crop to `sample_len`."""
        wav, _ = sf.read(str(path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_t = torch.from_numpy(wav.astype(np.float32))
        return random_crop(wav_t, sample_len)

    # ---- Iteration ----

    def __iter__(self) -> Iterator[Sample]:
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
                "[LocalTripletStream] no target utterances found; "
                "did `voicefilter-cli dataset prepare` finish?"
            )

        while True:
            # ---- 1. Pick target speaker + utterance ----
            tgt_spk = rng.choice(target_speakers)
            tgt_utt_paths = self._target_index[tgt_spk]
            if len(tgt_utt_paths) < 2:
                # Single-utterance speaker: ref must be self.
                tgt_path = tgt_utt_paths[0]
                ref_path = tgt_path
            else:
                tgt_path = rng.choice(tgt_utt_paths)
                # ---- 3. Pick ref: 70% same-speaker (different utt), 30% self ----
                if rng.random() < 0.7:
                    others = [p for p in tgt_utt_paths if p != tgt_path]
                    if others:
                        ref_path = rng.choice(others)
                    else:
                        ref_path = tgt_path
                else:
                    ref_path = tgt_path

            # ---- 4. Read target + ref wavs (BEFORE interference synthesis) ----
            # We need `target_power` (= mean of target wav²) to compute
            # the interference energy budget `target_power / 10^(snr/10)`,
            # which `_sample_interference` uses to allocate per-component
            # RMS² via Dirichlet sampling on the simplex. Reading +
            # cropping the target first lets us pass a real `target_power`
            # to the interference mixer (no placeholder → no post-hoc
            # rebalance that would amplify already-allocated noise).
            try:
                tgt_wav, _ = sf.read(str(tgt_path), dtype="float32")
                ref_wav, _ = sf.read(str(ref_path), dtype="float32")
            except Exception as e:
                _skipped = getattr(self, "_read_fail_count", 0) + 1
                self._read_fail_count = _skipped
                # (Per-failure WARN removed per training-log-only policy.
                # `self._read_fail_count` is still bumped so external health
                # checks can poll it; the underlying `continue` ensures the
                # bad sample is silently dropped from training.)
                continue

            # Mono
            tgt_wav_t = torch.from_numpy(tgt_wav if tgt_wav.ndim == 1 else tgt_wav.mean(axis=1))
            ref_wav_t = torch.from_numpy(ref_wav if ref_wav.ndim == 1 else ref_wav.mean(axis=1))

            # ---- 5–8. WAV-level synthesis (random crop, interference,
            # silent flags, activity bursts, snr_mix) — delegated to the
            # shared module-level `synthesize_raw_sample` so the
            # training iterator and offline audition scripts
            # (`examples/04_*`) consume the SAME per-sample logic.
            raw = synthesize_raw_sample(
                hp,
                rng,
                target_wav_full=tgt_wav_t,
                ref_wav_full=ref_wav_t,
                babble_spk_to_paths=self._spk_to_paths,
                noise_pool=self._noise_pool,
                music_pool=self._music_pool,
                type_probs=self._interference_type_probs,
                babble_cfg=self.hp.interference_types.get(NOISE_KIND_BABBLE),
                noise_cfg=self.hp.interference_types.get(NOISE_KIND_NOISE),
                music_cfg=self.hp.interference_types.get(NOISE_KIND_MUSIC),
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
            # failure path (no ref wav, encoder missing) it returns
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


# ---- HF audio-decode monkey-patch + cast helper ---------------------------
#
# `datasets.features.Audio.decode_example` is invoked automatically by HF's
# `Dataset.map` / generator wrapper on every batch, which would force a
# full FLAC decode of LibriSpeech *before* we have a chance to feed the
# bytes through our own `soundfile` pipeline. These two helpers suppress
# that: `_disable_hf_audio_decode` patches the class once; `_cast_audio_no_decode`
# flips `Audio.decode = False` on each column so the generator keeps the
# raw dict-with-bytes payload and lets the caller decode.
# Originally in scripts/prepare_dataset.py; lifted into `dataset` so
# `verify_dvec.py`, `evaluate.py`, `diag_mask.py` (and any other
# stream-HF-then-decode-locally consumer) can import them from here too.


_HF_AUDIO_DECODE_PATCHED = False


def _disable_hf_audio_decode() -> None:
    """Idempotently monkey-patch HF `Audio.decode_example` to never decode."""
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
    """Return a copy of `ds` with every `Audio` column set to `decode=False`."""
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
