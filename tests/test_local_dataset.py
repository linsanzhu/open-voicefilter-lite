"""Smoke tests for the HF-arrow-backed triplet stream pipeline.

These tests verify that:
    1. `HfTripletStream` finds target / interference-type pools /
       d-vector assets in mock HF `datasets.Dataset` objects and builds
       its in-memory `HfEntry` indices correctly under the
       function-oriented interference layout.
    2. `Sample` shapes match the model's expected inputs:
       - `stacked_mixed.shape == (T, n_mels * frame_stack)`
       - `target_logmel.shape == (T, n_mels)`
       - `overlap_label.shape == (T,)`, values in {-1, +1}
       - `ref_dvec.shape == (256,)`
    3. `collate_triplets` produces batched `(B, T, F)` tensors and the
       pre-baked d-vectors stack into `(B, 256)` — no None handling
       required.
    4. The noise-type predictor's per-frame overlap label is consistent
       with the burst windows sampled in the same iteration.
    5. Independent TYPE inclusion probabilities
       (`hp.interference.{babble,noise,music}.prob`) are honored
       per-sample.
    6. The `DVectorCache` lazy-fill path: a missing d-vector `.npy` is
       computed via `extract_dvec` and persisted to disk on first sight.

The test builds a *minimal* mock-HF-Dataset fixture set (3 target
speakers × 2 utts + 3 musan/speech clips + 3 musan/noise + 3
musan/music + 6 librispeech(other) utts across 3 spks) and monkey-
patches `_open_hf_dataset` to return them. No HuggingFace download,
no `models/embedder.pt` required (we monkey-patch `extract_dvec`).

Run with:
    python tests/test_local_dataset.py

Exit code 0 = all tests passed.
"""

from __future__ import annotations

import io
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Make the model + inference source dirs importable when running this
# file directly. Training data pipeline (`HfTripletStream`, mel extract,
# triplet collate) is in the workspace-only `model` package; HParams /
# SourceRef live in the published `open-voicefilter-lite` package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "model" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "open-voicefilter-lite" / "src"))

from model import dataset as _dataset_mod
from model.dataset import (
    HfTripletStream,
    _collect_babble_pool_entries,
    _pick_babble_speakers,
    _mix_babble_from_picks,
    _allocate_interference_budget,
    _per_frame_overlap_label,
    collate_triplets,
    default_hf_wav_loader,
    synthesize_raw_sample,
)
from voicefilter_lite.hparams import HParams, SourceRef


# ---- Test fixtures ----------------------------------------------------------


SR = 16000


def _flac_bytes(seconds: float = 4.0, *, seed: int | None = None) -> bytes:
    """Encode a random mono float32 wav to FLAC bytes."""
    T = int(seconds * SR)
    rng = np.random.default_rng(seed=seed if seed is not None else T)
    wav = (0.2 * rng.standard_normal(T).astype(np.float32))
    buf = io.BytesIO()
    sf.write(buf, wav, SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


class _MockHfRow(dict):
    """Row-like dict — supports `row.get(key)` (inherited from dict).

    HF's `datasets.Dataset[row_idx]` returns a dict-like row.
    """


class _MockHfDataset:
    """Mock `datasets.Dataset` for tests.

    Holds a list of row dicts; `__len__` and `__getitem__(int)` mirror
    the real interface. The `__getitem__` accepts an `int` only (the
    streaming iterator's slice-by-row path) and returns a fresh dict
    each time so callers can't mutate the underlying store by accident.
    """

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key):
        if isinstance(key, int):
            return _MockHfRow(self._rows[key])
        if isinstance(key, slice):
            return [_MockHfRow(r) for r in self._rows[key]]
        raise TypeError(
            f"_MockHfDataset only supports int/slice __getitem__; got {type(key).__name__}"
        )


# Per-fixture registry: `_dataset_id(resolved, split)` → _MockHfDataset
_MOCK_DATASETS: dict[str, _MockHfDataset] = {}
# Per-fixture registry: `_dataset_id` → list of `(row_idx, spk_id, utt_id)`
# populated at fixture-bake time so `_row_metadata` can return stable
# ids without walking the dataset again.


def _bake_fixtures(hp: HParams, dvec_dir_root: Path) -> None:
    """Populate `_MOCK_DATASETS` with the minimum assets a
    `HfTripletStream` iteration needs, and pre-seed per-speaker
    d-vectors at `<dvec_dir_root>/voicefilter-dvectors/<spec_id>/<spk>.npy`.

    The target / babble / noise / music pools are mounted under mock
    `(spec_id, subset, split)` keys that match the canonical
    `_dataset_id(resolved, split)` format so `HfTripletStream._scan_hf_datasets`
    picks them up via the monkey-patched `_open_hf_dataset`.
    """
    _MOCK_DATASETS.clear()

    # ---- Target: librispeech(train.clean.100) — 3 spks × 2 utts ----
    target_rows: list[dict] = []
    speakers = ["100", "200", "300"]
    for spk in speakers:
        for utt_idx in range(2):
            target_rows.append({
                "audio": {"bytes": _flac_bytes(4.0, seed=int(spk) * 100 + utt_idx),
                          "path": f"{spk}/utt_{utt_idx}.flac"},
                "speaker_id": int(spk),
                "id": f"utt_{utt_idx}",
                "file": f"{spk}/utt_{utt_idx}.flac",
            })
    _MOCK_DATASETS["librispeech/train.clean.360"] = _MockHfDataset(target_rows)

    # ---- Pool: librispeech(other)(train.other.500) — 3 spks × 2 utts ----
    other_rows: list[dict] = []
    for spk in ["1001", "1002", "1003"]:
        for utt_idx in range(2):
            other_rows.append({
                "audio": {"bytes": _flac_bytes(4.0, seed=int(spk) * 7 + utt_idx),
                          "path": f"{spk}/utt_{utt_idx}.flac"},
                "speaker_id": int(spk),
                "id": f"utt_{utt_idx}",
                "file": f"{spk}/utt_{utt_idx}.flac",
            })
    _MOCK_DATASETS["librispeech/train.other.500"] = _MockHfDataset(other_rows)

    # ---- Pool: musan(speech) — 3 clips, each one "speaker" identity ----
    speech_rows: list[dict] = []
    for clip_idx in range(3):
        path = f"musan_{clip_idx:02d}.flac"
        speech_rows.append({
            "audio": {"bytes": _flac_bytes(8.0, seed=1000 + clip_idx),
                      "path": path},
            "id": f"musan_{clip_idx:02d}",
            "file": path,
        })
    _MOCK_DATASETS["musan/speech/train"] = _MockHfDataset(speech_rows)

    # ---- Pool: musan(noise) — 3 clips ----
    noise_rows: list[dict] = []
    for clip_idx in range(3):
        path = f"noise_{clip_idx:02d}.flac"
        noise_rows.append({
            "audio": {"bytes": _flac_bytes(3.5, seed=2000 + clip_idx),
                      "path": path},
            "id": f"noise_{clip_idx:02d}",
            "file": path,
        })
    _MOCK_DATASETS["musan/noise/train"] = _MockHfDataset(noise_rows)

    # ---- Pool: musan(music) — 3 clips ----
    music_rows: list[dict] = []
    for clip_idx in range(3):
        path = f"music_{clip_idx:02d}.flac"
        music_rows.append({
            "audio": {"bytes": _flac_bytes(3.5, seed=3000 + clip_idx),
                      "path": path},
            "id": f"music_{clip_idx:02d}",
            "file": path,
        })
    _MOCK_DATASETS["musan/music/train"] = _MockHfDataset(music_rows)

    # ---- Per-speaker d-vectors (synthetic; no encoder needed) ----
    # DVectorCache writes to `<cache_dir>/voicefilter-dvectors/<spec_id>/<spk>.npy`
    # where `<cache_dir>` is the HfSourceSpec's `cache_dir` field. The
    # mock cache root is a temp dir; HfSourceSpec.cache_dir is set
    # there at the call site (`main` below).
    for spk in speakers:
        rng = np.random.default_rng(seed=int(spk) + 999)
        v = rng.standard_normal(hp.d_vector_dim).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)        # L2-normalise
        spk_path = dvec_dir_root / "librispeech" / f"{spk}.npy"
        spk_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(spk_path), v)


# ---- Tests ------------------------------------------------------------------


def _check_sample_shapes(sample, hp: HParams, step: int) -> None:
    """Assert `sample` carries the shape contract the model expects."""
    expected_T = hp.num_frames
    assert sample.stacked_mixed.shape == (expected_T, hp.n_mels * hp.frame_stack), (
        f"[step {step}] stacked_mixed shape "
        f"{tuple(sample.stacked_mixed.shape)} != "
        f"({expected_T}, {hp.n_mels * hp.frame_stack})"
    )
    assert sample.target_logmel.shape == (expected_T, hp.n_mels), (
        f"[step {step}] target_logmel shape "
        f"{tuple(sample.target_logmel.shape)} != ({expected_T}, {hp.n_mels})"
    )
    assert sample.overlap_label is not None
    assert sample.overlap_label.shape == (expected_T,), (
        f"[step {step}] overlap_label shape "
        f"{tuple(sample.overlap_label.shape)} != ({expected_T},)"
    )
    uniq = set(sample.overlap_label.unique().tolist())
    assert uniq.issubset({-1, 1}), (
        f"[step {step}] overlap_label values {uniq} ⊄ {{-1, +1}}"
    )
    assert sample.ref_dvec.shape == (hp.d_vector_dim,), (
        f"[step {step}] ref_dvec shape {tuple(sample.ref_dvec.shape)} != "
        f"({hp.d_vector_dim},)"
    )
    assert sample.ref_dvec.dtype == torch.float32, (
        f"[step {step}] ref_dvec dtype {sample.ref_dvec.dtype} != float32"
    )
    norm = float(sample.ref_dvec.norm().item())
    assert abs(norm - 1.0) < 1e-3, (
        f"[step {step}] ref_dvec not L2-normalised: |dvec|={norm:.4f}"
    )
    assert isinstance(sample.ref_spk_id, str) and sample.ref_spk_id, (
        f"[step {step}] ref_spk_id must be a non-empty string; "
        f"got {sample.ref_spk_id!r}"
    )
    assert isinstance(sample.noise_kind, tuple), (
        f"[step {step}] noise_kind must be a tuple; got "
        f"{type(sample.noise_kind).__name__}"
    )
    valid = {"babble", "noise", "music"}
    for k in sample.noise_kind:
        assert k in valid, f"[step {step}] unknown noise_kind element {k!r}"
    canonical = [k for k in sample.noise_kind if k in valid]
    assert canonical == sorted(canonical, key=lambda x: ("babble", "noise", "music").index(x)), (
        f"[step {step}] noise_kind order not canonical "
        f"(babble, noise, music); got {sample.noise_kind!r}"
    )


def test_local_triplet_iteration(hp: HParams) -> None:
    """Iterate `HfTripletStream` 50 times and assert shape contract."""
    ds = HfTripletStream(hp, worker_id=0, num_workers=1, seed=1234)
    n_seen = 0
    n_seen_by_subset: dict[tuple, int] = {}
    for sample in ds:
        _check_sample_shapes(sample, hp, n_seen)
        n_seen_by_subset[sample.noise_kind] = (
            n_seen_by_subset.get(sample.noise_kind, 0) + 1
        )
        n_seen += 1
        if n_seen >= 50:
            break
    assert n_seen == 50, f"only iterated {n_seen} samples; expected 50"
    print(
        f"  HfTripletStream OK — 50 samples; "
        f"interference subsets seen: {sorted(n_seen_by_subset.items())}"
    )


def test_collate_triplets(hp: HParams) -> None:
    """`collate_triplets` produces the (B, T, ...) batched tensors the
    training loop expects.
    """
    ds = HfTripletStream(hp, worker_id=0, num_workers=1, seed=999)
    batch_size = 4
    samples = []
    for i, sample in enumerate(ds):
        samples.append(sample)
        if len(samples) >= batch_size:
            break
    assert len(samples) == batch_size, (
        f"only collected {len(samples)} samples; expected {batch_size}"
    )

    batch = collate_triplets(samples)
    expected_T = hp.num_frames

    assert batch["stacked_mixed"].shape == (
        batch_size, expected_T, hp.n_mels * hp.frame_stack,
    ), f"stacked_mixed batch shape {tuple(batch['stacked_mixed'].shape)}"
    assert batch["target_logmel"].shape == (
        batch_size, expected_T, hp.n_mels,
    ), f"target_logmel batch shape {tuple(batch['target_logmel'].shape)}"
    assert batch["overlap_label"].shape == (batch_size, expected_T), (
        f"overlap_label batch shape {tuple(batch['overlap_label'].shape)}"
    )
    assert batch["ref_dvecs"].shape == (batch_size, hp.d_vector_dim), (
        f"ref_dvecs batch shape {tuple(batch['ref_dvecs'].shape)}"
    )
    assert batch["ref_dvecs"].dtype == torch.float32, (
        f"ref_dvecs dtype {batch['ref_dvecs'].dtype} != float32"
    )
    assert isinstance(batch["ref_spk_ids"], list)
    assert len(batch["ref_spk_ids"]) == batch_size
    assert all(isinstance(s, str) and s for s in batch["ref_spk_ids"]), (
        "ref_spk_ids must be a list of non-empty strings"
    )
    assert isinstance(batch["noise_kind"], list)
    assert len(batch["noise_kind"]) == batch_size
    print(
        f"  collate_triplets OK — B={batch_size}, T={expected_T}, "
        f"stacked={tuple(batch['stacked_mixed'].shape)} "
        f"target={tuple(batch['target_logmel'].shape)} "
        f"dvecs={tuple(batch['ref_dvecs'].shape)}"
    )


def test_missing_dvec_triggers_compute_and_caches(hp: HParams) -> None:
    """When a speaker's d-vector `.npy` is missing on disk, the
    `DVectorCache` should compute it on first sight and persist it.
    """
    import model.audio as _audio_mod

    mock_calls: list[str] = []

    def _mock_extract_dvec(wav, hp_, encoder, device=None, ref_lens=None):
        mock_calls.append("called")
        d = hp_.d_vector_dim
        out = torch.zeros(1, d, dtype=torch.float32)
        out[0, 0] = 1.0
        return out

    _audio_mod.extract_dvec = _mock_extract_dvec

    # Delete spk 200's dvec on disk so the next sight forces compute.
    target_path = (
        Path(hp.dataset.sources[0].cache_dir)
        / "voicefilter-dvectors" / "librispeech" / "200.npy"
    )
    assert target_path.is_file(), "fixture setup: 200.npy should exist"
    target_path.unlink()

    ds = HfTripletStream(hp, worker_id=0, num_workers=1, seed=0)
    # Lazy init: the in-memory `_target_index` is populated by the
    # first `__iter__` call (per worker process). Slice the index
    # AFTER triggering that one-shot scan so the test picks up the
    # expected single-speaker view.
    ds._ensure_initialized()
    ds._target_index = {"200": ds._target_index["200"]}

    seen = False
    n_steps = 0
    for sample in ds:
        if sample.ref_spk_id == "200":
            assert mock_calls, "DVectorCache should call extract_dvec on miss"
            assert sample.ref_dvec.shape == (hp.d_vector_dim,)
            assert sample.ref_dvec.dtype == torch.float32
            assert abs(float(sample.ref_dvec[0].item()) - 1.0) < 1e-5, (
                f"expected dvec[0]=1.0 from mock; got "
                f"{float(sample.ref_dvec[0].item()):.4f}"
            )
            assert target_path.is_file(), (
                "DVectorCache should persist the computed d-vector on miss"
            )
            on_disk = np.load(str(target_path))
            assert on_disk.shape == (hp.d_vector_dim,)
            assert abs(float(on_disk[0]) - 1.0) < 1e-5, (
                f"on-disk dvec mismatch: dvec[0]={float(on_disk[0]):.4f}"
            )
            seen = True
        n_steps += 1
        if n_steps >= 10 and seen:
            break
    assert seen, (
        "expected to draw speaker 200 after deleting dvectors/200.npy"
    )
    print("  missing d-vector → DVectorCache compute + persist OK")


def test_interference_type_probs_are_honored(hp: HParams) -> None:
    """v3 categorical sampler: bucket + single-weight rates match config.

    The HParams default categorical distribution is:

        interference_none_weight       = 0.10
        interference_single_weights    = {babble: 0.40, noise: 0.10, music: 0.10}
        interference_mixed_weight      = 0.30

    so over the long run the empirical rates should converge to those
    numbers. We don't compute closed-form per-TYPE rates (mixed-bucket
    composition is itself a weighted-sample), but we check three things:

        1. Empty-tuple rate matches `interference_none_weight` (within 5pp).
        2. Per-TYPE rates preserve the config ordering
           (babble weight >> noise weight == music weight) — babble
           inclusion rate must be at least 4× noise (babble's single
           weight is 4× larger).
        3. At least one multi-TYPE sample in 800 draws.

    Together these three cover the categorical-sampler contract without
    requiring an analytic closed-form for the mixed-bucket composition.
    """
    ds = HfTripletStream(hp, worker_id=0, num_workers=1, seed=42)
    n_seen = 0
    n_with_type: dict[str, int] = {"babble": 0, "noise": 0, "music": 0}
    n_empty = 0
    n_combinations: dict[tuple[str, ...], int] = {}
    for sample in ds:
        kinds = sample.noise_kind
        n_combinations[kinds] = n_combinations.get(kinds, 0) + 1
        if not kinds:
            n_empty += 1
        for k in kinds:
            n_with_type[k] = n_with_type.get(k, 0) + 1
        n_seen += 1
        if n_seen >= 800:
            break
    print(f"  TYPE inclusion counts over {n_seen} samples:")
    print(f"    empty={n_empty}, "
          f"{', '.join(f'{k}+={n_with_type[k]}' for k in ('babble','noise','music'))}")
    print(f"    combinations: {sorted(n_combinations.items())}")
    # (1) Empty-tuple (none) rate.
    expected_empty = hp.interference_none_weight
    got_empty = n_empty / n_seen
    assert abs(got_empty - expected_empty) < 0.05, (
        f"empty-tuple fraction {got_empty:.3f} deviates from "
        f"interference_none_weight={expected_empty:.3f} by more than "
        f"5pp: n_empty={n_empty}/{n_seen}"
    )
    # (2) Per-TYPE relative ordering: babble's single_weight is 4× noise's
    # AND music's, and the per-TYPE P(in any sample) ≈ single[t] + mixed
    # weighting that also favours babble. Empirically babble should be
    # at least 2× more frequent than noise/music in 800 samples.
    rates = {t: n_with_type[t] / n_seen for t in ("babble", "noise", "music")}
    assert rates["babble"] >= 2 * rates["noise"], (
        f"babble inclusion rate {rates['babble']:.3f} should be >= 2× "
        f"noise rate {rates['noise']:.3f} (babble single_weight=0.40 "
        f"vs noise 0.10): rates={rates}"
    )
    assert rates["babble"] >= 2 * rates["music"], (
        f"babble inclusion rate {rates['babble']:.3f} should be >= 2× "
        f"music rate {rates['music']:.3f}: rates={rates}"
    )
    # (3) At least one multi-TYPE sample.
    n_multi = sum(c for k, c in n_combinations.items() if len(k) >= 2)
    assert n_multi > 0, (
        f"expected at least one multi-TYPE sample in {n_seen}; got "
        f"{n_combinations}"
    )


def test_disabled_interference_type_never_appears(hp: HParams) -> None:
    """If a TYPE has `prob <= 0` or `enabled=False`, the stream must
    never include it.
    """
    hp.interference_types["babble"].prob = 0.0
    hp.interference_types["music"].prob = 0.0
    ds = HfTripletStream(hp, worker_id=0, num_workers=1, seed=7)
    n_seen = 0
    seen_kinds: set[str] = set()
    for sample in ds:
        seen_kinds.update(sample.noise_kind)
        n_seen += 1
        if n_seen >= 80:
            break
    assert seen_kinds <= {"noise"}, (
        f"disabled types leaked into stream; saw {seen_kinds}"
    )
    assert "babble" not in seen_kinds and "music" not in seen_kinds, (
        f"zero-prob types must never be sampled; saw {seen_kinds}"
    )
    print(f"  disabled-type filter OK (seen={seen_kinds})")


def test_babble_picks_distinct_speakers(hp: HParams) -> None:
    """Babble picker returns N distinct speakers per draw."""
    ds = HfTripletStream(hp, worker_id=0, num_workers=1, seed=11)
    # Lazy init: the babble pool (`_spk_to_entries`) is built by the
    # first `__iter__` call. Trigger the one-shot scan up front so
    # this test can introspect pool membership directly.
    ds._ensure_initialized()
    n_speakers = len(ds._spk_to_entries)
    assert n_speakers == 3, (
        f"fixture should yield 3 distinct musan_speech speakers; "
        f"got {n_speakers}: {sorted(ds._spk_to_entries.keys())}"
    )
    rng = random.Random(11)
    cfg = hp.interference_types["babble"]
    n_min = int(cfg.n_speakers_min)
    n_max = int(cfg.n_speakers_max)
    seen_counts: dict[str, int] = {spk: 0 for spk in ds._spk_to_entries}
    n_draws = 200
    for _ in range(n_draws):
        picks = _pick_babble_speakers(
            rng, ds._spk_to_entries, n_min, n_max,
        )
        assert n_min <= len(picks) <= n_max, (
            f"len(picks)={len(picks)} ∉ [{n_min}, {n_max}]"
        )
        spk_ids = [spk for spk, _p in picks]
        assert len(set(spk_ids)) == len(spk_ids), (
            f"duplicate speakers in single babble pick: {spk_ids}"
        )
        for spk, _p in picks:
            seen_counts[spk] = seen_counts.get(spk, 0) + 1
    missing = [spk for spk, c in seen_counts.items() if c == 0]
    assert not missing, (
        f"speakers never picked in {n_draws} babble draws: {missing}"
    )
    print(
        f"  babble dedup OK — {n_draws} draws, n∈[{n_min},{n_max}], "
        f"all {n_speakers} speakers covered; per-speaker counts "
        f"min={min(seen_counts.values())} "
        f"max={max(seen_counts.values())}"
    )


def test_babble_mixer_emits_full_length_wav(hp: HParams) -> None:
    """`_mix_babble_from_picks` should return a `(sample_len,)` wav
    that contains at least some non-zero samples when picks are non-empty.
    """
    ds = HfTripletStream(hp, worker_id=0, num_workers=1, seed=23)
    # Lazy init: trigger the one-shot scan so the babble pool is
    # populated before the test introspects it.
    ds._ensure_initialized()
    rng = random.Random(23)
    cfg = hp.interference_types["babble"]
    sr = hp.sample_rate
    sample_len = hp.sample_length_samples
    n_nonzero = 0
    n_total = 30
    for _ in range(n_total):
        picks = _pick_babble_speakers(
            rng, ds._spk_to_entries, cfg.n_speakers_min, cfg.n_speakers_max,
        )
        # `_mix_babble_from_picks` needs a wav_loader; the test
        # picks speakers from the babble pool and decodes each via
        # the default HF loader.
        wav = _mix_babble_from_picks(
            rng=rng,
            sample_len=sample_len,
            sr=sr,
            picks=picks,
            seg_min_sec=cfg.babble_seg_min_sec,
            seg_max_sec=cfg.babble_seg_max_sec,
            gain_min=cfg.gain_min,
            gain_max=cfg.gain_max,
            wav_loader=default_hf_wav_loader,
        )
        assert wav.shape == (sample_len,), (
            f"babble wav shape {tuple(wav.shape)} != ({sample_len},)"
        )
        assert wav.dtype == torch.float32, (
            f"babble wav dtype {wav.dtype} != float32"
        )
        if float(wav.abs().max()) > 1e-6:
            n_nonzero += 1
    assert n_nonzero >= n_total * 0.8, (
        f"too many silent babble outputs: {n_nonzero}/{n_total}"
    )
    print(
        f"  babble mixer OK — {n_nonzero}/{n_total} non-silent outputs, "
        f"all shape=({sample_len},)"
    )


def test_babble_pool_is_config_driven(hp: HParams) -> None:
    """The babble pool is CONFIG-DRIVEN — built from
    `hp.interference.babble.sources[*]`, NOT hardcoded.
    """
    babble_cfg = hp.interference_types["babble"]

    def _scan_with(sources, prob=0.5):
        babble_cfg.sources = sources
        babble_cfg.prob = prob
        # _collect_babble_pool_entries reads from `hp` directly + a
        # dataset_cache dict; we pass the stream's existing cache so
        # both the unchanged-default scan and the mutated scan
        # observe the same fixture data.
        return _collect_babble_pool_entries(hp, _dataset_mod_cache())

    # (a) musan only — fixture bakes 3 musan_speech clips → 3 spks
    pool_a = _scan_with((SourceRef(source="musan", subset="speech"),))
    assert {spk for spk, _ in pool_a} == {
        "musan_speech__musan_00",
        "musan_speech__musan_01",
        "musan_speech__musan_02",
    }, {spk for spk, _ in pool_a}
    # (b) librispeech only — fixture bakes 3 spks × 2 utts = 6 entries → 3 spks
    pool_b = _scan_with((
        SourceRef(source="librispeech", splits=("train.other.500",)),
    ))
    assert {spk for spk, _ in pool_b} == {
        "librispeech_other__1001",
        "librispeech_other__1002",
        "librispeech_other__1003",
    }, {spk for spk, _ in pool_b}
    # (c) both — union
    pool_c = _scan_with((
        SourceRef(source="musan", subset="speech"),
        SourceRef(source="librispeech", splits=("train.other.500",)),
    ))
    spk_ids_c = {spk for spk, _ in pool_c}
    assert spk_ids_c == {
        "musan_speech__musan_00",
        "musan_speech__musan_01",
        "musan_speech__musan_02",
        "librispeech_other__1001",
        "librispeech_other__1002",
        "librispeech_other__1003",
    }, spk_ids_c
    assert len(pool_c) == 9, f"expected 3 musan + 6 librispeech = 9; got {len(pool_c)}"
    # (d) disabled (prob=0) — empty pool even with sources configured
    pool_d = _scan_with(
        (SourceRef(source="musan", subset="speech"),),
        prob=0.0,
    )
    assert pool_d == [], f"disabled babble should yield empty pool; got {pool_d}"
    babble_cfg.enabled = False
    pool_e = _scan_with(
        (SourceRef(source="musan", subset="speech"),),
    )
    assert pool_e == [], f"enabled=False babble should yield empty pool; got {pool_e}"
    babble_cfg.enabled = True
    _scan_with((SourceRef(source="musan", subset="speech"),))
    print(
        f"  config-driven babble pool OK — "
        f"(a) musan only={len(pool_a)} clips; "
        f"(b) librispeech only={len(pool_b)} clips ({len({s for s, _ in pool_b})} spk); "
        f"(c) union={len(pool_c)} clips; "
        f"(d) disabled={len(pool_d)} clips; "
        f"(e) enabled=False={len(pool_e)} clips"
    )


def test_interference_budget_is_preserved(hp: HParams) -> None:
    """After budget allocation, the SUM RMS² must equal
    `target_power / 10^(snr/10)`.
    """
    sr = hp.sample_rate
    sample_len = sr * 3
    target_power = 0.04
    snr_db = 5.0
    expected_budget = target_power / (10.0 ** (snr_db / 10.0))

    rng = np.random.default_rng(0)
    babble = (1.0 * rng.standard_normal(sample_len).astype(np.float32))
    noise = (0.05 * rng.standard_normal(sample_len).astype(np.float32))
    music = (0.5 * rng.standard_normal(sample_len).astype(np.float32))
    components = [
        torch.from_numpy(babble),
        torch.from_numpy(noise),
        torch.from_numpy(music),
    ]
    nat_rms2 = [float((w ** 2).mean().item()) for w in components]
    assert nat_rms2[0] / nat_rms2[1] > 50, (
        f"babble RMS²/noise RMS² = {nat_rms2[0]/nat_rms2[1]:.0f}×; "
        f"expected > 50× for a meaningful budget test"
    )

    aggregate = _allocate_interference_budget(
        components=components,
        target_power=target_power,
        snr_db=snr_db,
        rng=random.Random(0),
    )
    actual_budget = float((aggregate ** 2).mean().item())
    assert abs(actual_budget - expected_budget) / expected_budget < 0.30, (
        f"aggregate RMS² {actual_budget:.5f} deviates from budget "
        f"{expected_budget:.5f} by > 30%: "
        f"target_power={target_power}, snr={snr_db} dB"
    )
    peak = float(aggregate.abs().max().item())
    assert peak <= 1.0 + 1e-5, (
        f"aggregate peak {peak:.3f} > 1.0 (soft-clip failed)"
    )

    only_babble = _allocate_interference_budget(
        components=[torch.from_numpy(babble)],
        target_power=target_power,
        snr_db=snr_db,
        rng=random.Random(1),
    )
    only_babble_budget = float((only_babble ** 2).mean().item())
    assert abs(only_babble_budget - expected_budget) / expected_budget < 0.30, (
        f"K=1 budget {only_babble_budget:.5f} deviates from "
        f"expected {expected_budget:.5f}"
    )

    empty = _allocate_interference_budget(
        components=[],
        target_power=target_power,
        snr_db=snr_db,
        rng=random.Random(2),
    )
    assert empty.numel() == 0, (
        f"empty-path aggregate should be length-0; got numel={empty.numel()}"
    )

    print(
        f"  budget OK — K=3 RMS² {actual_budget:.4f} ≈ target "
        f"{expected_budget:.4f} ({abs(actual_budget-expected_budget)/expected_budget*100:.1f}% drift); "
        f"K=1 RMS² {only_babble_budget:.4f} ≈ target"
    )


def test_silent_target_keeps_interference_at_natural_energy(hp: HParams) -> None:
    """Silent-target branch: when `target_power` is at the eps sentinel,
    the helper must NOT scale the aggregate to silence.
    """
    sr = hp.sample_rate
    sample_len = sr * 3
    rng = np.random.default_rng(0)
    noise = (0.3 * rng.standard_normal(sample_len).astype(np.float32))
    music = (0.7 * rng.standard_normal(sample_len).astype(np.float32))
    components = [
        torch.from_numpy(noise),
        torch.from_numpy(music),
    ]
    eps_target_power = 1e-12

    clipped = []
    for w in components:
        peak = float(w.abs().max().item())
        clipped.append(w / peak if peak > 1.0 else w)
    natural_sum = sum(clipped)
    nat_peak = float(natural_sum.abs().max().item())
    if nat_peak > 1.0:
        natural_sum = torch.tanh(natural_sum)
    natural_rms2 = float((natural_sum ** 2).mean().item())

    silent = _allocate_interference_budget(
        components=components,
        target_power=eps_target_power,
        snr_db=5.0,
        rng=random.Random(0),
        apply_budget=False,
    )
    silent_peak = float(silent.abs().max().item())
    assert silent_peak <= 1.0 + 1e-5, (
        f"silent-target aggregate peak {silent_peak:.3f} > 1.0 "
        f"(soft-clip failed)"
    )
    silent_rms2 = float((silent ** 2).mean().item())
    assert abs(silent_rms2 - natural_rms2) / natural_rms2 < 0.05, (
        f"silent-target RMS² {silent_rms2:.5f} deviates from natural "
        f"sum {natural_rms2:.5f} by > 5%"
    )
    assert silent_rms2 > 1e-3, (
        f"silent-target RMS² {silent_rms2:.6f} too low — the model "
        f"would see silence instead of interference"
    )

    print(
        f"  silent-target branch OK — natural RMS²={natural_rms2:.4f}, "
        f"silent-target RMS²={silent_rms2:.4f} "
        f"(≈audible, NOT scaled to ~0 by budget)"
    )


def test_target_rms_norm_pins_post_crop_rms(hp: HParams) -> None:
    """`target_rms_norm_db` pin post-burst-mask target RMS to the dBFS target.

    Three sub-cases:
      * disabled (None) → natural RMS passes through (scaled only by
        the burst-mask + crop, which is the legacy behaviour);
      * enabled (-20 dBFS = RMS=0.1) → post-crop RMS hits 0.1 within
        1% on a typical speech-like signal;
      * silent / near-silent target → guard skips the scale (no
        divide-by-zero, no NaN, peak = 0.0).
    """
    sr = hp.sample_rate
    # 1.5 s speech-ish signal (decaying sinusoid → RMS ~0.1 naturally).
    N = int(1.5 * sr)
    t = np.arange(N) / sr
    wav_full = torch.from_numpy(
        (0.3 * np.sin(2 * np.pi * 200.0 * t) * np.exp(-t * 2.0))
        .astype(np.float32)
    )
    ref_full = torch.from_numpy(
        (0.05 * np.random.default_rng(0).standard_normal(N).astype(np.float32))
    )

    def _run(rms_db):
        hp.target_rms_norm_db = rms_db
        rng = random.Random(0)
        return synthesize_raw_sample(
            hp, rng,
            target_wav_full=wav_full,
            ref_wav_full=ref_full,
            babble_spk_to_entries={}, noise_pool=[], music_pool=[],
            type_probs={"babble": 0.0, "noise": 0.0, "music": 0.0},
            babble_cfg=None, noise_cfg=None, music_cfg=None,
            multi_type_prob=0.0,
        )

    # Case A: disabled → no scaling (output RMS equals the natural
    # burst-masked + cropped RMS, NOT the configured 0.1).
    raw = _run(None)
    out_rms_disabled = float((raw.target_wav ** 2).mean().sqrt().item())
    assert not (abs(out_rms_disabled - 0.1) < 1e-3), (
        f"[disabled] output RMS {out_rms_disabled:.4f} accidentally "
        f"landed on the 0.1 target — the guard is probably broken"
    )

    # Case B: enabled → output RMS hits the target within 1%.
    raw = _run(-20.0)
    out_rms_enabled = float((raw.target_wav ** 2).mean().sqrt().item())
    assert abs(out_rms_enabled - 0.1) < 0.01, (
        f"[-20 dBFS] output RMS {out_rms_enabled:.4f} != 0.1 ± 1%"
    )
    # Defensive peak cap: RMS=0.1 should never push a speech peak
    # above 1.0 (LibriSpeech crest factor is ≲ 10).
    peak = float(raw.target_wav.abs().max().item())
    assert peak <= 1.0 + 1e-5, (
        f"[-20 dBFS] peak {peak:.3f} > 1.0 — peak cap failed"
    )

    # Case C: silent target → guard skipped, no NaN, peak = 0.0.
    hp.target_rms_norm_db = -20.0
    rng = random.Random(0)
    raw = synthesize_raw_sample(
        hp, rng,
        target_wav_full=torch.zeros(N, dtype=torch.float32),
        ref_wav_full=ref_full,
        babble_spk_to_entries={}, noise_pool=[], music_pool=[],
        type_probs={"babble": 0.0, "noise": 0.0, "music": 0.0},
        babble_cfg=None, noise_cfg=None, music_cfg=None,
        multi_type_prob=0.0,
    )
    assert not torch.isnan(raw.target_wav).any().item(), (
        "silent target produced NaN — divide-by-zero guard failed"
    )
    assert float(raw.target_wav.abs().max().item()) == 0.0, (
        "silent target post-norm peak should remain 0.0"
    )

    print(
        f"  target_rms_norm OK — disabled={out_rms_disabled:.4f} "
        f"(raw), enabled={out_rms_enabled:.4f} (target=0.1), "
        f"silent=0.0 (no NaN)"
    )


def test_interference_post_mix_sir_matches_configured(hp: HParams) -> None:
    """End-to-end smoke: iterate 200 times and verify TYPE inclusion
    distribution matches the independent-Bernoulli expectation.
    """
    hp.snr_min_db = 5.0
    hp.snr_max_db = 5.0
    hp.dataloader_workers = 0
    hp.prefetch_batches_max = 1

    ds = HfTripletStream(hp, worker_id=0, num_workers=1, seed=2025)
    n_seen = 0
    n_with_mix = 0
    n_kind_counts: dict[tuple[str, ...], int] = {}
    iterator = iter(ds)
    while n_seen < 200:
        sample = next(iterator)
        n_seen += 1
        assert sample.stacked_mixed.dtype == torch.float32
        assert sample.target_logmel.dtype == torch.float32
        assert sample.overlap_label.shape == (hp.num_frames,)
        assert sample.ref_dvec.shape == (hp.d_vector_dim,)
        n_kind_counts[sample.noise_kind] = (
            n_kind_counts.get(sample.noise_kind, 0) + 1
        )
        if sample.noise_kind:
            n_with_mix += 1
    got = n_with_mix / 200
    assert abs(got - 0.86) < 0.10, (
        f"non-empty noise_kind {got:.3f} deviates from expected "
        f"0.86 by > 10pp: n_kind_counts={n_kind_counts}"
    )
    n_multi = sum(c for k, c in n_kind_counts.items() if len(k) >= 2)
    assert n_multi > 0, (
        f"expected at least one multi-TYPE sample in {n_seen}; got "
        f"{n_kind_counts}"
    )
    print(
        f"  end-to-end iteration OK — {n_seen} samples, "
        f"{n_with_mix} with non-empty noise_kind ({got:.3f}); "
        f"kinds seen: {sorted(n_kind_counts.items())}"
    )


def test_short_source_target_act_is_padded_to_sample_len(hp: HParams) -> None:
    """Regression for `_per_frame_overlap_label` IndexError when the
    source utterance is shorter than `hp.sample_length_samples`.
    """
    from model.audio import wav_to_log_mel
    # `voicefilter_lite.mel_basis.make_mel_basis` is the inference-side
    # torch-free helper (librosa + numpy). The training-side
    # `wav_to_log_mel` expects a torchaudio MelSpectrogram, so build
    # it inline here. (Was the call `make_mel_basis(hp, device=...)`
    # before the train/inference split; the old helper was the
    # torchaudio-based one, which now lives only on the training side.)
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=hp.sample_rate,
        n_fft=hp.n_fft,
        win_length=hp.win_length,
        hop_length=hp.hop_length,
        n_mels=hp.n_mels,
        power=2.0,
        center=False,
        window_fn=torch.hann_window,
        mel_scale="htk",
        normalized=False,
    )
    sample_len = hp.sample_length_samples

    for src_dur_sec in (0.3, 0.5, 1.0, 2.5):
        T = int(src_dur_sec * hp.sample_rate)
        rng_np = np.random.default_rng(0)
        wav = torch.from_numpy(
            (0.2 * rng_np.standard_normal(T)).astype(np.float32)
        )
        rng = random.Random(0)
        raw = synthesize_raw_sample(
            hp, rng,
            target_wav_full=wav,
            ref_wav_full=wav.clone(),
            babble_spk_to_entries={}, noise_pool=[], music_pool=[],
            type_probs={},
            babble_cfg=None, noise_cfg=None, music_cfg=None,
        )
        assert raw.target_act.shape == (sample_len,), (
            f"[src={src_dur_sec}s] target_act shape "
            f"{tuple(raw.target_act.shape)} != ({sample_len},)"
        )
        assert raw.target_wav.shape == (sample_len,), (
            f"[src={src_dur_sec}s] target_wav shape "
            f"{tuple(raw.target_wav.shape)} != ({sample_len},)"
        )
        assert raw.interference_act.shape == (sample_len,), (
            f"[src={src_dur_sec}s] interference_act shape "
            f"{tuple(raw.interference_act.shape)} != ({sample_len},)"
        )
        mixed_logmel = wav_to_log_mel(raw.mixed_wav, hp, mel_fn)
        T_frames = mixed_logmel.shape[0]
        overlap = _per_frame_overlap_label(
            target_act=raw.target_act,
            interference_act=raw.interference_act,
            T_frames=T_frames,
            sample_len=sample_len,
            hp=hp,
        )
        assert overlap.shape == (T_frames,), (
            f"[src={src_dur_sec}s] overlap shape "
            f"{tuple(overlap.shape)} != ({T_frames},)"
        )
        assert set(overlap.unique().tolist()).issubset({-1, 1}), (
            f"[src={src_dur_sec}s] overlap values not in {{-1, +1}}: "
            f"{overlap.unique().tolist()}"
        )

    print(
        f"  short-source target_act padding OK — "
        f"src_dur ∈ [0.3, 2.5]s all produce target_act.shape=({sample_len},)"
    )


# ---- Driver -----------------------------------------------------------------


def _dataset_mod_cache() -> dict:
    """Return a fresh `dataset_id → MockHfDataset` dict to feed to
    `_collect_babble_pool_entries`. The mock patch below means
    `_open_hf_dataset` itself is never actually called, but the
    `_build_target_index` / `_collect_babble_pool_entries` code path
    asks for a `dataset_cache` dict we can satisfy with a literal
    snapshot.
    """
    return dict(_MOCK_DATASETS)


def main() -> None:
    print("== HfTripletStream smoke test ==")

    tmpdir = Path(tempfile.mkdtemp(prefix="voicefilter_test_"))
    try:
        # All HF mock fixtures share one cache dir; per-source dirs
        # resolve through `HfSourceSpec.cache_dir` which is set to
        # `tmpdir/librispeech` and `tmpdir/musan` below.
        libri_cache = tmpdir / "librispeech"
        musan_cache = tmpdir / "musan"
        libri_cache.mkdir(parents=True, exist_ok=True)
        musan_cache.mkdir(parents=True, exist_ok=True)
        dvec_dir_root = libri_cache / "voicefilter-dvectors"

        # Build HParams and override the cache_dir to point at the
        # tmpdir. `_bake_fixtures` writes d-vectors at
        # `<cache_dir>/voicefilter-dvectors/<spec_id>/<spk>.npy`.
        hp = HParams()
        hp.dataset.sources[0].cache_dir = str(libri_cache)  # librispeech
        hp.dataset.sources[1].cache_dir = str(musan_cache)  # musan
        # The default HParams specs don't pre-declare `splits` for the
        # librispeech-other / musan pools (production: paperyaml does),
        # so we pin them here to the keys our mock fixtures use.
        hp.dataset.sources[0].splits = ("train.clean.360", "train.other.500")
        hp.dataset.sources[1].splits = ("train",)
        # Use the default paper.yaml interference probs:
        # babble=0.5, noise=0.6, music=0.3 (independent Bernoulli).
        hp.dataloader_workers = 0
        hp.prefetch_batches_max = 1

        # Monkey-patch `_open_hf_dataset` to return our mocks. The
        # monkey-patch is set on the `voicefilter_cli.dataset` module
        # because both `_build_target_index` and the various `_iter_*`
        # helpers call the module-level name.
        def _mock_open_hf_dataset(resolved, split):
            ds_id = _dataset_mod._dataset_id(resolved, split)
            if ds_id not in _MOCK_DATASETS:
                raise RuntimeError(
                    f"test fixture: missing mock dataset for {ds_id!r}"
                )
            return _MOCK_DATASETS[ds_id]

        _dataset_mod._open_hf_dataset = _mock_open_hf_dataset

        # Build the mock datasets and d-vectors.
        _bake_fixtures(hp, dvec_dir_root)
        print(f"  baked fixtures under {tmpdir}")

        print("1/12] HfTripletStream iteration (50 samples)")
        test_local_triplet_iteration(hp)

        print("2/12] collate_triplets batched shapes")
        test_collate_triplets(hp)

        print("3/12] missing d-vector → DVectorCache compute + persist")
        _bake_fixtures(hp, dvec_dir_root)
        test_missing_dvec_triggers_compute_and_caches(hp)

        print("4/12] interference TYPE inclusion probs honored (800 samples)")
        _bake_fixtures(hp, dvec_dir_root)
        test_interference_type_probs_are_honored(hp)

        print("5/12] disabled TYPE never appears")
        _bake_fixtures(hp, dvec_dir_root)
        test_disabled_interference_type_never_appears(hp)

        print("6/12] babble picks N distinct speakers per sample")
        hp.interference_types["babble"].prob = 0.5
        _bake_fixtures(hp, dvec_dir_root)
        test_babble_picks_distinct_speakers(hp)

        print("7/12] babble mixer emits non-silent full-length wav")
        _bake_fixtures(hp, dvec_dir_root)
        test_babble_mixer_emits_full_length_wav(hp)

        print("8/12] babble pool is config-driven (musan/libri/union/disabled)")
        hp.interference_types["babble"].sources = (
            SourceRef(source="musan", subset="speech"),
        )
        hp.interference_types["babble"].enabled = True
        hp.interference_types["babble"].prob = 0.5
        _bake_fixtures(hp, dvec_dir_root)
        test_babble_pool_is_config_driven(hp)

        print("9/12] interference budget is preserved (K=1/3/empty)")
        hp.interference_types["babble"].prob = 0.5
        hp.interference_types["music"].prob = 0.3
        _bake_fixtures(hp, dvec_dir_root)
        test_interference_budget_is_preserved(hp)

        print("10/12] silent-target branch keeps interference audible")
        _bake_fixtures(hp, dvec_dir_root)
        test_silent_target_keeps_interference_at_natural_energy(hp)

        print("11/12] end-to-end post-mix SIR matches configured")
        _bake_fixtures(hp, dvec_dir_root)
        test_interference_post_mix_sir_matches_configured(hp)

        print("12/13] short-source target_act padded to sample_len (regression)")
        _bake_fixtures(hp, dvec_dir_root)
        test_short_source_target_act_is_padded_to_sample_len(hp)

        print("13/13] target_rms_norm_db pins post-crop RMS (3 cases)")
        test_target_rms_norm_pins_post_crop_rms(hp)

        print("== ALL LOCAL DATASET TESTS PASSED ==")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()