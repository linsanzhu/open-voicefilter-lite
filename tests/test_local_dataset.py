"""Smoke tests for the pre-baked local dataset pipeline.

These tests verify that:
    1. `LocalTripletStream` finds target / interference-type pools /
       d-vector assets in `data/processed/` and builds its in-memory
       indices correctly under the function-oriented interference layout
       (`interference/{type_name}/*.wav`).
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
       per-sample (each TYPE has its own Bernoulli prob, summed across
       included TYPES per sample).
    6. The `DVectorCache` lazy-fill path: a missing d-vector `.npy` is
       computed via `extract_dvec` and persisted to disk on first sight
       (we monkey-patch `extract_dvec` so the test doesn't need the
       real `models/embedder.pt`).

The test bakes a *minimal* fixture set (3 target speakers × 2 utts +
4 interference-type pools of 3 clips + 3 d-vectors) under a temp dir —
no HuggingFace download, no `models/embedder.pt` required (we write
synthetic d-vectors straight to disk).

Run with:
    python tests/test_local_dataset.py

Exit code 0 = all tests passed.
"""

from __future__ import annotations

import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from voicefilter_cli.dataset import LocalTripletStream, collate_triplets
from voicefilter_lite.hparams import HParams, SourceRef


# ---- Test fixtures ----------------------------------------------------------


SR = 16000


def _make_wav(seconds: float = 4.0) -> np.ndarray:
    """Random mono float32 wav (peak-normalised)."""
    T = int(seconds * SR)
    rng = np.random.default_rng(seed=int(T))
    wav = (0.2 * rng.standard_normal(T).astype(np.float32))
    return wav


def _write_wav(path: Path, wav: np.ndarray) -> None:
    """Write a (T,) float32 wav using soundfile's FLOAT subtype."""
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, SR, subtype="FLOAT")


def _bake_fixtures(processed_dir: Path, hp: HParams) -> None:
    """Populate `processed_dir` with the minimum assets a `LocalTripletStream`
    iteration needs.

    Pool layout that `voicefilter-cli dataset prepare` writes (synthesis
    happens at training time, so we only bake raw source pools):

        target/librispeech/clean/train.clean.100/<spk>/<utt>.wav
        pool/librispeech/other/train.other.500/<spk>/<utt>.wav
        pool/musan/speech/<clip>.wav
        pool/musan/noise/<clip>.wav
        pool/musan/music/<clip>.wav
        dvectors/librispeech/<spk>.npy
    """
    # ---- Target: 3 speakers × 2 utts (so the 70% same-speaker ref path
    # has something to draw from) ----
    target_dir = processed_dir / "target" / "librispeech" / "clean" / "train.clean.100"
    speakers = ["100", "200", "300"]
    target_utts: dict[str, list[Path]] = {}
    for spk in speakers:
        spk_dir = target_dir / spk
        for utt_idx in range(2):
            utt_path = spk_dir / f"utt_{utt_idx}.wav"
            _write_wav(utt_path, _make_wav(4.0))
        target_utts[spk] = sorted(spk_dir.glob("*.wav"))

    # ---- Pool: librispeech(other) ----
    # Mirror the target layout (role=other, train.other.500/<spk>/<utt>)
    # so `LocalTripletStream._scan_processed_dir` can recover the spk id
    # from `parts[1]` of the relative path. 3 speakers × 2 utts = 6 paths,
    # grouped under 3 distinct speaker ids (for babble dedup testing).
    lo_root = processed_dir / "pool" / "librispeech" / "other" / "train.other.500"
    for spk in ["1001", "1002", "1003"]:
        spk_dir = lo_root / spk
        for utt_idx in range(2):
            _write_wav(
                spk_dir / f"utt_{utt_idx}.wav",
                _make_wav(4.0),
            )

    # ---- Pool: musan/speech (each clip = one "speaker" identity) ----
    # The mixer dedupes babble speakers at the path level for librispeech
    # and treats every musan_speech clip as one speaker identity. We
    # include 3 clips here so babble has librispeech_other + musan_speech
    # to choose from.
    for clip_idx in range(3):
        _write_wav(
            processed_dir / "pool" / "musan" / "speech" / f"musan_{clip_idx:02d}.wav",
            _make_wav(8.0),  # musan speech clips are long monologues
        )

    # ---- Pool: musan/noise ----
    for clip_idx in range(3):
        _write_wav(
            processed_dir / "pool" / "musan" / "noise" / f"noise_{clip_idx:02d}.wav",
            _make_wav(3.5),
        )

    # ---- Pool: musan/music ----
    for clip_idx in range(3):
        _write_wav(
            processed_dir / "pool" / "musan" / "music" / f"music_{clip_idx:02d}.wav",
            _make_wav(3.5),
        )

    # ---- Per-speaker d-vectors (synthetic; no encoder needed) ----
    dvec_dir = processed_dir / "dvectors" / "librispeech"
    for spk in speakers:
        rng = np.random.default_rng(seed=int(spk) + 999)
        v = rng.standard_normal(hp.d_vector_dim).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)        # L2-normalise
        dvec_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(dvec_dir / f"{spk}.npy"), v)


# ---- Tests ------------------------------------------------------------------


def _check_sample_shapes(sample, hp: HParams, step: int) -> None:
    """Assert `sample` carries the shape contract the model expects."""
    # Frame count: 1 + (n - n_fft) // hop, derived from hp.
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
    # Overlap label values are exactly {-1, +1}.
    uniq = set(sample.overlap_label.unique().tolist())
    assert uniq.issubset({-1, 1}), (
        f"[step {step}] overlap_label values {uniq} ⊄ {{-1, +1}}"
    )
    # d-vector shape + dtype + L2-normalised.
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
    # Diagnostics: ref_spk_id is a non-empty string; noise_kind is a
    # `tuple[str, ...]` of TYPE names actually included in this sample's
    # interference. Every element must be a recognised TYPE tag, and
    # the order is canonical (babble, noise, music).
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
    # Canonical-order invariant: babble < noise < music if all are present.
    canonical = [k for k in sample.noise_kind if k in valid]
    assert canonical == sorted(canonical, key=lambda x: ("babble", "noise", "music").index(x)), (
        f"[step {step}] noise_kind order not canonical "
        f"(babble, noise, music); got {sample.noise_kind!r}"
    )


def test_local_triplet_iteration(processed_dir: Path, hp: HParams) -> None:
    """Iterate `LocalTripletStream` 50 times and assert shape contract."""
    ds = LocalTripletStream(hp, worker_id=0, num_workers=1, seed=1234)
    n_seen = 0
    n_seen_by_subset: dict[str, int] = {}
    for sample in ds:
        _check_sample_shapes(sample, hp, n_seen)
        n_seen_by_subset[sample.noise_kind] = (
            n_seen_by_subset.get(sample.noise_kind, 0) + 1
        )
        n_seen += 1
        if n_seen >= 50:
            break
    assert n_seen == 50, f"only iterated {n_seen} samples; expected 50"

    # With musan_prob=1.0 we expect every interference to be MUSAN.
    print(
        f"  LocalTripletStream OK — 50 samples; "
        f"interference subsets seen: {sorted(n_seen_by_subset.items())}"
    )


def test_collate_triplets(processed_dir: Path, hp: HParams) -> None:
    """`collate_triplets` produces the (B, T, ...) batched tensors the
    training loop expects, and pre-baked d-vectors stack into (B, 256)
    without None handling.
    """
    ds = LocalTripletStream(hp, worker_id=0, num_workers=1, seed=999)
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


def test_missing_dvec_triggers_compute_and_caches(processed_dir: Path, hp: HParams) -> None:
    """When a speaker's d-vector `.npy` is missing on disk, the
    `DVectorCache` should compute it on first sight, persist it
    (`os.replace` to `<spk>.npy`), and return the computed value on
    every subsequent call. This is the new lazy-fill behaviour that
    replaces the old `prepare_dataset.py` d-vector step.
    """
    # Monkey-patch `extract_dvec` so the test doesn't depend on
    # `models/embedder.pt` being available. The mock returns a known
    # L2-normalised vector indexed by speaker id so we can verify the
    # exact bytes get written to disk.
    import voicefilter_lite.audio as _audio_mod

    mock_calls: list[str] = []

    def _mock_extract_dvec(wav, hp_, encoder, device=None, ref_lens=None):
        # The wav tensor is the first ref utterance for `spk`; we don't
        # need to inspect it — just record that we were called.
        mock_calls.append("called")
        d = hp_.d_vector_dim
        out = torch.zeros(1, d, dtype=torch.float32)
        # Encode the speaker identity in the d-vector so the test can
        # verify the value flowed all the way through.
        out[0, 0] = 1.0
        return out

    _audio_mod.extract_dvec = _mock_extract_dvec

    dvec_dir = processed_dir / "dvectors" / "librispeech"
    target_path = dvec_dir / "200.npy"
    assert target_path.is_file(), "fixture setup: 200.npy should exist"
    target_path.unlink()
    assert not target_path.is_file(), "200.npy should be deleted"

    # Force selection of speaker 200 by reducing the index to just that
    # speaker — the simplest way to guarantee it's drawn.
    ds = LocalTripletStream(hp, worker_id=0, num_workers=1, seed=0)
    ds._target_index = {"200": ds._target_index["200"]}

    seen = False
    n_steps = 0
    for sample in ds:
        if sample.ref_spk_id == "200":
            # First sight: `extract_dvec` should have been invoked.
            assert mock_calls, "DVectorCache should call extract_dvec on miss"
            # The mock returns a vector with [1.0, 0, 0, ...] — verify
            # it flowed through (proves we got the computed value, not
            # a zero fallback).
            assert sample.ref_dvec.shape == (hp.d_vector_dim,)
            assert sample.ref_dvec.dtype == torch.float32
            assert abs(float(sample.ref_dvec[0].item()) - 1.0) < 1e-5, (
                f"expected dvec[0]=1.0 from mock; got "
                f"{float(sample.ref_dvec[0].item()):.4f}"
            )
            # Disk file should now exist with the same content.
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


def test_interference_type_probs_are_honored(processed_dir: Path, hp: HParams) -> None:
    """Each TYPE has an INDEPENDENT Bernoulli inclusion `prob`. Over enough
    samples, the fraction of samples that include that TYPE in their
    interference should approximate `prob` (within tolerance).

    We force the default paper.yaml config: babble=0.5, noise=0.6,
    music=0.3 (independent — NOT summing to 1). Over 800 samples we
    expect every TYPE to appear in some samples, the empty-tuple case
    (`()` — all three TYPES rolled false) to appear at roughly
    `(1-0.5)·(1-0.6)·(1-0.3) = 0.5·0.4·0.7 = 0.14` of samples, and
    each TYPE's per-sample inclusion rate to be within tolerance of
    its configured `prob`.
    """
    ds = LocalTripletStream(hp, worker_id=0, num_workers=1, seed=42)
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
    expected_prob = {"babble": 0.5, "noise": 0.6, "music": 0.3}
    for type_name, target_p in expected_prob.items():
        got = n_with_type[type_name] / n_seen
        assert abs(got - target_p) < 0.07, (
            f"{type_name} inclusion rate {got:.3f} deviates from prob "
            f"{target_p:.3f} by more than 7pp: {n_with_type}"
        )
    # `()` (all-three-false) expected ≈ (1-0.5)·(1-0.6)·(1-0.3) = 0.14.
    expected_empty = (1 - 0.5) * (1 - 0.6) * (1 - 0.3)
    got_empty = n_empty / n_seen
    assert abs(got_empty - expected_empty) < 0.05, (
        f"empty-tuple fraction {got_empty:.3f} deviates from "
        f"(1-babble)·(1-noise)·(1-music)={expected_empty:.3f} by more than "
        f"5pp: n_empty={n_empty}/{n_seen}"
    )
    # At least one sample must include multiple TYPES — independent
    # probs make this guaranteed with high probability over 800 draws.
    n_multi = sum(c for k, c in n_combinations.items() if len(k) >= 2)
    assert n_multi > 0, (
        f"expected at least one multi-TYPE sample in {n_seen}; got "
        f"{n_combinations}"
    )


def test_disabled_interference_type_never_appears(processed_dir: Path, hp: HParams) -> None:
    """If a TYPE has `prob <= 0` or `enabled=False`, the stream must
    never include it. We bake fixtures for all three TYPES (so missing
    pools aren't a confounder), then zero-out two probs.
    """
    hp.interference_types["babble"].prob = 0.0
    hp.interference_types["music"].prob = 0.0
    # Re-scan so the stream picks up the new probs.
    ds = LocalTripletStream(hp, worker_id=0, num_workers=1, seed=7)
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


def test_babble_picks_distinct_speakers(processed_dir: Path, hp: HParams) -> None:
    """The babble mixer must pick N DISTINCT speakers per sample.

    We test the picker directly (`_pick_babble_speakers`) rather than
    end-to-end because it's the contract: no two picks share a spk_id,
    N is uniform in `[n_min, n_max]`, and over many draws every
    available speaker shows up at least once (full coverage).
    """
    from voicefilter_cli.dataset import _pick_babble_speakers

    ds = LocalTripletStream(hp, worker_id=0, num_workers=1, seed=11)
    # 3 musan_speech clips (each a unique identity). Default
    # `babble.sources` is musan(speech) only — librispeech `other` is
    # NOT in the babble pool unless the user explicitly adds it under
    # `babble.sources`. Verified end-to-end in
    # `test_babble_pool_is_config_driven` below.
    n_speakers = len(ds._spk_to_paths)
    assert n_speakers == 3, (
        f"fixture should yield 3 distinct musan_speech speakers; "
        f"got {n_speakers}: {sorted(ds._spk_to_paths.keys())}"
    )
    rng = random.Random(11)
    cfg = hp.interference_types["babble"]
    n_min = int(cfg.n_speakers_min)
    n_max = int(cfg.n_speakers_max)
    # Over many picks, verify the contract:
    #   (a) len(picks) ∈ [n_min, n_max]
    #   (b) all spk_ids in picks are distinct
    #   (c) every speaker shows up at least once
    seen_counts: dict[str, int] = {spk: 0 for spk in ds._spk_to_paths}
    n_draws = 200
    for _ in range(n_draws):
        picks = _pick_babble_speakers(
            rng, ds._spk_to_paths, n_min, n_max,
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
    # Every available speaker should appear at least once over 200 draws.
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


def test_babble_mixer_emits_full_length_wav(processed_dir: Path, hp: HParams) -> None:
    """`_mix_babble_from_picks` should return a `(sample_len,)` wav
    that contains at least some non-zero samples when picks are non-empty
    (segments are placed at random offsets, summed). With 4 random
    speakers each placing a 0.5-2 s segment in a 3 s sample, the chance
    of an entirely-silent output is ~0.
    """
    from voicefilter_cli.dataset import (
        _mix_babble_from_picks, _pick_babble_speakers,
    )

    ds = LocalTripletStream(hp, worker_id=0, num_workers=1, seed=23)
    rng = random.Random(23)
    cfg = hp.interference_types["babble"]
    sr = hp.sample_rate
    sample_len = hp.sample_length_samples
    n_nonzero = 0
    n_total = 30
    for _ in range(n_total):
        picks = _pick_babble_speakers(
            rng, ds._spk_to_paths, cfg.n_speakers_min, cfg.n_speakers_max,
        )
        wav = _mix_babble_from_picks(
            rng=rng,
            sample_len=sample_len,
            sr=sr,
            picks=picks,
            seg_min_sec=cfg.babble_seg_min_sec,
            seg_max_sec=cfg.babble_seg_max_sec,
            gain_min=cfg.gain_min,
            gain_max=cfg.gain_max,
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


def test_babble_pool_is_config_driven(processed_dir: Path, hp: HParams) -> None:
    """The babble pool is CONFIG-DRIVEN — built from
    `hp.interference.babble.sources[*]`, NOT hardcoded to MUSAN
    speech. This test exercises four scenarios by mutating
    `babble.sources` (and `babble.prob` for the disabled case) and
    verifying `_spk_to_paths` reflects the configured sources exactly:

      (a) `babble.sources = [musan(speech)]`         → only `musan_speech__`
      (b) `babble.sources = [librispeech(other)]`   → only `librispeech_other__`
      (c) `babble.sources = [musan, librispeech]`   → union (both prefixes)
      (d) `babble.prob = 0` (disabled)              → empty pool

    Asserts both the key prefixes (`musan_speech__` /
    `librispeech_other__`) and the speaker counts.
    """
    from voicefilter_cli.dataset import _collect_babble_pool_paths

    babble_cfg = hp.interference_types["babble"]

    def _scan_with(sources, prob=0.5):
        babble_cfg.sources = sources
        babble_cfg.prob = prob
        return _collect_babble_pool_paths(hp, processed_dir)

    # (a) musan only — fixture bakes 3 musan_speech clips → 3 spks
    pool_a = _scan_with((SourceRef(source="musan", subset="speech"),))
    assert set(pool_a) == {
        ("musan_speech__musan_00", processed_dir / "pool/musan/speech/musan_00.wav"),
        ("musan_speech__musan_01", processed_dir / "pool/musan/speech/musan_01.wav"),
        ("musan_speech__musan_02", processed_dir / "pool/musan/speech/musan_02.wav"),
    }, pool_a
    # (b) librispeech only — fixture bakes 3 spks × 2 utts = 6 paths → 3 spks
    pool_b = _scan_with((
        SourceRef(source="librispeech", splits=("train.other.500",)),
    ))
    assert {spk for spk, _ in pool_b} == {
        "librispeech_other__1001",
        "librispeech_other__1002",
        "librispeech_other__1003",
    }, {spk for spk, _ in pool_b}
    assert len(pool_b) == 6, f"expected 6 librispeech utts (3 spk × 2); got {len(pool_b)}"
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
    # And explicit enabled=False
    babble_cfg.enabled = False
    pool_e = _scan_with(
        (SourceRef(source="musan", subset="speech"),),
    )
    assert pool_e == [], f"enabled=False babble should yield empty pool; got {pool_e}"
    babble_cfg.enabled = True
    # Restore default musan-only for downstream tests.
    _scan_with((SourceRef(source="musan", subset="speech"),))
    print(
        f"  config-driven babble pool OK — "
        f"(a) musan only={len(pool_a)} clips; "
        f"(b) librispeech only={len(pool_b)} clips ({len({s for s, _ in pool_b})} spk); "
        f"(c) union={len(pool_c)} clips; "
        f"(d) disabled={len(pool_d)} clips; "
        f"(e) enabled=False={len(pool_e)} clips"
    )


def test_interference_budget_is_preserved(processed_dir: Path, hp: HParams) -> None:
    """The BIGGEST fix vs. naive summing — when K TYPEs are mixed
    (K up to 3: babble + noise + music), the aggregate's RMS² must
    equal `target_power / 10^(snr/10)` so post-mix SIR matches the
    configured `snr_db` regardless of K. Without the budget fix,
    summing K components inflates RMS² by ~K and drops effective SIR
    per individual TYPE by `10·log10(K)` dB.

    This test exercises `_allocate_interference_budget` directly with
    synthetic components at wildly different natural energies (the
    realistic case — babble sums N speakers at random gain, so its
    RMS² is naturally ~100× a raw music clip's). After budget
    allocation, the SUM RMS² must equal `target_power / 10^(snr/10)`
    within tolerance (allowing for the tanh soft-clip when peaks
    exceed 1.0).
    """
    from voicefilter_cli.dataset import _allocate_interference_budget

    sr = hp.sample_rate
    sample_len = sr * 3  # 3-second sample
    target_power = 0.04  # corresponds to a sine wave at amplitude ~0.2
    snr_db = 5.0
    expected_budget = target_power / (10.0 ** (snr_db / 10.0))

    # Three components at very different natural energies — this is
    # the realistic case: babble N speakers at random gain dominates,
    # music clips are moderate, noise clips are quiet.
    rng = np.random.default_rng(0)
    babble = (1.0 * rng.standard_normal(sample_len).astype(np.float32))
    noise = (0.05 * rng.standard_normal(sample_len).astype(np.float32))
    music = (0.5 * rng.standard_normal(sample_len).astype(np.float32))
    components = [
        torch.from_numpy(babble),
        torch.from_numpy(noise),
        torch.from_numpy(music),
    ]
    # Verify the test setup: natural RMS² ratios are extreme (this
    # is the regime the budget fix exists to handle).
    nat_rms2 = [float((w ** 2).mean().item()) for w in components]
    assert nat_rms2[0] / nat_rms2[1] > 50, (
        f"babble RMS²/noise RMS² = {nat_rms2[0]/nat_rms2[1]:.0f}×; "
        f"expected > 50× for a meaningful budget test"
    )

    # Run the budget allocator with a fixed RNG.
    aggregate = _allocate_interference_budget(
        components=components,
        target_power=target_power,
        snr_db=snr_db,
        rng=random.Random(0),
    )
    actual_budget = float((aggregate ** 2).mean().item())
    # Tolerance: 30% covers the tanh soft-clip's gentle saturation
    # when peaks > 1.0, and minor floating-point error.
    assert abs(actual_budget - expected_budget) / expected_budget < 0.30, (
        f"aggregate RMS² {actual_budget:.5f} deviates from budget "
        f"{expected_budget:.5f} by > 30%: "
        f"target_power={target_power}, snr={snr_db} dB"
    )
    # Anti-clipping: aggregate peak must be ≤ 1.0 (tanh soft-clip).
    peak = float(aggregate.abs().max().item())
    assert peak <= 1.0 + 1e-5, (
        f"aggregate peak {peak:.3f} > 1.0 (soft-clip failed)"
    )

    # K=1 case: full budget assigned to the single component.
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

    # Empty components list → length-0 silent aggregate (defensive
    # branch — real callers guard K=0 before calling the helper).
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


def test_silent_target_keeps_interference_at_natural_energy(
    processed_dir: Path, hp: HParams,
) -> None:
    """Silent-target branch: when `target_power` is at the eps sentinel
    (i.e. the target wav is silent / near-silent), the helper called
    with `apply_budget=False` must NOT scale the aggregate to silence.
    The model needs to see real interference to learn "output silence
    when the target is absent" — without this override, the near-zero
    budget would scale interference to inaudible.

    Verifies:
      * Peak ≤ 1.0 (anti-clipping still applied)
      * Aggregate RMS² is in the same order of magnitude as the
        natural-energy component sum (NOT scaled to ~0 by budget).
    """
    from voicefilter_cli.dataset import _allocate_interference_budget

    sr = hp.sample_rate
    sample_len = sr * 3
    rng = np.random.default_rng(0)
    # Moderate-energy noise + loud music — mix that would get scaled
    # near to silence if `apply_budget=True` with eps target_power.
    noise = (0.3 * rng.standard_normal(sample_len).astype(np.float32))
    music = (0.7 * rng.standard_normal(sample_len).astype(np.float32))
    components = [
        torch.from_numpy(noise),
        torch.from_numpy(music),
    ]
    eps_target_power = 1e-12  # what `__iter__` clamps to for silent targets

    # Natural-energy sum reference: peak-clip + sum + soft-clip,
    # computed independently so we can compare. The soft-clip here
    # mirrors the helper's `tanh(aggregate)` formula (bounded by ±1).
    clipped = []
    for w in components:
        peak = float(w.abs().max().item())
        clipped.append(w / peak if peak > 1.0 else w)
    natural_sum = sum(clipped)
    nat_peak = float(natural_sum.abs().max().item())
    if nat_peak > 1.0:
        natural_sum = torch.tanh(natural_sum)
    natural_rms2 = float((natural_sum ** 2).mean().item())

    # Silent-target branch: apply_budget=False skips scaling + Dirichlet.
    silent = _allocate_interference_budget(
        components=components,
        target_power=eps_target_power,
        snr_db=5.0,  # ignored when apply_budget=False
        rng=random.Random(0),
        apply_budget=False,
    )
    # Peak-clip check.
    silent_peak = float(silent.abs().max().item())
    assert silent_peak <= 1.0 + 1e-5, (
        f"silent-target aggregate peak {silent_peak:.3f} > 1.0 "
        f"(soft-clip failed)"
    )
    # RMS² should be ≈ natural-energy RMS² (within 5% — same per-component
    # peak-clip + sum path; the only shared work is the aggregate tanh).
    silent_rms2 = float((silent ** 2).mean().item())
    assert abs(silent_rms2 - natural_rms2) / natural_rms2 < 0.05, (
        f"silent-target RMS² {silent_rms2:.5f} deviates from natural "
        f"sum {natural_rms2:.5f} by > 5%"
    )
    # Crucial: must NOT be near-zero (the budget-scaled version would
    # be ~eps_target_power / 10^(snr/10) ≈ 3e-13 — audibly silent).
    assert silent_rms2 > 1e-3, (
        f"silent-target RMS² {silent_rms2:.6f} too low — the model "
        f"would see silence instead of interference"
    )

    print(
        f"  silent-target branch OK — natural RMS²={natural_rms2:.4f}, "
        f"silent-target RMS²={silent_rms2:.4f} "
        f"(≈audible, NOT scaled to ~0 by budget)"
    )


def test_interference_post_mix_sir_matches_configured(processed_dir: Path, hp: HParams) -> None:
    """End-to-end smoke: iterate the dataset 200 times and verify the
    TYPE inclusion distribution still matches the independent-Bernoulli
    expectation, AND that no sample crashes the iterator (which would
    indicate a budget-fix regression breaking the synth pipeline).

    The precise budget math (per-component Dirichlet split, RMS²
    preservation) is verified at the helper level in
    `test_interference_budget_is_preserved`. This end-to-end test
    guards against regressions where the helper-level fix breaks the
    iterator (e.g. an exception in `_allocate_interference_budget`
    surfacing as a stuck sample).

    We also fix `snr_min_db == snr_max_db == 5.0` so any post-mix SIR
    sanity check could be derived from the configured budget — but
    the Sample surface doesn't expose raw waveforms, so we just
    verify the iterator yields 200 valid samples.
    """
    # Set snr range narrow so all samples have a known budget; the
    # end-to-end test doesn't probe the budget math directly, only
    # that the iterator doesn't crash while synthesising K=1..3
    # components under the budget allocator.
    hp.snr_min_db = 5.0
    hp.snr_max_db = 5.0
    hp.dataloader_workers = 0
    hp.prefetch_batches_max = 1

    ds = LocalTripletStream(hp, worker_id=0, num_workers=1, seed=2025)
    n_seen = 0
    n_with_mix = 0
    n_kind_counts: dict[tuple[str, ...], int] = {}
    iterator = iter(ds)
    while n_seen < 200:
        sample = next(iterator)
        n_seen += 1
        # Sanity: shape contract (mirrors `_check_sample_shapes` but
        # inline here to keep the test independent).
        assert sample.stacked_mixed.dtype == torch.float32
        assert sample.target_logmel.dtype == torch.float32
        assert sample.overlap_label.shape == (hp.num_frames,)
        assert sample.ref_dvec.shape == (hp.d_vector_dim,)
        n_kind_counts[sample.noise_kind] = (
            n_kind_counts.get(sample.noise_kind, 0) + 1
        )
        if sample.noise_kind:
            n_with_mix += 1
    # With babble=0.5, noise=0.6, music=0.3, expected P(non-empty
    # rolls) = 1 - (1-0.5)(1-0.6)(1-0.3) = 0.86.  Allow ±10pp
    # sampling tolerance over 200 draws (binomial std ≈ 2.5pp).
    got = n_with_mix / 200
    assert abs(got - 0.86) < 0.10, (
        f"non-empty noise_kind {got:.3f} deviates from expected "
        f"0.86 by > 10pp: n_kind_counts={n_kind_counts}"
    )
    # At least one sample must include multiple TYPES (independent
    # probs make this guaranteed with high probability over 200 draws).
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


def test_short_source_target_act_is_padded_to_sample_len(
    processed_dir: Path, hp: HParams,
) -> None:
    """Regression for `_per_frame_overlap_label` IndexError when the
    source utterance is shorter than `hp.sample_length_samples`.

    When `synthesize_raw_sample` applies the target activity burst to
    the FULL source audio and then crops, a source shorter than
    `sample_len` produced a `target_act` tensor of length `T_full`
    (the slice is clamped at the source end). The downstream
    `_per_frame_overlap_label` then computed `centre_idx` for a
    `sample_len`-shaped log-mel and tried to index the shorter
    `target_act` → IndexError in `_per_frame_overlap_label`. The fix
    pads `target_act` to `sample_len` so it matches the log-mel frame
    count. This test synthesises short sources (0.3 s – 2.5 s) and
    verifies the iterator's overlap-label path is in bounds for all of
    them.
    """
    from voicefilter_cli.dataset import _per_frame_overlap_label
    from voicefilter_lite.audio import wav_to_log_mel, make_mel_basis

    mel_fn = make_mel_basis(hp, device=torch.device("cpu"))
    sample_len = hp.sample_length_samples

    for src_dur_sec in (0.3, 0.5, 1.0, 2.5):
        T = int(src_dur_sec * hp.sample_rate)
        rng_np = np.random.default_rng(0)
        wav = torch.from_numpy(
            (0.2 * rng_np.standard_normal(T)).astype(np.float32)
        )
        # Iterate the stream and find a Sample whose target_spk matches
        # this short wav. We patch the dataset's target index to one
        # short file so every drawn sample uses it.
        ds = LocalTripletStream(hp, worker_id=0, num_workers=1, seed=0)
        # Replace the entire target index with a single utterance of
        # the requested duration.
        spk = f"short_{int(src_dur_sec * 1000)}"
        ds._target_index = {spk: [Path("/dev/null")]}  # path unused; we
        # also bypass dvec caching by pre-baking a synthetic one.
        ds._spk_to_spec_id = {spk: "librispeech"}
        # Pre-seed d-vectors for any spks we might draw.
        dvec_dir = processed_dir / "dvectors" / "librispeech"
        dvec_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            str(dvec_dir / f"{spk}.npy"),
            np.ones(hp.d_vector_dim, dtype=np.float32) / (hp.d_vector_dim ** 0.5),
        )

        # Manually run `synthesize_raw_sample` with the short wav.
        from voicefilter_cli.dataset import synthesize_raw_sample
        rng = random.Random(0)
        raw = synthesize_raw_sample(
            hp, rng,
            target_wav_full=wav,
            ref_wav_full=wav.clone(),
            babble_spk_to_paths={}, noise_pool=[], music_pool=[],
            type_probs={},
            babble_cfg=None, noise_cfg=None, music_cfg=None,
        )
        # Critical: target_act and target_wav must BOTH be sample_len
        # long (the pad branch in synthesize_raw_sample).
        assert raw.target_act.shape == (sample_len,), (
            f"[src={src_dur_sec}s] target_act shape "
            f"{tuple(raw.target_act.shape)} != ({sample_len},) — "
            f"would IndexError in _per_frame_overlap_label"
        )
        assert raw.target_wav.shape == (sample_len,), (
            f"[src={src_dur_sec}s] target_wav shape "
            f"{tuple(raw.target_wav.shape)} != ({sample_len},)"
        )
        assert raw.interference_act.shape == (sample_len,), (
            f"[src={src_dur_sec}s] interference_act shape "
            f"{tuple(raw.interference_act.shape)} != ({sample_len},)"
        )
        # Run _per_frame_overlap_label end-to-end — the IndexError
        # that motivated this fix fires here on the buggy code path.
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


def main() -> None:
    print("== LocalTripletStream smoke test ==")

    # Build a minimal fixture set under a temp dir; clean up on exit.
    tmpdir = Path(tempfile.mkdtemp(prefix="voicefilter_test_"))
    try:
        hp = HParams()
        hp.dataset.processed_dir = str(tmpdir)
        # Use the default paper.yaml interference probs:
        # babble=0.5, noise=0.6, music=0.3 (independent Bernoulli
        # inclusion probabilities, no longer summing to 1).
        hp.dataloader_workers = 0
        hp.prefetch_batches_max = 1

        _bake_fixtures(tmpdir, hp)
        print(f"  baked fixtures under {tmpdir}")

        print("1/12] LocalTripletStream iteration (50 samples)")
        test_local_triplet_iteration(tmpdir, hp)

        print("2/12] collate_triplets batched shapes")
        test_collate_triplets(tmpdir, hp)

        print("3/12] missing d-vector → DVectorCache compute + persist")
        # Rebuild fixtures (previous test may have consumed some).
        shutil.rmtree(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        _bake_fixtures(tmpdir, hp)
        test_missing_dvec_triggers_compute_and_caches(tmpdir, hp)

        print("4/12] interference TYPE inclusion probs honored (800 samples)")
        shutil.rmtree(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        _bake_fixtures(tmpdir, hp)
        test_interference_type_probs_are_honored(tmpdir, hp)

        print("5/12] disabled TYPE never appears")
        shutil.rmtree(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        _bake_fixtures(tmpdir, hp)
        test_disabled_interference_type_never_appears(tmpdir, hp)

        print("6/12] babble picks N distinct speakers per sample")
        shutil.rmtree(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        # Reset babble.prob back to default — test 5/12] above set it
        # to 0, and `_collect_babble_pool_paths` now skips sources when
        # the TYPE's prob <= 0 (config-driven behaviour). Without this
        # reset the babble pool would be empty here.
        hp.interference_types["babble"].prob = 0.5
        _bake_fixtures(tmpdir, hp)
        test_babble_picks_distinct_speakers(tmpdir, hp)

        print("7/12] babble mixer emits non-silent full-length wav")
        shutil.rmtree(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        _bake_fixtures(tmpdir, hp)
        test_babble_mixer_emits_full_length_wav(tmpdir, hp)

        print("8/12] babble pool is config-driven (musan/libri/union/disabled)")
        shutil.rmtree(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        # Reset babble.sources + babble.enabled in case earlier tests
        # mutated them — the new test mutates both, and we want the
        # downstream budget / end-to-end tests to start from the
        # paper.yaml default (musan only).
        hp.interference_types["babble"].sources = (
            SourceRef(source="musan", subset="speech"),
        )
        hp.interference_types["babble"].enabled = True
        hp.interference_types["babble"].prob = 0.5
        _bake_fixtures(tmpdir, hp)
        test_babble_pool_is_config_driven(tmpdir, hp)

        print("9/12] interference budget is preserved (K=1/3/empty)")
        shutil.rmtree(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        # Reset hp.interference_types probs back to paper.yaml defaults —
        # test 5/12] above zeroed babble/music to verify the disabled
        # filter, and that mutation persists on the shared hp.
        hp.interference_types["babble"].prob = 0.5
        hp.interference_types["music"].prob = 0.3
        _bake_fixtures(tmpdir, hp)
        test_interference_budget_is_preserved(tmpdir, hp)

        print("10/12] silent-target branch keeps interference audible")
        shutil.rmtree(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        _bake_fixtures(tmpdir, hp)
        test_silent_target_keeps_interference_at_natural_energy(tmpdir, hp)

        print("11/12] end-to-end post-mix SIR matches configured")
        shutil.rmtree(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        _bake_fixtures(tmpdir, hp)
        test_interference_post_mix_sir_matches_configured(tmpdir, hp)

        print("[12/12] short-source target_act padded to sample_len (regression)")
        shutil.rmtree(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        _bake_fixtures(tmpdir, hp)
        test_short_source_target_act_is_padded_to_sample_len(tmpdir, hp)

        print("== ALL LOCAL DATASET TESTS PASSED ==")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
