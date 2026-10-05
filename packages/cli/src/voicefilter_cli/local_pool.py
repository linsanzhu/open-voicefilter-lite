"""Read LibriSpeech wavs from the baked pool under `data/processed/`.

`scripts/prepare_dataset.py` bakes the upstream LibriSpeech + MUSAN assets
into a flat directory tree:

    data/processed/target/librispeech/clean/<split>/<spk_id>/<utt_id>.wav
    data/processed/pool/librispeech/other/<split>/<spk_id>/<utt_id>.wav

`<spk_id>` is the second path component (e.g. `2618`); `<utt_id>` is the
original HF arrow path string (long, includes hash). The wavs are 16 kHz
mono float32.

Both `evaluate voicefilter-lite` and `evaluate dvector` historically went
straight back to `datasets.load_dataset(..., streaming=True)` even when
the pool was already baked. This module makes the pool the default source
so the eval pass is reproducible, takes ~0 ms of network, and matches
exactly what the training-time `LocalTripletStream` consumed.

Usage from a CLI script:

    from voicefilter_cli.local_pool import LocalLibriSpeechPool

    pool = LocalLibriSpeechPool(
        root=Path("data/processed"),
        role="clean",                      # "clean" for target, "other" for interference
        splits=("train.clean.360",),
    )
    # Iterate (spk_id, wav_tensor) lazily:
    for spk, wav in pool.iter_wavs():
        ...

    # Or grab N utts per speaker for d-vector verification:
    by_spk = pool.sample_by_speaker(num_utts_per_speaker=6, seed=0)
    for spk, wavs in by_spk.items():
        ...
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import soundfile as sf
import torch

# Default root; pass `overridden` to use a different processed tree.
DEFAULT_POOL_ROOT = Path("data/processed")

# Where each role lives relative to `<root>`.
# `target/` is the special "I'm training on this" pool (used as the
# enrollment side), `pool/` is the shared interference pool.
ROLE_PATHS = {
    "clean":  "target/librispeech/clean",     # target role (paper §4.1)
    "other":  "pool/librispeech/other",       # interference role (LibriSpeech-other)
}


class LocalLibriSpeechPool:
    """Lazy reader over the baked LibriSpeech wav pool.

    Scanning is O(N) on the directory tree at construction time (one
    `Path.glob` per split dir; no audio is read). Audio is loaded
    on-demand via `soundfile.read`.

    Args:
        root: path to `data/processed` (or override root).
        role: `"clean"` (target) or `"other"` (interference). The role
            decides which sub-tree under `root` to scan.
        splits: tuple of split subdir names to include
            (e.g. `("train.clean.360",)` for clean; multiple OK).
        seed: PRNG seed for `sample_by_speaker`.

    Raises FileNotFoundError if the requested role/split tree has no
    wavs — caller should fall back to streaming from HF in that case.
    """

    def __init__(
        self,
        root: Path = DEFAULT_POOL_ROOT,
        role: str = "clean",
        splits: tuple[str, ...] = ("train.clean.360",),
        seed: int = 0,
    ) -> None:
        if role not in ROLE_PATHS:
            raise ValueError(
                f"[local-pool] role={role!r} not in {list(ROLE_PATHS)}; "
                f"expected one of these (paper.yaml roles)."
            )
        rel = ROLE_PATHS[role]
        self.root = root
        self.role = role
        self.splits = splits
        self.seed = seed

        wavs: list[tuple[str, Path]] = []
        for split in splits:
            split_dir = root / rel / split
            if not split_dir.is_dir():
                continue
            for spk_dir in split_dir.iterdir():
                if not spk_dir.is_dir():
                    continue
                for wav_path in spk_dir.glob("*.wav"):
                    wavs.append((spk_dir.name, wav_path))

        if not wavs:
            raise FileNotFoundError(
                f"[local-pool] no wavs under {root / rel} for splits={splits}. "
                f"Did `python scripts/prepare_dataset.py --config configs/paper.yaml` "
                f"bake role={role}? Falling back to HF streaming is the caller's job."
            )

        # Index by speaker for `sample_by_speaker` / `random_pair_*`.
        self.wavs: list[tuple[str, Path]] = wavs
        self.by_spk: dict[str, list[Path]] = {}
        for spk, p in wavs:
            self.by_spk.setdefault(spk, []).append(p)

        print(
            f"[local-pool] role={role} splits={splits}: "
            f"{len(wavs):,} wavs, {len(self.by_spk):,} distinct speakers "
            f"(root={root})"
        )

    # ------------------------------------------------------------------ #
    # Iteration / sampling
    # ------------------------------------------------------------------ #
    def iter_wavs(self, shuffle: bool = True) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield `(spk_id, wav_tensor)` lazily.

        `shuffle=True` randomises order at module load (the on-disk order
        is roughly alphabetical by spk_id → all-utt-of-one-speaker, which
        would crash any naive batch sampler that picks consecutive items
        as `target + ref`).
        """
        order = list(range(len(self.wavs)))
        if shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(order)
        for i in order:
            spk, path = self.wavs[i]
            yield spk, _read_wav(path)

    def sample_by_speaker(
        self,
        num_speakers: int,
        num_utts_per_speaker: int,
        seed: int | None = None,
    ) -> dict[str, list[torch.Tensor]]:
        """Pick `num_utts_per_speaker` random wavs for each of `num_speakers`
        random speakers. Used by `verify_dvec.py`.

        Speakers with < num_utts_per_speaker utts are skipped silently —
        LibriSpeech-clean's per-speaker counts are usually >=10 so this
        is a non-issue in practice.

        Returns: `{spk_id: [wav_tensor, ...]}` with len == num_utts_per_speaker
        per speaker (or fewer for skipped speakers).
        """
        rng = random.Random(self.seed if seed is None else seed)
        eligible = [spk for spk, ps in self.by_spk.items() if len(ps) >= num_utts_per_speaker]
        rng.shuffle(eligible)
        chosen = eligible[:num_speakers]
        out: dict[str, list[torch.Tensor]] = {}
        for spk in chosen:
            picks = rng.sample(self.by_spk[spk], num_utts_per_speaker)
            out[spk] = [_read_wav(p) for p in picks]
        return out

    def sample_eval_triple(
        self,
        rng: random.Random,
        buffer: dict[str, list[Path]],
        target_buffer_max: int = 4,
    ) -> tuple[Path, Path, Path] | None:
        """Mirror the existing `build_eval_pairs` flow used by `evaluate.py`:
        pick a random target, accumulate up to `target_buffer_max` utts per
        speaker for the d-vector ref, and pair it with a random other-speaker
        wav as interference. Returns `None` when both pools are exhausted.

        `buffer` is mutated in-place (it tracks per-speaker target history).
        The interference wav comes from a separate `LocalLibriSpeechPool`
        for role=`other`; pass a separate rng for it.
        """
        # ---- Pick a target + add to buffer ----
        try:
            tgt_spk, tgt_path = self._random_one(rng)
        except IndexError:
            return None
        buf = buffer.setdefault(tgt_spk, [])
        if len(buf) < target_buffer_max:
            buf.append(tgt_path)
            # Need at least 2 utts for the same speaker to serve as ref.
            return None
        ref_path = rng.choice(buf)
        return tgt_path, ref_path, tgt_spk

    def random_path(self, rng: random.Random) -> tuple[str, Path]:
        """Pick one random (spk, path)."""
        idx = rng.randrange(len(self.wavs))
        return self.wavs[idx]

    def _random_one(self, rng: random.Random) -> tuple[str, Path]:
        return self.random_path(rng)


def _read_wav(path: Path) -> torch.Tensor:
    """Read a baked wav → (T,) float32 tensor at the pool's native SR.

    The pool is always 16 kHz mono (enforced by `prepare_dataset.py`'s
    `_decode_to_wav` + `_write_wav`). We don't resample here — the
    encoder / model both assume `hp.sample_rate == 16000`.
    """
    data, _sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return torch.from_numpy(data)


def has_local_pool(
    root: Path = DEFAULT_POOL_ROOT,
    role: str = "clean",
    split: str = "train.clean.360",
) -> bool:
    """Cheap existence check used by the CLI scripts to decide whether
    to opt into the local pool by default. Returns True iff at least one
    wav exists under `root/<role_path>/<role>/<split>/`."""
    if role not in ROLE_PATHS:
        return False
    rel = ROLE_PATHS[role]
    p = root / rel / split
    if not p.is_dir():
        return False
    return any(p.rglob("*.wav"))


def resolve_eval_pair(
    pool_root: Path,
    target_role: str = "clean",
    target_split: str = "train.clean.360",
    inter_role: str = "other",
    inter_split: str = "train.other.500",
) -> tuple[LocalLibriSpeechPool, LocalLibriSpeechPool]:
    """Open both target + interference pools used by `evaluate.py`.

    Returns `(target_pool, inter_pool)`. Both pools are independent;
    pass two RNGs (one per pool) for determinism.
    """
    target_pool = LocalLibriSpeechPool(
        root=pool_root, role=target_role, splits=(target_split,),
    )
    inter_pool = LocalLibriSpeechPool(
        root=pool_root, role=inter_role, splits=(inter_split,),
    )
    return target_pool, inter_pool