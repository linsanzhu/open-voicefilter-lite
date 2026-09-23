"""Per-speaker ECAPA-TDNN d-vector cache with optional on-disk persistence.

Holds the in-memory + on-disk cache that previously lived inside
`LibriSpeechTripletStream`. By lifting the cache (and the ECAPA forward that
populates it) out of the dataset, all MPS work moves onto the main training
thread — the dataset becomes pure CPU and the cross-thread MPS race that
crashed `python3.13` with `BUG IN CLIENT OF LIBMALLOC: memory corruption of
free block` goes away.

Threading contract: `DVecCache` is **single-threaded**. Construct once on
the main process and call `get_dvecs` from the main thread only. The cache
mutates `self._mem` and `self._disk` without locks; calling concurrently
will race on `self._mem[spk] = ...` and on concurrent `torch.save` to the
same file. The crash report that motivated this refactor (`/Library/Logs/
DiagnosticReports/python3.13-*.ips`) was caused by exactly this race
running across `PrefetchIterator`'s daemon thread + the main thread on MPS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from open_voicefilter_lite.hparams import HParams
from open_voicefilter_lite.speaker_encoder import FrozenSpeakerEncoder


def _safe_filename(s: str) -> str:
    """Sanitise an arbitrary speaker_id for use as a filename.

    LibriSpeech speaker_ids are numeric ("1272"), but if downstream users
    point `target_config` at a different dataset the id could be any string.
    Hash-trim to a fixed length and replace anything non-filesystem-safe
    with `_` so `torch.save(dvec, dir / f"{spk}.pt")` always succeeds.
    """
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))
    return cleaned[:200] or "unknown"


class DVecCache:
    """Per-speaker ECAPA-TDNN d-vector cache.

    The d-vector is a function of the *speaker*, not the utterance, so once
    we've computed one for a given speaker_id we never need to re-run
    ECAPA for them. In-memory cache is always-on; set `hp.dvec_cache_dir`
    to also persist to disk so subsequent training runs (or checkpoint
    resumes) don't pay the ECAPA warm-up cost again. `None` = in-memory
    only (cache lost on process exit).
    """

    def __init__(
        self,
        hp: HParams,
        speaker_encoder: FrozenSpeakerEncoder,
        device: torch.device,
    ) -> None:
        self._hp = hp
        self._encoder = speaker_encoder
        self._device = torch.device(device)

        # In-memory cache: speaker_id -> CPU float32 dvec (raw ECAPA output).
        self._mem: dict[str, torch.Tensor] = {}

        # On-disk cache (None → in-memory only).
        self._dir: Optional[Path] = None
        if hp.dvec_cache_dir:
            path = Path(hp.dvec_cache_dir)
            try:
                path.mkdir(parents=True, exist_ok=True)
                self._dir = path
            except OSError as e:
                print(f"[dvec] WARNING could not create dvec_cache_dir={path}: {e}")

        # ECAPA input-length floor (TDNN collapses on very short inputs)
        # and ceiling (matches `hp.ref_max_sec`, which the dataset already
        # enforces before yielding `ref_wav` to the collate). The 30-s
        # hardcode was the legacy defensive cap against pathological
        # LibriSpeech files; routing it through the same knob keeps the
        # cache miss path consistent with what the dataset hands us.
        self._ref_min_len = int(0.5 * hp.sample_rate)
        self._cap_samples = int(hp.ref_max_sec * hp.sample_rate)

        # Best-effort warm-start from disk.
        self._load_disk_cache()

    # ---- Disk cache I/O ----

    def _load_disk_cache(self) -> None:
        """Load persisted d-vectors from disk into the in-memory cache.

        Called once from `__init__`. Idempotent. Corrupt / partial files
        from a previous interrupted run are skipped — we'll re-compute and
        overwrite them on the next miss.
        """
        if self._dir is None:
            return
        loaded = 0
        for f in self._dir.glob("*.pt"):
            spk = f.stem
            try:
                self._mem[spk] = torch.load(f, map_location="cpu")
                loaded += 1
            except Exception:
                continue
        if loaded:
            print(
                f"[dvec] loaded {loaded} persisted d-vectors from {self._dir} "
                f"(warm-start; first few batches will hit cache instead of ECAPA)"
            )

    def _persist(self, spk: str, dvec_cpu: torch.Tensor) -> None:
        """Best-effort `torch.save` of a freshly computed d-vector. Failures
        are non-fatal — the in-memory cache is the source of truth for the
        current run."""
        if self._dir is None:
            return
        try:
            torch.save(dvec_cpu, self._dir / f"{_safe_filename(spk)}.pt")
        except Exception:
            pass

    # ---- Public API ----

    @torch.no_grad()
    def get_dvecs(
        self,
        ref_wavs: torch.Tensor,
        ref_lens: torch.Tensor,
        ref_spks: list[str],
    ) -> torch.Tensor:
        """Return (B, hp.speaker_encoder_raw_dim) d-vectors on `self._device`.

        Args:
            ref_wavs: (B, T_max_in_batch) variable-length padded waveform
                batch on CPU (or any device — gets moved to `self._device`).
            ref_lens: (B,) relative lengths in [0, 1]; longest in the batch
                == 1.0. After re-padding inside this method, lengths are
                recomputed against the new padded length.
            ref_spks: list[str] of length B, parallel to rows of
                `ref_wavs` / `ref_lens`. Used as the dvec cache key.

        Returns:
            (B, hp.speaker_encoder_raw_dim) float32 d-vectors on
            `self._device`, in the original row order.
        """
        assert ref_wavs.dim() == 2, f"[dvec] ref_wavs must be (B, T); got {ref_wavs.shape}"
        assert ref_wavs.shape[0] == ref_lens.shape[0] == len(ref_spks), (
            f"[dvec] shape mismatch: ref_wavs {tuple(ref_wavs.shape)} vs "
            f"ref_lens {tuple(ref_lens.shape)} vs ref_spks {len(ref_spks)}"
        )

        B = ref_wavs.shape[0]
        T_max = ref_wavs.shape[1]
        device = self._device

        # ---- Split into cache hits and cache misses ----
        result: list[Optional[torch.Tensor]] = [None] * B
        miss_indices: list[int] = []
        miss_spks: list[str] = []

        for i, spk in enumerate(ref_spks):
            cached = self._mem.get(spk)
            if cached is not None:
                result[i] = cached.to(device)
            else:
                miss_indices.append(i)
                miss_spks.append(spk)

        if not miss_indices:
            return torch.stack(result)  # type: ignore[arg-type]

        # ---- ECAPA forward on cache misses ----
        # Recover the actual per-ref sample counts from the padded batch:
        # the collate function pads to T_max and reports ref_lens as
        # actual_len / T_max, so actual_len = ref_len * T_max.
        uncached_wavs = ref_wavs[miss_indices]                 # (N, T_max)
        uncached_lens = ref_lens[miss_indices]                 # (N,)
        actual_lens = (uncached_lens * T_max).long()            # (N,) sample counts

        # Determine the padded length for the ECAPA forward:
        #   - at least the longest actual ref in this miss subset
        #   - at least the ECAPA floor (TDNN receptive field)
        #   - at most sample_rate * 30 (pathological refs)
        target_len = int(actual_lens.max().item())
        target_len = max(target_len, self._ref_min_len)
        target_len = min(target_len, self._cap_samples)

        N = len(miss_indices)
        padded = torch.zeros(N, target_len, dtype=torch.float32)
        new_lens = torch.zeros(N, dtype=torch.float32)
        for j in range(N):
            L = min(int(actual_lens[j].item()), target_len)
            padded[j, :L] = uncached_wavs[j, :L]
            new_lens[j] = L / target_len

        padded = padded.to(device)
        new_lens = new_lens.to(device)

        # Run the batched ECAPA forward. `_encoder` is the frozen
        # `FrozenSpeakerEncoder` wrapper around speechbrain's TDNN.
        new_dvecs = self._encoder(padded, new_lens).cpu()

        # Update cache (in-memory + best-effort disk) and fill `result`.
        for orig_idx, spk, dvec in zip(miss_indices, miss_spks, new_dvecs):
            self._mem[spk] = dvec
            self._persist(spk, dvec)
            result[orig_idx] = dvec.to(device)

        return torch.stack(result)  # type: ignore[arg-type]

    # ---- Diagnostics ----

    @property
    def cache_size(self) -> int:
        """Number of speakers currently in the in-memory cache."""
        return len(self._mem)