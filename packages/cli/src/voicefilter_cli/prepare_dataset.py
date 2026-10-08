#!/usr/bin/env python3
"""Populate the HF arrow cache for VoiceFilter-Lite training.

The dataset is described by a function-oriented YAML config
(`configs/paper.yaml`). The config drives:

  - `dataset.sources[*]`     : upstream HF dataset declarations
                               (`HfSourceSpec` by id), shared by both
                               target and interference.
  - Target sources           : `SourceRef`s referencing `dataset.sources`
                               by `id`, with optional `splits:` override.
  - Interference TYPEs       : each TYPE pools over `SourceRef`s — every
                               TYPE that points at the same upstream
                               `HfSourceSpec` shares one HF arrow cache
                               (no per-TYPE baking).
  - Per-speaker d-vectors    : 256-d .npy keyed by speaker id; filled
                               lazily during training by `DVectorCache`
                               under
                               `<cache_dir>/voicefilter-dvectors/<spec_id>/<spk>.npy`.

Output layout — per HF cache directory:

    <cache_dir>/                              # arrow cache from HF datasets
      openslr___librispeech_asr/<config>/...   # (HF's own layout)
      voicefilter-index/                      # per-source JSON row metadata
        librispeech__train.clean.360.json     # ({row_idx, spk_id, utt_id,
                                              #   source_sr}, …)
        musan_noise__train.json
        ...
      voicefilter-dvectors/<spec_id>/<spk>.npy # populated lazily by training
      voicefilter-manifest.json               # per-cache-dir manifest

This script writes no wav files — decoding happens lazily at training
time via `default_hf_wav_loader` (reads `{"bytes": ...}` payload,
decodes via `soundfile`, resamples to `hp.sample_rate` when needed).
The on-disk footprint is therefore ~10-20 GB for a full
LibriSpeech-500 + MUSAN bake (HF arrow + metadata) instead of the
~200-300 GB the old wav-baking pipeline produced.

Why per-cache-dir layout (not a central `processed_dir`)?  Each HF
cache is self-describing: deleting `data/librispeech/` reclaims
LibriSpeech only; `data/musan/` survives. `voicefilter-cli dataset
clear` follows the same shape.

Usage
-----
    # Populate HF caches for the paper-faithful set:
    `voicefilter-cli dataset prepare` \\
        --config configs/paper.yaml \\
        --hf-endpoint https://hf-mirror.com
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Optional

from cyclopts import App, Parameter

from voicefilter_lite.hparams import HParams, ResolvedSource, resolve_source
from voicefilter_cli.hf import apply_hf_endpoint
from voicefilter_cli.dataset import (
    INDEX_DIRNAME,
    MANIFEST_FILENAME,
    _cast_audio_no_decode,
    _dataset_id,
    _disable_hf_audio_decode,
    _open_hf_dataset,
    _row_metadata,
)


# Default cache dirs for each HF family — paper.yaml values; pass at the
# call site so the policy is not hardcoded inside the closure.
DEFAULT_LIBRISPEECH_CACHE = "data/librispeech"
DEFAULT_MUSAN_CACHE = "data/musan"


# ---- Per-source prepare: walk dataset, build JSON index ------------------
def _ensure_hf_dataset(
    resolved: ResolvedSource,
    split: str,
) -> tuple[object, list[dict]]:
    """Open a HF `datasets.Dataset` and extract lightweight row metadata.

    Returns `(dataset, entries)` where `entries` is the per-row
    metadata list:

      [
        {"row_idx": int, "spk_id": str, "utt_id": str, "source_sr": int},
        ...
      ]

    Used by `HfTripletStream._scan_hf_datasets` to skip the same walk
    on training startup (saves ~1-2 s on a cold LibriSpeech clean-360
    scan; ~30 s on a cold-spinning-disk cache).

    `dataset` is returned too so the caller can free the underlying
    arrow file (we don't hold onto it after `entries` is built).
    """
    ds = _open_hf_dataset(resolved, split)
    source_sr = 16000  # both LibriSpeech + MUSAN (audio is per-source)
    if resolved.is_musan():
        # MUSAN `noise` / `music` are 44.1 kHz; MUSAN `speech` is 16 kHz.
        source_sr = 16000 if resolved.subset == "speech" else 44100

    entries: list[dict] = []
    for row_idx in range(len(ds)):
        row = ds[row_idx]
        spk_id, utt_id = _row_metadata(resolved, row, source_sr)
        entries.append({
            "row_idx": row_idx,
            "spk_id": spk_id,
            "utt_id": utt_id,
            "source_sr": source_sr,
        })
    return ds, entries


def _prepare_source(
    resolved: ResolvedSource,
    cache_dir: Path,
) -> dict:
    """Populate the HF arrow cache + JSON index for one resolved source.

    Iterates over every split in `resolved.splits`, opens each via
    `load_dataset(... , split=...)` (HF populates the arrow cache
    under `cache_dir` on first call), and writes
    `<cache_dir>/voicefilter-index/<dataset_id>.json`.

    Returns stats: `{spec_id, n_splits, n_rows_total, cache_dir,
    per_split: [...]}`.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_root = cache_dir / INDEX_DIRNAME
    index_root.mkdir(parents=True, exist_ok=True)

    per_split: list[dict] = []
    n_rows_total = 0
    for split in resolved.splits:
        ds_id = _dataset_id(resolved, split)
        print(f"[{resolved.id}] {split}: loading {resolved.repo}…")
        t0 = time.time()
        ds, entries = _ensure_hf_dataset(resolved, split)
        dt = time.time() - t0

        # Write the JSON index file (the iterator reads this on startup
        # to skip the dataset walk). Overwrites any previous file —
        # the source of truth is the HF arrow cache.
        safe_ds = ds_id.replace("/", "__")
        index_path = index_root / f"{safe_ds}.json"
        index_payload = {
            "dataset_id": ds_id,
            "spec_id": resolved.id,
            "subset": resolved.subset,
            "split": split,
            "repo": resolved.repo,
            "cache_dir": str(cache_dir),
            "n_rows": len(entries),
            "entries": entries,
        }
        index_path.write_text(json.dumps(index_payload))
        # Drop the dataset object — the iterator will reopen it.
        del ds

        per_split.append({
            "split": split,
            "n_rows": len(entries),
            "seconds": round(dt, 2),
            "index_path": str(index_path.relative_to(cache_dir)),
        })
        n_rows_total += len(entries)
        print(
            f"[{resolved.id}] {split}: {len(entries)} rows in {dt:.1f}s "
            f"(index: {index_path})"
        )

    return {
        "spec_id": resolved.id,
        "repo": resolved.repo,
        "subset": resolved.subset,
        "splits": list(resolved.splits),
        "cache_dir": str(cache_dir),
        "n_rows_total": n_rows_total,
        "per_split": per_split,
    }


# ---- Per-cache-dir manifest writer ---------------------------------------

def _write_manifest(
    cache_dir: Path,
    sources: list[dict],
    args: dict,
) -> Path:
    """Write `<cache_dir>/voicefilter-manifest.json`.

    `sources` is a list of per-source stat dicts returned by
    `_prepare_source`. `args` is the runtime config snapshot. The
    manifest is per-cache-dir (no central `processed_dir`) so each
    cache is self-describing — `voicefilter-cli dataset clear`
    targets the right dir directly.

    Returns the manifest path.
    """
    manifest_path = cache_dir / MANIFEST_FILENAME
    payload = {
        "started_at": args.get("started_at"),
        "finished_at": time.time(),
        "cache_dir": str(cache_dir),
        "args": args,
        "sources": sources,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, default=str))
    return manifest_path


# ---- YAML config → dataset-baking defaults -------------------------------

def _yaml_to_prepare_defaults(cfg: dict) -> dict:
    """Extract prepare-dataset defaults from a loaded YAML config.

    Mirrors the function-oriented `dataset.*` block of `configs/paper.yaml`.
    Returns only fields that are explicitly set in the YAML; missing fields
    fall back to the built-in defaults at resolution time.

    Returned keys (all optional):
      * `target_sources`         : list[dict] — SourceRef dicts (passed
                                   through verbatim).
      * `interference_types`     : dict[type_name → dict] — overlay of the
                                   three built-in TYPEs' fields (only keys
                                   explicitly mentioned in the YAML).
      * `librispeech_subsets`    : tuple of LibriSpeech role tags
                                   (`"clean"` / `"other"`) to bake, unioned
                                   across every referenced LibriSpeech
                                   `SourceRef.subset`.
      * `train_<role>_<size>`    : bake flags for LibriSpeech splits named
                                   `train.<role>.<size>`. e.g.
                                   `train_clean_360=True` iff
                                   `train.clean.360` is referenced anywhere.
      * `splits`                 : tuple of non-train split names that
                                   need baking (e.g. `validation` / `test`).
      * `sample_rate`            : from `model.audio.sample_rate`.

    `target.sources[*]` and `interference.{babble,noise,music}.sources[*]`
    are `SourceRef`s referencing `dataset.sources[*]` (an `HfSourceSpec`
    by `id`). We don't resolve them here — `main()` builds an `HParams`
    and calls `resolve_source` for the actual prepare.
    """
    out: dict = {}
    ds = cfg.get("dataset") or {}

    # Build a `id → spec` map so we can walk target / interference
    # `SourceRef`s back to their `HfSourceSpec` for splits / configs.
    sources_by_id: dict[str, dict] = {
        s.get("id"): s for s in (ds.get("sources") or [])
    }

    # target.sources — pass through verbatim; the YAML keys match the
    # SourceRef dataclass (source / subset / splits).
    target = ds.get("target") or {}
    if "sources" in target:
        out["target_sources"] = target["sources"]

    # Union of LibriSpeech role tags needed across all referenced sources.
    # Role tags are parsed from `train.<role>.<size>` split names, not
    # carried on the SourceRef (`subset:` is reserved for HF config names
    # like MUSAN's speech/noise/music).
    interference = ds.get("interference") or {}
    # Interference TYPES are direct fields on `interference` (no `types`
    # wrapper): `babble`, `noise`, `music`. Only include keys the YAML
    # actually mentioned — the dataclass fills the rest with defaults.
    types_block = {
        k: v for k, v in interference.items()
        if k in ("babble", "noise", "music") and isinstance(v, dict)
    }
    roles: list[str] = []

    def _librispeech_role_of(ref: dict) -> Optional[str]:
        """Parse the LibriSpeech role tag from `ref.splits`, if any."""
        for s in ref.get("splits") or []:
            parts = s.split(".")
            if (
                len(parts) == 3
                and parts[0] == "train"
                and parts[1] in ("clean", "other")
            ):
                return parts[1]
        return None

    for src in target.get("sources") or []:
        role = _librispeech_role_of(src)
        if role and role not in roles:
            roles.append(role)
    for type_name, type_cfg in types_block.items():
        for src in type_cfg.get("sources") or []:
            role = _librispeech_role_of(src)
            if role and role not in roles:
                roles.append(role)
    if roles:
        out["librispeech_subsets"] = tuple(roles)

    # Aggregate LibriSpeech splits across referenced specs + SourceRef overrides.
    all_splits: list[str] = []
    for src in target.get("sources") or []:
        override_splits = src.get("splits") or []
        if override_splits:
            for s in override_splits:
                if s not in all_splits:
                    all_splits.append(s)
        else:
            spec = sources_by_id.get(src.get("source") or "")
            if spec:
                for s in spec.get("splits") or []:
                    if s not in all_splits:
                        all_splits.append(s)
    for type_name, type_cfg in types_block.items():
        for src in type_cfg.get("sources") or []:
            override_splits = src.get("splits") or []
            if override_splits:
                for s in override_splits:
                    if s not in all_splits:
                        all_splits.append(s)
            else:
                spec = sources_by_id.get(src.get("source") or "")
                if spec:
                    for s in spec.get("splits") or []:
                        if s not in all_splits:
                            all_splits.append(s)
    if all_splits:
        # Split names like `train.clean.360` / `train.other.500` carry the
        # role tag inline; derive one `train_<role>_<size>` flag per match
        # so consumers can check `train_clean_360` / `train_other_500`
        # without parsing split names themselves. Non-train splits
        # (`validation`, `test`) fall through to the generic `splits` tuple.
        for s in all_splits:
            parts = s.split(".")
            if (
                len(parts) == 3
                and parts[0] == "train"
                and parts[1] in ("clean", "other")
                and parts[2] in ("100", "360", "500")
            ):
                out[f"train_{parts[1]}_{parts[2]}"] = True
        extra = [s for s in all_splits if s in ("validation", "test")]
        out["splits"] = tuple(extra) if extra else ()

    # Interference TYPEs the YAML explicitly mentioned — only fields
    # actually set are kept; the dataclass fills the rest with defaults.
    if types_block:
        out["interference_types"] = types_block

    # audio.sample_rate → reuses for the d-vector / STFT path
    audio = cfg.get("model", {}).get("audio") or {}
    if "sample_rate" in audio:
        out["sample_rate"] = int(audio["sample_rate"])

    return out


# ---- CLI --------------------------------------------------------------------

app = App(name="prepare", help="Populate HF arrow caches for VoiceFilter-Lite training (no wav baking).", help_format="plaintext")


@app.default
def main(
    *,
    config: Annotated[
        Optional[str],
        Parameter(
            name="--config",
            help="Path to a YAML config (e.g. configs/paper.yaml). "
                 "Provides the full target / interference / paths blocks. "
                 "CLI flags still override these.",
        ),
    ] = None,
    librispeech_cache: Annotated[
        Optional[str], Parameter(name="--librispeech-cache")
    ] = None,
    musan_cache: Annotated[Optional[str], Parameter(name="--musan-cache")] = None,
    sample_rate: Annotated[Optional[int], Parameter(name="--sample-rate")] = None,
    # HF mirror knobs
    hf_endpoint: Annotated[
        Optional[str],
        Parameter(
            name="--hf-endpoint",
            help="Override HF Hub base URL (e.g. https://hf-mirror.com).",
        ),
    ] = None,
    hf_transfer: Annotated[
        Optional[bool],
        Parameter(
            negative="--no-hf-transfer",
            help="Force-enable HF hf_transfer protocol.",
        ),
    ] = None,
) -> None:
    """Populate HF arrow caches for VoiceFilter-Lite training.

    Run once before ``voicefilter-cli train voicefilter-lite``.

    Configuration precedence is CLI flag > ``--config`` YAML > built-in
    default. The YAML config provides the dataset shape (target sources,
    interference types + probs + sources, paths).
    CLI flags override individual knobs.
    """
    started_at = time.time()

    # 1. Load YAML config (if any) once and reuse it for both
    #    `_yaml_to_prepare_defaults` (CLI summary / sample_rate) and
    #    `apply_overrides` (HParams).
    cfg: dict = {}
    if config:
        from voicefilter_lite.config import load_config
        cfg = load_config(config)

    yaml_defaults: dict = _yaml_to_prepare_defaults(cfg) if cfg else {}
    if cfg:
        print(
            f"[cli] loaded config: {config} "
            f"({len(yaml_defaults)} dataset-baking field(s) applied)"
        )

    sample_rate = sample_rate if sample_rate is not None else yaml_defaults.get("sample_rate", 16000)
    sr = int(sample_rate)

    # 2. Build HParams from YAML BEFORE apply_hf_endpoint so the YAML's
    #    `hf_endpoint` / `hf_transfer` keys are layered into `hp` and
    #    visible to the resolver.
    from voicefilter_lite.config import apply_overrides

    hp = HParams()
    hp.sample_rate = sr
    if cfg:
        apply_overrides(hp, cfg, strict=True)
        if hp.sample_rate != sr:
            hp.sample_rate = sr

    # CLI flag > YAML > env > default. `hp` is built so the YAML's
    # `hf_endpoint` field participates.
    apply_hf_endpoint(
        hp=hp,
        cli_override=hf_endpoint,
        cli_transfer=hf_transfer,
    )

    # 3. CLI cache-dir overrides stamp onto every matching
    #    `HfSourceSpec.cache_dir`. We don't modify `HfSourceSpec` in
    #    place (it's the user-declared source); instead we layer the
    #    override at resolve time via `_resolve_cache`.
    def _resolve_cache(
        resolved: ResolvedSource,
        cli_override: Optional[str],
        default_for_family: str,
    ) -> str:
        """CLI override > spec.cache_dir > dataset-family default."""
        if cli_override is not None:
            return cli_override
        if resolved.cache_dir:
            return resolved.cache_dir
        return default_for_family

    # 4. Walk every (source, subset, split) referenced anywhere in the
    #    config, deduped by `cache_dir`, and populate the cache + JSON
    #    index. The same source can be referenced by multiple TYPEs
    #    (e.g. `musan(noise)` under `noise.sources` AND `babble.sources`);
    #    we dedupe by `(id, subset)` so each upstream pool is populated
    #    exactly once per cache_dir.
    seen_sources: set[tuple[str, Optional[str]]] = set()
    # cache_dir → list of per-source stats
    per_cache: dict[str, list[dict]] = {}

    def _process(resolved: ResolvedSource) -> None:
        cli_override = musan_cache if resolved.is_musan() else librispeech_cache
        default = DEFAULT_MUSAN_CACHE if resolved.is_musan() else DEFAULT_LIBRISPEECH_CACHE
        cache_dir_str = _resolve_cache(resolved, cli_override, default)
        cache_dir = Path(cache_dir_str)
        try:
            stats = _prepare_source(resolved, cache_dir)
        except Exception as e:
            print(f"[{resolved.id}] FAILED: {e!r}")
            return
        per_cache.setdefault(cache_dir_str, []).append(stats)

    # 4a. Target sources.
    for ref in hp.target_sources:
        try:
            resolved = resolve_source(hp, ref)
        except KeyError as e:
            print(f"[target] {e}; skipping")
            continue
        key = (resolved.id, resolved.subset)
        if key in seen_sources:
            continue
        seen_sources.add(key)
        _process(resolved)

    # 4b. Interference source pools (musan + librispeech(other)).
    for type_name, type_cfg in hp.interference_types.items():
        if not type_cfg.enabled or type_cfg.prob <= 0:
            continue
        for ref in type_cfg.sources:
            try:
                resolved = resolve_source(hp, ref)
            except KeyError as e:
                print(f"[interference.{type_name}] {e}; skipping")
                continue
            key = (resolved.id, resolved.subset)
            if key in seen_sources:
                continue
            seen_sources.add(key)
            _process(resolved)

    # 5. Per-cache-dir manifest.
    args_snapshot = {
        "config": config,
        "sample_rate": sr,
        "hf_endpoint": hf_endpoint,
        "hf_transfer": hf_transfer,
        "librispeech_cache": librispeech_cache,
        "musan_cache": musan_cache,
        "target_sources": [
            {"source": r.source, "splits": list(r.splits)}
            for r in hp.target_sources
        ],
        "interference_types": {
            t: {
                "enabled": c.enabled,
                "prob": c.prob,
                "sources": [
                    {"source": r.source, "splits": list(r.splits)}
                    for r in c.sources
                ],
            }
            for t, c in hp.interference_types.items()
        },
        "dataset_sources": [
            {
                "id": s.id, "repo": s.repo,
                "configs": list(s.configs),
                "splits": list(s.splits), "cache_dir": s.cache_dir,
            }
            for s in hp.dataset.sources
        ],
    }
    for cache_dir_str, sources in per_cache.items():
        manifest_path = _write_manifest(
            Path(cache_dir_str), sources,
            {**args_snapshot, "started_at": started_at},
        )
        print(f"[manifest] wrote {manifest_path}")

    print(f"\n[done] HF caches populated; "
          f"run `voicefilter-cli train voicefilter-lite`")


if __name__ == "__main__":
    app()