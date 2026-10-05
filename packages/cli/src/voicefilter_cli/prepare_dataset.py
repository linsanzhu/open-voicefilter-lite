#!/usr/bin/env python3
"""Pre-bake the VoiceFilter-Lite training dataset as local wav files.

The dataset is described by a function-oriented YAML config
(`configs/paper.yaml`). The config drives:

  - `dataset.sources[*]`     : upstream HF dataset declarations
                               (`HfSourceSpec` by id), shared by both
                               target and interference.
  - Target sources           : `SourceRef`s referencing `dataset.sources`
                               by `id`, with optional `splits:` override.
  - Interference TYPEs       : each TYPE pools over `SourceRef`s — every
                               TYPE that points at the same upstream
                               `HfSourceSpec` shares one baked pool under
                               `pool/<spec.id>/...`. Synthesis of the
                               actual interference waveform (babble / noise
                               / music) happens at TRAINING time inside
                               `LocalTripletStream`,
                               not here.
  - Per-speaker d-vectors    : 256-d .npy keyed by speaker id; filled lazily
                               during training by `DVectorCache`.

Output layout under `data/processed/`:

    manifest.json
    target/librispeech/clean/{split}/{spk}/{utt}.wav
    pool/librispeech/other/{split}/{spk}/{utt}.wav
    pool/musan/{subset}/{clip_id}.wav
    # NOTE — dvectors/ is NOT written by this script. The d-vector cache
    # is filled lazily by the training-time `DVectorCache` (in
    # `voicefilter_cli.dataset`); per-speaker files appear under
    # `data/processed/dvectors/librispeech/{spk}.npy` on first encounter.

Why pools (not per-type pools)?  Pre-baking `babble` clips at prepare
time was both buggy (the babble mixer exited after the first source clip
when that clip alone already exceeded the target length, so most "babble"
clips were actually single-speaker) and wasteful (each clip was a
synthesised 3-6 s wav). Now `prepare` only decodes audio; the training
iterator picks N distinct speakers and mixes them on the fly, so every
sample gets a freshly-mixed babble waveform and the on-disk footprint
shrinks by ~hundreds of MB on a full LibriSpeech-500 bake.

Once this script finishes, the training loop
(`voicefilter-cli train voicefilter-lite`) reads only from `data/processed/`
— zero HF traffic, zero per-batch decode.

Usage
-----
    # Bake the paper-faithful set (function-oriented defaults from
    # configs/paper.yaml):
    `voicefilter-cli dataset prepare` \\
        --config configs/paper.yaml \\
        --hf-endpoint https://hf-mirror.com

    # Quick smoke to a separate output dir:
    `voicefilter-cli dataset prepare` \\
        --config configs/paper.yaml \\
        --output-dir data/processed_smoke

Note: d-vector baking has been removed from this stage. d-vectors are
computed lazily during training (see `DVectorCache` in
`voicefilter_cli.dataset`), keyed per speaker, and cached under
`data/processed/dvectors/librispeech/{spk}.npy`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import soundfile as sf
from cyclopts import App, Parameter

from voicefilter_lite.hparams import HParams, ResolvedSource, resolve_source
from voicefilter_cli.hf import apply_hf_endpoint
from voicefilter_cli.dataset import (
    _cast_audio_no_decode,
    _disable_hf_audio_decode,
)

# ---- Per-utterance decode + write ------------------------------------------

def _decode_to_wav(
    audio_field: dict,
    target_sr: int,
) -> Optional[np.ndarray]:
    """Decode a HF `Audio(decode=False)` field to a (T,) float32 mono array.

    Returns None if the field is malformed. HF stores the audio as either
    `{"bytes": <FLAC bytes>, "path": ...}` or `{"array": <np.ndarray>,
    "sampling_rate": ...}`. We use `soundfile` (libsndfile) which is more
    lenient than FFmpeg with corrupt FLAC packets.
    """
    import io
    if not isinstance(audio_field, dict):
        return None
    try:
        if "bytes" in audio_field and audio_field["bytes"] is not None:
            data, sr = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32")
        elif "array" in audio_field and audio_field["array"] is not None:
            data = np.asarray(audio_field["audio" if "audio" in audio_field else "array"], dtype=np.float32)
            sr = int(audio_field.get("sampling_rate", target_sr))
        elif "path" in audio_field and audio_field["path"]:
            data, sr = sf.read(audio_field["path"], dtype="float32")
        else:
            return None
    except Exception:
        return None
    if data.ndim > 1:
        data = data.mean(axis=1)
    # Resample if needed (librosa's resampling is the codebase standard).
    if sr != target_sr:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    # Defensive: clip huge peaks so downstream arithmetic doesn't overflow.
    peak = float(np.abs(data).max()) if data.size else 0.0
    if peak > 5.0:
        data = data / peak
    return data.astype(np.float32)

def _write_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    """Write a (T,) float32 wav with libsndfile FLOAT subtype (lossless)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr, subtype="FLOAT")

def _already_baked(path: Path) -> bool:
    """Skip writes for files that exist and are non-empty (resumable)."""
    return path.is_file() and path.stat().st_size > 0

# ---- HF dataset loader (HF datasets 5.x compatibility) ---------------------

def _load_hf_split(repo: str, config: Optional[str], split: str, cache_dir: Path):
    """Load an HF dataset split, monkey-patching the audio decoder to no-op.

    We bypass `Audio(decode=True)` because it eagerly instantiates a
    torchcodec `AudioDecoder` (FFmpeg) which crashes on certain corrupt
    FLAC packets. We decode the raw bytes ourselves via `soundfile`.
    """
    from datasets import load_dataset

    _disable_hf_audio_decode()

    kwargs: dict = {"cache_dir": str(cache_dir)}
    if config is not None:
        kwargs["name"] = config
    ds = load_dataset(repo, split=split, **kwargs)
    return _cast_audio_no_decode(ds)


# ---- Per-asset bakers ------------------------------------------------------

def _bake_librispeech(
    out_root: Path,
    subset: str,
    split: str,
    target_sr: int,
    cache_dir: Path,
    role: str,
    spec_id: str,
) -> dict:
    """Bake every utterance in `openslr/librispeech_asr[all][split]`.

    LibriSpeech is loaded via the `all` config; `subset` (`"clean"` /
    `"other"`) becomes the per-role directory under `<role>/<spec_id>/`
    so the target pool lands at `target/<spec_id>/clean/<split>/...`
    and interference pools at `interference/<spec_id>/other/<split>/...`.
    `spec_id` is the HfSourceSpec.id of the source (default "librispeech")
    — passed in by the caller so the dir segment follows the config, not a
    hardcoded literal. Returns stats: {n_utterances, n_skipped, n_errors,
    n_speakers}.
    """
    print(f"[{spec_id}] loading all/{split}…")
    ds = _load_hf_split(
        "openslr/librispeech_asr", config="all", split=split, cache_dir=cache_dir,
    )
    out_dir = out_root / role / spec_id / subset / split
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    n_errors = 0
    speakers: set[str] = set()

    for i, ex in enumerate(ds):
        spk = str(ex.get("speaker_id", f"unknown_{i}"))
        utt_id = str(ex.get("file", ex.get("id", i)))
        # Strip path prefixes that show up in some HF mirrors (e.g.
        # `train-clean-100/0000/0000.flac` → `0000`).
        utt_id = utt_id.replace("/", "_").replace("\\", "_")
        if utt_id.endswith(".flac"):
            utt_id = utt_id[:-5]

        speakers.add(spk)
        out_path = out_dir / spk / f"{utt_id}.wav"
        if _already_baked(out_path):
            n_skipped += 1
            continue

        wav = _decode_to_wav(ex.get("audio", {}), target_sr)
        if wav is None or wav.size == 0:
            n_errors += 1
            continue
        try:
            _write_wav(out_path, wav, target_sr)
            n_written += 1
        except Exception:
            n_errors += 1

        if (n_written + n_skipped + n_errors) % 1000 == 0:
            print(
                f"[{spec_id}] {split}: "
                f"{n_written + n_skipped}/{len(ds)} ({n_errors} errors)",
                flush=True,
            )

    print(
        f"[{spec_id}] {split}: "
        f"wrote={n_written} skipped={n_skipped} errors={n_errors} "
        f"speakers={len(speakers)}"
    )
    return {
        "n_utterances": n_written + n_skipped,
        "n_written": n_written,
        "n_skipped": n_skipped,
        "n_errors": n_errors,
        "n_speakers": len(speakers),
    }

def _iter_librispeech_clips(
    resolved: ResolvedSource,
    cache_dir: Path,
    target_sr: int,
):
    """Yield `(clip_id, wav)` for each utterance in a LibriSpeech pool.

    LibriSpeech is loaded via the `all` config (single arrow cache
    holding every `train.<role>.<size>` split); the role tag is parsed
    from the split name and baked into the clip id so different roles
    don't collide on speaker ids.
    """
    repo = resolved.repo
    role = resolved.librispeech_role() or "librispeech"
    for split in resolved.splits:
        print(f"[{resolved.id}] loading all/{split}…")
        ds = _load_hf_split(
            repo, config="all", split=split, cache_dir=cache_dir,
        )
        for ex in ds:
            spk = str(ex.get("speaker_id", "unknown"))
            utt_id = str(ex.get("file", ex.get("id", "x")))
            utt_id = utt_id.replace("/", "_").replace("\\", "_")
            if utt_id.endswith(".flac"):
                utt_id = utt_id[:-5]
            wav = _decode_to_wav(ex.get("audio", {}), target_sr)
            if wav is None or wav.size == 0:
                continue
            yield f"{role}__{spk}__{utt_id}", wav


def _iter_musan_clips(
    resolved: ResolvedSource,
    cache_dir: Path,
    target_sr: int,
    subset: str,
):
    """Yield `(clip_id, wav)` for each clip in a MUSAN subset.

    `resolved.subset` is the (per-usage) MUSAN subset; `resolved.spec`
    carries the HF repo and cache. The split is fixed to `"train"` —
    MUSAN exposes one split per subset on HF.
    """
    repo = resolved.repo
    print(f"[{resolved.id}/{subset}] loading {repo}/{subset}…")
    ds = _load_hf_split(
        repo, config=subset, split="train", cache_dir=cache_dir,
    )
    for i, ex in enumerate(ds):
        clip_id = str(ex.get("id", ex.get("file", i)))
        clip_id = clip_id.replace("/", "_").replace("\\", "_")
        if clip_id.endswith((".wav", ".flac", ".mp3")):
            clip_id = clip_id.rsplit(".", 1)[0]
        wav = _decode_to_wav(ex.get("audio", {}), target_sr)
        if wav is None or wav.size == 0:
            continue
        yield clip_id, wav


def _iter_source_clips(
    resolved: ResolvedSource,
    cache_dir: Path,
    target_sr: int,
):
    """Dispatch a `ResolvedSource` to the right clip iterator."""
    if resolved.is_librispeech():
        yield from _iter_librispeech_clips(resolved, cache_dir, target_sr)
        return
    if resolved.is_musan():
        subset = resolved.subset
        if not subset:
            raise ValueError(
                f"MUSAN source {resolved.id!r} must specify `subset:` "
                "when referenced (multiple subsets declared on the spec)."
            )
        yield from _iter_musan_clips(resolved, cache_dir, target_sr, subset)
        return
    raise ValueError(
        f"unknown source spec: id={resolved.id!r} "
        f"repo={resolved.repo!r}"
    )


def _bake_pool_librispeech(
    out_root: Path,
    target_sr: int,
    resolved: ResolvedSource,
    cache_dir: Path,
) -> dict:
    """Bake `pool/<spec.id>/<role>/<split>/<spk>/<utt>.wav`.

    LibriSpeech is loaded via the `all` config (single arrow cache
    holding every `train.<role>.<size>` split); the role tag is parsed
    from the split name (`train.clean.360` → `clean`) and becomes the
    per-role pool directory so `babble` (and any future speech-source
    TYPE) can pull from `pool/librispeech/other/...` without colliding
    with the `clean` target pool under `pool/librispeech/clean/...`. The
    directory layout (`{spk}` as the fourth path component) is what
    `LocalTripletStream` parses at scan time to recover `spk_id` for
    babble speaker-dedup — each speaker's utterances live under their
    own directory.
    """
    pool_id = resolved.id
    role = resolved.librispeech_role()
    assert role in ("clean", "other"), (
        f"LibriSpeech pool ref must list a `train.clean.*` or "
        f"`train.other.*` split; got {resolved.splits!r}"
    )
    out_dir = out_root / "pool" / pool_id / role
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = resolved.repo
    n_clips = 0
    n_skipped = 0
    n_errors = 0
    speakers: set[str] = set()

    for split in resolved.splits:
        print(f"[pool/{pool_id}/{role}] loading all/{split}…")
        ds = _load_hf_split(
            repo, config="all", split=split, cache_dir=cache_dir,
        )
        for i, ex in enumerate(ds):
            spk = str(ex.get("speaker_id", f"unknown_{i}"))
            utt_id = str(ex.get("file", ex.get("id", i)))
            utt_id = utt_id.replace("/", "_").replace("\\", "_")
            if utt_id.endswith(".flac"):
                utt_id = utt_id[:-5]

            speakers.add(spk)
            out_path = out_dir / split / spk / f"{utt_id}.wav"
            if _already_baked(out_path):
                n_skipped += 1
                continue

            wav = _decode_to_wav(ex.get("audio", {}), target_sr)
            if wav is None or wav.size == 0:
                n_errors += 1
                continue
            try:
                _write_wav(out_path, wav, target_sr)
                n_clips += 1
            except Exception:
                n_errors += 1

            if (n_clips + n_skipped + n_errors) % 1000 == 0:
                print(
                    f"[pool/{pool_id}/{role}] {split}: "
                    f"{n_clips + n_skipped}/{len(ds)} ({n_errors} errors)",
                    flush=True,
                )

    print(
        f"[pool/{pool_id}/{role}] wrote={n_clips} skipped={n_skipped} "
        f"errors={n_errors} speakers={len(speakers)}"
    )
    return {
        "n_clips": n_clips,
        "n_skipped": n_skipped,
        "n_errors": n_errors,
        "n_speakers": len(speakers),
    }


def _bake_pool_musan(
    out_root: Path,
    target_sr: int,
    resolved: ResolvedSource,
    cache_dir: Path,
    subset: str,
) -> dict:
    """Bake `pool/<spec.id>/{subset}/{clip_id}.wav` for MUSAN `subset`.

    For `subset == "speech"`, every clip is treated as one speaker identity
    by `LocalTripletStream`'s babble picker (musan speech is a long monologue;
    picking a 0.5-2 s random segment of it stands in for "one speaker").
    For `subset == "noise" / "music"` the per-clip identity is irrelevant —
    those pools feed the noise/music TYPEs that take 1 clip per sample.

    The pool's parent directory is `pool/<spec.id>/` (e.g. `pool/musan/`)
    so all three subsets share one parent; the per-clip wavs land under
    `pool/<spec.id>/<subset>/<clip_id>.wav`.
    """
    assert subset in ("speech", "noise", "music"), subset
    pool_id = resolved.id
    out_dir = out_root / "pool" / pool_id / subset
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = resolved.repo
    print(f"[pool/{pool_id}/{subset}] loading {repo}/{subset}…")
    ds = _load_hf_split(
        repo, config=subset, split="train", cache_dir=cache_dir,
    )
    n_clips = 0
    n_skipped = 0
    n_errors = 0
    for i, ex in enumerate(ds):
        clip_id = str(ex.get("id", ex.get("file", i)))
        clip_id = clip_id.replace("/", "_").replace("\\", "_")
        if clip_id.endswith((".wav", ".flac", ".mp3")):
            clip_id = clip_id.rsplit(".", 1)[0]
        out_path = out_dir / f"{clip_id}.wav"
        if _already_baked(out_path):
            n_skipped += 1
            continue
        wav = _decode_to_wav(ex.get("audio", {}), target_sr)
        if wav is None or wav.size == 0:
            n_errors += 1
            continue
        try:
            _write_wav(out_path, wav, target_sr)
            n_clips += 1
        except Exception:
            n_errors += 1

    print(
        f"[pool/{pool_id}/{subset}] wrote={n_clips} skipped={n_skipped} "
        f"errors={n_errors}"
    )
    return {
        "n_clips": n_clips,
        "n_skipped": n_skipped,
        "n_errors": n_errors,
    }


def _bake_pool_one(
    out_root: Path,
    target_sr: int,
    resolved: ResolvedSource,
    cache_dir: Path,
) -> dict:
    """Dispatch one resolved source to the right pool baker.

    Writes to `pool/<spec.id>/...`. Multiple TYPEs may reference the
    same `HfSourceSpec` (e.g. `babble` lists both `librispeech(other)`
    and `musan(speech)` as its source pool); the caller dedupes by
    `resolved.spec.id` (+ subset) so each upstream pool is baked
    exactly once.
    """
    if resolved.is_librispeech():
        return _bake_pool_librispeech(out_root, target_sr, resolved, cache_dir)
    if resolved.is_musan():
        subset = resolved.subset
        if not subset:
            raise ValueError(
                f"MUSAN source {resolved.id!r} must specify `subset:` "
                "when referenced."
            )
        return _bake_pool_musan(out_root, target_sr, resolved, cache_dir, subset)
    raise ValueError(
        f"unknown source spec: id={resolved.id!r} "
        f"repo={resolved.repo!r}"
    )

# ---- YAML config → dataset-baking defaults -------------------------------

def _yaml_to_prepare_defaults(cfg: dict) -> dict:
    """Extract prepare-dataset defaults from a loaded YAML config.

    Mirrors the function-oriented `dataset.*` block of `configs/paper.yaml`.
    Returns only fields that are explicitly set in the YAML; missing fields
    fall back to the built-in defaults at resolution time.

    Returned keys (all optional):
      * `target_sources`         : list[dict] — each entry is one source spec
                                   (with `cache_dir` attached for HF arrow
                                   caching).
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
      * `output_dir`             : filesystem path (`dataset.processed_dir`).
      * `sample_rate`            : from `model.audio.sample_rate`.

    `target.sources[*]` and `interference.{babble,noise,music}.sources[*]`
    are `SourceRef`s referencing `dataset.sources[*]` (an `HfSourceSpec`
    by `id`). We don't resolve them here — main() builds an `HParams`
    (which applies the defaults + YAML overrides) and calls
    `resolve_source` for the actual baking.
    """
    out: dict = {}
    ds = cfg.get("dataset") or {}

    # processed_dir (top-level `dataset.processed_dir`).
    if "processed_dir" in ds:
        out["output_dir"] = ds["processed_dir"]

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
        # Per-ref override takes precedence; else the spec's default splits.
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

app = App(name="prepare", help_format="plaintext")


@app.default
def main(
    *,
    # YAML config overlay — populates the full dataset.* block (target
    # sources, interference types, augmentation, paths, …). The single
    # source of truth for dataset baking decisions.
    config: Annotated[
        Optional[str],
        Parameter(
            name="--config",
            help="Path to a YAML config (e.g. configs/paper.yaml). "
                 "Provides the full target / interference / paths blocks. "
                 "CLI flags still override these.",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[str],
        Parameter(name="--output-dir", help="Where to write wav / npy / manifest.json."),
    ] = None,
    librispeech_cache: Annotated[
        Optional[str], Parameter(name="--librispeech-cache")
    ] = None,
    musan_cache: Annotated[Optional[str], Parameter(name="--musan-cache")] = None,
    # Audio params for STFT + d-vector (must match training config)
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
    # Skip options
    # (d-vector baking has moved to training time — see `DVectorCache`
    # in `voicefilter_cli.dataset`. There is no `--skip-dvectors` flag
    # here anymore; the cache fills on first use.)
) -> None:
    """Pre-bake VoiceFilter-Lite training data into local wav files.

    Run once before ``voicefilter-cli train voicefilter-lite``.

    Configuration precedence is CLI flag > ``--config`` YAML > built-in
    default. The YAML config provides the dataset shape (target sources,
    interference types + probs + sources, paths).
    CLI flags override individual knobs.
    """
    # 1. Load YAML config overlay (if any) → dataset-baking defaults.
    yaml_defaults: dict = {}
    if config:
        from voicefilter_lite.config import load_config
        cfg = load_config(config)
        yaml_defaults = _yaml_to_prepare_defaults(cfg)
        print(
            f"[cli] loaded config: {config} "
            f"({len(yaml_defaults)} dataset-baking field(s) applied)"
        )

    # 2. Resolve CLI > YAML > built-in fallback for paths.
    def _r(key: str, fallback):
        return yaml_defaults.get(key, fallback)

    output_dir = output_dir if output_dir is not None else _r("output_dir", "data/processed")
    sample_rate = sample_rate if sample_rate is not None else _r("sample_rate", 16000)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    apply_hf_endpoint(
        hp=None,
        cli_override=hf_endpoint,
        cli_transfer=hf_transfer,
    )

    sr = sample_rate

    # 3. Build the dataset from YAML + CLI. We construct HParams and let
    # `apply_overrides` walk the YAML — that path already understands the
    # function-oriented schema (dataset.sources / target.sources /
    # interference.{babble,noise,music}) AND the new `HfSourceSpec` /
    # `SourceRef` split.
    from voicefilter_lite.config import apply_overrides

    hp = HParams()
    hp.sample_rate = sr
    if config:
        cfg = load_config(config)
        apply_overrides(hp, cfg, strict=True)
        if hp.sample_rate != sr:
            hp.sample_rate = sr

    # 4. CLI cache-dir overrides (`--librispeech-cache` / `--musan-cache`)
    #    stamp onto every matching `HfSourceSpec.cache_dir`. We don't
    #    modify `HfSourceSpec` in place (it's the user-declared source);
    #    instead we layer the override at resolve time via `_resolve_cache`.
    # No-explicit-cache_dir fallbacks. Paper-faithful values match
    # `configs/paper.yaml` (`cache_dir: data/librispeech` /
    # `cache_dir: data/musan`). Pass at the call site so the policy is
    # not hardcoded inside the closure.
    default_librispeech_cache = "data/librispeech"
    default_musan_cache = "data/musan"

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
        # No explicit cache_dir → fall back to the dataset-family
        # default supplied by the caller (paper.yaml values by default).
        return default_for_family

    # 5. Persist what we baked for this run.
    target_manifest_entries: list[dict] = []
    for ref in hp.target_sources:
        try:
            resolved = resolve_source(hp, ref)
        except KeyError as e:
            print(f"[target] {e}; skipping")
            continue
        cache_dir_used = _resolve_cache(
            resolved,
            librispeech_cache,
            default_musan_cache if resolved.is_musan() else default_librispeech_cache,
        )
        target_manifest_entries.append({
            "source": resolved.id,
            "subset": resolved.subset,
            "splits": list(resolved.splits),
            "repo": resolved.repo,
            "cache_dir": cache_dir_used,
        })

    interference_manifest: dict[str, dict] = {}
    for type_name, type_cfg in hp.interference_types.items():
        per_type_sources: list[dict] = []
        for ref in type_cfg.sources:
            try:
                resolved = resolve_source(hp, ref)
            except KeyError as e:
                print(f"[interference.{type_name}] {e}; skipping")
                continue
            cli_override = musan_cache if resolved.is_musan() else librispeech_cache
            cache_dir_used = _resolve_cache(
                resolved,
                cli_override,
                default_musan_cache if resolved.is_musan() else default_librispeech_cache,
            )
            per_type_sources.append({
                "source": resolved.id,
                "subset": resolved.subset,
                "splits": list(resolved.splits),
                "repo": resolved.repo,
                "cache_dir": cache_dir_used,
            })
        interference_manifest[type_name] = {
            "enabled": type_cfg.enabled,
            "prob": type_cfg.prob,
            "sources": per_type_sources,
            # Synthesis is at training time; no `pool_size` here
            # (used to be the pre-bake pool count for babble).
            "n_speakers_min": type_cfg.n_speakers_min,
            "n_speakers_max": type_cfg.n_speakers_max,
            "babble_seg_min_sec": type_cfg.babble_seg_min_sec,
            "babble_seg_max_sec": type_cfg.babble_seg_max_sec,
            "gain_min": type_cfg.gain_min,
            "gain_max": type_cfg.gain_max,
        }

    manifest: dict = {
        "started_at": time.time(),
        "args": {
            "config": config,
            "output_dir": output_dir,
            "sample_rate": sr,
            "hf_endpoint": hf_endpoint,
            "hf_transfer": hf_transfer,
            "target_sources": target_manifest_entries,
            "interference_types": interference_manifest,
            "dataset_sources": [
                {
                    "id": s.id, "repo": s.repo,
                    "configs": list(s.configs),
                    "splits": list(s.splits), "cache_dir": s.cache_dir,
                }
                for s in hp.dataset.sources
            ],
        },
    }

    # 6. Bake target sources. We resolve each `SourceRef` against the
    # top-level `dataset.sources` declarations; the resolved spec gives
    # us `repo`, `subset`, `splits`, and (after CLI override) the
    # cache_dir to load from. LibriSpeech is loaded via the `all` config
    # (single arrow cache holding every `train.<role>.<size>` split);
    # the role tag (`clean` / `other`) becomes the per-role dir under
    # `target/librispeech/<role>/<split>/...`.
    target_stats: dict[str, dict] = {}
    for ref in hp.target_sources:
        try:
            resolved = resolve_source(hp, ref)
        except KeyError as e:
            print(f"[target] {e}; skipping")
            continue
        if not resolved.is_librispeech():
            print(
                f"[target] skipping non-LibriSpeech source "
                f"{resolved.id!r} (repo={resolved.repo!r})"
            )
            continue
        role = resolved.librispeech_role()
        assert role in ("clean", "other"), (
            f"target LibriSpeech ref needs a `train.clean.*` or "
            f"`train.other.*` split; got {resolved.splits!r}"
        )
        cache_dir = Path(_resolve_cache(resolved, librispeech_cache, default_librispeech_cache))
        for split in resolved.splits:
            key = f"librispeech/{role}/{split}"
            try:
                target_stats[key] = _bake_librispeech(
                    out_root,
                    subset=role,
                    split=split,
                    target_sr=sr,
                    cache_dir=cache_dir,
                    role="target",
                    spec_id=resolved.id,
                )
            except Exception as e:
                print(f"[target] FAILED {key}: {e!r}")
                target_stats[key] = {"error": str(e)}

    # 7. Bake interference SOURCE POOLS, deduped by resolved spec.
    # The runtime iterator (`LocalTripletStream`) pulls from these flat
    # pools and synthesises `babble` / `noise` / `music` per sample.
    # Multiple TYPEs may share a source (e.g. `babble` previously listed
    # both `librispeech(other)` and `musan(speech)`; now babble lists
    # only `musan(speech)`); we walk `seen_pools` so each upstream pool
    # is baked exactly once. The librispeech `other` pool is still
    # baked if any TYPE references it (e.g. legacy configs).
    pool_stats: dict[str, dict] = {}
    seen_pools: set[tuple[str, Optional[str]]] = set()
    for type_name, type_cfg in hp.interference_types.items():
        if not type_cfg.enabled or type_cfg.prob <= 0:
            continue
        for ref in type_cfg.sources:
            try:
                resolved = resolve_source(hp, ref)
            except KeyError as e:
                print(f"[interference.{type_name}] {e}; skipping")
                continue
            pool_key = (resolved.id, resolved.subset)
            if pool_key in seen_pools:
                continue
            seen_pools.add(pool_key)
            cli_override = musan_cache if resolved.is_musan() else librispeech_cache
            cache_dir = Path(_resolve_cache(
                resolved,
                cli_override,
                default_musan_cache if resolved.is_musan() else default_librispeech_cache,
            ))
            try:
                pool_stats[resolved.id + (
                    f"/{resolved.subset}" if resolved.subset else ""
                )] = _bake_pool_one(
                    out_root, sr, resolved, cache_dir,
                )
            except Exception as e:
                key = resolved.id + (
                    f"/{resolved.subset}" if resolved.subset else ""
                )
                print(f"[pool] FAILED {key}: {e!r}")
                pool_stats[key] = {"error": str(e)}

    # 8. Write manifest. d-vectors are filled lazily during training by
    # `DVectorCache` (see `voicefilter_cli.dataset`) — they are not part
    # of the prepare step anymore.
    manifest.update({
        "finished_at": time.time(),
        "target": target_stats,
        "pool": pool_stats,
    })
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\n[manifest] wrote {manifest_path}")
    print(f"[done] all preprocessing complete; run `voicefilter-cli train voicefilter-lite`")


if __name__ == "__main__":
    app()
