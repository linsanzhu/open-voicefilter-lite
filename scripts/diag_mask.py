#!/usr/bin/env python3
"""Diagnostic: print the mask distribution of a trained VoiceFilter-Lite ckpt.

For each mel-bin in each frame of each test sample, collect the raw sigmoid
mask value the model emits. Then print:

  * global stats (mean, std, p1/p5/p25/p50/p75/p95/p99)
  * 10-bucket histogram
  * per-bin mean mask (low-freq → high-freq) — reveals if the model only
    suppresses a specific band
  * mask split by input frame energy (silent / low / mid / high)
  * Pearson correlation between mask and input energy — near 0 means the
    mask collapsed to a constant; positive means it tracks input presence

Usage
-----
    python scripts/diag_mask.py --checkpoint runs/mps/ckpt_step047000.pt
    python scripts/diag_mask.py --checkpoint runs/mps/ckpt_final.pt \\
        --num-samples 100 --results-json mask_stats.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from open_voicefilter_lite.audio import (
    make_mel_basis,
    pad_or_trim,
    random_crop,
    snr_mix,
    wav_to_log_mel,
)
from open_voicefilter_lite.cli import apply_hf_endpoint
from open_voicefilter_lite.dataset import (
    _cast_audio_no_decode,
    _disable_hf_audio_decode,
)
from open_voicefilter_lite.hparams import HParams
from open_voicefilter_lite.model import VoiceFilterLite
from open_voicefilter_lite.speaker_encoder import FrozenSpeakerEncoder


# Threshold for input frame energy buckets (log-mel mean across bins).
# log_mel_floor=1e-6 → log(1e-6)≈-13.8 is the absolute silence floor.
# "silent"    : energy ≤ -10   (≈ near-floor — pure silence / noise-only)
# "low"       : -10 < energy ≤ -5 (very quiet)
# "mid"       : -5  < energy ≤ 0  (normal speech)
# "high"      : energy > 0        (loud)
ENERGY_BUCKETS: list[tuple[str, float, float]] = [
    ("silent",   float("-inf"), -10.0),
    ("low",      -10.0,           -5.0),
    ("mid",       -5.0,            0.0),
    ("high",      0.0,       float("inf")),
]


def decode_audio(audio_field: dict, target_sr: int) -> torch.Tensor:
    """Decode HF audio bytes → mono float32 tensor at target_sr.

    Mirrors `open_voicefilter_lite.dataset._decode_audio` so this script
    behaves identically to the training-time path (handles dict-with-bytes,
    dict-with-array, and AudioDecoder-with-bytes/path). The latter shows up
    if `_disable_hf_audio_decode` monkey-patch didn't apply for any reason.
    """
    # Path 1: raw dict with bytes/array (the `_cast_audio_no_decode` path).
    if isinstance(audio_field, dict):
        if "bytes" in audio_field and audio_field["bytes"] is not None:
            data, sr = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32")
        elif "array" in audio_field and audio_field["array"] is not None:
            data = np.asarray(audio_field["array"], dtype=np.float32)
            sr = int(audio_field.get("sampling_rate", target_sr))
        elif "path" in audio_field and audio_field["path"]:
            data, sr = sf.read(audio_field["path"], dtype="float32")
        else:
            raise ValueError(
                f"unrecognised audio dict keys: {list(audio_field.keys())}"
            )
    # Path 2: AudioDecoder object (fallback if HF monkey-patch missed).
    elif hasattr(audio_field, "get_all_samples") and callable(
        getattr(audio_field, "get_all_samples", None)
    ):
        try:
            recovered = _soundfile_fallback(audio_field, target_sr)
            if recovered is not None:
                return recovered
        except Exception:
            pass
        samples = audio_field.get_all_samples()
        wav = samples.data
        sr = int(samples.sample_rate)
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
    else:
        raise ValueError(
            f"unrecognised audio field type: {type(audio_field).__name__}"
        )

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


def _soundfile_fallback(audio_field, target_sr: int):
    """Last-resort: try soundfile on an AudioDecoder's `.bytes` / `.path`."""
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
            wav = torchaudio.functional.resample(
                wav.unsqueeze(0), sr, target_sr
            ).squeeze(0)
        peak = wav.abs().max().clamp(min=1e-8)
        if peak > 5.0:
            wav = wav / peak
        return wav
    except Exception:
        return None


def build_eval_pairs(
    hp: HParams,
    num_samples: int,
    seed: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Same construction as `scripts/evaluate.build_eval_pairs`."""
    from datasets import load_dataset

    load_kwargs: dict = {"streaming": hp.streaming}
    if not hp.streaming:
        load_kwargs["cache_dir"] = hp.librispeech_cache

    _disable_hf_audio_decode()

    def _load(config: str, split: str):
        ds = load_dataset(
            "openslr/librispeech_asr",
            config,
            split=split,
            **load_kwargs,
        )
        return _cast_audio_no_decode(ds)

    target_iter = iter(
        chain.from_iterable(_load(hp.target_config, s) for s in hp.target_splits)
    )
    interference_iter = iter(_load(hp.interference_config, hp.interference_split))

    # Probe the first row so we can fail loudly if the schema is wrong
    # (instead of silently consuming the whole iterator on bad `audio`).
    try:
        first_tgt = next(target_iter)
        first_intf = next(interference_iter)
    except StopIteration:
        print(f"[diag] ERROR: target/interference iterators empty on probe")
        return

    audio_type = type(first_tgt.get("audio")).__name__
    keys = sorted(first_tgt.keys())
    print(f"[diag] first target row: keys={keys} audio_type={audio_type}")
    if "speaker_id" in first_tgt:
        print(f"[diag]   speaker_id={first_tgt['speaker_id']!r}")
    else:
        print(f"[diag]   WARNING: 'speaker_id' field missing — will use random IDs")

    # Re-prime: `next()` already consumed the first row of each; reuse them.
    rng = random.Random(seed)
    spk_buf: dict[str, list] = {}

    def _spk_key(row: dict) -> str:
        spk = row.get("speaker_id")
        if spk is None:
            return f"_anon_{rng.randint(0, 10_000)}"
        return str(spk)

    # First row pair goes into the buffer to seed same-speaker ref picking.
    tgt = first_tgt
    intf = first_intf
    spk_buf.setdefault(_spk_key(tgt), []).append(tgt)

    n = 0
    attempts = 0
    max_attempts = max(50, num_samples * 30)
    decode_fail = 0
    last_decode_err: str = ""

    while n < num_samples and attempts < max_attempts:
        attempts += 1
        try:
            tgt = next(target_iter)
            intf = next(interference_iter)
        except StopIteration:
            print(f"[diag] iterators exhausted after {attempts} attempts "
                  f"(yielded {n}/{num_samples}, decode_fail={decode_fail})")
            return

        spk = _spk_key(tgt)
        buf = spk_buf.setdefault(spk, [])

        # Prefer a same-speaker ref, but fall back to using tgt itself for
        # the d-vector if no buffer entry yet — dvec quality is acceptable
        # either way for a *mask-distribution* diagnostic.
        ref = rng.choice(buf) if buf else tgt

        try:
            target_wav = decode_audio(tgt["audio"], hp.sample_rate)
            interference_wav = decode_audio(intf["audio"], hp.sample_rate)
            ref_wav = decode_audio(ref["audio"], hp.sample_rate)
        except Exception as e:
            decode_fail += 1
            last_decode_err = f"{type(e).__name__}: {e}"
            if decode_fail <= 3 or decode_fail % 50 == 0:
                print(f"[diag] decode failed ({decode_fail}× so far): "
                      f"audio_type={type(tgt.get('audio')).__name__}, "
                      f"err={last_decode_err}")
            buf.append(tgt)
            continue

        sample_len = hp.sample_length_samples
        target_wav = random_crop(target_wav, sample_len)
        interference_wav = random_crop(interference_wav, sample_len)
        snr = rng.uniform(hp.snr_min_db, hp.snr_max_db)
        mixed = snr_mix(target_wav, interference_wav, snr)

        yield mixed, target_wav, ref_wav, interference_wav
        buf.append(tgt)
        n += 1

        if n % 5 == 0 or n <= 3:
            print(f"[diag] collected {n}/{num_samples} samples "
                  f"(attempts={attempts}, decode_fail={decode_fail})")

    if n < num_samples:
        print(f"[diag] hit max_attempts={max_attempts} with only "
              f"{n}/{num_samples} samples; last_decode_err={last_decode_err}")


def _mask_per_frame_hook(model: VoiceFilterLite) -> dict:
    """Install a forward hook on `model.out_proj` so we capture every sigmoid
    mask the model emits during streaming inference (without re-running the
    full model). Returns a dict with `frames: list[(n_mels,)]`.

    We use a `register_forward_hook` (NOT pre-hook) so we receive `out_proj`'s
    output directly. Recomputing `sigmoid(out_proj(x))` inside a pre-hook
    would re-trigger the pre-hook and recurse infinitely — see the
    RecursionError that motivated this design.
    """
    store: dict = {"frames": []}

    def hook(_module, _inputs, output):
        # `output` is `out_proj(x)`, shape (1, 1, n_mels), pre-sigmoid.
        # Apply sigmoid ourselves; same arithmetic as `model.forward`.
        with torch.no_grad():
            m = torch.sigmoid(output.detach().cpu().flatten())
        store["frames"].append(m)

    handle = model.out_proj.register_forward_hook(hook)
    store["__handle__"] = handle
    return store


def _print_histogram(counts: np.ndarray, total: int, n_buckets: int = 10) -> str:
    """ASCII histogram. `counts[i]` = #samples in bucket i."""
    width = 50
    lines = []
    max_count = counts.max() if counts.size and counts.max() > 0 else 1
    for i in range(n_buckets):
        lo = i / n_buckets
        hi = (i + 1) / n_buckets
        c = int(counts[i])
        pct = 100.0 * c / max(1, total)
        bar_len = int(width * c / max_count)
        bar = "█" * bar_len
        lines.append(
            f"  [{lo:.1f}, {hi:.1f}): {c:>7d} ({pct:5.2f}%) {bar}"
        )
    return "\n".join(lines)


def _format_pct(p: float) -> str:
    return f"{100.0 * p:5.2f}%"


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose mask distribution of a VoiceFilter ckpt")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-samples", type=int, default=50,
                   help="Number of LibriSpeech (target, interference) pairs.")
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--results-json", type=str, default=None,
                   help="If set, dump stats to this JSON file.")
    p.add_argument("--n-bins-show", type=int, default=16,
                   help="Number of mel-bin groups in the per-bin report "
                        "(bins are averaged within each group for readability).")
    p.add_argument("--hf-endpoint", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hp = HParams.from_dict(ckpt["hparams"])
    hp.device = str(device)

    apply_hf_endpoint(
        hp,
        cli_override=args.hf_endpoint,
        cli_transfer=None,
    )

    print(f"[diag] loading {args.checkpoint} (step={ckpt.get('step')}) on {device}")
    model = VoiceFilterLite(hp).to(device).eval()
    model.load_state_dict(ckpt["model_state"])

    speaker_encoder = FrozenSpeakerEncoder(hp).to(device)
    mel_transform = make_mel_basis(hp, device=torch.device("cpu"))

    # ---- Collect masks & input energies ----
    n_mels = hp.n_mels
    n_buckets = 10
    hist_counts = np.zeros(n_buckets, dtype=np.int64)

    per_bin_mask_sum = np.zeros(n_mels, dtype=np.float64)
    per_bin_count = np.zeros(n_mels, dtype=np.int64)

    energy_bucket_masks: dict[str, list[float]] = defaultdict(list)
    energy_bucket_counts: dict[str, int] = defaultdict(int)

    all_masks: list[np.ndarray] = []
    all_input_energies: list[np.ndarray] = []
    n_frames_collected = 0

    for sample_i, (mixed, target, ref, _interference) in enumerate(
        build_eval_pairs(hp, args.num_samples, hp.seed)
    ):
        # Compute d-vector.
        ref_for_dvec = pad_or_trim(ref, int(0.5 * hp.sample_rate))
        with torch.no_grad():
            dvec = speaker_encoder(ref_for_dvec.unsqueeze(0).to(device)).squeeze(0)

        # Install the mask-capture hook on `out_proj`. The streaming
        # pipeline calls `model.out_proj(x)` once per frame inside
        # `_run_model`, so the hook fires every frame.
        store = _mask_per_frame_hook(model)
        try:
            from open_voicefilter_lite.streaming import StreamingVoiceFilter
            sf_model = StreamingVoiceFilter(model, hp)
            sf_model.set_dvector(dvec)

            mixed_logmel = wav_to_log_mel(mixed, hp, mel_transform)  # (T, F)
            T = mixed_logmel.shape[0]

            for t in range(T):
                sf_model.step(mixed_logmel[t])

            masks_this_sample = torch.stack(store["frames"], dim=0).numpy()  # (T, n_mels)
        finally:
            store["__handle__"].remove()

        # Per-frame input energy (mean over mel bins).
        input_energy = mixed_logmel.mean(dim=-1).numpy()  # (T,)

        # Global histogram + per-bin.
        flat_masks = masks_this_sample.flatten()
        bucket_idx = np.clip(
            (flat_masks * n_buckets).astype(np.int64), 0, n_buckets - 1
        )
        hist_counts += np.bincount(bucket_idx, minlength=n_buckets)

        per_bin_mask_sum += masks_this_sample.sum(axis=0)
        per_bin_count += masks_this_sample.shape[0]

        # Energy bucketization (per frame, per bin — same shape).
        energy_tiled = np.broadcast_to(
            input_energy[:, None], masks_this_sample.shape
        )
        for name, lo, hi in ENERGY_BUCKETS:
            sel = (energy_tiled > lo) & (energy_tiled <= hi)
            if sel.any():
                energy_bucket_masks[name].extend(flat_masks[sel.flatten()].tolist())
                energy_bucket_counts[name] += int(sel.sum())

        all_masks.append(flat_masks)
        all_input_energies.append(input_energy)
        n_frames_collected += T

        if (sample_i + 1) % 10 == 0:
            print(f"[diag] processed {sample_i + 1}/{args.num_samples} samples "
                  f"(frames so far: {n_frames_collected})")

    if not all_masks:
        print("[diag] no samples processed; aborting")
        return

    masks_flat = np.concatenate(all_masks)
    energies_flat = np.concatenate(all_input_energies)
    total = masks_flat.size

    # ---- Report ----
    print()
    print(f"[diag] ===== MASK DISTRIBUTION REPORT =====")
    print(f"[diag] checkpoint : {args.checkpoint}")
    print(f"[diag] samples    : {len(all_masks)}")
    print(f"[diag] frames     : {n_frames_collected}")
    print(f"[diag] total bins : {total:,}  (frames × {n_mels} bins/frame)")
    print()

    print("[diag] Global mask stats (all bins, all frames):")
    print(f"        mean  = {masks_flat.mean():.4f}")
    print(f"        std   = {masks_flat.std():.4f}")
    print(f"        min   = {masks_flat.min():.4f}")
    print(f"        max   = {masks_flat.max():.4f}")
    print(f"        p1    = {np.percentile(masks_flat,  1):.4f}")
    print(f"        p5    = {np.percentile(masks_flat,  5):.4f}")
    print(f"        p25   = {np.percentile(masks_flat, 25):.4f}")
    print(f"        p50   = {np.percentile(masks_flat, 50):.4f}")
    print(f"        p75   = {np.percentile(masks_flat, 75):.4f}")
    print(f"        p95   = {np.percentile(masks_flat, 95):.4f}")
    print(f"        p99   = {np.percentile(masks_flat, 99):.4f}")
    print(f"        >0.5  = {_format_pct((masks_flat > 0.5).mean())}")
    print(f"        >0.9  = {_format_pct((masks_flat > 0.9).mean())}")
    print(f"        >0.95 = {_format_pct((masks_flat > 0.95).mean())}")
    print(f"        <0.1  = {_format_pct((masks_flat < 0.1).mean())}")
    print(f"        <0.5  = {_format_pct((masks_flat < 0.5).mean())}")
    print()
    print("[diag] Histogram:")
    print(_print_histogram(hist_counts, total, n_buckets))
    print()

    # ---- Per-bin (grouped) ----
    n_groups = max(1, args.n_bins_show)
    bin_group_size = n_mels // n_groups
    print(f"[diag] Per-bin mean mask ({n_groups} groups of "
          f"~{bin_group_size} bins, low-freq → high-freq):")
    for g in range(n_groups):
        lo = g * bin_group_size
        hi = (g + 1) * bin_group_size if g < n_groups - 1 else n_mels
        m = per_bin_mask_sum[lo:hi].sum() / max(1, per_bin_count[lo:hi].sum())
        bar_len = int(40 * m)
        bar = "█" * bar_len
        print(f"        bins [{lo:>2d}-{hi - 1:>2d}]: {m:.4f}  {bar}")
    print()

    # ---- Energy-bucket mask stats ----
    print("[diag] Mask split by input frame energy:")
    for name, lo, hi in ENERGY_BUCKETS:
        vals = np.array(energy_bucket_masks.get(name, []), dtype=np.float64)
        n = vals.size
        if n == 0:
            print(f"        {name:>7s} (energy ∈ ({lo:.0f}, {hi:.0f}]):  no frames")
            continue
        print(
            f"        {name:>7s} (energy ∈ ({lo:.0f}, {hi:.0f}]):  "
            f"n_bins={n:>6d}  mean={vals.mean():.4f}  "
            f"p50={np.percentile(vals, 50):.4f}  "
            f"p10={np.percentile(vals, 10):.4f}  "
            f"p90={np.percentile(vals, 90):.4f}"
        )
    print()

    # ---- Pearson correlation ----
    if masks_flat.std() > 1e-3 and energies_flat.std() > 1e-3:
        # Expand energies to per-bin (one energy value per bin of that frame).
        # We repeat energy once per frame → already done in energy_tiled above,
        # but that was per-sample. Stack all samples:
        energies_per_bin = np.concatenate([
            np.broadcast_to(e[:, None], (e.shape[0], n_mels)).flatten()
            for e in all_input_energies
        ])
        assert energies_per_bin.shape == masks_flat.shape, (
            f"shape mismatch: energies {energies_per_bin.shape} vs masks {masks_flat.shape}"
        )
        corr = float(np.corrcoef(energies_per_bin, masks_flat)[0, 1])
    else:
        corr = float("nan")
    print(f"[diag] Pearson r(mask, input_energy) = {corr:+.4f}")
    print(f"        (≈ 0  → mask is constant w.r.t. input — collapsed to 1)")
    print(f"        (> 0  → mask tracks input presence — selective filtering)")
    print()

    # ---- Optional JSON dump ----
    if args.results_json:
        # Summary statistics in JSON form (drop per-bin detail to keep file small).
        summary = {
            "checkpoint": args.checkpoint,
            "num_samples": len(all_masks),
            "frames": n_frames_collected,
            "total_bins": int(total),
            "global": {
                "mean": float(masks_flat.mean()),
                "std": float(masks_flat.std()),
                "min": float(masks_flat.min()),
                "max": float(masks_flat.max()),
                "p1": float(np.percentile(masks_flat, 1)),
                "p5": float(np.percentile(masks_flat, 5)),
                "p25": float(np.percentile(masks_flat, 25)),
                "p50": float(np.percentile(masks_flat, 50)),
                "p75": float(np.percentile(masks_flat, 75)),
                "p95": float(np.percentile(masks_flat, 95)),
                "p99": float(np.percentile(masks_flat, 99)),
                "frac_gt_0.5":  float((masks_flat > 0.5).mean()),
                "frac_gt_0.9":  float((masks_flat > 0.9).mean()),
                "frac_gt_0.95": float((masks_flat > 0.95).mean()),
                "frac_lt_0.1":  float((masks_flat < 0.1).mean()),
                "frac_lt_0.5":  float((masks_flat < 0.5).mean()),
            },
            "histogram": {
                f"[{i / n_buckets:.1f},{(i + 1) / n_buckets:.1f})": int(hist_counts[i])
                for i in range(n_buckets)
            },
            "per_bin_mean": (
                per_bin_mask_sum / np.maximum(per_bin_count, 1)
            ).tolist(),
            "energy_buckets": {
                name: {
                    "n": int(energy_bucket_counts.get(name, 0)),
                    "mean": (
                        float(np.mean(energy_bucket_masks[name]))
                        if energy_bucket_masks.get(name) else None
                    ),
                    "p10": (
                        float(np.percentile(energy_bucket_masks[name], 10))
                        if energy_bucket_masks.get(name) else None
                    ),
                    "p50": (
                        float(np.percentile(energy_bucket_masks[name], 50))
                        if energy_bucket_masks.get(name) else None
                    ),
                    "p90": (
                        float(np.percentile(energy_bucket_masks[name], 90))
                        if energy_bucket_masks.get(name) else None
                    ),
                }
                for name, _, _ in ENERGY_BUCKETS
            },
            "corr_mask_input_energy": corr,
        }
        Path(args.results_json).write_text(json.dumps(summary, indent=2))
        print(f"[diag] wrote {args.results_json}")


if __name__ == "__main__":
    main()
