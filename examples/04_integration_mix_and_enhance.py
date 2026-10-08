#!/usr/bin/env python3
"""Example 04 — Integration test using the SAME per-sample synthesis as training.

Synthesises `--num-samples` mixed clips from the HF arrow cache that
`voicefilter-cli dataset prepare` populated and runs the trained
VoiceFilter-Lite model on them so you can audition the result
directly. The synthesis path is the module-level `synthesize_raw_sample`
from `voicefilter_cli.dataset` — i.e. the exact same code the training
iterator runs — so the audition samples have the same interference
statistics as training:

  * target picked from LibriSpeech (first declared target source,
    `hp.target_sources[0]`); random-cropped to `hp.sample_length_samples`.
  * reference utterance picked 70% same-spk-different-utt, 30% self
    (matches `HfTripletStream.__iter__`); capped at
    `hp.ref_max_sec * sample_rate`.
  * SNR drawn unconditionally from `[snr_min_db, snr_max_db]`.
  * TYPE-inclusion Bernoulli rolls (babble / noise / music) with the
    `prob` from `hp.interference_types`.
  * Energy budget across the K included TYPES via Dirichlet(α=1) on
    the simplex so post-mix SIR matches `snr_db` regardless of K.
  * `silent_target_prob` / `silent_interference_prob` silent flags
    (rolled AFTER synthesis so the Bernoulli draws land on the same
    RNG state regardless of silent outcome).
  * Per-channel activity-burst masks (with linear fades).
  * `snr_mix` branch: target_silent → interference, interference_silent
    → target, both silent → zeros, else `snr_mix(target, interference,
    snr)`.

Then runs the model frame-by-frame via `StreamingVoiceFilter.step`
exactly the way a production ASR frontend would consume it, and
reconstructs the waveform with Griffin-Lim.

Data is read directly from the HF arrow cache that `voicefilter-cli
dataset prepare` populated — there are no wav files baked on disk.
Audio is decoded on-the-fly per sample via `default_hf_wav_loader`
(reads `{"bytes": ...}` payload and decodes via `soundfile`, with
librosa resampling when source_sr differs from `hp.sample_rate`).

Prerequisite
------------
Run once before the first invocation so the HF cache exists:

    voicefilter-cli dataset prepare --config configs/paper.yaml

This populates `<cache_dir>/` for every configured `HfSourceSpec`
(default: `data/librispeech/` and `data/musan/` for paper.yaml).

Output layout (one folder per sample, plus a summary):

    out-dir/
      00_target.wav          # clean target speaker (after burst mask)
      00_enrollment.wav      # d-vector source (same spk, different utt
                             # 70% of the time, else self)
      00_interference.wav    # the "other" sound (after burst mask)
      00_mixed.wav           # snr_mix result (matches training)
      00_enhanced.wav        # model output — listen to this
      ...
      summary.txt            # per-sample SNR / sources / paths
      summary.json           # same data as JSON

Usage
-----
    python examples/04_integration_mix_and_enhance.py \\
        --checkpoint runs/exp1/ckpt_final.pt \\
        --out-dir    data/integration_test \\
        --num-samples 5

Note
----
This is for audition, not for SI-SDR; for numerical evaluation use
`voicefilter-cli evaluate`. For ONNX parity checks use
`examples/03_onnx_inference.py --verify`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

# Self-contained: add the `src/` layouts to sys.path so this works
# without `pip install -e .`.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))
# voicefilter_cli (training CLI) sits in the workspace under packages/cli.
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(HERE, "..", "packages", "cli", "src")
    ),
)

from voicefilter_lite.audio import (  # noqa: E402
    extract_dvec,
    make_mel_basis,
    wav_to_log_mel,
)
from voicefilter_lite.hparams import HParams, resolve_source  # noqa: E402
from voicefilter_lite.model import VoiceFilterLite  # noqa: E402
from voicefilter_lite.speaker_encoder import (  # noqa: E402
    SpeakerEncoder,
)
from voicefilter_lite.streaming import (  # noqa: E402
    StreamingVoiceFilter,
)
from voicefilter_cli.dataset import (  # noqa: E402
    HfEntry,
    _build_target_index,
    _collect_babble_pool_entries,
    _collect_passthrough_pool_entries,
    default_hf_wav_loader,
    synthesize_raw_sample,
)


# ---- TYPE name constants (mirror HParams.interference_types keys) ----
NOISE_KIND_BABBLE = "babble"
NOISE_KIND_NOISE = "noise"
NOISE_KIND_MUSIC = "music"


# --------------------------------------------------------------------------- #
# Device + WAV helpers                                                         #
# --------------------------------------------------------------------------- #


def _select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _model_device(model: VoiceFilterLite) -> torch.device:
    return next(model.parameters()).device


def _save_wav(path: str, wav: torch.Tensor, sample_rate: int) -> None:
    """Peak-normalise then write a 16-bit WAV.

    Peak normalisation keeps audibility consistent across the per-sample
    WAVs — `mixed.wav` may come in quieter than `target.wav` because of
    the SNR scaling, but the listener's volume knob shouldn't have to
    chase the per-file energy. Hard-clips to ±1 only if a clip would clip.
    """
    peak = wav.abs().max().clamp(min=1e-8)
    if peak > 1.0:
        wav = wav / peak
    sf.write(path, wav.numpy().astype(np.float32), sample_rate)


def _logmel_to_wav(log_mel: torch.Tensor, hp: HParams, n_iter: int = 16) -> torch.Tensor:
    """Inverse log-mel → waveform via Griffin-Lim (audition only).

    `hp` is the HParams restored from the loaded checkpoint's `hparams`
    dict (see `main()` — `ckpt["hparams"]` was serialized at training
    time from whichever `--config` the model was trained with, e.g.
    `configs/paper.yaml` → `n_mels=128, n_fft=1200`). All audio params
    used in this function — `n_mels`, `n_fft`, `hop_length`,
    `win_length`, `sample_rate`, `log_mel_floor` — are read from `hp`,
    so this helper tracks the training config with no hard-coded
    mel-bin counts. Same end-to-end inverse as
    `examples/01_offline_separation.py` and
    `examples/02_streaming_inference.py`, modulo the inverse operator.

    Uses the Moore-Penrose pseudo-inverse of the configured mel basis
    rather than `torchaudio.transforms.InverseMelScale`: the pseudo-
    inverse is a one-shot least-squares solve (no `max_iter` knob to
    tune), and it degrades gracefully if the basis is rank-deficient
    (which can happen when a config sets `n_mels` close to or above
    `1 + n_fft // 2`). `examples/01/02` use `InverseMelScale` instead
    — both produce audibly equivalent waveforms for the paper.yaml
    config (`n_mels=128, n_fft=1200`); the choice here is just cheaper
    and more explicit.
    """
    import torchaudio

    mel = torch.exp(log_mel) - hp.log_mel_floor
    mel = mel.transpose(0, 1).unsqueeze(0)              # (1, F, T)

    mel_transform = make_mel_basis(hp, device=torch.device("cpu"))
    fb = mel_transform.mel_scale.fb                    # (n_freq, n_mels)
    fb_pinv = torch.linalg.pinv(fb.transpose(-1, -2))  # (n_freq, n_mels)
    linear = torch.relu(fb_pinv.to(mel.dtype).to(mel.device) @ mel)  # (1, n_freq, T)

    target_len = int(hp.sample_length_sec * hp.sample_rate)
    needed_frames = target_len // hp.hop_length + 1
    if linear.shape[-1] < needed_frames:
        pad = torch.zeros(
            1, linear.shape[1], needed_frames - linear.shape[-1],
            dtype=linear.dtype, device=linear.device,
        )
        linear = torch.cat([linear, pad], dim=-1)

    griffin = torchaudio.transforms.GriffinLim(
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        power=1.0,
        n_iter=n_iter,
        length=target_len,
    )
    return griffin(linear).squeeze(0)


# --------------------------------------------------------------------------- #
# Per-sample construction                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class SampleSummary:
    """Per-sample metadata persisted to `summary.txt` and `summary.json`."""
    index: int
    target_speaker: str
    target_utt_id: str
    enrollment_utt_id: str
    # `tuple[str, ...]` of TYPE names actually mixed (in canonical order).
    interference_kind: tuple[str, ...]
    interference_detail: str
    snr_db: float
    mixed_path: str
    enhanced_path: str
    target_silent: bool = False
    interference_silent: bool = False
    notes: list[str] = field(default_factory=list)


def _pick_enrollment_training_style(
    rng: random.Random,
    spk_entries: list[HfEntry],
    target_entry: HfEntry,
) -> Optional[HfEntry]:
    """Pick the enrollment entry following `HfTripletStream.__iter__` rules.

    Mirrors the training logic:
      * 70% same-spk different-utt (if available)
      * 30% self (use the target entry — caller falls back to target wav)

    Returns None iff the speaker has only one utterance OR the 30%
    self-roll came up "self". Caller treats None as "fall back to the
    target entry's wav".
    """
    candidates = spk_entries
    if len(candidates) < 2:
        # Single-utt speaker: training always uses self.
        return None
    if rng.random() < 0.7:
        for _ in range(20):
            e = rng.choice(candidates)
            if e != target_entry:
                return e
        # Couldn't find a different utt in 20 tries — fall through.
        return None
    return None


def build_one_sample(
    idx: int,
    hp: HParams,
    rng: random.Random,
    target_spk: str,
    target_wav_full: torch.Tensor,
    enroll_wav: Optional[torch.Tensor],
    babble_spk_to_entries: dict[str, list[HfEntry]],
    noise_pool: list[HfEntry],
    music_pool: list[HfEntry],
    babble_cfg,
    noise_cfg,
    music_cfg,
    type_probs: dict[str, float],
    model: VoiceFilterLite,
    encoder,
    mel_fn,
    out_dir: Path,
    gl_iter: int,
) -> SampleSummary:
    """Construct, mix, and enhance one sample using the training's synthesis.

    `target_wav_full` is the full (uncropped) LibriSpeech utterance for
    the picked target. `enroll_wav` is the full reference (None means
    "use target_wav_full"). The interference pools come from the HF
    arrow cache (`babble_spk_to_entries`, `noise_pool`, `music_pool`);
    `synthesize_raw_sample` decodes via `default_hf_wav_loader` and
    resamples MUSAN entries from 44.1 kHz to `hp.sample_rate`.

    Returns a `SampleSummary` with paths to the per-sample WAVs.
    """
    notes: list[str] = []

    # 2) Compute d-vector from the enrollment clip. The enrollment
    #    wav is capped to `ref_max_sec` at this layer so the d-vec
    #    sees the same input as training.
    if enroll_wav is None:
        enroll_full = target_wav_full
        notes.append("enrollment: self (single-utt speaker or 30%-roll)")
    else:
        enroll_full = enroll_wav

    cap = int(hp.ref_max_sec * hp.sample_rate)
    enroll_for_dvec = enroll_full[..., :cap] if enroll_full.shape[-1] > cap else enroll_full
    model_dev = _model_device(model)
    with torch.no_grad():
        enroll_dev = enroll_for_dvec.to(model_dev)
        # Keep dvec on the model's device — `StreamingVoiceFilter._run_model`
        # uses `dvec.device` as the model_device for the LSTM forward.
        dvec = extract_dvec(
            enroll_dev, hp, encoder, device=enroll_dev.device
        ).squeeze(0)
    notes.append(f"d-vector dim={dvec.shape[-1]} (extract_dvec → in_proj directly, on {dvec.device})")

    # 3) Synthesize the WAV-level sample using the SAME module-level
    #    function the training iterator calls. `synthesize_raw_sample`
    #    does the v3 categorical interference synthesis (full-length
    #    interference from sample 0, no random segment placement) +
    #    silent flags + target activity burst + snr_mix — all with the
    #    training RNG semantics. `multi_type_prob=1.0` here disables the
    #    curriculum for the offline audition script (we want the FULL
    #    target mixed distribution immediately so the audition wav
    #    matches the post-warmup training distribution).
    raw = synthesize_raw_sample(
        hp,
        rng,
        target_wav_full=target_wav_full,
        ref_wav_full=enroll_full,
        babble_spk_to_entries=babble_spk_to_entries,
        noise_pool=noise_pool,
        music_pool=music_pool,
        type_probs=type_probs,
        babble_cfg=babble_cfg,
        noise_cfg=noise_cfg,
        music_cfg=music_cfg,
        multi_type_prob=1.0,
    )
    target_wav = raw.target_wav
    interference_wav = raw.interference_wav
    mixed_wav = raw.mixed_wav
    noise_kind = raw.included_kinds
    snr_db = raw.snr_db
    detail = "+".join(noise_kind) if noise_kind else "silence"
    notes.append(
        f"SNR={snr_db:+.1f} dB; interference={detail}; "
        f"target_silent={raw.target_silent} interference_silent={raw.interference_silent}"
    )

    # 4) Streaming inference frame-by-frame (matches training's
    #    forward path: log-mel input → per-frame enhanced log-mel).
    sf_model = StreamingVoiceFilter(model, hp)
    sf_model.set_dvector(dvec)

    mixed_logmel = wav_to_log_mel(mixed_wav, hp, mel_fn)            # (T, 80)
    enhanced_frames: list[torch.Tensor] = []
    for t in range(mixed_logmel.shape[0]):
        enhanced_frames.append(sf_model.step(mixed_logmel[t]).cpu())
    enhanced_logmel = torch.stack(enhanced_frames, dim=0)            # (T, 80)

    # 5) Reconstruct waveform via Griffin-Lim (audition only).
    enhanced_wav = _logmel_to_wav(enhanced_logmel, hp, n_iter=gl_iter)

    # 6) Persist WAVs.
    tag = f"{idx:02d}"
    mixed_path = out_dir / f"{tag}_mixed.wav"
    enhanced_path = out_dir / f"{tag}_enhanced.wav"

    _save_wav(str(mixed_path), mixed_wav, hp.sample_rate)
    _save_wav(str(enhanced_path), enhanced_wav, hp.sample_rate)

    return SampleSummary(
        index=idx,
        target_speaker=target_spk,
        target_utt_id="",  # filled by main
        enrollment_utt_id="",  # filled by main
        interference_kind=noise_kind,
        interference_detail=detail,
        snr_db=float(snr_db),
        mixed_path=str(mixed_path.name),
        enhanced_path=str(enhanced_path.name),
        target_silent=raw.target_silent,
        interference_silent=raw.interference_silent,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True, help="Path to ckpt_*.pt")
    p.add_argument("--out-dir", required=True,
                   help="Directory for per-sample WAVs + summary")
    p.add_argument("--num-samples", type=int, default=5,
                   help="Number of mixed clips to generate")
    p.add_argument("--sample-length-sec", type=float, default=None,
                   help="Override hp.sample_length_sec (default: use checkpoint's)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--gl-iter", type=int, default=16,
                   help="Griffin-Lim iterations for waveform reconstruction")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _select_device(args.device)
    print(f"[example04] device={device}")

    # ---- Load checkpoint + restore hparams ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hp = HParams.from_dict(ckpt["hparams"])
    hp.device = str(device)
    if args.sample_length_sec is not None:
        hp.sample_length_sec = float(args.sample_length_sec)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = VoiceFilterLite(hp).to(device).eval()
    # Training wraps the model with `torch.compile` (see train.py), which
    # saves state_dict keys prefixed by `_orig_mod.` — strip it so the
    # un-compiled model here can load the weights unchanged.
    model_state = {
        k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state"].items()
    }
    model.load_state_dict(model_state)
    print(f"[example04] loaded {args.checkpoint} (step={ckpt.get('step')})")

    encoder = SpeakerEncoder(hp).to(device)
    if Path(hp.dvector_pt_path).is_file():
        encoder.load_pretrained(hp.dvector_pt_path)
        print(f"[example04] SpeakerEncoder loaded from {hp.dvector_pt_path}")
    else:
        print(
            f"[example04] WARNING: dvector weights not found at {hp.dvector_pt_path}; "
            "encoder will use RANDOM init."
        )
        encoder.freeze()
    print(f"[example04] d_vector_dim={hp.d_vector_dim}")
    mel_fn = make_mel_basis(hp, device=torch.device("cpu"))

    # ---- Open LibriSpeech target + babble / noise / music pools ----
    # Uses the training's module-level helpers — opens one HF
    # `datasets.Dataset` per (source, subset, split) and builds the same
    # `{spk_id → [HfEntry, …]}` maps as `HfTripletStream`. The cache_dir
    # comes from the resolved spec.
    if not hp.target_sources:
        raise SystemExit(
            "[example04] hp.target_sources is empty; nothing to mix into target."
        )
    dataset_cache: dict[str, object] = {}
    target_index, _spk_to_spec_id = _build_target_index(hp, dataset_cache)
    if not target_index:
        raise SystemExit(
            "[example04] target_index is empty; did `voicefilter-cli dataset "
            "prepare` finish?"
        )
    n_speakers = len(target_index)
    n_target_utts = sum(len(v) for v in target_index.values())
    print(
        f"[example04] target pool: {n_target_utts:,} utts, "
        f"{n_speakers:,} speakers"
    )

    # ---- Build the pool structures via the training's module-level helpers ----
    babble_entries = _collect_babble_pool_entries(hp, dataset_cache)
    babble_spk_to_entries: dict[str, list[HfEntry]] = {}
    for spk_id, entry in babble_entries:
        babble_spk_to_entries.setdefault(spk_id, []).append(entry)
    noise_pool = _collect_passthrough_pool_entries(hp, dataset_cache, NOISE_KIND_NOISE)
    music_pool = _collect_passthrough_pool_entries(hp, dataset_cache, NOISE_KIND_MUSIC)
    type_probs: dict[str, float] = {
        name: float(cfg.prob)
        for name, cfg in hp.interference_types.items()
        if cfg.enabled and cfg.prob > 0
    }
    babble_cfg = hp.interference_types.get(NOISE_KIND_BABBLE)
    noise_cfg = hp.interference_types.get(NOISE_KIND_NOISE)
    music_cfg = hp.interference_types.get(NOISE_KIND_MUSIC)
    print(
        f"[example04] pools ready: babble spks={len(babble_spk_to_entries)} "
        f"noise clips={len(noise_pool)} music clips={len(music_pool)}"
    )

    # ---- Build N samples ----
    summaries: list[SampleSummary] = []
    t0 = time.time()
    for i in range(args.num_samples):
        ts = time.time()
        try:
            # 1) Pick target speaker + utterance from HF cache.
            target_spk = rng.choice(list(target_index.keys()))
            spk_entries = target_index[target_spk]
            tgt_entry = rng.choice(spk_entries)
            target_wav_full = torch.from_numpy(
                default_hf_wav_loader(tgt_entry, hp.sample_rate)
            )

            # 2) Pick enrollment (training's 70/30 rule).
            enroll_entry = _pick_enrollment_training_style(rng, spk_entries, tgt_entry)
            if enroll_entry is None:
                enroll_wav = target_wav_full.clone()
                enroll_utt_id = tgt_entry.utt_id
            else:
                enroll_wav = torch.from_numpy(
                    default_hf_wav_loader(enroll_entry, hp.sample_rate)
                )
                enroll_utt_id = enroll_entry.utt_id

            # 3) Synthesize + enhance + persist via the shared
            #    module-level `synthesize_raw_sample`.
            summary = build_one_sample(
                idx=i, hp=hp, rng=rng,
                target_spk=target_spk,
                target_wav_full=target_wav_full,
                enroll_wav=enroll_wav,
                babble_spk_to_entries=babble_spk_to_entries,
                noise_pool=noise_pool,
                music_pool=music_pool,
                babble_cfg=babble_cfg,
                noise_cfg=noise_cfg,
                music_cfg=music_cfg,
                type_probs=type_probs,
                model=model, encoder=encoder, mel_fn=mel_fn,
                out_dir=out_dir,
                gl_iter=args.gl_iter,
            )
            summary.target_utt_id = tgt_entry.utt_id
            summary.enrollment_utt_id = enroll_utt_id
        except Exception as e:
            print(f"[example04] sample {i:02d} FAILED: {type(e).__name__}: {e}")
            continue
        summaries.append(summary)
        kind_str = (
            "+".join(summary.interference_kind) if summary.interference_kind else "silence"
        )
        print(
            f"[example04] sample {i:02d} done in {time.time() - ts:.1f}s "
            f"(target_spk={summary.target_speaker}, interference={kind_str}, "
            f"SNR={summary.snr_db:+.1f} dB)"
        )

    # ---- Write summaries ----
    json_path = out_dir / "summary.json"
    txt_path = out_dir / "summary.txt"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in summaries], f, indent=2, ensure_ascii=False)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("# Integration test summary\n")
        f.write(f"# generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# checkpoint: {args.checkpoint}\n")
        f.write(f"# num_samples: {len(summaries)} (requested {args.num_samples})\n")
        f.write(f"# sample_length_sec: {hp.sample_length_sec}\n")
        f.write(f"# snr_range_db: [{hp.snr_min_db}, {hp.snr_max_db}]\n")
        f.write(f"# total_runtime_sec: {time.time() - t0:.1f}\n\n")
        for s in summaries:
            kind_str = (
                "+".join(s.interference_kind) if s.interference_kind else "silence"
            )
            f.write(f"--- sample {s.index:02d} ---\n")
            f.write(f"  target_speaker    : {s.target_speaker}\n")
            f.write(f"  target_utt_id     : {s.target_utt_id}\n")
            f.write(f"  enrollment_utt_id : {s.enrollment_utt_id}\n")
            f.write(f"  interference      : {kind_str}  ({s.interference_detail})\n")
            f.write(f"  snr_db            : {s.snr_db:+.1f}\n")
            f.write(f"  target_silent     : {s.target_silent}\n")
            f.write(f"  interference_silent: {s.interference_silent}\n")
            f.write(f"  mixed wav         : {s.mixed_path}\n")
            f.write(f"  enhanced wav      : {s.enhanced_path}\n")
            for note in s.notes:
                f.write(f"  note              : {note}\n")
            f.write("\n")

    print(f"\n[example04] wrote {len(summaries)} samples to {out_dir}")
    print(f"[example04] summary: {txt_path}")
    print(f"[example04] summary: {json_path}")
    print(f"[example04] total runtime: {time.time() - t0:.1f}s")
    print("[example04] Listen to *_mixed.wav vs *_enhanced.wav to compare "
          "input vs model output.")


if __name__ == "__main__":
    main()