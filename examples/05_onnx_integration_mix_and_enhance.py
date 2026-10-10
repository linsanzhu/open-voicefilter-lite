#!/usr/bin/env python3
"""Example 05 — Integration mix-and-enhance with a custom mask ONNX.

Same synthesis pipeline as `examples/04_integration_mix_and_enhance.py`
(LibriSpeech target + MUSAN / LibriSpeech interference, identical RNG
semantics), but the **mask predictor** is the user-supplied ONNX (e.g.
the INT8 quantised export for production deployment) rather than the
bundled FP32 graph. The d-vector is still extracted via the bundled
INT8 embedder ONNX; pass `--embedder-onnx` to override that too.

The fixed seed (`--seed 0`) makes the mixed samples bit-identical to
example 04's run, so `xx_enhanced.wav` from the two scripts differ only
by the underlying mask model — useful for A/B listening (FP32 vs INT8).

Output layout (identical to example 04):

    out-dir/
      00_target.wav          # clean target speaker
      00_enrollment.wav      # d-vector source (same spk, different utt)
      00_interference.wav    # the "other" sound
      00_mixed.wav           # snr_mix result
      00_enhanced.wav        # custom-ONNX model output — listen to this
      ...
      summary.txt            # per-sample SNR / sources / paths
      summary.json           # same data as JSON

Usage
-----
    python examples/05_onnx_integration_mix_and_enhance.py \\
        --mask-onnx   voicefilter_lite.int8.onnx \\
        --out-dir     data/integration_test_int8 \\
        --num-samples 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

# Self-contained: add the lib's `src/` and the CLI's `src/` so this
# works without `pip install -e .`.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..",
    "packages", "open-voicefilter-lite", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..",
    "packages", "cli", "src")))

from voicefilter_lite import StreamingVoiceFilter  # noqa: E402
from voicefilter_lite.config import load_and_apply  # noqa: E402
from voicefilter_lite.hparams import HParams  # noqa: E402
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


def _save_wav(path: str, wav: torch.Tensor, sample_rate: int) -> None:
    """Peak-normalise then write a 16-bit WAV."""
    peak = wav.abs().max().clamp(min=1e-8)
    if peak > 1.0:
        wav = wav / peak
    sf.write(path, wav.numpy().astype(np.float32), sample_rate)


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
    interference_kind: tuple[str, ...]
    interference_detail: str
    snr_db: float
    mixed_path: str
    enhanced_path: str
    target_silent: bool = False
    interference_silent: bool = False
    notes: list[str] = field(default_factory=list)


def _stream_pcm_through(model: StreamingVoiceFilter, pcm: torch.Tensor) -> torch.Tensor:
    """Slice `pcm` into `hop`-sized chunks, run `step`, concat the outputs."""
    hop = model.hop_length
    n_full = pcm.shape[-1] // hop
    if n_full == 0:
        return torch.zeros(0, dtype=torch.float32)
    out_chunks = [
        model.step(pcm[i * hop:(i + 1) * hop].numpy())
        for i in range(n_full)
    ]
    return torch.cat(out_chunks)[: n_full * hop]


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
    model: StreamingVoiceFilter,
    out_dir: Path,
) -> SampleSummary:
    """Construct, mix, and enhance one sample — see example 04 for details."""
    notes: list[str] = []

    if enroll_wav is None:
        enroll_full = target_wav_full
        notes.append("enrollment: self (single-utt speaker or 30%-roll)")
    else:
        enroll_full = enroll_wav

    raw = synthesize_raw_sample(
        hp, rng,
        target_wav_full=target_wav_full, ref_wav_full=enroll_full,
        babble_spk_to_entries=babble_spk_to_entries,
        noise_pool=noise_pool, music_pool=music_pool,
        type_probs=type_probs,
        babble_cfg=babble_cfg, noise_cfg=noise_cfg, music_cfg=music_cfg,
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

    cap = int(hp.ref_max_sec * hp.sample_rate)
    enroll_for_dvec = enroll_full[..., :cap] if enroll_full.shape[-1] > cap else enroll_full
    model.set_enrollment(enroll_for_dvec)
    notes.append(f"d-vector from {enroll_for_dvec.shape[-1] / hp.sample_rate:.2f}s of enrollment audio")

    enhanced_wav = _stream_pcm_through(model, mixed_wav)

    tag = f"{idx:02d}"
    target_path = out_dir / f"{tag}_target.wav"
    enroll_path = out_dir / f"{tag}_enrollment.wav"
    interference_path = out_dir / f"{tag}_interference.wav"
    mixed_path = out_dir / f"{tag}_mixed.wav"
    enhanced_path = out_dir / f"{tag}_enhanced.wav"

    _save_wav(str(target_path), target_wav, hp.sample_rate)
    _save_wav(str(enroll_path), enroll_full, hp.sample_rate)
    _save_wav(str(interference_path), interference_wav, hp.sample_rate)
    _save_wav(str(mixed_path), mixed_wav, hp.sample_rate)
    _save_wav(str(enhanced_path), enhanced_wav, hp.sample_rate)

    return SampleSummary(
        index=idx, target_speaker=target_spk,
        target_utt_id="", enrollment_utt_id="",
        interference_kind=noise_kind, interference_detail=detail,
        snr_db=float(snr_db),
        mixed_path=str(mixed_path.name), enhanced_path=str(enhanced_path.name),
        target_silent=raw.target_silent, interference_silent=raw.interference_silent,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mask-onnx",
                   help="Path to a custom mask ONNX (e.g. voicefilter_lite.int8.onnx). "
                        "Default: use the bundled FP32 graph.")
    p.add_argument("--embedder-onnx",
                   help="Path to a custom embedder ONNX. "
                        "Default: use the bundled INT8 embedder.int8.onnx.")
    p.add_argument("--out-dir", required=True,
                   help="Directory for per-sample WAVs + summary")
    p.add_argument("--num-samples", type=int, default=5,
                   help="Number of mixed clips to generate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default="configs/paper.yaml",
                   help="Path to a HParams YAML matching the training config "
                        "(needed so the synthesis matches training statistics)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hp = HParams()
    load_and_apply(hp, args.config)
    print(f"[example05] loaded hparams from {args.config}")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    wav_loader = default_hf_wav_loader
    target_index = _build_target_index(hp)
    if not target_index:
        raise SystemExit(
            f"No LibriSpeech target entries found at {hp.target_sources!r}. "
            "Did you run `voicefilter-cli dataset prepare`?"
        )
    babble_spk_to_entries = _collect_babble_pool_entries(hp, wav_loader)
    noise_pool = _collect_passthrough_pool_entries(hp, wav_loader, NOISE_KIND_NOISE)
    music_pool = _collect_passthrough_pool_entries(hp, wav_loader, NOISE_KIND_MUSIC)

    babble_cfg = hp.interference_types.get(NOISE_KIND_BABBLE)
    noise_cfg = hp.interference_types.get(NOISE_KIND_NOISE)
    music_cfg = hp.interference_types.get(NOISE_KIND_MUSIC)
    if any(c is None for c in (babble_cfg, noise_cfg, music_cfg)):
        raise SystemExit(
            f"interference_types missing one of babble/noise/music: {hp.interference_types}"
        )
    type_probs = {NOISE_KIND_BABBLE: babble_cfg.prob,
                  NOISE_KIND_NOISE: noise_cfg.prob,
                  NOISE_KIND_MUSIC: music_cfg.prob}

    # ---- Model: bundled or custom mask ONNX ----
    kwargs: dict = {}
    if args.mask_onnx:
        kwargs["mask_model"] = args.mask_onnx
    if args.embedder_onnx:
        kwargs["dvector_model"] = args.embedder_onnx
    model = StreamingVoiceFilter(**kwargs)
    print(f"[example05] model ready (mask={args.mask_onnx or '(bundled FP32)'}, "
          f"embedder={args.embedder_onnx or '(bundled INT8)'}, "
          f"n_fft={model.hp.n_fft}, hop={model.hop_length})")

    # ---- Generate + enhance `num_samples` clips ----
    summaries: list[SampleSummary] = []
    speakers = list(target_index.keys())
    if not speakers:
        raise SystemExit("No speakers in target index")

    for i in range(args.num_samples):
        target_spk = rng.choice(speakers)
        spk_entries = target_index[target_spk]
        target_entry = rng.choice(spk_entries)
        with wav_loader(target_entry) as loader:
            target_wav_full = loader.get_full_wav()
        enroll_wav_full: Optional[torch.Tensor] = None
        if len(spk_entries) > 1 and rng.random() < 0.7:
            for _ in range(20):
                e = rng.choice(spk_entries)
                if e != target_entry:
                    with wav_loader(e) as loader:
                        enroll_wav_full = loader.get_full_wav()
                    break

        summary = build_one_sample(
            idx=i, hp=hp, rng=rng,
            target_spk=target_spk, target_wav_full=target_wav_full,
            enroll_wav=enroll_wav_full,
            babble_spk_to_entries=babble_spk_to_entries,
            noise_pool=noise_pool, music_pool=music_pool,
            babble_cfg=babble_cfg, noise_cfg=noise_cfg, music_cfg=music_cfg,
            type_probs=type_probs, model=model, out_dir=out_dir,
        )
        summary.target_utt_id = target_entry.utt_id
        if enroll_wav_full is not None:
            summary.enrollment_utt_id = f"(same spk as {target_entry.utt_id})"
        summaries.append(summary)
        print(f"[example05] sample {i:02d} — target_spk={target_spk} "
              f"SNR={summary.snr_db:+.1f} dB interference={summary.interference_detail}")

    # ---- Persist summaries ----
    (out_dir / "summary.json").write_text(
        json.dumps([asdict(s) for s in summaries], indent=2)
    )
    with (out_dir / "summary.txt").open("w") as f:
        f.write(f"# {len(summaries)} sample(s) — see {out_dir}\n\n")
        for s in summaries:
            f.write(
                f"#{s.index:02d}  spk={s.target_speaker}  "
                f"target_utt={s.target_utt_id}  enroll_utt={s.enrollment_utt_id}  "
                f"SNR={s.snr_db:+.1f}dB  interference={s.interference_detail}  "
                f"mixed={s.mixed_path}  enhanced={s.enhanced_path}\n"
            )
            for n in s.notes:
                f.write(f"        note: {n}\n")
    print(f"[example05] wrote {len(summaries)} samples + summary to {out_dir}")


if __name__ == "__main__":
    main()
