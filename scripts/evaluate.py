#!/usr/bin/env python3
"""Evaluate a trained VoiceFilter-Lite checkpoint on LibriSpeech test-clean/other.

Metrics
-------
    Input  SI-SNR / SDR : baseline of the mixed signal (target vs mixture)
    Output SI-SNR / SDR : after VoiceFilter-Lite
    SI-SNRi / SDRi      : improvement (output - input)

Usage
-----
    python scripts/evaluate.py --checkpoint runs/exp1/ckpt_step050000.pt
    python scripts/evaluate.py --checkpoint runs/exp1/ckpt_final.pt --num-samples 50
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
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
from open_voicefilter_lite.hparams import HParams
from open_voicefilter_lite.model import StreamingVoiceFilter, VoiceFilterLite
from open_voicefilter_lite.speaker_encoder import FrozenSpeakerEncoder


# ---- SI-SNR / SDR ----


def si_snr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Scale-invariant SNR (dB). Both inputs are (T,) mono."""
    target = target - target.mean()
    estimate = estimate - estimate.mean()

    alpha = (estimate * target).sum() / (target.pow(2).sum() + eps)
    proj = alpha * target
    noise = estimate - proj
    return 10.0 * torch.log10(proj.pow(2).sum() / (noise.pow(2).sum() + eps) + eps)


def sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Plain SDR without scale invariance."""
    return 10.0 * torch.log10(
        target.pow(2).sum() / ((estimate - target).pow(2).sum() + eps) + eps
    )


# ---- Audio decoding ----


def decode_audio(audio_field: dict, target_sr: int) -> torch.Tensor:
    if isinstance(audio_field, dict):
        if "bytes" in audio_field and audio_field["bytes"] is not None:
            data, sr = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32")
        elif "array" in audio_field and audio_field["array"] is not None:
            data = np.asarray(audio_field["array"], dtype=np.float32)
            sr = int(audio_field.get("sampling_rate", target_sr))
        else:
            raise ValueError("no audio bytes/array")
    else:
        data = np.asarray(audio_field, dtype=np.float32)
        sr = target_sr

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


# ---- Main eval loop ----


def build_eval_pairs(
    hp: HParams,
    num_samples: int,
    seed: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield (mixed_wav, target_wav, ref_wav, interference_wav) tuples.

    target + ref come from the same speaker (ref for d-vector),
    interference is a different speaker from the other split.
    """
    from datasets import load_dataset
    from itertools import chain

    target_iter = iter(
        chain.from_iterable(
            load_dataset(
                "openslr/librispeech_asr",
                hp.target_config,
                split=s,
                streaming=True,
            )
            for s in hp.target_splits
        )
    )
    interference_iter = iter(
        load_dataset(
            "openslr/librispeech_asr",
            hp.interference_config,
            split="test",
            streaming=True,
        )
    )

    rng = random.Random(seed)
    spk_buf: dict[str, dict] = {}

    n = 0
    while n < num_samples:
        try:
            tgt = next(target_iter)
            intf = next(interference_iter)
        except StopIteration:
            return

        spk = str(tgt.get("speaker_id", rng.randint(0, 10_000)))
        buf = spk_buf.setdefault(spk, [])
        if len(buf) < 4:
            buf.append(tgt)
            continue

        ref = rng.choice(buf)
        try:
            target_wav = decode_audio(tgt["audio"], hp.sample_rate)
            interference_wav = decode_audio(intf["audio"], hp.sample_rate)
            ref_wav = decode_audio(ref["audio"], hp.sample_rate)
        except Exception:
            continue

        sample_len = hp.sample_length_samples
        target_wav = random_crop(target_wav, sample_len)
        interference_wav = random_crop(interference_wav, sample_len)
        snr = rng.uniform(hp.snr_min_db, hp.snr_max_db)
        mixed = snr_mix(target_wav, interference_wav, snr)

        yield mixed, target_wav, ref_wav, interference_wav
        n += 1


def evaluate(hp: HParams, args: argparse.Namespace) -> None:
    device = torch.device(hp.device if torch.backends.mps.is_available() or torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}")

    # Load checkpoint.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hp = HParams.from_dict(ckpt["hparams"])
    hp.device = str(device)

    # Resolve HF endpoint after hp is rebuilt from the checkpoint — the ckpt
    # may carry a different `hf_endpoint` than the default HParams, and that's
    # the one we want to honour. CLI flag still wins.
    apply_hf_endpoint(
        hp,
        cli_override=getattr(args, "hf_endpoint", None),
        cli_transfer=getattr(args, "hf_transfer", None),
    )

    model = VoiceFilterLite(hp).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    print(f"[eval] loaded {args.checkpoint} (step={ckpt.get('step')})")

    speaker_encoder = FrozenSpeakerEncoder(hp).to(device)

    mel_transform = make_mel_basis(hp, device=torch.device("cpu"))

    si_snr_in, si_snr_out = [], []
    sdr_in, sdr_out = [], []

    for i, (mixed, target, ref, interference) in enumerate(
        build_eval_pairs(hp, args.num_samples, hp.seed)
    ):
        # Compute d-vector from ref.
        ref_for_dvec = pad_or_trim(ref, int(0.5 * hp.sample_rate))
        with torch.no_grad():
            dvec = speaker_encoder(ref_for_dvec.unsqueeze(0).to(device)).squeeze(0)

        # Streaming inference.
        sf_model = StreamingVoiceFilter(model, hp)
        sf_model.set_dvector(dvec)

        mixed_logmel = wav_to_log_mel(mixed, hp, mel_transform)  # (T, F)
        enhanced_frames = []
        for t in range(mixed_logmel.shape[0]):
            ef = sf_model.step(mixed_logmel[t])
            enhanced_frames.append(ef.cpu())
        enhanced_logmel = torch.stack(enhanced_frames, dim=0)  # (T, F)

        # SI-SNR / SDR in the time domain: reconstruct via Griffin-Lim is
        # expensive and not the paper's primary metric. For a quick eval
        # proxy we invert log-mel → magnitude → Griffin-Lim (10 iters).
        try:
            enhanced_wav = _logmel_to_wav(
                enhanced_logmel, hp, mel_transform,
                n_iter=args.gl_iter,
            )
            input_wav = mixed
            target_wav_eval = target

            si_in = si_snr(input_wav, target_wav_eval)
            si_out = si_snr(enhanced_wav, target_wav_eval)
            sd_in = sdr(input_wav, target_wav_eval)
            sd_out = sdr(enhanced_wav, target_wav_eval)

            si_snr_in.append(si_in)
            si_snr_out.append(si_out)
            sdr_in.append(sd_in)
            sdr_out.append(sd_out)

            if (i + 1) % 10 == 0:
                print(
                    f"[eval] {i+1}/{args.num_samples} "
                    f"SI-SNRi={np.mean(si_snr_out) - np.mean(si_snr_in):.2f} dB "
                    f"SDRi={np.mean(sdr_out) - np.mean(sdr_in):.2f} dB"
                )
        except Exception as e:
            print(f"[eval] sample {i} failed: {e}")
            continue

    if not si_snr_in:
        print("[eval] no successful samples")
        return

    summary = {
        "num_samples": len(si_snr_in),
        "input_si_snr_mean": float(np.mean(si_snr_in)),
        "output_si_snr_mean": float(np.mean(si_snr_out)),
        "si_snri_mean": float(np.mean(si_snr_out) - np.mean(si_snr_in)),
        "input_sdr_mean": float(np.mean(sdr_in)),
        "output_sdr_mean": float(np.mean(sdr_out)),
        "sdri_mean": float(np.mean(sdr_out) - np.mean(sdr_in)),
    }
    print(json.dumps(summary, indent=2))
    if args.results_json:
        Path(args.results_json).write_text(json.dumps(summary, indent=2))


def _logmel_to_wav(
    log_mel: torch.Tensor,
    hp: HParams,
    mel_transform,
    n_iter: int = 10,
) -> torch.Tensor:
    """Approximate log-mel → waveform via inverse-mel + Griffin-Lim.

    This is *not* the production inference path (the real path skips waveform
    entirely and feeds enhanced log-mel straight to ASR); it exists only so
    we can compute SI-SDR proxies during evaluation.
    """
    import torchaudio

    mel = torch.exp(log_mel) - hp.log_mel_floor  # (T, F)
    mel = mel.transpose(0, 1).unsqueeze(0)      # (1, F, T)
    # Inverse mel.
    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=hp.n_fft // 2 + 1,
        n_mels=hp.n_mels,
        sample_rate=hp.sample_rate,
        max_iter=128,
        tolerance_loss=1e-5,
        tolerance_change=1e-7,
    )
    linear = inv_mel(mel)                       # (1, n_freq, T)

    # Griffin-Lim to invert phase.
    griffin = torchaudio.transforms.GriffinLim(
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        power=1.0,
        n_iter=n_iter,
    )
    wav = griffin(linear).squeeze(0)
    # Align length to `mixed` (the caller expects same length).
    return wav


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate VoiceFilter-Lite")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--results-json", type=str, default=None)
    p.add_argument("--gl-iter", type=int, default=10)
    # Highest-priority HF endpoint override — see `cli.apply_hf_endpoint`.
    p.add_argument("--hf-endpoint", type=str, default=None,
                   help="Override HF Hub base URL (e.g. https://hf-mirror.com).")
    # HF hf_transfer protocol toggle. Default is whatever the env says;
    # use --no-hf-transfer when targeting a mirror that doesn't support it.
    xf = p.add_mutually_exclusive_group()
    xf.add_argument("--hf-transfer", dest="hf_transfer", action="store_true",
                    default=None,
                    help="Force-enable HF hf_transfer protocol.")
    xf.add_argument("--no-hf-transfer", dest="hf_transfer", action="store_false",
                    help="Force-disable HF hf_transfer protocol.")
    args = p.parse_args()

    hp = HParams()
    hp.device = args.device
    evaluate(hp, args)


if __name__ == "__main__":
    main()
