#!/usr/bin/env python3
"""Run a trained VoiceFilter-Lite ckpt on a directory of mix_dump samples.

For each `sample_XX_*/` directory in the mix dump:
  - 03_ref.wav      → d-vector via frozen ECAPA
  - 02_mixed.wav    → STFT → magnitude + phase
  - magnitude       → mel → log-mel → streaming model → mask per frame
  - apply mask in mel domain → pseudo-inverse back to linear magnitude
  - combine with ORIGINAL phase → inverse STFT → enhanced wav
  - save as 04_enhanced.wav

Why original phase (and not Griffin-Lim)? The model was trained on the
mixed wav's phase implicitly (its log-mel input is computed from the
mixed wav's STFT). Replacing the phase with Griffin-Lim's estimate makes
the speech sound buzzy/robotic and inflates time-domain SI-SNR. Keeping
the original phase gives the cleanest listening result.

Usage
-----
    python scripts/enhance_mix_dump.py --checkpoint runs/v2_fix/ckpt_final.pt
    python scripts/enhance_mix_dump.py --checkpoint runs/mps/ckpt_step047000.pt \\
        --mix-dump-dir runs/mps/mix_dump
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.linalg as la

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE.parent / "src"))

from open_voicefilter_lite.audio import (
    make_mel_basis,
    pad_or_trim,
)
from open_voicefilter_lite.hparams import HParams
from open_voicefilter_lite.model import VoiceFilterLite
from open_voicefilter_lite.speaker_encoder import FrozenSpeakerEncoder


def compute_si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Scale-invariant SDR in dB. Both inputs (T,) mono."""
    target = target - target.mean()
    estimate = estimate - estimate.mean()
    alpha = (estimate * target).sum() / (target.pow(2).sum() + eps)
    proj = alpha * target
    noise = estimate - proj
    return float(
        10 * torch.log10(proj.pow(2).sum() / (noise.pow(2).sum() + eps) + eps)
    )


def enhance_sample(
    mixed_wav: torch.Tensor,
    target_wav: torch.Tensor,
    ref_wav: torch.Tensor,
    model: VoiceFilterLite,
    speaker_encoder: FrozenSpeakerEncoder,
    hp: HParams,
    fb: torch.Tensor,
    pinv_fb: torch.Tensor,
    window: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    """Enhance `mixed_wav` using the model. Returns (enhanced_wav, si_sdr_in, si_sdr_out)."""
    T_samples = mixed_wav.shape[0]

    # ---- D-vector from reference ----
    ref_for_dvec = pad_or_trim(ref_wav, int(0.5 * hp.sample_rate))
    with torch.no_grad():
        dvec = speaker_encoder(ref_for_dvec.unsqueeze(0).to(device)).squeeze(0)

    # ---- STFT of mixed ----
    mixed_stft = torch.stft(
        mixed_wav,
        n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length,
        window=window, center=True, return_complex=True,
    )  # (n_freq, T_frames)
    n_freq, T_frames = mixed_stft.shape
    magnitude = mixed_stft.abs()
    phase = mixed_stft.angle()

    # ---- magnitude → mel → log-mel (per frame) ----
    # fb is (n_freq, n_mels); mel = fb.T @ mag → (n_mels, T_frames)
    mel = fb.T @ magnitude
    log_mel = torch.log(mel.T + hp.log_mel_floor).to(device)  # (T_frames, n_mels)

    # ---- Streaming inference: collect mask per frame ----
    # Re-implement the streaming forward so we can grab the raw mask (the
    # StreamingVoiceFilter wrapper swallows it inside the adaptive blend).
    # Math is identical to streaming._run_model — same layers, same order.
    dvec_proj = model.dvec_proj(dvec).view(1, 1, -1)              # (1, 1, d_vector_dim)
    h0 = torch.zeros(hp.num_lstm_layers, 1, hp.hidden_dim, device=device)
    c0 = torch.zeros(hp.num_lstm_layers, 1, hp.hidden_dim, device=device)
    hidden = (h0, c0)

    context: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    with torch.no_grad():
        for t in range(T_frames):
            context.append(log_mel[t])
            if len(context) > hp.frame_stack:
                context = context[-hp.frame_stack:]
            stack = hp.frame_stack
            stacked = torch.zeros(stack, hp.n_mels, device=device)
            offset = stack - len(context)
            for i, f in enumerate(context):
                stacked[offset + i] = f
            stacked_flat = stacked.reshape(1, 1, -1)
            x = model.in_proj(torch.cat([stacked_flat, dvec_proj], dim=-1))
            x = model.in_norm(x)
            x, hidden = model.lstm(x, hidden)
            mask = torch.sigmoid(model.out_proj(x))              # (1, 1, n_mels)
            masks.append(mask.squeeze(0).squeeze(0).cpu())
    masks = torch.stack(masks, dim=0)                             # (T_frames, n_mels)

    # ---- Apply mask in mel domain → invert to linear magnitude ----
    # masked_mel = mask * mix_mel;  then mag = pinv(fb) @ masked_mel
    masked_mel = masks.T.to(mel.dtype) * mel                     # (n_mels, T_frames)
    masked_magnitude = (pinv_fb.T @ masked_mel).clamp(min=0.0)   # (n_freq, T_frames)

    # ---- Reconstruct with ORIGINAL phase ----
    enhanced_stft = masked_magnitude * torch.exp(1j * phase)
    enhanced_wav = torch.istft(
        enhanced_stft,
        n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length,
        window=window, center=True,
    )

    # Length-align to original (istft may differ by a few samples).
    if enhanced_wav.shape[0] < T_samples:
        enhanced_wav = torch.nn.functional.pad(
            enhanced_wav, (0, T_samples - enhanced_wav.shape[0])
        )
    elif enhanced_wav.shape[0] > T_samples:
        enhanced_wav = enhanced_wav[:T_samples]

    si_sdr_in = compute_si_sdr(mixed_wav, target_wav)
    si_sdr_out = compute_si_sdr(enhanced_wav, target_wav)
    return enhanced_wav, si_sdr_in, si_sdr_out


def main() -> None:
    p = argparse.ArgumentParser(description="Enhance mix_dump samples with a VoiceFilter ckpt")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mix-dump-dir", default="runs/mps/mix_dump")
    p.add_argument("--device", default="mps")
    p.add_argument("--output-name", default="04_enhanced.wav",
                   help="Filename inside each sample dir.")
    p.add_argument("--no-normalize", action="store_true",
                   help="Don't normalize the enhanced wav peak to 0.95.")
    args = p.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hp = HParams.from_dict(ckpt["hparams"])
    hp.device = str(device)

    model = VoiceFilterLite(hp).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    print(f"[enhance] loaded {args.checkpoint} (step={ckpt.get('step')}) on {device}")

    speaker_encoder = FrozenSpeakerEncoder(hp).to(device)
    mel_transform = make_mel_basis(hp, device=torch.device("cpu"))
    fb = mel_transform.mel_scale.fb                              # (n_freq, n_mels)
    pinv_fb = la.pinv(fb, rcond=1e-4)                            # (n_mels, n_freq)
    window = torch.hann_window(hp.win_length)

    mix_dump = Path(args.mix_dump_dir)
    manifest = json.loads((mix_dump / "manifest.json").read_text())
    print(f"[enhance] found {len(manifest)} samples in {mix_dump}")

    results = []
    for entry in manifest:
        sample_dir = Path(entry["dir"])
        snr = entry["snr_target_db"]
        print(f"\n[enhance] sample {entry['sample']}: "
              f"spk={entry['spk']} snr={snr:+.2f}dB  ({sample_dir.name})")

        ref_wav, _ = sf.read(str(sample_dir / "03_ref.wav"), dtype="float32")
        mixed_wav, _ = sf.read(str(sample_dir / "02_mixed.wav"), dtype="float32")
        target_wav, _ = sf.read(str(sample_dir / "00_target.wav"), dtype="float32")

        ref_wav_t = torch.from_numpy(ref_wav)
        mixed_wav_t = torch.from_numpy(mixed_wav)
        target_wav_t = torch.from_numpy(target_wav)

        enhanced_wav, si_in, si_out = enhance_sample(
            mixed_wav_t, target_wav_t, ref_wav_t,
            model, speaker_encoder, hp, fb, pinv_fb, window, device,
        )

        if not args.no_normalize:
            peak = enhanced_wav.abs().max().clamp(min=1e-8)
            if peak > 1.0:
                enhanced_wav = enhanced_wav / peak * 0.95

        out_path = sample_dir / args.output_name
        sf.write(str(out_path), enhanced_wav.numpy(), hp.sample_rate)
        print(f"[enhance]   → {out_path.name}  "
              f"peak={enhanced_wav.abs().max():.3f}  "
              f"SI-SDR in={si_in:+.2f}dB  out={si_out:+.2f}dB  "
              f"Δ={si_out - si_in:+.2f}dB")

        results.append({
            "sample": entry["sample"],
            "snr_target_db": snr,
            "si_sdr_in": si_in,
            "si_sdr_out": si_out,
            "si_sdr_delta": si_out - si_in,
            "out_path": str(out_path),
        })

    # ---- Summary ----
    print("\n[enhance] ===== SUMMARY =====")
    avg_in = float(np.mean([r["si_sdr_in"] for r in results]))
    avg_out = float(np.mean([r["si_sdr_out"] for r in results]))
    avg_delta = float(np.mean([r["si_sdr_delta"] for r in results]))
    print(f"  avg SI-SDR in  : {avg_in:+.2f} dB")
    print(f"  avg SI-SDR out : {avg_out:+.2f} dB")
    print(f"  avg improvement: {avg_delta:+.2f} dB")

    out_json = mix_dump / "enhance_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"[enhance] per-sample results → {out_json}")


if __name__ == "__main__":
    main()
