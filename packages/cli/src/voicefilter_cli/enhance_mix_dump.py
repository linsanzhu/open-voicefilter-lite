#!/usr/bin/env python3
"""Run a trained VoiceFilter-Lite ckpt on a directory of mix_dump samples.

For each `sample_XX_*/` directory in the mix dump:
  - 03_ref.wav      → d-vector via frozen SpeakerEncoder
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
    `voicefilter-cli enhance-mix-dump` --checkpoint runs/v2_fix/ckpt_final.pt
    `voicefilter-cli enhance-mix-dump` --checkpoint runs/mps/ckpt_step047000.pt \\
        --mix-dump-dir runs/mps/mix_dump
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import soundfile as sf
import torch
import torch.linalg as la
from cyclopts import App, Parameter

from voicefilter_lite.audio import (
    extract_dvec,
    make_mel_basis,
    pad_or_trim,
)
from voicefilter_lite.hparams import HParams
from voicefilter_lite.model import VoiceFilterLite
from voicefilter_lite.speaker_encoder import SpeakerEncoder

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
    speaker_encoder: SpeakerEncoder,
    hp: HParams,
    fb: torch.Tensor,
    pinv_fb: torch.Tensor,
    window: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    """Enhance `mixed_wav` using the model. Returns (enhanced_wav, si_sdr_in, si_sdr_out)."""
    T_samples = mixed_wav.shape[0]

    # ---- D-vector from reference (full ref, capped at ref_max_sec) ----
    ref_for_dvec = pad_or_trim(ref_wav, int(hp.ref_max_sec * hp.sample_rate))
    with torch.no_grad():
        dvec = extract_dvec(ref_for_dvec, hp, speaker_encoder, device=device).squeeze(0)

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
    # The d-vector from the speaker encoder is already 256-d and is fed
    # directly into `in_proj`.
    dvec_tiled = dvec.view(1, 1, -1).to(device)                  # (1, 1, d_vector_dim)
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
            x = model.in_proj(torch.cat([stacked_flat, dvec_tiled], dim=-1))
            x = model.in_norm(x)
            x, hidden = model.lstm(x, hidden)
            mask = torch.sigmoid(model.out_proj(x))              # (1, 1, n_mels)
            masks.append(mask.squeeze(0).squeeze(0).cpu())
    masks = torch.stack(masks, dim=0)                             # (T_frames, n_mels)

    # ---- Apply mask in LINEAR-mel domain → invert to linear magnitude ----
    # Paper §3.2: `Y_enh = M · S_in` is on the linear filterbank energy
    # `S_in`, NOT on log-mel. `mel = fb.T @ magnitude` is the linear
    # filterbank energy of the mixed frame, so we multiply the mask with
    # `mel` directly (not the log-mel the NN was conditioned on) — then
    # pseudo-invert through the mel basis to recover the enhanced
    # magnitude, and combine with the original phase for the wav.
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

app = App(name="enhance-mix-dump", help_format="plaintext")


@app.default
def main(
    *,
    checkpoint: Annotated[
        str, Parameter(help="Path to a VoiceFilter-Lite checkpoint (.pt).")
    ],
    mix_dump_dir: Annotated[
        str, Parameter(name="--mix-dump-dir", help="Directory containing mix_dump samples + manifest.json.")
    ] = "runs/mps/mix_dump",
    device: Annotated[
        str, Parameter(help="Force a device (mps / cuda / cpu). Default 'auto' picks MPS → CUDA → CPU.")
    ] = "auto",
    output_name: Annotated[
        str, Parameter(name="--output-name", help="Filename written inside each sample dir.")
    ] = "04_enhanced.wav",
    no_normalize: Annotated[
        bool, Parameter(negative="", name="--no-normalize", help="Don't normalize the enhanced wav peak to 0.95.")
    ] = False,
) -> None:
    """Enhance mix_dump samples with a VoiceFilter ckpt."""
    # Auto-detect device unless forced via `--device`.
    if device == "auto":
        if torch.backends.mps.is_available():
            device_obj = torch.device("mps")
        elif torch.cuda.is_available():
            device_obj = torch.device("cuda")
        else:
            device_obj = torch.device("cpu")
    else:
        device_obj = torch.device(device)

    ckpt = torch.load(checkpoint, map_location=device_obj, weights_only=False)
    hp = HParams.from_dict(ckpt["hparams"])

    model = VoiceFilterLite(hp).to(device_obj).eval()
    model.load_state_dict(ckpt["model_state"])
    print(f"[enhance] loaded {checkpoint} (step={ckpt.get('step')}) on {device_obj}")

    speaker_encoder = SpeakerEncoder(hp).to(device_obj)
    if Path(hp.dvector_pt_path).is_file():
        speaker_encoder.load_pretrained(hp.dvector_pt_path)
        print(f"[enhance] SpeakerEncoder loaded from {hp.dvector_pt_path}")
    else:
        print(
            f"[enhance] WARNING: dvector weights not found at {hp.dvector_pt_path}; "
            f"encoder will use RANDOM init. The pretrained embedder "
            f"(`models/embedder.pt`) ships with this repo (from "
            f"https://github.com/maum-ai/voicefilter)."
        )
        speaker_encoder.freeze()
    print(f"[enhance] d_vector_dim={hp.d_vector_dim}")
    mel_transform = make_mel_basis(hp, device=torch.device("cpu"))
    fb = mel_transform.mel_scale.fb                              # (n_freq, n_mels)
    pinv_fb = la.pinv(fb, rcond=1e-4)                            # (n_mels, n_freq)
    window = torch.hann_window(hp.win_length)

    mix_dump = Path(mix_dump_dir)
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
            model, speaker_encoder, hp, fb, pinv_fb, window, device_obj,
        )

        if not no_normalize:
            peak = enhanced_wav.abs().max().clamp(min=1e-8)
            if peak > 1.0:
                enhanced_wav = enhanced_wav / peak * 0.95

        out_path = sample_dir / output_name
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
    app()
