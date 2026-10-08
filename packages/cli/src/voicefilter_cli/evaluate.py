#!/usr/bin/env python3
"""Evaluate a trained VoiceFilter-Lite checkpoint on LibriSpeech test-clean/other.

Metrics
-------
    Input  SI-SNR / SDR : baseline of the mixed signal (target vs mixture)
    Output SI-SNR / SDR : after VoiceFilter-Lite
    SI-SNRi / SDRi      : improvement (output - input)

Usage
-----
    `voicefilter-cli evaluate` --checkpoint runs/exp1/ckpt_step050000.pt
    `voicefilter-cli evaluate` --checkpoint runs/exp1/ckpt_final.pt --num-samples 50
"""

from __future__ import annotations

import io
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Iterator, Optional

import numpy as np
import soundfile as sf
import torch
from cyclopts import App, Parameter

from voicefilter_lite.audio import (
    extract_dvec,
    make_mel_basis,
    pad_or_trim,
    random_crop,
    snr_mix,
    wav_to_log_mel,
)
from voicefilter_cli.dataset import first_librispeech_other_source
from voicefilter_cli.hf import apply_hf_endpoint
from voicefilter_cli.dataset import (
    _cast_audio_no_decode,
    _disable_hf_audio_decode,
)
from voicefilter_lite.hparams import HParams, resolve_source
from voicefilter_lite.model import VoiceFilterLite
from voicefilter_lite.streaming import StreamingVoiceFilter
from voicefilter_lite.speaker_encoder import SpeakerEncoder

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

def log_mel_snr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    mix: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[float, float, float]:
    """Compute SI-SNR in the **linear-mel** domain. The model itself operates
    on linear-mel filterbank energies (`voicefilter_lite.model.apply_mask` applies
    the mask to magnitude then re-logs), so this metric is the cleanest
    phase-free proxy for how well VoiceFilter recovers the target's spectral
    envelope.

    Avoids the Griffin-Lim phase pitfall that makes time-domain SI-SNR
    unreliable (Griffin-Lim's phase is essentially random and destroys the
    time-alignment SI-SNR requires).

    All inputs are (T, F) **log-mel** filterbank energies; we convert to
    linear-mel internally before computing SI-SNR.
    Returns (input_snr_db, output_snr_db, improvement_db).
    """
    log_mel_floor = 1e-6   # matches hp.log_mel_floor default
    est_lin = (torch.exp(estimate) - log_mel_floor).clamp(min=0.0)
    tgt_lin = (torch.exp(target) - log_mel_floor).clamp(min=0.0)
    mix_lin = (torch.exp(mix) - log_mel_floor).clamp(min=0.0)

    def _snr_db(est: torch.Tensor, ref: torch.Tensor) -> float:
        # Per-mel-bin SI-SNR (DC-removed), then averaged across the 80 bins.
        est_c = est - est.mean(dim=0, keepdim=True)
        ref_c = ref - ref.mean(dim=0, keepdim=True)
        alpha = (est_c * ref_c).sum(dim=0) / (ref_c.pow(2).sum(dim=0) + eps)
        proj = alpha.unsqueeze(0) * ref_c
        noise = est_c - proj
        per_bin_db = 10.0 * torch.log10(
            proj.pow(2).sum(dim=0) / (noise.pow(2).sum(dim=0) + eps) + eps
        )
        return float(per_bin_db.mean())

    snr_in = _snr_db(mix_lin, tgt_lin)
    snr_out = _snr_db(est_lin, tgt_lin)
    return snr_in, snr_out, snr_out - snr_in

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

    Reads from the HF arrow cache directly (no wav-baked pool) — the
    `voicefilter-cli dataset prepare` step populates this cache.
    """
    from datasets import load_dataset
    from itertools import chain

    # Resolve target + interference SourceRefs against the top-level
    # `dataset.sources` declarations.
    if not hp.target_sources:
        raise RuntimeError("[evaluate] hp.target_sources is empty")
    target_resolved = resolve_source(hp, hp.target_sources[0])
    inter_resolved = first_librispeech_other_source(hp)

    # `cache_dir` is honoured by `load_dataset` only when `streaming=False`;
    # when streaming, the rows come straight from the HF Hub stream. We
    # build a kwargs dict per source so target / interference can point at
    # different caches if a config wants that.
    def _load_kwargs(cache_dir: str) -> dict:
        kw: dict = {"streaming": hp.streaming}
        if not hp.streaming:
            kw["cache_dir"] = cache_dir or "data/librispeech"
        return kw

    # Disable HF datasets' eager torchcodec/FFmpeg audio decode so we can
    # hand the bytes to `soundfile` (which is already used by
    # `decode_audio` below) — see `dataset._disable_hf_audio_decode`.
    _disable_hf_audio_decode()

    def _load(repo: str, config: str, split: str, cache_dir: str):
        ds = load_dataset(repo, config, split=split, **_load_kwargs(cache_dir))
        return _cast_audio_no_decode(ds)

    # Target: use the first declared target source (function-oriented
    # `hp.target_sources`). Eval scripts assume a single LibriSpeech
    # target pool; multi-source targets would need a per-source rotation.
    # LibriSpeech is loaded via the `all` config — the role tag (`clean` /
    # `other`) lives in the split name (`train.clean.360`).
    assert target_resolved.is_librispeech(), (
        f"eval target must be a LibriSpeech source; got {target_resolved}"
    )
    target_iter = iter(
        chain.from_iterable(
            _load(target_resolved.repo, "all", s, target_resolved.cache_dir)
            for s in target_resolved.splits
        )
    )
    # Interference: pick the first enabled LibriSpeech `other` source from
    # `hp.interference` (the canonical "other" pool used by `babble` in
    # paper.yaml).
    assert inter_resolved.is_librispeech(), (
        f"eval interference must be a LibriSpeech source; got {inter_resolved}"
    )
    interference_iter = iter(
        _load(
            inter_resolved.repo, "all",
            inter_resolved.splits[0], inter_resolved.cache_dir,
        )
    )

    rng = random.Random(seed)
    spk_buf_hf: dict[str, dict] = {}

    n = 0
    while n < num_samples:
        try:
            tgt = next(target_iter)
            intf = next(interference_iter)
        except StopIteration:
            return

        spk = str(tgt.get("speaker_id", rng.randint(0, 10_000)))
        buf = spk_buf_hf.setdefault(spk, [])
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


def evaluate(hp: HParams, args: "EvalArgs") -> None:
    # Auto-detect device at runtime unless `--device` forces one.
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[eval] device={device}")

    # Load checkpoint.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hp = HParams.from_dict(ckpt["hparams"])

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

    # Diagnostic: when --no-adaptive is set, disable the per-frame adaptive
    # suppression `w^(t)` so the model emits pure mask output. Useful for
    # sanity-checking whether the trained mask itself is sensible vs. whether
    # the adaptive blending is over-suppressing.
    if getattr(args, "no_adaptive", False):
        hp.use_noise_type_predictor = False

    speaker_encoder = SpeakerEncoder(hp).to(device)
    if Path(hp.dvector_pt_path).is_file():
        speaker_encoder.load_pretrained(hp.dvector_pt_path)
        print(f"[eval] SpeakerEncoder loaded from {hp.dvector_pt_path}")
    else:
        print(
            f"[eval] WARNING: dvector weights not found at {hp.dvector_pt_path}; "
            f"encoder will use RANDOM init. The pretrained embedder "
            f"(`models/embedder.pt`) ships with this repo (from "
            f"https://github.com/maum-ai/voicefilter)."
        )
        speaker_encoder.freeze()
    print(f"[eval] d_vector_dim={hp.d_vector_dim}")

    mel_transform = make_mel_basis(hp, device=torch.device("cpu"))

    si_snr_in, si_snr_out = [], []
    sdr_in, sdr_out = [], []
    log_mel_snr_in: list[float] = []
    log_mel_snr_out: list[float] = []

    for i, (mixed, target, ref, interference) in enumerate(
        build_eval_pairs(hp, args.num_samples, hp.seed)
    ):
        # Compute d-vector from ref. The VoiceFilter d-vector recipe expects
        # arbitrary-length refs (it does its own windowed aggregation),
        # so we no longer crop to a fixed length as we did with the
        # pre-refactor wrapper.
        ref_for_dvec = pad_or_trim(ref, int(hp.ref_max_sec * hp.sample_rate))
        with torch.no_grad():
            dvec = extract_dvec(ref_for_dvec, hp, speaker_encoder, device=device).squeeze(0)

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

            # Griffin-Lim produces ~`(num_frames-1)*hop + win_length` samples;
            # for `center=False` Mel-spec that's typically 240 samples short of
            # the original waveform. Pad / trim to the length of `mixed` so
            # SI-SNR / SDR have matching lengths.
            if enhanced_wav.shape[0] < input_wav.shape[0]:
                enhanced_wav = torch.nn.functional.pad(
                    enhanced_wav, (0, input_wav.shape[0] - enhanced_wav.shape[0])
                )
            elif enhanced_wav.shape[0] > input_wav.shape[0]:
                enhanced_wav = enhanced_wav[: input_wav.shape[0]]

            si_in = si_snr(input_wav, target_wav_eval)
            si_out = si_snr(enhanced_wav, target_wav_eval)
            sd_in = sdr(input_wav, target_wav_eval)
            sd_out = sdr(enhanced_wav, target_wav_eval)

            # Log-mel-domain SI-SNR — the time-domain metric above is unreliable
            # because Griffin-Lim's reconstructed phase is essentially random
            # and SI-SNR requires time-alignment. The log-mel-domain version
            # skips phase entirely and measures how well the filter recovered
            # the target's spectral envelope.
            target_logmel = wav_to_log_mel(target_wav_eval, hp, mel_transform)
            lm_in, lm_out, lm_imp = log_mel_snr(
                enhanced_logmel, target_logmel, mixed_logmel
            )

            si_snr_in.append(si_in)
            si_snr_out.append(si_out)
            sdr_in.append(sd_in)
            sdr_out.append(sd_out)
            log_mel_snr_in.append(lm_in)
            log_mel_snr_out.append(lm_out)

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
        # Time-domain (Griffin-Lim reconstructed — phase is essentially random,
        # so these numbers are an unreliable proxy; reported for completeness).
        "input_si_snr_mean": float(np.mean(si_snr_in)),
        "output_si_snr_mean": float(np.mean(si_snr_out)),
        "si_snri_mean": float(np.mean(si_snr_out) - np.mean(si_snr_in)),
        "input_sdr_mean": float(np.mean(sdr_in)),
        "output_sdr_mean": float(np.mean(sdr_out)),
        "sdri_mean": float(np.mean(sdr_out) - np.mean(sdr_in)),
        # Log-mel-domain SI-SNR — phase-free, the meaningful proxy for this
        # filter. Numbers are per-mel-bin SI-SNR averaged across the 80 mel
        # channels; positive `logMelNetValue` means VoiceFilter is recovering
        # the target's spectral envelope better than the unfiltered mixture.
        "logmel_input_si_snr_mean": float(np.mean(log_mel_snr_in)) if log_mel_snr_in else 0.0,
        "logmel_output_si_snr_mean": float(np.mean(log_mel_snr_out)) if log_mel_snr_out else 0.0,
        "logmel_snri_mean": (
            float(np.mean(log_mel_snr_out) - np.mean(log_mel_snr_in))
            if log_mel_snr_out else 0.0
        ),
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

    Inverse-mel is done with `torch.linalg.pinv` on the filterbank matrix
    (Tikhonov-regularised via `rcond`). Older torchaudio versions exposed an
    iterative `InverseMelScale(max_iter, tolerance_loss, tolerance_change)`
    constructor; the newer API (>=2.10) dropped those kwargs and uses a
    direct LAPACK `gels` solve that fails on rank-deficient filterbank
    matrices — which is exactly our case (n_mels=128, n_freq//2+1=201, so
    the basis is rectangular). Doing the pseudoinverse ourselves is more
    robust and matches what the earlier iterative constructor converged on.
    """
    import torch.linalg as la

    fb = mel_transform.mel_scale.fb                              # (n_freq, n_mels)
    pinv_fb = la.pinv(fb, rcond=1e-4)                            # (n_mels, n_freq)

    # Forward path:  magnitude.T @ fb  →  (T, n_mels). Inverse path:
    # mel.T  (T, n_mels) @  pinv(fb) (n_mels, n_freq)  →  (T, n_freq).
    mel_T = torch.exp(log_mel).clamp(min=0.0)                    # (T, n_mels)
    linear_T = mel_T @ pinv_fb                                    # (T, n_freq)
    linear = linear_T.transpose(0, 1).unsqueeze(0)               # (1, n_freq, T)

    # Griffin-Lim to invert phase.
    import torchaudio
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

@dataclass
class EvalArgs:
    """Typed CLI surface for ``evaluate voicefilter-lite``."""

    checkpoint: str
    num_samples: int = 200
    device: str = "auto"
    results_json: Optional[str] = None
    gl_iter: int = 10
    no_adaptive: bool = False
    hf_endpoint: Optional[str] = None
    hf_transfer: Optional[bool] = None


app = App(name="voicefilter-lite", help_format="plaintext")


@app.default
def main(
    checkpoint: Annotated[str, Parameter(help="VoiceFilter-Lite trained ckpt_*.pt.")],
    *,
    num_samples: Annotated[
        int, Parameter(name="--num-samples", help="Number of evaluation samples (target/interference pairs).")
    ] = 200,
    device: Annotated[
        str,
        Parameter(
            name="--device",
            help="Force a device (mps / cuda / cpu). Default 'auto' picks "
                 "MPS → CUDA → CPU based on availability.",
        ),
    ] = "auto",
    results_json: Annotated[
        Optional[str],
        Parameter(name="--results-json", help="Write summary metrics to this JSON file path."),
    ] = None,
    gl_iter: Annotated[
        int,
        Parameter(
            name="--gl-iter",
            help="Griffin-Lim iterations (used for time-domain SI-SNR/SDR approximation).",
        ),
    ] = 10,
    no_adaptive: Annotated[
        bool,
        Parameter(
            negative="",
            name="--no-adaptive",
            help="Disable per-frame adaptive suppression (isolate the raw mask's contribution).",
        ),
    ] = False,
    hf_endpoint: Annotated[
        Optional[str],
        Parameter(
            name="--hf-endpoint",
            help="Override HF Hub base URL (e.g. https://hf-mirror.com).",
        ),
    ] = None,
    hf_transfer: Annotated[
        Optional[bool],
        Parameter(negative="--no-hf-transfer", help="Force-enable HF hf_transfer protocol."),
    ] = None,
) -> None:
    """Evaluate a trained VoiceFilter-Lite checkpoint on LibriSpeech.

    Metrics: SI-SNR / SDR / log-mel SNR improvements over the mixed signal.
    """
    args = EvalArgs(
        checkpoint=checkpoint,
        num_samples=num_samples,
        device=device,
        results_json=results_json,
        gl_iter=gl_iter,
        no_adaptive=no_adaptive,
        hf_endpoint=hf_endpoint,
        hf_transfer=hf_transfer,
    )
    hp = HParams()
    evaluate(hp, args)


if __name__ == "__main__":
    app()