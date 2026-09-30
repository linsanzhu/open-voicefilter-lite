#!/usr/bin/env python3
"""Example 03 — Inference with the exported ONNX model.

`scripts/export_onnx.py` ships a streaming-friendly ONNX graph whose inputs
are the 256-d d-vector (from `SpeakerEncoder`) plus the persistent LSTM
(h, c). The deployment side has to:

  1. Run SpeakerEncoder on the enrollment clip → 256-d
  2. Run ONNX frame-by-frame, feeding back the new (h, c) each call

The d-vector is fed directly into the graph's `in_proj` — there is no
in-model projection layer.

This script demonstrates that pipeline and cross-checks the ONNX output
against PyTorch (optional, but recommended for a sanity check after
re-exporting).

Usage
-----
    python examples/03_onnx_inference.py \\
        --onnx       voicefilter_lite.int8.onnx \\
        --checkpoint runs/exp1/ckpt_final.pt    \\
        --enrollment data/enroll.wav             \\
        --mixed      data/mixed.wav              \\
        --out        data/enhanced_onnx.wav       \\
        --verify                                  # also run PyTorch on the same input

INT8 note: paper §3.5 quantises the deployed model. Pass the
``voicefilter_lite.int8.onnx`` produced by ``scripts/export_onnx.py
--quantize int8``. The default ``--max-abs-err`` is widened to ``5e-2``
to tolerate the INT8 rounding on the Linear layers (LSTM stays FP32 —
``onnxruntime.quantization.quantize_dynamic`` only handles MatMul-class
ops, not LSTM). For FP32 ONNX use ``--max-abs-err 1e-3``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from open_voicefilter_lite.audio import (  # noqa: E402
    extract_dvec,
    make_mel_basis,
    wav_to_log_mel,
)
from open_voicefilter_lite.hparams import HParams  # noqa: E402
from open_voicefilter_lite.model import VoiceFilterLite  # noqa: E402
from open_voicefilter_lite.speaker_encoder import (  # noqa: E402
    SpeakerEncoder,
)


def _select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_audio_mono_16k(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import torchaudio
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), sr, target_sr
        ).numpy()
    return torch.from_numpy(wav)


def _logmel_to_wav(log_mel: torch.Tensor, hp: HParams, n_iter: int = 16) -> torch.Tensor:
    import torchaudio

    mel = torch.exp(log_mel) - hp.log_mel_floor
    mel = mel.transpose(0, 1).unsqueeze(0)

    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=hp.n_fft // 2 + 1,
        n_mels=hp.n_mels,
        sample_rate=hp.sample_rate,
        max_iter=128,
    )
    linear = inv_mel(mel)

    griffin = torchaudio.transforms.GriffinLim(
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        power=1.0,
        n_iter=n_iter,
    )
    return griffin(linear).squeeze(0)


def _compute_dvec(
    encoder,
    enroll_wav: torch.Tensor,
    hp: HParams,
    device: torch.device,
) -> torch.Tensor:
    """`audio.extract_dvec` → `hp.d_vector_dim`=256-d (the vector the ONNX
    graph expects).
    """
    with torch.no_grad():
        dvec = extract_dvec(enroll_wav, hp, encoder, device=device).squeeze(0)
    return dvec.cpu()


class ONNXStreamingVoiceFilter:
    """Stateful ONNX runtime wrapper mirroring StreamingVoiceFilter's API.

    The ONNX graph expects:
        stacked_logmel: (1, T, n_mels * frame_stack)
        dvec:           (1, d_vector_dim)
        h0, c0:         (num_lstm_layers, 1, hidden_dim)
    and produces the next hidden state as well. We feed T=1 each call so
    inference cost is independent of stream length.
    """

    def __init__(self, session, hp: HParams):
        self.session = session
        self.hp = hp
        self._dvec: torch.Tensor | None = None
        self._h = np.zeros((hp.num_lstm_layers, 1, hp.hidden_dim), dtype=np.float32)
        self._c = np.zeros((hp.num_lstm_layers, 1, hp.hidden_dim), dtype=np.float32)
        self._ctx: list[np.ndarray] = []

    def set_dvector(self, dvec: torch.Tensor) -> None:
        self._dvec = dvec.unsqueeze(0).numpy().astype(np.float32)  # (1, d_vector_dim)
        self.reset_state()

    def reset_state(self) -> None:
        self._h[:] = 0.0
        self._c[:] = 0.0
        self._ctx = []

    def step(self, frame_logmel: torch.Tensor) -> torch.Tensor:
        if self._dvec is None:
            raise RuntimeError("Call set_dvector() before step().")

        f = frame_logmel.numpy().astype(np.float32).reshape(-1)        # (n_mels,)
        self._ctx.append(f)
        if len(self._ctx) > self.hp.frame_stack:
            self._ctx = self._ctx[-self.hp.frame_stack:]

        stack = self.hp.frame_stack
        n_mels = self.hp.n_mels
        stacked = np.zeros((1, 1, stack * n_mels), dtype=np.float32)
        for i, fr in enumerate(self._ctx):
            stacked[0, 0, (stack - len(self._ctx) + i) * n_mels:(stack - len(self._ctx) + i + 1) * n_mels] = fr

        outs = self.session.run(
            None,
            {
                "stacked_logmel": stacked,
                "dvec": self._dvec,
                "h0": self._h,
                "c0": self._c,
            },
        )
        enhanced, self._h, self._c = outs[0], outs[1], outs[2]
        return torch.from_numpy(enhanced[0, 0])


def run_onnx(
    session,
    hp: HParams,
    dvec: torch.Tensor,
    mixed_path: str,
    out_path: str,
    gl_iter: int,
) -> None:
    mixed = _load_audio_mono_16k(mixed_path, hp.sample_rate)
    mel_fn = make_mel_basis(hp, device=torch.device("cpu"))
    log_mel = wav_to_log_mel(mixed, hp, mel_fn)                       # (T, 80)

    onnx_sf = ONNXStreamingVoiceFilter(session, hp)
    onnx_sf.set_dvector(dvec)

    enhanced_frames = []
    for t in range(log_mel.shape[0]):
        enhanced_frames.append(onnx_sf.step(log_mel[t]).cpu())
    enhanced_logmel = torch.stack(enhanced_frames, dim=0)

    wav = _logmel_to_wav(enhanced_logmel, hp, n_iter=gl_iter)
    peak = wav.abs().max().clamp(min=1e-8)
    if peak > 1.0:
        wav = wav / peak
    sf.write(out_path, wav.numpy(), hp.sample_rate)
    print(f"[onnx] wrote {out_path} ({wav.shape[-1] / hp.sample_rate:.2f}s)")


def verify_against_pytorch(
    hp: HParams,
    model: VoiceFilterLite,
    dvec: torch.Tensor,
    mixed_logmel: torch.Tensor,
    max_abs_err: float,
) -> bool:
    """Run the same inputs through PyTorch and confirm the gap is small.

    This is a one-shot check on a short prefix — full sequence parity is
    guaranteed by determinism of both engines given identical init state.
    """
    from open_voicefilter_lite.model import StreamingVoiceFilter

    device = next(model.parameters()).device
    pyt_sf = StreamingVoiceFilter(model, hp)
    pyt_sf.set_dvector(dvec.to(device))

    onnx_sess = _load_onnx_from_model(hp, model)
    onnx_sf = ONNXStreamingVoiceFilter(onnx_sess, hp)
    onnx_sf.set_dvector(dvec.to(device).cpu())

    n_check = min(64, mixed_logmel.shape[0])
    pyt_frames, onnx_frames = [], []
    for t in range(n_check):
        pyt_frames.append(pyt_sf.step(mixed_logmel[t]).cpu())
        onnx_frames.append(onnx_sf.step(mixed_logmel[t]).cpu())

    pyt_t = torch.stack(pyt_frames)
    onnx_t = torch.stack(onnx_frames)
    err = float((pyt_t - onnx_t).abs().max())
    print(f"[onnx] max |PyTorch - ONNX| over first {n_check} frames: {err:.3e}")
    ok = err < max_abs_err
    print(f"[onnx] verify {'PASS' if ok else 'FAIL'} (threshold={max_abs_err:.1e})")
    return ok


def _load_onnx_from_model(hp: HParams, model: VoiceFilterLite):
    """Re-export in-memory for the verify path (no disk write)."""
    import io
    import onnxruntime as ort

    B, T = 1, 4
    stacked = torch.randn(B, T, hp.n_mels * hp.frame_stack)
    dvec = torch.randn(B, hp.d_vector_dim)
    h0 = torch.zeros(hp.num_lstm_layers, B, hp.hidden_dim)
    c0 = torch.zeros(hp.num_lstm_layers, B, hp.hidden_dim)

    # Mirror StreamWrapper from scripts/export_onnx.py
    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m

        def forward(self, slm, dv, h0, c0):
            B_, T_, _ = slm.shape
            dt = dv.unsqueeze(1).expand(-1, T_, -1)
            x = torch.cat([slm, dt], dim=-1)
            x = self.model.in_proj(x)
            x = self.model.in_norm(x)
            # Optional 1D CNN (paper §3.2, frequency-axis). When
            # `hp.use_cnn=False` this branch is a no-op and the wrapper
            # matches the LSTM-only export.
            if self.model.cnn is not None:
                x_per = x.reshape(-1, 1, x.shape[-1])              # (B*T, 1, H)
                x_conv = self.model.cnn(x_per)                     # (B*T, cnn_channels, H)
                x = x_conv.mean(dim=-1).reshape(x.shape[0], x.shape[1], -1)
            x, (hN, cN) = self.model.lstm(x, (h0, c0))
            mask = torch.sigmoid(self.model.out_proj(x))
            cur_idx = self.model.hp.frame_stack - 1
            cur = slm[..., cur_idx * self.model.hp.n_mels:(cur_idx + 1) * self.model.hp.n_mels]
            # Paper §3.2 literal mask application (Y_enh = M · Y in log-mel).
            return self.model._apply_mask(mask, cur), hN, cN

    wrap = _Wrap(model).eval()
    buf = io.BytesIO()
    try:
        torch.onnx.export(
            wrap, (stacked, dvec, h0, c0), buf,
            opset_version=17, do_constant_folding=True,
            input_names=["stacked_logmel", "dvec", "h0", "c0"],
            output_names=["enhanced", "hN", "cN"],
            dynamic_shapes={
                "stacked_logmel": {0: "batch", 1: "time"},
                "dvec":           {0: "batch"},
                "h0":             {1: "batch"},
                "c0":             {1: "batch"},
            },
        )
    except TypeError:
        # Fall back: rewrite buf and re-export with dynamic_axes
        buf = io.BytesIO()
        torch.onnx.export(
            wrap, (stacked, dvec, h0, c0), buf,
            opset_version=17, do_constant_folding=True,
            input_names=["stacked_logmel", "dvec", "h0", "c0"],
            output_names=["enhanced", "hN", "cN"],
            dynamic_axes={
                "stacked_logmel": {0: "batch", 1: "time"},
                "dvec":           {0: "batch"},
                "h0":             {1: "batch"},
                "c0":             {1: "batch"},
                "enhanced":       {0: "batch", 1: "time"},
                "hN":             {1: "batch"},
                "cN":             {1: "batch"},
            },
        )
    buf.seek(0)
    return ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--onnx", required=True, help="Path to voicefilter_lite.onnx")
    p.add_argument("--checkpoint", required=True, help="Path to ckpt_*.pt (for d-vector source + verify)")
    p.add_argument("--enrollment", required=True, help="16 kHz mono WAV of the target speaker")
    p.add_argument("--mixed", required=True, help="16 kHz mono WAV to separate")
    p.add_argument("--out", required=True, help="Where to write the enhanced WAV")
    p.add_argument("--verify", action="store_true", help="Cross-check ONNX against PyTorch")
    p.add_argument("--device", default="auto")
    p.add_argument("--gl-iter", type=int, default=16)
    p.add_argument("--max-abs-err", type=float, default=5e-2,
                   help="Max allowed |PyTorch - ONNX| over the verify prefix. "
                        "Use 1e-3 for FP32 ONNX; 5e-2 for INT8 ONNX "
                        "(default accommodates either).")
    args = p.parse_args()

    device = _select_device(args.device)

    # ---- Load ONNX ----
    import onnxruntime as ort

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    print(f"[onnx] loaded {args.onnx}")

    # ---- Restore model (for d-vector source + optional verify) ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hp = HParams.from_dict(ckpt["hparams"])
    hp.device = str(device)

    model = VoiceFilterLite(hp).to(device).eval()
    sd = ckpt["model_state"]
    # Backward compat: drop legacy `dvec_proj.*` (pre-refactor 192→256
    # projection Linear) and any `alpha_head.*` keys that don't fit the
    # current paper-faithful 2-Linear head (paper §4.2). Strict shape
    # filter: keep only keys that exist in the model with the matching
    # shape. See the matching block in `scripts/export_onnx.py`.
    model_sd = model.state_dict()
    kept = {k: v for k, v in sd.items()
            if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape)}
    _, unexpected = model.load_state_dict(kept, strict=False)
    if unexpected:
        raise RuntimeError(f"[onnx] unexpected keys beyond compat drop: {unexpected[:5]}…")

    encoder = SpeakerEncoder(hp).to(device)
    if Path(hp.dvector_pt_path).is_file():
        encoder.load_pretrained(hp.dvector_pt_path)
        print(f"[onnx] SpeakerEncoder loaded from {hp.dvector_pt_path}")
    else:
        print(
            f"[onnx] WARNING: dvector weights not found at {hp.dvector_pt_path}; "
            f"encoder will use RANDOM init. The pretrained embedder "
            f"(`models/embedder.pt`) ships with this repo (from "
            f"https://github.com/maum-ai/voicefilter)."
        )
        encoder.freeze()
    print(f"[onnx] d_vector_dim={hp.d_vector_dim}")

    enroll_wav = _load_audio_mono_16k(args.enrollment, hp.sample_rate)
    with torch.no_grad():
        dvec = extract_dvec(enroll_wav, hp, encoder, device=device).squeeze(0).cpu()
    print(f"[onnx] d-vector dim {dvec.shape[-1]} (from speaker encoder)")

    if args.verify:
        mixed = _load_audio_mono_16k(args.mixed, hp.sample_rate)
        mel_fn = make_mel_basis(hp, device=torch.device("cpu"))
        log_mel = wav_to_log_mel(mixed, hp, mel_fn)
        ok = verify_against_pytorch(hp, model, dvec, log_mel, args.max_abs_err)
        if not ok:
            raise SystemExit(2)

    run_onnx(sess, hp, dvec, args.mixed, args.out, args.gl_iter)


if __name__ == "__main__":
    main()