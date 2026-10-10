#!/usr/bin/env python3
"""Compare d-vector outputs from PyTorch embedder vs INT8 ONNX embedder.

Loads a handful of .wav files (positional CLI args, or the integration
test set by default), runs the canonical `extract_dvec` recipe through
both the PyTorch `SpeakerEncoder` and the INT8 ONNX graph, and prints
per-utterance cosine similarity + summary stats.

Usage
-----
    uv run python scripts/compare_embedder_quant.py \\
        --weights models/embedder.pt \\
        --onnx models/embedder.int8.onnx \\
        --wavs data/integration_test/00_enhanced.wav \\
               data/integration_test/01_enhanced.wav \\
               data/integration_test/02_enhanced.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torch.nn.functional as F

from model.audio import extract_dvec, pad_or_trim
from voicefilter_lite.hparams import HParams
from model.speaker_encoder import SpeakerEncoder


def dvec_onnx(onnx_path: str, wav: torch.Tensor, hp: HParams) -> torch.Tensor:
    """Replicate `extract_dvec` against an ONNX d-vector encoder."""
    # Reuse the same audio preprocessing the PyTorch path uses.
    from model.audio import wav_to_dvec_mel

    feat = wav_to_dvec_mel(wav, hp)
    if feat.dim() == 2:
        feat = feat.unsqueeze(0)
    B, T_frames, F_dim = feat.shape
    win, hop = hp.dvector_window, hp.dvector_stride

    if T_frames < win:
        pad = torch.zeros(B, win - T_frames, F_dim, dtype=feat.dtype)
        feat = torch.cat([feat, pad], dim=1)
        T_frames = win

    n_win = 1 + (T_frames - win) // hop
    feat_np = feat.numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    if n_win == 1:
        embs = sess.run(None, {"mel_window": feat_np})[0]      # (B, 256)
        embs = embs / (np.linalg.norm(embs, axis=-1, keepdims=True) + 1e-12)
        mean = embs                                            # (B, 256)
    else:
        # Sliding window → (n_win, B, win, F_dim) → (n_win*B, win, F_dim)
        windows = (
            torch.tensor(feat_np)
            .unfold(1, win, hop)
            .permute(1, 0, 3, 2)
            .reshape(n_win * B, win, F_dim)
            .numpy()
        )
        embs = sess.run(None, {"mel_window": windows})[0]       # (n_win*B, 256)
        embs = embs.reshape(n_win, B, -1)
        embs = embs / (np.linalg.norm(embs, axis=-1, keepdims=True) + 1e-12)
        mean = embs.mean(axis=0)                                # (B, 256)

    out = mean / (np.linalg.norm(mean, axis=-1, keepdims=True) + 1e-12)
    return torch.from_numpy(out).squeeze(0)                    # (256,)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="models/embedder.pt")
    p.add_argument("--onnx", default="models/embedder.int8.onnx",
                   help="INT8 ONNX path; pass the FP32 ONNX here too if you "
                        "want to isolate the INT8-specific cost.")
    p.add_argument("--onnx-fp32", default="models/embedder.onnx",
                   help="FP32 ONNX baseline; pass empty string to skip.")
    p.add_argument(
        "--wavs",
        nargs="*",
        default=[
            "data/integration_test/00_enhanced.wav",
            "data/integration_test/01_enhanced.wav",
            "data/integration_test/02_enhanced.wav",
            "data/integration_test/03_enhanced.wav",
            "data/integration_test/04_enhanced.wav",
        ],
    )
    args = p.parse_args()

    hp = HParams()
    enc = SpeakerEncoder(hp).eval()
    enc.load_pretrained(args.weights)

    cos_per_utt: list[float] = []
    l2_per_utt: list[float] = []
    cos_fp32_per_utt: list[float] = []
    l2_fp32_per_utt: list[float] = []

    fp32_onnx = args.onnx_fp32 if args.onnx_fp32 else None

    print(f"{'wav':<48} "
          f"{'cos(fp32)':>10} {'L2(fp32)':>11} "
          f"{'cos(int8)':>10} {'L2(int8)':>11}")
    for wav_path in args.wavs:
        wav_path = Path(wav_path)
        if not wav_path.is_file():
            print(f"{wav_path.name:<48} (missing — skipped)")
            continue

        data, sr = sf.read(str(wav_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        wav = torch.from_numpy(data).float()
        if sr != hp.sample_rate:
            import torchaudio
            wav = torchaudio.functional.resample(
                wav.unsqueeze(0), sr, hp.sample_rate,
            ).squeeze(0)
        wav = pad_or_trim(wav, int(hp.ref_max_sec * hp.sample_rate))

        with torch.no_grad():
            d_pt = extract_dvec(wav, hp, enc).squeeze(0)
        d_int8 = dvec_onnx(args.onnx, wav, hp)

        cos_i8 = F.cosine_similarity(d_pt.unsqueeze(0), d_int8.unsqueeze(0)).item()
        l2_i8 = float(torch.linalg.norm(d_pt - d_int8).item())
        cos_per_utt.append(cos_i8)
        l2_per_utt.append(l2_i8)

        if fp32_onnx:
            d_fp32 = dvec_onnx(fp32_onnx, wav, hp)
            cos_f = F.cosine_similarity(d_pt.unsqueeze(0), d_fp32.unsqueeze(0)).item()
            l2_f = float(torch.linalg.norm(d_pt - d_fp32).item())
            cos_fp32_per_utt.append(cos_f)
            l2_fp32_per_utt.append(l2_f)
        else:
            cos_f = l2_f = float("nan")

        print(
            f"{wav_path.name:<48} "
            f"{cos_f:>10.6f} {l2_f:>11.4e} "
            f"{cos_i8:>10.6f} {l2_i8:>11.4e}"
        )

    if not cos_per_utt:
        print("\n(no wavs found — nothing to compare)", file=sys.stderr)
        return 1

    print(
        f"\nINT8 ONNX ↔ PyTorch:  cosine mean={float(np.mean(cos_per_utt)):.6f}  "
        f"L2 mean={float(np.mean(l2_per_utt)):.4e}"
    )
    if fp32_onnx:
        print(
            f"FP32 ONNX ↔ PyTorch:  cosine mean={float(np.mean(cos_fp32_per_utt)):.6f}  "
            f"L2 mean={float(np.mean(l2_fp32_per_utt)):.4e}"
        )
        # INT8-specific cost = (INT8 drift) − (FP32 drift) on cosine.
        delta = float(np.mean(cos_per_utt)) - float(np.mean(cos_fp32_per_utt))
        print(
            f"\n→ INT8 cost vs FP32 ONNX: Δcos = {delta:+.6f}. "
            f"Negative = INT8 is worse; positive = INT8 happens to be closer "
            f"to PyTorch (rare; depends on whether the activation "
            f"calibration drifted the projection in a fortuitous way)."
        )
    print(
        f"\nFor d-vector downstream use (L2 → cosine sim → mask), "
        f"cosine ≥ 0.98 between two backends is functionally identical: "
        f"the speaker-discrimination gap between same-speaker and cross-"
        f"speaker is typically 0.4-0.6, so 1e-2 backend drift is noise."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
