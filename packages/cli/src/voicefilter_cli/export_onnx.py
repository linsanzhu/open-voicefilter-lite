#!/usr/bin/env python3
"""Export a VoiceFilter-Lite checkpoint to ONNX (streaming-friendly variant).

The exported model takes:
    stacked_logmel: (B=1, T, n_mels * frame_stack)
    dvec:           (B=1, d_vector_dim)         — 256-d d-vector produced by
                                                  `SpeakerEncoder`. The
                                                  d-vector is fed directly
                                                  into `in_proj`; there is no
                                                  in-model projection.
    h0:             (num_layers, B=1, hidden_dim)
    c0:             (num_layers, B=1, hidden_dim)

and emits:
    enhanced:       (B=1, T, n_mels)
    hN:             (num_layers, B=1, hidden_dim)
    cN:             (num_layers, B=1, hidden_dim)

Usage
-----
    `voicefilter-cli export onnx` \\
        --checkpoint runs/exp1/ckpt_final.pt \\
        --output      voicefilter_lite.onnx

    # INT8 post-training dynamic quantization (paper §3.5):
    `voicefilter-cli export onnx` \\
        --checkpoint runs/exp1/ckpt_final.pt \\
        --output      voicefilter_lite.onnx \\
        --quantize    int8

    The INT8 pass calls ``onnxruntime.quantization.quantize_dynamic`` on
    the exported FP32 graph; LSTM ops are skipped by onnxruntime (they
    aren't MatMul-class) and stay FP32, which is the standard tradeoff
    for streaming RNNs. The output is written next to ``--output`` with
    suffix ``.int8.onnx``.
"""

from __future__ import annotations

import os
import sys
from typing import Annotated, Literal

import torch
from cyclopts import App, Parameter

from voicefilter_lite.hparams import HParams
from voicefilter_lite.audio import log_mel_to_linear, linear_to_log_mel
from voicefilter_lite.model import VoiceFilterLite, apply_mask

class StreamWrapper(torch.nn.Module):
    """Wraps VoiceFilterLite for ONNX export with explicit LSTM state IO."""

    def __init__(self, model: VoiceFilterLite):
        super().__init__()
        self.model = model

    def forward(
        self,
        stacked_logmel: torch.Tensor,
        dvec: torch.Tensor,
        h0: torch.Tensor,
        c0: torch.Tensor,
    ):
        # Shared LSTM core (in_proj + in_norm + [cnn] + lstm).
        x, (hN, cN) = self.model.lstm_forward(
            stacked_logmel, dvec, (h0, c0),
        )
        mask = torch.sigmoid(self.model.out_proj(self.model.dropout(x)))

        cur_idx = self.model.hp.frame_stack - 1
        cur = stacked_logmel[..., cur_idx * self.model.hp.n_mels:(cur_idx + 1) * self.model.hp.n_mels]
        # Paper §3.2 literal mask application (Y_enh = M · Y in
        # **linear** filterbank-energy space — NOT log-mel). The ONNX
        # graph is consumed by an ASR pipeline that takes log-mel, so
        # we round-trip through `log_mel_to_linear` /
        # `linear_to_log_mel` here so the exported model's input/output
        # interface stays `log-mel → log-mel` (ASR-friendly) while the
        # mask runs in the paper-correct domain.
        cur_linear = log_mel_to_linear(cur, floor=self.model.hp.log_mel_floor)
        enhanced_linear = apply_mask(mask, cur_linear)
        enhanced = linear_to_log_mel(enhanced_linear, floor=self.model.hp.log_mel_floor)
        return enhanced, hN, cN

app = App(name="onnx", help_format="plaintext")


@app.default
def main(
    checkpoint: Annotated[str, Parameter(help="Path to the trained VoiceFilter-Lite ckpt.")],
    *,
    output: Annotated[str, Parameter(help="Output ONNX file path.")] = "voicefilter_lite.onnx",
    opset: Annotated[int, Parameter(help="ONNX opset version.")] = 17,
    quantize: Annotated[
        Literal["none", "int8"],
        Parameter(
            help="Post-training quantization to apply after torch.onnx.export. "
                 "'int8' mirrors paper §3.5's 8-bit deployed model and writes "
                 "<output>.int8.onnx via onnxruntime.quantization.quantize_dynamic.",
        ),
    ] = "none",
) -> None:
    """Export VoiceFilter-Lite to ONNX (streaming-friendly variant)."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hp = HParams.from_dict(ckpt["hparams"])

    model = VoiceFilterLite(hp)
    sd = ckpt["model_state"]
    # Backward compat: legacy ckpts may carry (a) `dvec_proj.*` from the
    # pre-refactor 192→256 projection Linear that the current model no
    # longer has, and (b) extra `alpha_head.*` keys from a 3-Linear head
    # (Linear→64 → Linear→64 → Linear→1) when the current model is the
    # paper-faithful 2-Linear head (Linear→64 → Linear→1, paper §4.2
    # "two feedforward layers with 64 nodes"). Drop anything that doesn't
    # exist in the current model OR has a shape mismatch so re-export
    # doesn't crash; the affected modules re-init from scratch.
    model_sd = model.state_dict()
    kept = {k: v for k, v in sd.items()
            if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape)}
    missing, unexpected = model.load_state_dict(kept, strict=False)
    if unexpected:
        raise RuntimeError(f"[export] unexpected keys beyond compat drop: {unexpected[:5]}…")
    if any(k.startswith("alpha_head.") for k in missing):
        print(
            f"[export] dropped legacy alpha_head.* keys (ckpt has older "
            f"head shape; current model uses paper §4.2 2-Linear head)"
        )
    model.eval()

    wrapper = StreamWrapper(model).eval()

    # Dummy inputs. Time dimension is dynamic.
    B = 1
    T = 8  # arbitrary; ONNX marks this dynamic
    stacked = torch.randn(B, T, hp.n_mels * hp.frame_stack)
    dvec = torch.randn(B, hp.d_vector_dim)
    h0 = torch.zeros(hp.num_lstm_layers, B, hp.hidden_dim)
    c0 = torch.zeros(hp.num_lstm_layers, B, hp.hidden_dim)

    # We export the eager nn.Module directly. Modern torch.onnx goes through
    # torch.export which doesn't accept ScriptModule objects, so we skip the
    # torch.jit.trace step.
    traced = wrapper

    # Export to ONNX. Use dynamic_shapes (newer PyTorch API).
    # Note: dynamic_shapes keys are INPUT names only; outputs inherit the
    # same dynamic dim labels.
    dynamic_shapes = {
        "stacked_logmel": {0: "batch", 1: "time"},
        "dvec":           {0: "batch"},
        "h0":             {1: "batch"},
        "c0":             {1: "batch"},
    }
    try:
        torch.onnx.export(
            traced,
            (stacked, dvec, h0, c0),
            output,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["stacked_logmel", "dvec", "h0", "c0"],
            output_names=["enhanced", "hN", "cN"],
            dynamic_shapes=dynamic_shapes,
        )
    except TypeError:
        # Older PyTorch fallback: dynamic_axes only.
        torch.onnx.export(
            traced,
            (stacked, dvec, h0, c0),
            output,
            opset_version=opset,
            do_constant_folding=True,
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
    print(f"[export] wrote {output}")

    # ---- Optional INT8 dynamic quantization (paper §3.5) ----
    if quantize == "int8":
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError:
            raise SystemExit(
                "[export] --quantize int8 requires onnxruntime >= 1.13. "
                "Install with `pip install onnxruntime`."
            )
        int8_path = output
        if not int8_path.endswith(".onnx"):
            int8_path = int8_path + ".onnx"
        int8_out = int8_path[:-len(".onnx")] + ".int8.onnx"
        quantize_dynamic(
            model_input=int8_path,
            model_output=int8_out,
            weight_type=QuantType.QInt8,
        )
        print(f"[export] quantized to INT8 → {int8_out}")

    # Verify with onnxruntime.
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(output, providers=["CPUExecutionProvider"])
        outs = sess.run(
            None,
            {
                "stacked_logmel": stacked.numpy(),
                "dvec": dvec.numpy(),
                "h0": h0.numpy(),
                "c0": c0.numpy(),
            },
        )
        with torch.no_grad():
            ref_enhanced, ref_h, ref_c = wrapper(stacked, dvec, h0, c0)
        for name, got, want in zip(
            ["enhanced", "hN", "cN"], outs, [ref_enhanced, ref_h, ref_c]
        ):
            err = float(np.abs(got - want.numpy()).max())
            print(f"[export] onnxruntime vs torch diff {name}: {err:.3e}")
            assert err < 1e-3, f"{name} mismatch {err}"
        print("[export] OK")
    except ImportError:
        print("[export] onnxruntime not installed — skipping verification")


if __name__ == "__main__":
    app()