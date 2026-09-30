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
    python scripts/export_onnx.py \\
        --checkpoint runs/exp1/ckpt_final.pt \\
        --output      voicefilter_lite.onnx

    # INT8 post-training dynamic quantization (paper §3.5):
    python scripts/export_onnx.py \\
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

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from open_voicefilter_lite.hparams import HParams
from open_voicefilter_lite.model import VoiceFilterLite


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
        B, T, _ = stacked_logmel.shape
        dvec_tiled = dvec.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([stacked_logmel, dvec_tiled], dim=-1)
        x = self.model.in_proj(x)
        x = self.model.in_norm(x)
        # Optional 1D CNN (paper §3.2). Mirrors `VoiceFilterLite.forward` —
        # `nn.Conv1d` sliding along the frequency axis (H positions are
        # Conv1d's L axis after the `(B, T, H) → (B*T, 1, H)` reshape),
        # then mean-pool over the preserved H positions. When
        # `hp.use_cnn=False` this branch is a no-op and the exported
        # graph matches the legacy LSTM-only variant.
        if self.model.cnn is not None:
            x_per = x.reshape(-1, 1, x.shape[-1])              # (B*T, 1, H)
            x_conv = self.model.cnn(x_per)                     # (B*T, cnn_channels, H)
            x = x_conv.mean(dim=-1).reshape(x.shape[0], x.shape[1], -1)
        x, (hN, cN) = self.model.lstm(x, (h0, c0))
        mask = torch.sigmoid(self.model.out_proj(x))

        cur_idx = self.model.hp.frame_stack - 1
        cur = stacked_logmel[..., cur_idx * self.model.hp.n_mels:(cur_idx + 1) * self.model.hp.n_mels]
        # Paper §3.2 literal mask application (Y_enh = M · Y in log-mel).
        enhanced = self.model._apply_mask(mask, cur)
        return enhanced, hN, cN


def main() -> None:
    p = argparse.ArgumentParser(description="Export VoiceFilter-Lite to ONNX")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="voicefilter_lite.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument(
        "--quantize",
        choices=["none", "int8"],
        default="none",
        help="Post-training quantization to apply after torch.onnx.export. "
             "'int8' mirrors paper §3.5's 8-bit deployed model and writes "
             "<output>.int8.onnx via onnxruntime.quantization.quantize_dynamic.",
    )
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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
            args.output,
            opset_version=args.opset,
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
            args.output,
            opset_version=args.opset,
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
    print(f"[export] wrote {args.output}")

    # ---- Optional INT8 dynamic quantization (paper §3.5) ----
    if args.quantize == "int8":
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError:
            raise SystemExit(
                "[export] --quantize int8 requires onnxruntime >= 1.13. "
                "Install with `pip install onnxruntime`."
            )
        int8_path = args.output
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

        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
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
    main()