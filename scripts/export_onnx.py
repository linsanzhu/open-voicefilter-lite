#!/usr/bin/env python3
"""Export a VoiceFilter-Lite checkpoint to ONNX (streaming-friendly variant).

The exported model takes:
    stacked_logmel: (B=1, T, n_mels * frame_stack)
    dvec:           (B=1, d_vector_dim)         — pre-projected (raw ECAPA 192 is not
                                                    in the graph; callers must run
                                                    ECAPA on the enrollment audio
                                                    and pass the projected 256-d
                                                    vector through `dvec_proj`).
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
        dvec_proj: torch.Tensor,
        h0: torch.Tensor,
        c0: torch.Tensor,
    ):
        # Concat input projection (so the dvec projection stays outside).
        B, T, _ = stacked_logmel.shape
        dvec_tiled = dvec_proj.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([stacked_logmel, dvec_tiled], dim=-1)
        x = self.model.in_proj(x)
        x = self.model.in_norm(x)
        x, (hN, cN) = self.model.lstm(x, (h0, c0))
        mask = torch.sigmoid(self.model.out_proj(x))

        cur_idx = self.model.hp.frame_stack - 1
        cur = stacked_logmel[..., cur_idx * self.model.hp.n_mels:(cur_idx + 1) * self.model.hp.n_mels]
        enhanced = mask * cur
        return enhanced, hN, cN


def main() -> None:
    p = argparse.ArgumentParser(description="Export VoiceFilter-Lite to ONNX")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="voicefilter_lite.onnx")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hp = HParams.from_dict(ckpt["hparams"])

    model = VoiceFilterLite(hp)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    wrapper = StreamWrapper(model).eval()

    # Dummy inputs. Time dimension is dynamic.
    B = 1
    T = 8  # arbitrary; ONNX marks this dynamic
    stacked = torch.randn(B, T, hp.n_mels * hp.frame_stack)
    dvec_proj = torch.randn(B, hp.d_vector_dim)
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
        "dvec_proj":      {0: "batch"},
        "h0":             {1: "batch"},
        "c0":             {1: "batch"},
    }
    try:
        torch.onnx.export(
            traced,
            (stacked, dvec_proj, h0, c0),
            args.output,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["stacked_logmel", "dvec_proj", "h0", "c0"],
            output_names=["enhanced", "hN", "cN"],
            dynamic_shapes=dynamic_shapes,
        )
    except TypeError:
        # Older PyTorch fallback: dynamic_axes only.
        torch.onnx.export(
            traced,
            (stacked, dvec_proj, h0, c0),
            args.output,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["stacked_logmel", "dvec_proj", "h0", "c0"],
            output_names=["enhanced", "hN", "cN"],
            dynamic_axes={
                "stacked_logmel": {0: "batch", 1: "time"},
                "dvec_proj":      {0: "batch"},
                "h0":             {1: "batch"},
                "c0":             {1: "batch"},
                "enhanced":       {0: "batch", 1: "time"},
                "hN":             {1: "batch"},
                "cN":             {1: "batch"},
            },
        )
    print(f"[export] wrote {args.output}")

    # Verify with onnxruntime.
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        outs = sess.run(
            None,
            {
                "stacked_logmel": stacked.numpy(),
                "dvec_proj": dvec_proj.numpy(),
                "h0": h0.numpy(),
                "c0": c0.numpy(),
            },
        )
        with torch.no_grad():
            ref_enhanced, ref_h, ref_c = wrapper(stacked, dvec_proj, h0, c0)
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
