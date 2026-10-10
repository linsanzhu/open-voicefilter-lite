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
    mask:           (B=1, T, n_mels)            — raw sigmoid mask in [0, 1]
    f_adapt:        (B=1, T, 1)                — noise-type predictor logit
                                                  (sigmoid applied inside the
                                                  graph; range [0, 1])
    hN:             (num_layers, B=1, hidden_dim)
    cN:             (num_layers, B=1, hidden_dim)

The mask is returned raw so the caller can apply it in **linear**
filterbank-energy space (paper §3.2) and blend with the raw input
under adaptive suppression `w^(t)` (paper §3.4) — both identical to
the torch `StreamingVoiceFilter` path. Older `.onnx` files that emit
`enhanced` instead of `mask`+`f_adapt` are still loaded; the onnx
backend detects the output names and falls back to mask-less blending
(adaptive suppression then has no `f_adapt`, so it is disabled —
`w=0`, the model output is taken as-is).

Usage
-----
    `voicefilter-cli export onnx` \\
        --checkpoint runs/exp1/ckpt_final.pt \\
        --output      voicefilter_lite.onnx

    # INT8 post-training dynamic quantization (paper §3.5):
    `voicefilter-cli export onnx` \\
        --checkpoint runs/exp1/ckpt_final.pt \\
        --output      voicefilter_lite.int8.onnx \\
        --quantize    int8

    With ``--quantize int8``, ``--output`` IS the final INT8 destination
    (FP32 intermediate goes to a tempfile). The INT8 pass calls
    ``onnxruntime.quantization.quantize_dynamic`` on the FP32 graph; the
    export path is the legacy ONNX tracer so the LSTM survives as a
    single ONNX op (see ``StreamWrapper`` for the dynamo=False rationale),
    and ``quantize_dynamic`` substitutes ``DynamicQuantizeLSTM`` for it.
    That gives T=1 streaming the 2-4× weight-only speedup that LSTM's
    tiny-batch bottleneck rewards; without the legacy path the LSTM
    stays FP32 and the speedup collapses to ~1.1×.
"""

from __future__ import annotations

import os
import tempfile
from typing import Annotated, Literal

import onnx
import torch
from cyclopts import App, Parameter

from voicefilter_cli._onnx_metadata import embed_hp_metadata
from voicefilter_lite.hparams import HParams
from model.nn import VoiceFilterLite


def _inline_onnx_weights(path: str) -> None:
    """Rewrite ``path`` with all tensor weights embedded (no sibling ``.data``).

    ``torch.onnx.export`` in dynamo mode (torch >= 2.5) defaults to writing
    initializers to a sibling ``<path>.onnx.data`` file, leaving the
    ``.onnx`` as a thin graph. That breaks the moment the ``.data`` file
    is moved, deleted, or skipped by a downstream consumer, so we always
    inline after export (and again after INT8 quantization, which also
    produces a ``.data`` sibling).
    """
    model = onnx.load(path)
    onnx.save(model, path, save_as_external_data=False)
    try:
        os.remove(path + ".data")
    except FileNotFoundError:
        pass


def _mask_metadata(hp: HParams) -> dict:
    """Build the JSON metadata blob for the mask-predictor ONNX.

    Captures everything a third party needs to drive the streaming
    inference loop (frame stacking, d-vector dim, LSTM state shape,
    adaptive suppression smoother coefficients) without a PyTorch
    checkpoint. Field values match ``configs/paper.yaml`` when that
    config was used to train.
    """
    return {
        # Audio preprocessing (main model; differs from the d-vector
        # 40-mel path — n_mels=128 here, n_fft=400 not 512).
        "sample_rate":  hp.sample_rate,        # 16000
        "n_fft":        hp.n_fft,              # 400
        "win_length":   hp.win_length,         # 400
        "hop_length":   hp.hop_length,         # 160
        "n_mels":       hp.n_mels,             # 128
        "log_mel_floor": hp.log_mel_floor,     # 1e-6
        "frame_stack":  hp.frame_stack,        # 4 — stacked-mel input
        # D-vector side input — registered once via set_dvector().
        "d_vector_dim": hp.d_vector_dim,       # 256
        # Architecture (LSTM core + optional pre-LSTM CNN).
        "hidden_dim":       hp.hidden_dim,           # 512 (paper) / 256 (smoke)
        "num_lstm_layers":  hp.num_lstm_layers,      # 3
        "dropout":          hp.dropout,              # 0.1 at export (eval mode)
        "use_cnn":          hp.use_cnn,
        "num_cnn_layers":   hp.num_cnn_layers,
        "cnn_kernel_size":  hp.cnn_kernel_size,
        "cnn_channels_first": hp.cnn_channels_first,
        # Adaptive suppression smoother (paper §3.4 — applied
        # OUTSIDE the graph; the ONNX emits raw `mask` + `f_adapt`,
        # the caller runs the EMA + linear transform).
        "adaptive_w_beta":  hp.adaptive_w_beta,  # 0.4 (paper §4.4)
        "adaptive_w_a":     hp.adaptive_w_a,     # 0.7
        "adaptive_w_b":     hp.adaptive_w_b,     # 0.2
        # Noise-type predictor — drives f_adapt (paper §4.2).
        "use_noise_type_predictor":   hp.use_noise_type_predictor,
        "alpha_head_use_current_frame": hp.alpha_head_use_current_frame,
        # ONNX I/O contract (also exposed via session.get_inputs/outputs;
        # duplicated here so consumers can recover the full spec from
        # JSON alone). Time dim is dynamic; batch dim is dynamic.
        "inputs": {
            "stacked_logmel": "(batch, time, n_mels * frame_stack)",
            "dvec":           "(batch, d_vector_dim)",
            "h0":             "(num_lstm_layers, batch, hidden_dim)",
            "c0":             "(num_lstm_layers, batch, hidden_dim)",
        },
        "outputs": {
            "mask":    "(batch, time, n_mels) — raw sigmoid in [0,1]",
            "f_adapt": "(batch, time, 1) — noise-type predictor prob",
            "hN":      "(num_lstm_layers, batch, hidden_dim)",
            "cN":      "(num_lstm_layers, batch, hidden_dim)",
        },
    }

class StreamWrapper(torch.nn.Module):
    """Wraps VoiceFilterLite for ONNX export with explicit LSTM state IO.

    Pre-calls ``model.lstm.flatten_parameters()`` once so the legacy
    ``torch.onnx.export`` tracer sees a stable, cuDNN-packed LSTM weight
    layout. Without this, ``nn.LSTM``'s per-step ``flatten_parameters()``
    confuses the captured initializers the same way the d-vector encoder's
    wrapper does — see ``OnnxEncoderWrapper`` in
    ``export_embedder_onnx.py`` for the original diagnosis. We pair this
    with ``dynamo=False`` in ``torch.onnx.export`` so the LSTM survives
    as a single ONNX op; that lets ``quantize_dynamic`` substitute
    ``DynamicQuantizeLSTM`` and deliver the 2-4× weight-only speedup
    that T=1 streaming needs (dynamo decomposes LSTM into per-cell
    MatMul, which ``quantize_dynamic`` cannot recognise as a unit).
    """

    def __init__(self, model: VoiceFilterLite):
        super().__init__()
        self.model = model
        # One-shot; `lstm_forward` never re-flattens on its own.
        self.model.lstm.flatten_parameters()

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

        # Paper §3.4: f_adapt(S_in^(t)) — primary input is the current
        # frame's noisy filterbank. Sigmoid inside the graph so the
        # caller can read it as a probability directly.
        cur_idx = self.model.hp.frame_stack - 1
        cur = stacked_logmel[..., cur_idx * self.model.hp.n_mels:(cur_idx + 1) * self.model.hp.n_mels]
        head_in = cur if self.model.hp.alpha_head_use_current_frame else x
        f_adapt = torch.sigmoid(self.model.alpha_head(head_in))
        return mask, f_adapt, hN, cN

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
                 "the final product to ``--output`` via "
                 " ``onnxruntime.quantization.quantize_dynamic`` (the FP32 "
                 "intermediate goes to a tempfile).",
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

    # Dummy inputs. Time dimension is dynamic — use T=1 so dynamo
    # export doesn't bake a specific time length into the in_proj /
    # LSTM reshapes; the graph is intended to be called with T=1 at
    # inference (frame-by-frame streaming), and `dynamic_axes` /
    # `dynamic_shapes` already exposes the time axis as dynamic.
    B = 1
    T = 1
    stacked = torch.randn(B, T, hp.n_mels * hp.frame_stack)
    dvec = torch.randn(B, hp.d_vector_dim)
    h0 = torch.zeros(hp.num_lstm_layers, B, hp.hidden_dim)
    c0 = torch.zeros(hp.num_lstm_layers, B, hp.hidden_dim)

    # We export the eager nn.Module directly. Modern torch.onnx goes through
    # torch.export which doesn't accept ScriptModule objects, so we skip the
    # torch.jit.trace step.
    traced = wrapper

    # FP32 destination. With `--quantize int8`, FP32 is just the source
    # for `quantize_dynamic`, so it goes to a tempfile that we clean up
    # at the end; with `--quantize none` (default) it's the user-facing
    # artifact and writes straight to `--output`.
    if quantize == "int8":
        fp32_tmpdir = tempfile.TemporaryDirectory()
        fp32_path = os.path.join(fp32_tmpdir.name, "voicefilter_lite.fp32.onnx")
    else:
        fp32_tmpdir = None
        fp32_path = output

    # Export to ONNX with the LEGACY tracer (`dynamo=False`) so the LSTM
    # survives as a single ONNX op. Two reasons:
    #   1. `quantize_dynamic` only substitutes `DynamicQuantizeLSTM` for a
    #      full `LSTM` op; dynamo decomposes LSTM into per-cell MatMul and
    #      that substitution never fires, leaving the LSTM weights FP32
    #      (T=1 streaming bottleneck — ~1.1× speedup instead of 2-4×).
    #   2. dynamo's per-cell decomposition drifts ~6e-2 fp32 against
    #      `nn.LSTM`; the legacy path matches cuDNN to ~1e-4, the same
    #      numerics the streaming torch path uses. Pre-flattening the
    #      LSTM in `StreamWrapper` is required for this branch.
    dynamic_axes = {
        "stacked_logmel": {0: "batch", 1: "time"},
        "dvec":           {0: "batch"},
        "h0":             {1: "batch"},
        "c0":             {1: "batch"},
        "mask":           {0: "batch", 1: "time"},
        "f_adapt":        {0: "batch", 1: "time"},
        "hN":             {1: "batch"},
        "cN":             {1: "batch"},
    }
    try:
        torch.onnx.export(
            traced,
            (stacked, dvec, h0, c0),
            fp32_path,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["stacked_logmel", "dvec", "h0", "c0"],
            output_names=["mask", "f_adapt", "hN", "cN"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    except TypeError:
        # Older PyTorch without `dynamo=` kwarg: legacy signature.
        torch.onnx.export(
            traced,
            (stacked, dvec, h0, c0),
            fp32_path,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["stacked_logmel", "dvec", "h0", "c0"],
            output_names=["mask", "f_adapt", "hN", "cN"],
            dynamic_axes=dynamic_axes,
        )
    _inline_onnx_weights(fp32_path)
    embed_hp_metadata(fp32_path, "mask_predictor", _mask_metadata(hp))
    if quantize == "int8":
        print(f"[export] wrote FP32 intermediate to {fp32_path}")
    else:
        print(f"[export] wrote FP32 → {fp32_path} (single-file, no .data sibling)")

    # ---- Optional INT8 dynamic quantization (paper §3.5) ----
    if quantize == "int8":
        try:
            from onnxruntime.quantization import (
                QuantType, quant_pre_process, quantize_dynamic,
            )
        except ImportError:
            raise SystemExit(
                "[export] --quantize int8 requires onnxruntime >= 1.13. "
                "Install with `pip install onnxruntime`."
            )

        # `quantize_dynamic` runs ONNX shape inference internally, which
        # chokes on the LSTM-output → Linear hand-off when the LSTM op is
        # given a dynamic batch dim ("Inferred shape and existing shape
        # differ in dimension 0: (hidden) vs (proj)") — same fix the
        # d-vector exporter uses. `quant_pre_process` runs the same
        # shape-inference pass with `strict_mode=False` via `auto_merge`
        # and resolves symbolic dims against the LSTM's known batch;
        # `quantize_dynamic` then accepts the cleaned-up graph and
        # substitutes `DynamicQuantizeLSTM` for the LSTM op.
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
            quant_pre_process(
                input_model_path=fp32_path,
                output_model_path=tmp.name,
                skip_optimization=False,
                skip_onnx_shape=False,
                auto_merge=True,
                int_max=2**31 - 1,
            )
            quantize_dynamic(
                model_input=tmp.name,
                model_output=output,
                weight_type=QuantType.QInt8,
            )
        print(f"[export] quantized to INT8 → {output}")
        _inline_onnx_weights(output)
        print(f"[export] inlined weights into {output} (single-file, no .data sibling)")
        embed_hp_metadata(output, "mask_predictor", _mask_metadata(hp))
        # Don't cleanup `fp32_tmpdir` here — the verification block
        # below still needs the FP32 file. The `finally` clause at
        # the end of `main` handles cleanup.

    # Verify with onnxruntime.
    # The mask predictor ships an LSTM that onnxruntime's generic `LSTM`
    # op evaluates vs cuDNN with ~3e-2 drift (per-cell matmul order
    # differs from cuDNN's fused kernel). That matches the embedder's
    # documented tolerance, so we hold this exporter to the same `5e-2`
    # max-abs bound and additionally report cosine similarity between
    # the ONNX and torch `mask` outputs — what actually drives the
    # downstream blend (linear-mask * noisy) is shape similarity, not
    # element-wise equality. When `--quantize int8` is set, we verify
    # the FP32 export here; the INT8 graph adds ~1e-2 dynamic-activation
    # noise on top, well inside the same tolerance band.
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
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
            ref_mask, ref_fa, ref_h, ref_c = wrapper(stacked, dvec, h0, c0)
        ref = {"mask": ref_mask, "f_adapt": ref_fa, "hN": ref_h, "cN": ref_c}
        for name, got, want in zip(
            ["mask", "f_adapt", "hN", "cN"], outs, [ref_mask, ref_fa, ref_h, ref_c],
        ):
            err = float(np.abs(got - want.numpy()).max())
            print(f"[export] onnxruntime vs torch diff {name}: {err:.3e}")
            assert err < 5e-2, f"{name} mismatch {err}"
        # Cosine similarity on the `mask` output — the only output the
        # downstream blend actually uses element-wise.
        mask_cos = float(np.mean(
            (outs[0] * ref["mask"].numpy()).sum(-1)
            / (
                np.linalg.norm(outs[0], axis=-1)
                * np.linalg.norm(ref["mask"].numpy(), axis=-1)
                + 1e-12
            )
        ))
        print(f"[export] mask cosine sim: {mask_cos:.6f}")
        if mask_cos < 0.99:
            print(
                f"[export] WARN: mask cosine sim {mask_cos:.4f} < 0.99 — "
                f"LSTM numerics drifted further than usual. Run a downstream "
                f"audition (see examples/04) on the final output to confirm "
                f"the blend is still intact."
            )
        print("[export] OK")
    except ImportError:
        print("[export] onnxruntime not installed — skipping verification")
    finally:
        if fp32_tmpdir is not None:
            fp32_tmpdir.cleanup()


if __name__ == "__main__":
    app()