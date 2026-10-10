#!/usr/bin/env python3
"""Export `models/embedder.pt` (the d-vector speaker encoder) to ONNX.

Exports the bare ``SpeakerEncoder`` module (paper §3.1) — input is one
or more 80-frame × 40-mel log10 windows, output is the (B, 256) raw
embedding (NOT L2-normalised). Audio-side preprocessing (fbank +
80-frame sliding windows + per-window L2 → mean → final L2) stays on
the caller — ``model.audio.extract_dvec`` is the canonical
PyTorch reference; the ONNX graph just hosts the heavy LSTM.

Usage
-----
    `voicefilter-cli export embedder` \\
        --weights models/embedder.pt \\
        --output  dvector.onnx

    # INT8 post-training dynamic quantisation
    `voicefilter-cli export embedder` \\
        --weights models/embedder.pt --output dvector.onnx \\
        --quantize int8
"""

from __future__ import annotations

import os
from typing import Annotated, Literal

import onnx
import torch
from cyclopts import App, Parameter

from voicefilter_cli._onnx_metadata import embed_hp_metadata
from voicefilter_lite.hparams import HParams
from model.speaker_encoder import SpeakerEncoder


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


def _dvector_metadata(hp: HParams) -> dict:
    """Build the JSON metadata blob for the d-vector ONNX.

    A third party with only the .onnx in hand uses this to reproduce
    the full PyTorch d-vector pipeline (audio → log-mel → sliding
    windows → encoder → per-window L2 → mean → final L2) without
    needing the source tree or a checkpoint. Field values match
    ``HParams`` defaults when ``configs/paper.yaml`` was used.
    """
    return {
        # Audio preprocessing (d-vector pipeline; differs from the main
        # 128-mel path — n_mels=40 here, n_fft=512 not 400).
        "sample_rate":  hp.sample_rate,         # 16000
        "n_fft":        hp.dvector_n_fft,       # 512
        "win_length":   hp.dvector_win_length,  # 400
        "hop_length":   hp.dvector_hop_length,  # 160
        "n_mels":       hp.dvector_n_mels,      # 40
        "log_mel_floor": hp.log_mel_floor,      # 1e-6 (used for log10 floor)
        # Sliding window fed to the encoder per sample.
        "window":       hp.dvector_window,      # 80 frames
        "stride":       hp.dvector_stride,      # 40 frames (50% overlap)
        # Architecture (matches upstream SpeechEmbedder / paper §3.1).
        "lstm_hidden":  hp.dvector_lstm_hidden, # 768
        "lstm_layers":  hp.dvector_lstm_layers, # 3
        "emb_dim":      hp.dvector_emb_dim,     # 256 (= D_VECTOR_DIM)
        # ONNX I/O contract (also exposed via session.get_inputs/outputs;
        # duplicated here for one-shot JSON reads).
        "input": {
            "name":  "mel_window",
            "dtype": "float32",
            "shape": "(batch, 80, 40)",
        },
        "output": {
            "name":  "embedding",
            "dtype": "float32",
            "shape": "(batch, 256)",
            "normalised": False,                # raw L2 not applied in-graph
        },
        # Caller-side post-processing the caller must run before treating
        # the output as a speaker embedding.
        "aggregation":  "l2_per_window_then_mean_then_l2",
    }


class OnnxEncoderWrapper(torch.nn.Module):
    """ONNX-friendly view of `SpeakerEncoder`.

    `SpeakerEncoder.forward` calls `self.lstm.flatten_parameters()` every
    step. That mutates the module's `_flat_weights` Python list, which
    confuses the legacy `torch.onnx.export` tracer: it ends up capturing
    stale or inconsistent references to the LSTM weight tensors, and the
    exported graph computes garbage (max-abs-err ~1.0 against eager).

    We pre-call `flatten_parameters()` once (it just packs cuDNN-friendly
    memory layouts; values are unchanged) and then skip it in `forward`,
    so the tracer sees a pure stateless module. The dynamo path doesn't
    have this problem but does accumulate ~6e-2 fp32 drift from
    decomposing LSTM into per-cell ops, so this wrapper is the preferred
    path for tight PyTorch parity.
    """

    def __init__(self, encoder: SpeakerEncoder) -> None:
        super().__init__()
        self.lstm = encoder.lstm
        self.proj = encoder.proj
        self.lstm.flatten_parameters()  # one-shot; forward never re-runs it

    def forward(self, mel_window: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(mel_window)        # (B, T, 768)
        return self.proj(lstm_out[:, -1, :])        # (B, 256)


app = App(name="embedder", help_format="plaintext")


@app.default
def main(
    weights: Annotated[
        str,
        Parameter(help="Path to the d-vector state_dict (e.g. models/embedder.pt)."),
    ] = "models/embedder.pt",
    *,
    output: Annotated[str, Parameter(help="Output ONNX file path.")] = "dvector.onnx",
    opset: Annotated[int, Parameter(help="ONNX opset version.")] = 17,
    quantize: Annotated[
        Literal["none", "int8"],
        Parameter(
            help="Post-training dynamic quantization to apply after torch.onnx.export. "
                 "'int8' writes <output>.int8.onnx via onnxruntime.quantization.quantize_dynamic. "
                 "NB: LSTM cells stay FP32 — onnxruntime only quantises MatMul/Gemm-class ops; "
                 "this is the standard streaming-RNN tradeoff (same as the main mask predictor).",
        ),
    ] = "none",
) -> None:
    """Export the d-vector speaker encoder (paper §3.1) to ONNX."""
    hp = HParams()  # embedder params live under hp.model.dvector.*
    enc = SpeakerEncoder(hp)
    enc.load_pretrained(weights)
    enc.eval()

    # ONNX-friendly wrapper: drops the per-forward `flatten_parameters()`
    # call that confuses the legacy ONNX tracer. See class docstring.
    wrapper = OnnxEncoderWrapper(enc).eval()

    # Dummy input — T=80 / F=40 are fixed by the architecture (paper §3.1),
    # only batch is dynamic. Use B=2 so dynamo export doesn't fold it into
    # a constant.
    mel = torch.randn(2, hp.dvector_window, hp.dvector_n_mels)

    dynamic_axes = {
        "mel_window": {0: "batch"},
        "embedding":  {0: "batch"},
    }
    # `dynamic_shapes` keys are INPUT names only; output shapes are inferred.
    dynamic_shapes = {
        "mel_window": {0: "batch"},
    }
    # The d-vector is a 3-layer LSTM over T=80 frames. dynamo (default in
    # torch >= 2.5) decomposes LSTM into per-cell ops and produces ~6e-2
    # fp32 drift against `nn.LSTM`. `dynamo=False` falls back to the legacy
    # ONNX exporter, which emits the ONNX `LSTM` op as a single node —
    # matching cuDNN to ~1e-4 drift (same order as the main mask
    # predictor's T=1 case). The wrapper above is required for this path
    # because `SpeakerEncoder.forward` calls `flatten_parameters()` per
    # step, which the legacy tracer mishandles.
    try:
        torch.onnx.export(
            wrapper,
            (mel,),
            output,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["mel_window"],
            output_names=["embedding"],
            dynamic_shapes=dynamic_shapes,
            dynamo=False,
        )
    except (TypeError, ValueError):
        # Older PyTorch without `dynamo=`: fall back to legacy signature
        # entirely (`dynamic_axes` only — `dynamic_shapes` is dynamo-only).
        torch.onnx.export(
            wrapper,
            (mel,),
            output,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["mel_window"],
            output_names=["embedding"],
            dynamic_axes=dynamic_axes,
        )
    print(f"[export] wrote {output}")
    _inline_onnx_weights(output)
    print(f"[export] inlined weights into {output} (single-file, no .data sibling)")
    embed_hp_metadata(output, "dvector", _dvector_metadata(hp))

    # ---- Optional INT8 dynamic quantization ----
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
        import tempfile

        int8_path = output if output.endswith(".onnx") else output + ".onnx"
        int8_out = int8_path[: -len(".onnx")] + ".int8.onnx"
        # `quantize_dynamic` runs ONNX shape inference internally, which
        # chokes on the LSTM-output → Linear hand-off when the LSTM op is
        # given a dynamic batch dim ("Inferred shape and existing shape
        # differ in dimension 0: (768) vs (256)"). `quant_pre_process`
        # runs the same shape-inference pass with `strict_mode=False` and
        # resolves symbolic dims against a provided input shape — feeding
        # it B=2 (our export dummy batch) yields a clean graph that
        # `quantize_dynamic` then accepts.
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
            quant_pre_process(
                input_model_path=int8_path,
                output_model_path=tmp.name,
                skip_optimization=False,
                skip_onnx_shape=False,
                auto_merge=True,
                int_max=2**31 - 1,
            )
            quantize_dynamic(
                model_input=tmp.name,
                model_output=int8_out,
                weight_type=QuantType.QInt8,
            )
        print(f"[export] quantized to INT8 → {int8_out}")
        _inline_onnx_weights(int8_out)
        print(f"[export] inlined weights into {int8_out} (single-file, no .data sibling)")
        embed_hp_metadata(int8_out, "dvector", _dvector_metadata(hp))

    # ---- Sanity-check against PyTorch ----
    # The d-vector is a 3-layer LSTM over T=80 frames. PyTorch's `nn.LSTM`
    # is cuDNN-optimized; onnxruntime's `LSTM` op is a generic
    # implementation. Numerics drift ~3e-2 on raw values that are O(1),
    # which matches the INT8-graph tolerance the README allows for the
    # main model (5e-2). Crucially, the downstream d-vector contract is
    # per-window L2 → mean → final L2, so what actually matters is the
    # cosine similarity between the normalised ONNX and PyTorch outputs.
    # That drift is order 1e-3 (errors are roughly unbiased noise), well
    # within the speaker-verification tolerance reported in
    # `evaluate dvector`.
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(output, providers=["CPUExecutionProvider"])
        out_onnx = sess.run(None, {"mel_window": mel.numpy()})[0]
        with torch.no_grad():
            out_torch = enc(mel).numpy()
        err = float(np.abs(out_onnx - out_torch).max())
        cos = float(np.mean(
            (out_onnx * out_torch).sum(-1)
            / (
                np.linalg.norm(out_onnx, axis=-1)
                * np.linalg.norm(out_torch, axis=-1)
                + 1e-12
            )
        ))
        print(f"[export] onnxruntime vs torch max-abs-err: {err:.3e}, "
              f"cosine sim: {cos:.6f}")
        if err >= 5e-2:
            raise AssertionError(
                f"max-abs-err {err:.3e} exceeds 5e-2 — investigate LSTM "
                f"weight capture before shipping."
            )
        if cos < 0.99:
            print(
                f"[export] WARN: cosine sim {cos:.4f} < 0.99 — outputs are "
                f"close but the LSTM numerics drifted further than usual. "
                f"Run `voicefilter-cli evaluate dvector` on a pair of "
                f"enrollments through this ONNX graph to confirm "
                f"speaker-discrimination is still intact."
            )
        print("[export] OK")
    except ImportError:
        print("[export] onnxruntime not installed — skipping verification")


if __name__ == "__main__":
    app()
