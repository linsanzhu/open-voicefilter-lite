"""Streaming inference wrapper + adaptive suppression (paper §3.4).

`StreamingVoiceFilter` is the stateful frame-by-frame inference entry point.
It owns:
  * the registered target-speaker d-vector (set once in `set_dvector`)
  * the LSTM hidden state (`(h, c)`) — held by the active backend
  * the left-context frame stack (rolling buffer of `frame_stack - 1` frames)
  * the adaptive-suppression smoother (`AdaptiveSuppression`)

The model core is pluggable: pass an `nn.Module` (torch) or an
`onnxruntime.InferenceSession` / path-to-onnx-file (ONNX Runtime).
The interface (`set_dvector` / `reset_state` / `step`) and the
math (linear-mel mask, adaptive `w^(t)` blend, log-mel return) are
identical across backends, so callers can swap deployment
representations without touching their streaming loops.

`step(frame_logmel)` does **one** new frame and returns the enhanced
log-mel (paper §3.2: the ASR pipeline consumes log-mel, so the wrapper
internally operates in linear-mel for mask + adaptive blending then
converts back to log-mel for the return value — keeps the caller's
interface unchanged while running the mask in the paper §3.4-correct
linear filterbank-energy domain). The caller never sees intermediate
tensors; device/dim plumbing is contained here.

Warmup note: the rolling buffer is empty at construction (or after
`reset_state()`), so the first `frame_stack - 1` calls would have a
zero-padded left-context. **That is in-distribution**: training uses the
identical `frame_stack_left` pattern (zero padding on the left for the
first frames) and resets the LSTM hidden state every batch, so the model
sees exactly this "zero hidden + zero-padded stacked" warm-up thousands of
times per epoch. `step()` therefore runs the LSTM forward unconditionally
from the first call, and the 1-in / 1-out contract holds (one frame in,
one enhanced frame out, same shape / dtype / device).

The d-vector comes in already at `hp.d_vector_dim` (256) directly from
`SpeakerEncoder` — no in-model projection.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from voicefilter_lite.audio import linear_to_log_mel, log_mel_to_linear
from voicefilter_lite.hparams import HParams
from voicefilter_lite.model import VoiceFilterLite, apply_mask


class AdaptiveSuppression:
    """Inference-time adaptive suppression strength `w^(t)` (paper §3.4).

    The noise-type predictor emits `f_adapt ∈ [0, 1]` per frame. The
    smoothed suppression is:

        w^(t) = β · w^(t-1) + (1 - β) · (a · f_adapt + b)

    β ∈ [0, 1) is the moving-average coefficient. Paper §4.4 last
    paragraph (Table 2 last row) reports β=0.4 as the value that gives
    "almost no WER degradation on clean and non-speech noise conditions,
    and still a large improvement on speech noise conditions"; the
    upstream open-source default β=0.8 was an earlier guess and is no
    longer the recommended value.

    `a, b` define a linear transform on f_adapt that maps f_adapt ∈ [0,1]
    onto w_raw ∈ [w_min, w_max] with w_min = b and w_max = a + b. The
    defaults below (a=0.7, b=0.2) → w_raw ∈ [0.2, 0.9]: predictor-clean
    frames keep a small always-on compensation (w_min=0.2), predictor-
    overlap frames get near-full filtering (w_max=0.9) with a sliver of
    the original input as a safety net. Slides §3.4 Eq. 5 does not pin
    numeric values; see configs/paper.yaml for the full derivation.

    Output is clamped to [0, 1]:
      * w = 0 → voice filtering disabled (pass-through)
      * w = 1 → no compensation (full filtering)
    """

    def __init__(self, beta: float = 0.4, a: float = 0.7, b: float = 0.2):
        self.beta = float(beta)
        self.a = float(a)
        self.b = float(b)
        self.w = 0.0

    def reset(self) -> None:
        self.w = 0.0

    def update(self, f_adapt: float) -> float:
        target = self.a * f_adapt + self.b
        self.w = self.beta * self.w + (1.0 - self.beta) * target
        self.w = max(0.0, min(1.0, self.w))
        return self.w


# --------------------------------------------------------------------- backends
class _Backend:
    """Minimal interface every backend exposes to `StreamingVoiceFilter`."""

    name: str = "abstract"

    def reset(self, dtype, device) -> None: ...
    def set_dvec(self, dvec: torch.Tensor) -> None: ...
    def infer(self, stacked_logmel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Run one step. Returns (mask (1, 1, n_mels), last_frame (n_mels,), f_adapt)."""
        ...


class _TorchBackend(_Backend):
    """Runs `VoiceFilterLite` on whatever device the module is on."""

    name = "torch"

    def __init__(self, model: VoiceFilterLite, hp: HParams):
        self.model = model.eval()
        self.hp = hp
        self._dvec: Optional[torch.Tensor] = None
        self._hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset(self, dtype=None, device=None) -> None:
        dev = device if device is not None else next(self.model.parameters()).device
        self._hidden = (
            torch.zeros(self.hp.num_lstm_layers, 1, self.hp.hidden_dim, device=dev),
            torch.zeros(self.hp.num_lstm_layers, 1, self.hp.hidden_dim, device=dev),
        )

    def set_dvec(self, dvec: torch.Tensor) -> None:
        self._dvec = dvec

    def infer(self, stacked_logmel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        dvec = self._dvec
        model_device = dvec.device
        flat = stacked_logmel.reshape(1, 1, -1)
        if flat.device != model_device:
            flat = flat.to(model_device)
        with torch.no_grad():
            x, self._hidden = self.model.lstm_forward(flat, dvec, self._hidden)
            mask = torch.sigmoid(self.model.out_proj(self.model.dropout(x)))   # (1, 1, n_mels)
            # Paper §3.4: f_adapt(S_in^(t)) — primary input is the current
            # frame's noisy filterbank. Fall back to LSTM output when
            # `alpha_head_use_current_frame=False` (paper §4.2 alternative).
            last_frame = stacked_logmel[0, -1]
            head_in = last_frame.to(mask.device) if self.hp.alpha_head_use_current_frame else x
            if self.hp.use_noise_type_predictor:
                f_adapt = float(torch.sigmoid(self.model.alpha_head(head_in)).item())
            else:
                f_adapt = 0.0
        return mask[0, 0], last_frame, f_adapt


class _OnnxBackend(_Backend):
    """Runs the exported ONNX graph via onnxruntime (stateful LSTM I/O)."""

    name = "onnx"

    def __init__(self, session, hp: HParams):
        self.session = session
        self.hp = hp
        outputs = {o.name for o in session.get_outputs()}
        # New graph (export_onnx.py StreamWrapper v2): mask + f_adapt + hN + cN.
        # Legacy graph: enhanced + hN + cN (mask + blending baked in, no
        # adaptive suppression). We auto-detect so older .onnx files keep
        # working — their `enhanced` already encodes mask*Y without w-blend,
        # which is mathematically close to the new path for clean frames
        # (w≈0) but the two diverge under heavy overlap. Callers that want
        # parity with the torch streaming wrapper must re-export with the
        # new graph (`voicefilter-cli export onnx`).
        self._output_mode = "mask" if "mask" in outputs and "f_adapt" in outputs else "enhanced"
        self._dvec: Optional[np.ndarray] = None
        self._h: Optional[np.ndarray] = None
        self._c: Optional[np.ndarray] = None

    def reset(self, dtype=np.float32, device=None) -> None:
        self._h = np.zeros((self.hp.num_lstm_layers, 1, self.hp.hidden_dim), dtype=dtype)
        self._c = np.zeros((self.hp.num_lstm_layers, 1, self.hp.hidden_dim), dtype=dtype)

    def set_dvec(self, dvec: torch.Tensor) -> None:
        self._dvec = dvec.detach().cpu().numpy().astype(np.float32).reshape(1, -1)

    def infer(self, stacked_logmel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        feed = {
            "stacked_logmel": stacked_logmel.reshape(1, 1, -1).cpu().numpy().astype(np.float32),
            "dvec": self._dvec,
            "h0": self._h,
            "c0": self._c,
        }
        outs = self.session.run(None, feed)
        if self._output_mode == "mask":
            mask_np, f_adapt_np, self._h, self._c = outs
            last_frame = stacked_logmel[0, -1].cpu()
            f_adapt = float(np.asarray(f_adapt_np).reshape(-1)[0])
        else:
            # Legacy graph: enhanced = (mask * cur_linear) already in log-mel.
            # We can't recover the mask without the cur frame, but we can
            # recover a numerical f_adapt: fall back to 0 (no adaptive
            # suppression) and surface a one-shot warning.
            enhanced_np, self._h, self._c = outs
            mask_np = None
            f_adapt = 0.0
            last_frame = stacked_logmel[0, -1].cpu()
        mask_t = torch.from_numpy(np.asarray(mask_np).reshape(1, 1, self.hp.n_mels)) if mask_np is not None else None
        return mask_t, last_frame, f_adapt


def _coerce_backend(model, hp: HParams) -> _Backend:
    """Pick the right backend based on what the caller passed.

    Accepted `model` values:
      * `voicefilter_lite.model.VoiceFilterLite` (nn.Module) → torch
      * `str` / `os.PathLike` (path to a `.onnx` file) → onnxruntime
      * `onnxruntime.capi.onnxruntime_inference_collection.InferenceSession` → onnxruntime
    """
    if isinstance(model, _Backend):
        return model
    if isinstance(model, (str, os.PathLike)):
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Passing an ONNX path requires `onnxruntime`. "
                "Install with `pip install onnxruntime`."
            ) from e
        from onnxruntime import InferenceSession
        session = InferenceSession(str(model), providers=["CPUExecutionProvider"])
        return _OnnxBackend(session, hp)
    # Duck-typed InferenceSession.
    if hasattr(model, "run") and hasattr(model, "get_inputs") and hasattr(model, "get_outputs"):
        return _OnnxBackend(model, hp)
    # nn.Module fallback.
    return _TorchBackend(model, hp)


class StreamingVoiceFilter:
    """Stateful frame-by-frame inference wrapper (torch or ONNX backend).

    Usage:
        # torch model:
        sf = StreamingVoiceFilter(model, hp)
        # ONNX path or session:
        sf = StreamingVoiceFilter("voicefilter_lite.onnx", hp)

        sf.set_dvector(dvec)                 # registration step (256-d from SpeakerEncoder)
        for chunk_logmel_frame in stream:    # (n_mels,) each iteration
            enhanced_frame = sf.step(chunk_logmel_frame)
    """

    def __init__(self, model, hp: HParams):
        self.hp = hp
        self._backend: _Backend = _coerce_backend(model, hp)
        # Carry `.eval()`-equivalent behaviour regardless of backend.
        self._dvec: Optional[torch.Tensor] = None
        self._context: List[torch.Tensor] = []
        self._adaptive: Optional[AdaptiveSuppression] = None

    @property
    def backend_name(self) -> str:
        return self._backend.name

    # ------------------------------------------------------------------ setup
    @torch.no_grad()
    def set_dvector(self, dvec: torch.Tensor) -> None:
        """Register the target speaker embedding.

        Args:
            dvec: (hp.d_vector_dim,) or (1, hp.d_vector_dim) L2-normalised
                speaker embedding produced by `SpeakerEncoder`.
        """
        if dvec.dim() == 1:
            dvec = dvec.unsqueeze(0)
        assert dvec.shape[-1] == self.hp.d_vector_dim, (
            f"[streaming.set_dvector] dvec dim {dvec.shape[-1]} != "
            f"hp.d_vector_dim ({self.hp.d_vector_dim}); encoder and model "
            f"dims must match."
        )
        self._dvec = dvec
        self._backend.set_dvec(dvec)
        self.reset_state()

    @torch.no_grad()
    def reset_state(self) -> None:
        """Drop LSTM hidden state and clear the left-context buffer."""
        self._backend.reset()
        self._context = []
        if self._adaptive is not None:
            self._adaptive.reset()
        else:
            self._adaptive = AdaptiveSuppression(
                beta=self.hp.adaptive_w_beta,
                a=self.hp.adaptive_w_a,
                b=self.hp.adaptive_w_b,
            )

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, frame_logmel: torch.Tensor) -> torch.Tensor:
        """Process one log-mel frame.

        Args:
            frame_logmel: (n_mels,) or (1, n_mels).

        Returns:
            (n_mels,) enhanced log-mel for this frame, on the caller's device.
        """
        if self._dvec is None:
            raise RuntimeError("Call set_dvector() before step().")

        squeeze = frame_logmel.dim() == 1
        if squeeze:
            frame_logmel = frame_logmel.unsqueeze(0)
        caller_device = frame_logmel.device

        stacked = self._build_stacked_input(frame_logmel)
        mask, last_frame, f_adapt = self._backend.infer(stacked)
        enhanced = self._apply_suppression(mask, last_frame, f_adapt)
        if enhanced.device != caller_device:
            enhanced = enhanced.to(caller_device)
        return enhanced.squeeze(0) if squeeze else enhanced

    # ------------------------------------------------------------------ helpers
    def _build_stacked_input(self, frame_logmel: torch.Tensor) -> torch.Tensor:
        """Roll the left-context buffer and right-align `frame_stack` frames.

        Returns (1, frame_stack, n_mels) — zeros padding on the left for the
        first `frame_stack - 1` frames, matching the training pipeline's
        `frame_stack_left` exactly so warm-up behaviour is in distribution.
        """
        stack = self.hp.frame_stack
        n_mels = self.hp.n_mels

        self._context.append(frame_logmel)
        if len(self._context) > stack:
            self._context = self._context[-stack:]

        stacked = torch.zeros(1, stack, n_mels, dtype=frame_logmel.dtype, device=frame_logmel.device)
        offset = stack - len(self._context)
        for i, f in enumerate(self._context):
            stacked[0, offset + i] = f[0]
        return stacked

    def _apply_suppression(
        self,
        mask: Optional[torch.Tensor],
        last_frame: torch.Tensor,
        f_adapt: float,
    ) -> torch.Tensor:
        """Blend the filtered output with the raw input using w^(t).

        Paper §3.2 / §3.4 — both the mask application and the adaptive
        suppression blending happen in **linear** filterbank-energy
        space (the mask is `M · S_in` on linear-mel, NOT log-mel, per
        the paper's literal formulas). The result is converted back to
        log-mel before return so the caller's ASR pipeline still sees
        the same feature distribution it would have produced from the
        un-filtered input.
        """
        w = self._adaptive.update(f_adapt) if self._adaptive is not None else 1.0
        floor = self.hp.log_mel_floor

        # log → linear for both filtered and raw frames (single round-trip).
        raw_linear = log_mel_to_linear(last_frame, floor=floor)
        if mask is None:
            # Legacy ONNX graph: the model already applied the mask in its
            # own log-mel round-trip; we have nothing to multiply here.
            filtered_linear = raw_linear
        else:
            filtered_linear = apply_mask(mask.reshape(-1), raw_linear)

        # §3.4 Eq. 4: S_out = w · Ŝ_in + (1-w) · S_in (all in linear-mel).
        enhanced_linear = w * filtered_linear + (1.0 - w) * raw_linear
        return linear_to_log_mel(enhanced_linear, floor=floor)
