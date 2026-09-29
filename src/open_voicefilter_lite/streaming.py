"""Streaming inference wrapper + adaptive suppression (paper §3.4).

`StreamingVoiceFilter` is the stateful frame-by-frame inference entry point.
It owns:
  * the ECAPA d-vector projection (precomputed in `set_dvector`)
  * the LSTM hidden state (`(h, c)`)
  * the left-context frame stack (rolling buffer of `frame_stack - 1` frames)
  * the adaptive-suppression smoother (`AdaptiveSuppression`)

`step(frame_logmel)` does **one** new frame and returns the enhanced
log-mel. The caller never sees intermediate tensors; device/dim plumbing is
contained here.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from open_voicefilter_lite.hparams import HParams
from open_voicefilter_lite.model import VoiceFilterLite


class AdaptiveSuppression:
    """Inference-time adaptive suppression strength `w^(t)` (paper §3.4).

    The noise-type predictor emits `f_adapt ∈ [0, 1]` per frame. The
    smoothed suppression is:

        w^(t) = β · w^(t-1) + (1 - β) · (a · f_adapt + b)

    β ∈ [0, 1) is the moving-average coefficient (paper §4.4: β = 0.8).
    a, b define a linear transform on f_adapt (default a=1, b=0).

    Output is clamped to [0, 1]:
      * w = 0 → voice filtering disabled (pass-through)
      * w = 1 → no compensation (full filtering)
    """

    def __init__(self, beta: float = 0.8, a: float = 1.0, b: float = 0.0):
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


class StreamingVoiceFilter:
    """Stateful frame-by-frame inference wrapper.

    Usage:
        sf = StreamingVoiceFilter(model, hp)
        sf.set_dvector(dvec_raw)             # registration step (raw ECAPA output)
        for chunk_logmel_frame in stream:    # (n_mels,) each iteration
            enhanced_frame = sf.step(chunk_logmel_frame)
    """

    def __init__(self, model: VoiceFilterLite, hp: HParams):
        self.model = model.eval()
        self.hp = hp
        self._dvec: Optional[torch.Tensor] = None
        self._dvec_proj: Optional[torch.Tensor] = None
        self._hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._context: List[torch.Tensor] = []   # most recent frame_stack - 1 frames
        # Initialised lazily so it inherits dtype/device from the model.
        self._adaptive: Optional[AdaptiveSuppression] = None

    # ------------------------------------------------------------------ setup
    @torch.no_grad()
    def set_dvector(self, dvec_raw: torch.Tensor) -> None:
        """Register the target speaker embedding (raw ECAPA output).

        Args:
            dvec_raw: (hp.speaker_encoder_raw_dim,) or (1, hp.speaker_encoder_raw_dim)
                L2-normalised ECAPA embedding.
        """
        if dvec_raw.dim() == 1:
            dvec_raw = dvec_raw.unsqueeze(0)
        self._dvec = dvec_raw
        # Pre-project so step() never has to touch dvec_proj for each frame.
        self._dvec_proj = self.model.dvec_proj(dvec_raw)
        self.reset_state()

    @torch.no_grad()
    def reset_state(self) -> None:
        """Drop LSTM hidden state and clear the left-context buffer."""
        device = next(self.model.parameters()).device
        self._hidden = (
            torch.zeros(self.hp.num_lstm_layers, 1, self.hp.hidden_dim, device=device),
            torch.zeros(self.hp.num_lstm_layers, 1, self.hp.hidden_dim, device=device),
        )
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
        if self._dvec_proj is None:
            raise RuntimeError("Call set_dvector() before step().")

        squeeze = frame_logmel.dim() == 1
        if squeeze:
            frame_logmel = frame_logmel.unsqueeze(0)
        caller_device = frame_logmel.device

        stacked = self._build_stacked_input(frame_logmel)
        mask, x, last_frame = self._run_model(stacked)
        f_adapt = self._predict_f_adapt(x) if self.hp.use_noise_type_predictor else 0.0

        enhanced = self._apply_suppression(mask, last_frame, f_adapt, caller_device)
        return enhanced.squeeze(0) if squeeze else enhanced

    # ------------------------------------------------------------------ helpers
    def _build_stacked_input(self, frame_logmel: torch.Tensor) -> torch.Tensor:
        """Roll the left-context buffer and right-align `frame_stack` frames.

        Returns (1, frame_stack, n_mels) — zeros padding at the start until
        the buffer is full.
        """
        stack = self.hp.frame_stack
        n_mels = self.hp.n_mels

        self._context.append(frame_logmel)
        if len(self._context) > stack:
            self._context = self._context[-stack:]

        stacked = torch.zeros(1, stack, n_mels, dtype=frame_logmel.dtype, device=frame_logmel.device)
        # Right-align: most recent frame goes into the last slot.
        offset = stack - len(self._context)
        for i, f in enumerate(self._context):
            stacked[0, offset + i] = f[0]
        return stacked

    def _run_model(self, stacked: torch.Tensor):
        """One-step LSTM forward → mask, x (last hidden state for alpha head), last frame.

        The model may live on a different device than the caller's frame
        (e.g. CPU frame + MPS model). We align the input to the model's
        device for the LSTM forward, then bring outputs back to the
        caller's device before returning.
        """
        stacked_flat = stacked.reshape(1, 1, -1)
        dvec_proj = self._dvec_proj
        if dvec_proj.dim() == 2:
            dvec_proj = dvec_proj.unsqueeze(1)        # (1, 1, d_vector_dim)

        model_device = dvec_proj.device
        if stacked_flat.device != model_device:
            stacked_flat = stacked_flat.to(model_device)

        x = self.model.in_proj(torch.cat([stacked_flat, dvec_proj], dim=-1))
        x = self.model.in_norm(x)
        x, self._hidden = self.model.lstm(x, self._hidden)
        mask = torch.sigmoid(self.model.out_proj(x))   # (1, 1, n_mels)

        last_frame = stacked[0, -1]
        if last_frame.device != mask.device:
            last_frame = last_frame.to(mask.device)
        return mask, x, last_frame

    def _predict_f_adapt(self, x: torch.Tensor) -> float:
        """Run the noise-type head on the last LSTM hidden state → scalar."""
        logit = self.model.alpha_head(x)               # (1, 1, 1)
        return float(torch.sigmoid(logit).item())

    def _apply_suppression(
        self,
        mask: torch.Tensor,
        last_frame: torch.Tensor,
        f_adapt: float,
        caller_device: torch.device,
    ) -> torch.Tensor:
        """Blend the filtered output with the raw input using w^(t).

        Both filtering and blending happen in **linear-mel** space
        (paper §3.2); the result is converted back to log-mel before return.
        """
        w = self._adaptive.update(f_adapt) if self._adaptive is not None else 1.0

        floor = self.hp.log_mel_floor
        mix_mel = (torch.exp(last_frame) - floor).clamp(min=0.0)
        # Equivalently `exp(model._apply_mask(mask[0,0], last_frame)) - floor`,
        # but we already have `mask * mix_mel` in hand and skip a log/exp round-trip.
        filtered_mel = mask[0, 0] * mix_mel

        enhanced_mel = w * filtered_mel + (1.0 - w) * mix_mel
        enhanced = torch.log(enhanced_mel + floor)
        if enhanced.device != caller_device:
            enhanced = enhanced.to(caller_device)
        return enhanced