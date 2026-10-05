"""Streaming inference wrapper + adaptive suppression (paper §3.4).

`StreamingVoiceFilter` is the stateful frame-by-frame inference entry point.
It owns:
  * the registered target-speaker d-vector (set once in `set_dvector`)
  * the LSTM hidden state (`(h, c)`)
  * the left-context frame stack (rolling buffer of `frame_stack - 1` frames)
  * the adaptive-suppression smoother (`AdaptiveSuppression`)

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

from typing import List, Optional, Tuple

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


class StreamingVoiceFilter:
    """Stateful frame-by-frame inference wrapper.

    Usage:
        sf = StreamingVoiceFilter(model, hp)
        sf.set_dvector(dvec)                 # registration step (256-d from SpeakerEncoder)
        for chunk_logmel_frame in stream:    # (n_mels,) each iteration
            enhanced_frame = sf.step(chunk_logmel_frame)
    """

    def __init__(self, model: VoiceFilterLite, hp: HParams):
        self.model = model.eval()
        self.hp = hp
        # Registered d-vector (already 256-d from SpeakerEncoder; no in-model
        # projection).
        self._dvec: Optional[torch.Tensor] = None
        self._hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._context: List[torch.Tensor] = []   # most recent frame_stack - 1 frames
        # Initialised lazily so it inherits dtype/device from the model.
        self._adaptive: Optional[AdaptiveSuppression] = None

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
            One input frame always yields one output frame (same shape, dtype,
            device) — `step()` runs the LSTM forward unconditionally so the
                first `frame_stack - 1` calls see the same zero-padded
                left-context / zero-hidden-state warm-up pattern that training uses
                (training's `frame_stack_left` + per-batch hidden reset).
        """
        if self._dvec is None:
            raise RuntimeError("Call set_dvector() before step().")

        squeeze = frame_logmel.dim() == 1
        if squeeze:
            frame_logmel = frame_logmel.unsqueeze(0)
        caller_device = frame_logmel.device

        stacked = self._build_stacked_input(frame_logmel)
        mask, x, last_frame = self._run_model(stacked)
        # Paper §3.4: `f_adapt(S_in^(t))` — the noise-type predictor's
        # primary input is the current frame's noisy filterbank, not the
        # LSTM hidden state. `last_frame` is the right-most slot of the
        # right-aligned stack, i.e. S_in^(t). When
        # `hp.alpha_head_use_current_frame=False` we fall back to the
        # LSTM output `x` (paper §4.2 alternative).
        head_in = last_frame if self.hp.alpha_head_use_current_frame else x
        f_adapt = self._predict_f_adapt(head_in) if self.hp.use_noise_type_predictor else 0.0

        enhanced = self._apply_suppression(mask, last_frame, f_adapt, caller_device)
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
        # Right-align: most recent frame goes into the last slot.
        offset = stack - len(self._context)
        for i, f in enumerate(self._context):
            stacked[0, offset + i] = f[0]
        return stacked

    def _run_model(self, stacked: torch.Tensor):
        """One-step model forward → mask, lstm_out (for alpha head), last frame.

        Delegates the LSTM core to `model.lstm_forward` so this stays in
        lockstep with `VoiceFilterLite.forward`. The streaming path needs
        `lstm_out` separately for the `hp.alpha_head_use_current_frame=False`
        variant (paper §4.2 alternative input to the noise-type head).

        The model may live on a different device than the caller's frame
        (e.g. CPU frame + MPS model). We align the input to the model's
        device for the LSTM forward, then bring outputs back to the
        caller's device before returning.
        """
        stacked_flat = stacked.reshape(1, 1, -1)
        # `lstm_forward` expects `dvec` as `(B, d_vector_dim)` — `self._dvec`
        # is already at this shape (set_dvector adds the batch dim). The
        # legacy inline path used `(1, 1, 256)` here; we no longer need
        # the extra T dim because `lstm_forward` tiles internally.
        dvec = self._dvec

        model_device = dvec.device
        if stacked_flat.device != model_device:
            stacked_flat = stacked_flat.to(model_device)

        # `lstm_forward` already runs in_proj + in_norm + [cnn] + lstm.
        x, self._hidden = self.model.lstm_forward(stacked_flat, dvec, self._hidden)
        mask = torch.sigmoid(self.model.out_proj(self.model.dropout(x)))   # (1, 1, n_mels)

        last_frame = stacked[0, -1]
        if last_frame.device != mask.device:
            last_frame = last_frame.to(mask.device)
        return mask, x, last_frame

    def _predict_f_adapt(self, head_in: torch.Tensor) -> float:
        """Run the noise-type head on `head_in` (current frame or LSTM output) → scalar.

        Paper §3.4: `f_adapt(S_in^(t))` — primary input is the current frame's
        noisy filterbank (80-d). Paper §4.2 alternative: feed the LSTM output
        features (`x`). The `step` caller picks which one based on
        `hp.alpha_head_use_current_frame`.
        """
        logit = self.model.alpha_head(head_in)               # (1, 1, 1)
        return float(torch.sigmoid(logit).item())

    def _apply_suppression(
        self,
        mask: torch.Tensor,
        last_frame: torch.Tensor,
        f_adapt: float,
        caller_device: torch.device,
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
        filtered_linear = apply_mask(mask[0, 0], raw_linear)

        # §3.4 Eq. 4: S_out = w · Ŝ_in + (1-w) · S_in (all in linear-mel).
        enhanced_linear = w * filtered_linear + (1.0 - w) * raw_linear
        enhanced = linear_to_log_mel(enhanced_linear, floor=floor)
        if enhanced.device != caller_device:
            enhanced = enhanced.to(caller_device)
        return enhanced