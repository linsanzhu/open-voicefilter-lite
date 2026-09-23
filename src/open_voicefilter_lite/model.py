"""VoiceFilter-Lite main model + streaming inference wrapper.

Architecture (≈ 2.2-2.6 M params; matches paper §2 closely):
    Linear(d_vector_dim + n_mels * frame_stack → hidden_dim)
    LayerNorm
    LSTM(hidden_dim) × num_lstm_layers (unidirectional — streaming)
    Dropout
    Linear(hidden_dim → n_mels)
    Sigmoid → soft mask M ∈ [0, 1]^F

The mask is element-wise multiplied with the *current frame's* log-mel to
produce the enhanced log-mel. Training targets are clean log-mel;
the loss is the asymmetric L2 penalty on (enhanced, clean).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from open_voicefilter_lite.hparams import HParams


class VoiceFilterLite(nn.Module):
    """Streaming targeted voice-separation mask predictor.

    Inputs (at training time):
        stacked_logmel: (B, T, n_mels * frame_stack)  — left-context stacked input
        dvec:           (B, d_vector_dim)             — speaker embedding, broadcast over T

    Output:
        enhanced:       (B, T, n_mels)                — log-mel of target speaker
    """

    def __init__(self, hp: HParams):
        super().__init__()
        self.hp = hp

        # Project raw ECAPA embedding down to d_vector_dim (trainable).
        # Source dim comes from `hp.speaker_encoder_raw_dim` (192 for
        # `spkrec-ecapa-voxceleb`).
        self.dvec_proj = nn.Linear(hp.speaker_encoder_raw_dim, hp.d_vector_dim)

        in_dim = hp.input_dim
        self.in_proj = nn.Linear(in_dim, hp.hidden_dim)
        self.in_norm = nn.LayerNorm(hp.hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hp.hidden_dim,
            hidden_size=hp.hidden_dim,
            num_layers=hp.num_lstm_layers,
            batch_first=True,
            bidirectional=False,   # streaming: no future context
            dropout=hp.dropout if hp.num_lstm_layers > 1 else 0.0,
        )

        self.out_proj = nn.Linear(hp.hidden_dim, hp.output_dim)
        self.dropout = nn.Dropout(hp.dropout)

        # Noise-type predictor (paper §4.2): two feedforward layers (256 → 64
        # → 64 → 1), each with 64 nodes per layer, trained with a hinge loss
        # for **per-frame** "is this frame overlapped speech?" binary
        # classification. At inference, sigmoid(logits) drives the adaptive
        # suppression `w^(t)` per paper §3.4. ~20.6 K params.
        self.alpha_head = nn.Sequential(
            nn.Linear(hp.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Xavier init on the linears (LSTM defaults are fine). `alpha_head` is a
        # nn.Sequential of Linear+ReLU layers; iterate its children so each
        # Linear gets xavier_uniform_(weight) and zero bias.
        init_modules = [self.dvec_proj, self.in_proj, self.out_proj]
        init_modules.extend(self.alpha_head)  # add Linear children of Sequential
        for m in init_modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        stacked_logmel: torch.Tensor,
        dvec: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run the model (inference path).

        Args:
            stacked_logmel: (B, T, n_mels * frame_stack).
            dvec:           (B, hp.speaker_encoder_raw_dim) raw ECAPA embedding.
            hidden:         optional (h, c) tuple to seed the LSTM state.

        Returns:
            enhanced: (B, T, n_mels) — input log-mel element-wise masked.
            hidden:   final (h, c) tuple for stateful inference.
        """
        B, T, _ = stacked_logmel.shape

        # Project the raw ECAPA embedding (hp.speaker_encoder_raw_dim-d) to d_vector_dim.
        dvec_proj = self.dvec_proj(dvec)               # (B, d_vector_dim)
        dvec_tiled = dvec_proj.unsqueeze(1).expand(-1, T, -1)  # (B, T, d_vector_dim)

        x = torch.cat([stacked_logmel, dvec_tiled], dim=-1)   # (B, T, in_dim)
        x = self.in_proj(x)
        x = self.in_norm(x)
        x = self.dropout(x)

        x, hidden = self.lstm(x, hidden)               # (B, T, hidden)

        mask = torch.sigmoid(self.out_proj(self.dropout(x)))   # (B, T, n_mels)

        # Recover the *current frame's* log-mel from the stacked input
        # (rightmost slot when stacking is right-aligned: ..., t-1, t).
        cur_idx = self.hp.frame_stack - 1
        cur = stacked_logmel[..., cur_idx * self.hp.n_mels:(cur_idx + 1) * self.hp.n_mels]

        enhanced = mask * cur
        return enhanced, hidden

    def forward_train(
        self,
        stacked_logmel: torch.Tensor,
        dvec: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Training path that additionally returns the **per-frame** logits
        from the noise-type predictor head (paper §4.2).

        Returns:
            enhanced:     (B, T, n_mels)
            hidden:       final (h, c) tuple
            frame_logits: (B, T) — raw logits, one per frame, for "is this
                          frame overlapped speech?" binary classification.
                          `sigmoid(frame_logits)` is `f_adapt` in §3.4.
        """
        T = stacked_logmel.shape[1]

        dvec_proj = self.dvec_proj(dvec)
        dvec_tiled = dvec_proj.unsqueeze(1).expand(-1, T, -1)

        x = torch.cat([stacked_logmel, dvec_tiled], dim=-1)
        x = self.in_proj(x)
        x = self.in_norm(x)
        x = self.dropout(x)

        x, hidden = self.lstm(x, hidden)               # (B, T, hidden)

        mask = torch.sigmoid(self.out_proj(self.dropout(x)))   # (B, T, n_mels)

        # Per-frame noise-type logits (paper §4.2: 2 FC × 64 + sigmoid).
        frame_logits = self.alpha_head(x).squeeze(-1)   # (B, T)

        cur_idx = self.hp.frame_stack - 1
        cur = stacked_logmel[..., cur_idx * self.hp.n_mels:(cur_idx + 1) * self.hp.n_mels]
        enhanced = mask * cur

        return enhanced, hidden, frame_logits


class AdaptiveSuppression:
    """Inference-time adaptive suppression strength `w^(t)` (paper §3.4).

    The noise-type predictor emits f_adapt ∈ [0, 1] per frame ("this frame
    is overlapped speech"). The final suppression strength is the
    exponentially-smoothed linear transform of f_adapt:

        w^(t) = β · w^(t-1) + (1 - β) · (a · f_adapt(S^(t)_in) + b)

    β ∈ [0, 1) is a moving-average coefficient (paper §4.4: β = 0.8) that
    smooths w across frames — when overlapped speech stops, w slowly
    decays back to 0 instead of snapping. a > 0 and b ≥ 0 define a linear
    transform (default a=1, b=0 → w tracks f_adapt directly).

    The output is clamped to [0, 1] because:
      - w = 0 → voice filtering fully disabled (pass-through)
      - w = 1 → no compensation (full filtering)
    """

    def __init__(self, beta: float = 0.8, a: float = 1.0, b: float = 0.0):
        self.beta = float(beta)
        self.a = float(a)
        self.b = float(b)
        self.w = 0.0  # initial suppression strength

    def reset(self) -> None:
        self.w = 0.0

    def update(self, f_adapt: float) -> float:
        """Compute the new w^(t) given the predictor output f_adapt for the
        current frame. Returns the new w, and updates the internal state."""
        target = self.a * f_adapt + self.b
        self.w = self.beta * self.w + (1.0 - self.beta) * target
        # Clamp to [0, 1]: w=0 disables filtering, w=1 is full filtering.
        self.w = max(0.0, min(1.0, self.w))
        return self.w


class StreamingVoiceFilter:
    """Stateful inference wrapper for live audio (real-time telephony).

    Usage:
        sf = StreamingVoiceFilter(model)
        sf.set_dvector(dvec_raw)           # registration step
        for chunk_logmel_frame in stream:  # (1, n_mels) each iteration
            enhanced_frame = sf.step(chunk_logmel_frame)

    The LSTM hidden state persists across calls. The left-context frame
    stack is internal — the caller only needs to provide one log-mel
    frame at a time.
    """

    def __init__(self, model: VoiceFilterLite, hp: HParams):
        self.model = model.eval()
        self.hp = hp
        self._dvec: Optional[torch.Tensor] = None
        self._dvec_proj: Optional[torch.Tensor] = None
        self._hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._context: list[torch.Tensor] = []   # most recent N-1 frames
        # Adaptive suppression (paper §3.4) — initialized lazily so we know the
        # dtype/device of the model. Reset whenever `reset_state` is called.
        self._adaptive: Optional[AdaptiveSuppression] = None

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
        # Pre-project for efficiency at inference time.
        self._dvec_proj = self.model.dvec_proj(dvec_raw)
        self.reset_state()

    @torch.no_grad()
    def reset_state(self) -> None:
        """Drop LSTM hidden state and context buffer (call on stream start)."""
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

    @torch.no_grad()
    def step(self, frame_logmel: torch.Tensor) -> torch.Tensor:
        """Process one new log-mel frame.

        Args:
            frame_logmel: (n_mels,) or (1, n_mels).

        Returns:
            (n_mels,) enhanced log-mel for this frame.
        """
        if self._dvec_proj is None:
            raise RuntimeError("Call set_dvector() before step().")

        squeeze = False
        if frame_logmel.dim() == 1:
            frame_logmel = frame_logmel.unsqueeze(0)
            squeeze = True

        # Update rolling context.
        self._context.append(frame_logmel)
        if len(self._context) > self.hp.frame_stack:
            self._context = self._context[-self.hp.frame_stack:]

        # Build stacked input: pad with zeros at the start if we don't yet
        # have frame_stack frames of context.
        stack = self.hp.frame_stack
        n_mels = self.hp.n_mels
        stacked = torch.zeros(1, stack, n_mels, dtype=frame_logmel.dtype, device=frame_logmel.device)
        ctx = self._context
        # Right-align: most recent frame goes into the last slot.
        for i, f in enumerate(ctx):
            stacked[0, stack - len(ctx) + i, :] = f[0]

        # Flatten the time+mel dims into (1, 1, stack*n_mels).
        stacked_flat = stacked.reshape(1, 1, -1)

        dvec_proj = self._dvec_proj
        if dvec_proj.dim() == 2:
            dvec_proj = dvec_proj.unsqueeze(1)  # (1, 1, d_vector_dim)

        # Concat and run a 1-step LSTM.
        x = torch.cat([stacked_flat, dvec_proj], dim=-1)
        x = self.model.in_proj(x)
        x = self.model.in_norm(x)
        x, self._hidden = self.model.lstm(x, self._hidden)
        mask = torch.sigmoid(self.model.out_proj(x))   # (1, 1, n_mels)

        # ---- Adaptive suppression (paper §3.4) ----
        # The noise-type predictor (a child module of `self.model`) emits a
        # per-frame logit; sigmoid → [0,1] gives the predicted probability
        # that this frame is overlapped speech (paper §4.2 binary head).
        # AdaptiveSuppression smooths it into w^(t) and we blend the filtered
        # output with the raw input:
        #     out = w · mask·input + (1-w) · input
        #        = input · (w·mask + (1-w))
        if self.hp.use_noise_type_predictor and self._adaptive is not None:
            logit = self.model.alpha_head(x)                  # (1, 1, 1)
            f_adapt = float(torch.sigmoid(logit).item())
            w = self._adaptive.update(f_adapt)
        else:
            w = 1.0   # no adaptive suppression → apply mask as-is
        effective_mask = w * mask[0, 0] + (1.0 - w)            # (1, n_mels)
        enhanced = effective_mask * stacked[0, -1]

        if squeeze:
            return enhanced.squeeze(0)
        return enhanced
