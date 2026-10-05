"""VoiceFilter-Lite: streaming targeted voice-separation mask predictor.

Architecture (paper §3.2, §3.3, §4.2):
    Linear(n_mels*frame_stack + d_vector_dim → hidden) → LayerNorm
        → [optional Conv1d × num_cnn_layers]   # paper §3.2 1D CNN (default OFF; 2 layers when on)
        → LSTM (unidirectional) × num_lstm_layers
        → Linear(hidden → n_mels) → Sigmoid   # mask branch
        → Linear(n_mels → 64) → ReLU → Linear(64 → 1)   # noise-type head (paper §3.4: S_in^(t); slides p.28 "2 FF × 64 nodes")

`d_vector_dim` is the 256-d d-vector from the official VoiceFilter
paper's GE2E-trained LSTM encoder (`SpeakerEncoder`). The d-vector is
fed **directly** into `in_proj` — there is no in-model projection layer.

The mask is applied in **linear filterbank-energy** space (paper §3.2,
literal `Y_enh(t,f) = M(t,f) · Y(t,f)` on `S_in`, NOT on log-mel). The
model's input/output remains log-mel (ASR-side convention) — callers
convert via `voicefilter_lite.audio.log_mel_to_linear` /
`linear_to_log_mel` before/after `apply_mask`. At `M=0` the enhanced
output is 0 (silence); at `M=1` it is identity. The noise-type head
emits per-frame logits for an "overlapped speech?" binary classifier;
by default it consumes the **current frame's noisy log-mel**
(`S_in^(t)`, paper §3.4) rather than the LSTM output (§4.2 alternative).
At inference its sigmoid drives adaptive suppression `w^(t)` — see
`streaming.py`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from voicefilter_lite.hparams import HParams


class VoiceFilterLite(nn.Module):
    """Streaming target-speaker mask predictor.

    Forward signature:
        forward(stacked_logmel, dvec, hidden=None) → (mask, hidden)

    `mask` is the raw sigmoid output (B, T, n_mels) ∈ [0, 1] — the only
    thing the model emits. Applying the mask in **linear** mel-energy
    space (`Y_enh = M · Y`, paper §3.2) is the caller's job — see the
    module-level `apply_mask` helper. Use
    `voicefilter_lite.audio.log_mel_to_linear` /
    `linear_to_log_mel` to bridge the log-mel ↔ linear-mel boundary.

    Public surface for callers that need extra pieces of the model:
      * `lstm_forward(...)` — returns the LSTM output before `out_proj`, for
        callers that want to feed it into `alpha_head` when
        `hp.alpha_head_use_current_frame=False` (paper §4.2 alternative).
      * `alpha_head` — a public `nn.Sequential` head. Training and the
        streaming wrapper call it externally to obtain `frame_logits`.
      * `apply_mask(mask, cur_linear)` — module-level free function
        implementing paper §3.2's literal mask application.

    `hidden` is the LSTM (h, c) state. Pass it back on the next call to
    enable stateful (streaming) inference. Training does NOT pass `hidden`
    — each batch resets state to zeros, which matches the reference
    project (`maum-ai/voicefilter`) and the streaming wrapper's first-call
    warmup pattern.
    """

    def __init__(self, hp: HParams):
        super().__init__()
        self.hp = hp

        # The d-vector from `SpeakerEncoder` is already 256-dim (paper §3.1);
        # it is fed directly into `in_proj`.

        in_dim = hp.n_mels * hp.frame_stack + hp.d_vector_dim
        self.in_proj = nn.Linear(in_dim, hp.hidden_dim)
        self.in_norm = nn.LayerNorm(hp.hidden_dim)
        self.dropout = nn.Dropout(hp.dropout)

        # Unidirectional (streaming: no future context).
        self.lstm = nn.LSTM(
            input_size=hp.hidden_dim,
            hidden_size=hp.hidden_dim,
            num_layers=hp.num_lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=hp.dropout if hp.num_lstm_layers > 1 else 0.0,
        )

        # Optional 1D CNN (paper §3.2). Slides along the **frequency**
        # axis (the last dim of `x`, i.e. `hidden_dim`); never mixes
        # across time. Implementation: `nn.Sequential` of `nn.Conv1d`
        # modules (the paper's "1D CNN"). Conv1d's kernel slides along
        # its `L` (length) axis, so we reshape `(B, T, H)` → `(B*T, 1, H)`
        # — each time step becomes an independent 1-channel "signal" of
        # length `H`, and the K=3 kernel slides along that length
        # (i.e. along the frequency axis). Same-padding K//2 keeps H
        # preserved. The H positions are collapsed by mean-pooling in
        # `lstm_forward` so the LSTM downstream gets
        # `(B, T, cnn_channels)`. The last Conv1d's output channels are
        # forced to `hidden_dim` so it feeds the LSTM stack directly
        # without an extra projection bridge.
        #
        # Slides p.17 Fig. 1b draws the 1D CNN explicitly (2 stacked
        # Conv1d layers → LSTM, Figure 2); slides p.28 Models text
        # doesn't list it separately. The paper admits "most of our
        # models actually remove these CNN layers and purely consist
        # of LSTM layers". Default `use_cnn=False` matches the
        # LSTM-only topology; flip it on for the slides-p.17
        # main-experiment architecture (set `num_cnn_layers=2`).
        self.cnn: Optional[nn.Sequential] = None
        if hp.use_cnn:
            layers: list[nn.Conv1d] = []
            in_ch = 1
            for i in range(hp.num_cnn_layers):
                is_last = (i == hp.num_cnn_layers - 1)
                # Last layer projects to hidden_dim so LSTM input shape matches.
                out_ch = hp.hidden_dim if is_last else hp.cnn_channels_first
                layers.append(nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=hp.cnn_kernel_size,
                    padding=hp.cnn_kernel_size // 2,
                ))
                in_ch = out_ch
            self.cnn = nn.Sequential(*layers)

        self.out_proj = nn.Linear(hp.hidden_dim, hp.n_mels)

        # Per-frame "is this frame overlapped speech?" head (paper §4.2).
        # Slides page 28 says literally "2 feedforward layers, each with
        # 64 nodes for noise type prediction" — i.e. `Linear(128 → 64) →
        # ReLU → Linear(64 → 1)` (input projection to 64 hidden nodes,
        # then a single logit output). ~3.5 K params.
        #
        # Input source is paper-§3.4-literal by default:
        #   * `alpha_head_use_current_frame=True`  → input is S_in^(t), the
        #     current frame's noisy filterbank (n_mels=128 dims, paper §3.4).
        #     Paper-faithful default.
        #   * `alpha_head_use_current_frame=False` → input is the LSTM's
        #     output features (hidden_dim dims). Paper §4.2 also allows
        #     this variant; kept as a legacy option for backward
        #     compatibility with already-trained checkpoints.
        head_in = hp.n_mels if hp.alpha_head_use_current_frame else hp.hidden_dim
        # Attribute name `alpha_head` kept for checkpoint compatibility.
        self.alpha_head = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Xavier init on every Linear; LSTM defaults are fine.
        xavier_modules: list[nn.Module] = [
            self.in_proj, self.out_proj,
            *self.alpha_head,  # alpha_head is a Sequential; init each child.
        ]
        if self.cnn is not None:
            xavier_modules.extend(list(self.cnn))  # unpack Conv1d children.
        for m in xavier_modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # `out_proj` feeds the sigmoid mask. Bias=0 starts the
                    # mask at sigmoid(0)=0.5, but under the asymmetric L2
                    # loss (paper §3.3, α≥2 amplifies over-suppression by α²)
                    # the gradient landscape has a stable local minimum at
                    # mask≈1 (pass-through): the model learns "always emit 1"
                    # to dodge the α² penalty, and once sigmoid saturates
                    # the discrimination signal vanishes (range → 0,
                    # corr_target → 0). Starting the mask at sigmoid(-1)=0.27
                    # gives the model room to climb into the healthy
                    # 0.6–0.8 range before late-training bias drift pulls
                    # it back toward 1.
                    if m is self.out_proj:
                        nn.init.constant_(m.bias, -1.0)
                    else:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def lstm_forward(
        self,
        stacked_logmel: torch.Tensor,
        dvec: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run in_proj → in_norm → [cnn] → lstm.

        Exposed so callers that need the LSTM output (e.g. to feed
        `alpha_head` under the `hp.alpha_head_use_current_frame=False`
        variant, paper §4.2) share the exact same LSTM stack as
        `forward` — no duplicated in_proj/in_norm/cnn logic.

        Args:
            stacked_logmel: (B, T, n_mels * frame_stack) — right-aligned.
            dvec:           (B, hp.d_vector_dim) speaker embedding.
            hidden:         optional (h, c) tuple to seed the LSTM.

        Returns:
            lstm_out: (B, T, hp.hidden_dim)  — pre-`out_proj` features.
            hidden:   (h, c) tuple          — updated LSTM state.
        """
        B, T, _ = stacked_logmel.shape

        dvec_tiled = dvec.unsqueeze(1).expand(-1, T, -1)
        x = self.dropout(self.in_norm(self.in_proj(
            torch.cat([stacked_logmel, dvec_tiled], dim=-1)
        )))
        # Optional 1D CNN (paper §3.2). See `__init__` for the slides-p.17
        # vs slides-p.28 default-off rationale; we just apply what was built.
        if self.cnn is not None:
            x_per = x.reshape(-1, 1, x.shape[-1])              # (B*T, 1, H)
            x_conv = self.cnn(x_per)                           # (B*T, cnn_channels, H)
            x = x_conv.mean(dim=-1).reshape(x.shape[0], x.shape[1], -1)  # (B, T, cnn_channels)
        x, hidden = self.lstm(x, hidden)
        return x, hidden

    def forward(
        self,
        stacked_logmel: torch.Tensor,
        dvec: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run the model.

        Args:
            stacked_logmel: (B, T, n_mels * frame_stack) — right-aligned.
            dvec:           (B, hp.d_vector_dim) speaker embedding (already
                            256-dim) from `SpeakerEncoder`.
            hidden:         optional (h, c) tuple to seed the LSTM. Pass
                            back the previous return value for stateful
                            inference; omit (or pass None) to reset state
                            — correct for chunked training.

        Returns:
            mask:    (B, T, n_mels)   — raw sigmoid output, [0, 1]. The
                                        caller applies the mask in
                                        **linear** mel-energy space
                                        (see `apply_mask`); convert
                                        log-mel ↔ linear-mel with
                                        `voicefilter_lite.audio`.
            hidden:  (h, c) tuple     — pass back next call for stateful
                                        inference.
        """
        lstm_out, hidden = self.lstm_forward(stacked_logmel, dvec, hidden)
        mask = torch.sigmoid(self.out_proj(self.dropout(lstm_out)))
        return mask, hidden


def apply_mask(mask: torch.Tensor, cur_linear: torch.Tensor) -> torch.Tensor:
    """Paper §3.2 literal mask application: `Y_enh = M · Y` in linear-mel space.

    Pure elementwise multiplication in the **linear filterbank-energy**
    domain — paper §3.2 / §3.4 (the mask operates on `S_in`, the
    pre-log-mel filterbank energy). Exposed at module level so training,
    the streaming wrapper, and ONNX export all apply the mask
    identically.

    `cur_linear` must already be in linear-mel (use
    `voicefilter_lite.audio.log_mel_to_linear` to recover it from
    log-mel). The streaming wrapper blends the result with the raw
    linear-mel input at strength `w^(t)` to drive the adaptive
    suppression from §3.4 — see `streaming.py`.

    At `M=0` this returns exactly 0 (silence floor); at `M=1` it is the
    identity. Numerical caveat: when `M ≈ 0`, the output may be ~1e-7
    rather than exactly 0 due to the `floor` subtracted by
    `log_mel_to_linear` — callers comparing M=0 to exact 0 should
    compare with `atol=1e-6` or larger.
    """
    return mask * cur_linear