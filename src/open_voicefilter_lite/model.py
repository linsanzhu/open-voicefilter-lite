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

Mask is applied in **log-mel** space using paper §3.2's literal formula
`Y_enh(t,f) = M(t,f) · Y(t,f)` — direct multiplication in log-domain.
At `M=0` the enhanced output is 0; at `M=1` it is identity. The noise-
type head emits per-frame logits for an "overlapped speech?" binary
classifier; by default it consumes the **current frame's noisy log-mel**
(`S_in^(t)`, paper §3.4) rather than the LSTM output (§4.2 alternative).
At inference its sigmoid drives adaptive suppression `w^(t)` — see
`streaming.py`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from open_voicefilter_lite.hparams import HParams


class VoiceFilterLite(nn.Module):
    """Streaming target-speaker mask predictor.

    Forward signature:
        forward(stacked_logmel, dvec, hidden=None)
            → (enhanced, hidden, frame_logits, mask)

    `frame_logits` is the per-frame noise-type predictor output; training
    consumes it for the hinge loss, inference callers (`StreamingVoiceFilter`,
    ONNX) ignore it.
    `mask` is the raw sigmoid output (B, T, n_mels) ∈ [0, 1] — training uses
    it for diagnostic logging, inference callers ignore it.
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

        # Optional 1D CNN (paper §3.2). Paper §3.2 specifies "Convolutional
        # layers are 1D (kernels for frequency dimension only)" — the
        # kernel slides along the **frequency** axis (the last dim after
        # in_proj, which carries the per-frame feature) and never mixes
        # across time. Same Section 4.4 admits "most of our models
        # actually remove these CNN layers and purely consist of LSTM
        # layers", so the canonical 2.2 MB variant leaves the CNN out
        # (`use_cnn=False`).
        #
        # Paper Figure 2 topology: **2 stacked Conv1d** layers feeding into
        # the LSTM stack (paper-faithful `num_cnn_layers=2`). Implementation:
        # an `nn.Sequential` of `nn.Conv1d` modules (the paper's "1D CNN").
        # Conv1d's kernel slides along its `L` (length) axis, so we
        # reshape `(B, T, H)` → `(B*T, 1, H)` — each time step becomes
        # an independent 1-channel "signal" of length `H`, and the
        # K=3 kernel slides along that length (i.e. along the frequency
        # axis). Same-padding K//2 keeps H preserved. The H positions
        # are collapsed by mean-pooling in `forward` so the LSTM
        # downstream gets `(B, T, last_conv_out)`. The last Conv1d's
        # output channels are forced to `hidden_dim` so it feeds the
        # LSTM stack directly without an extra projection bridge.
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
        # 64 nodes for noise type prediction" — i.e. `Linear(80 → 64) →
        # ReLU → Linear(64 → 1)` (input projection to 64 hidden nodes,
        # then a single logit output). ~3.3 K params.
        #
        # Input source is paper-§3.4-literal by default:
        #   * `alpha_head_use_current_frame=True`  → input is S_in^(t), the
        #     current frame's noisy filterbank (n_mels=80 dims, paper §3.4).
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
                    # 0.6–0.8 range observed during the brief phase-②
                    # learning window before late-training bias drift pulls
                    # it back toward 1. The companion mask-saturation
                    # regulariser in `losses.AsymmetricL2Loss` keeps the
                    # late-training bias drift in check.
                    if m is self.out_proj:
                        nn.init.constant_(m.bias, -1.0)
                    else:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        stacked_logmel: torch.Tensor,
        dvec: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Run the model.

        Args:
            stacked_logmel: (B, T, n_mels * frame_stack) — right-aligned.
            dvec:           (B, hp.d_vector_dim) speaker embedding (already
                            256-dim) from `SpeakerEncoder`.
            hidden:         optional (h, c) tuple to seed the LSTM.

        Returns:
            enhanced:     (B, T, n_mels)   — log-mel of target speaker.
            hidden:       (h, c) tuple     — for stateful inference.
            frame_logits: (B, T)           — per-frame logits; ignore at inference.
            mask:         (B, T, n_mels)   — raw sigmoid output, [0, 1]. Training
                                         uses it for diagnostics; inference paths
                                         (StreamingVoiceFilter, ONNX) ignore it.
        """
        B, T, _ = stacked_logmel.shape

        dvec_tiled = dvec.unsqueeze(1).expand(-1, T, -1)
        x = self.dropout(self.in_norm(self.in_proj(
            torch.cat([stacked_logmel, dvec_tiled], dim=-1)
        )))
        # Optional 1D CNN (paper §3.2). Slides along the **frequency** axis
        # (the last dim of `x`, i.e. `hidden_dim`); never mixes across
        # time. View each time step as an independent 1-channel signal
        # of length `H` so `nn.Conv1d`'s L axis falls on the frequency
        # dim. After the conv H is preserved (same-padding K//2) and we
        # collapse it via mean to land at `(B, T, cnn_channels)` for
        # the LSTM downstream.
        if self.cnn is not None:
            x_per = x.reshape(-1, 1, x.shape[-1])              # (B*T, 1, H)
            x_conv = self.cnn(x_per)                           # (B*T, cnn_channels, H)
            x = x_conv.mean(dim=-1).reshape(x.shape[0], x.shape[1], -1)  # (B, T, cnn_channels)
        x, hidden = self.lstm(x, hidden)

        mask = torch.sigmoid(self.out_proj(self.dropout(x)))

        # Right-aligned stacking: the current frame is the last slot. We
        # extract it before calling the alpha_head because paper §3.4's
        # primary spec is to feed `f_adapt(S_in^(t))` — the noisy
        # filterbank at time t — into the noise-type predictor, NOT the
        # LSTM's hidden state. §4.2 also allows the LSTM-output variant;
        # we honour that via `hp.alpha_head_use_current_frame=False`.
        cur_idx = self.hp.frame_stack - 1
        cur = stacked_logmel[..., cur_idx * self.hp.n_mels : (cur_idx + 1) * self.hp.n_mels]
        head_in = cur if self.hp.alpha_head_use_current_frame else x
        frame_logits = self.alpha_head(head_in).squeeze(-1)

        enhanced = self._apply_mask(mask, cur)
        return enhanced, hidden, frame_logits, mask

    def _apply_mask(self, mask: torch.Tensor, cur_logmel: torch.Tensor) -> torch.Tensor:
        """Paper §3.2 literal mask application: `Y_enh = M · Y` in log-mel space.

        At `M=0` this returns exactly 0 (which is `log(1)`, not silence
        floor); at `M=1` it is the identity. Streaming callers
        (`StreamingVoiceFilter._apply_suppression`) blend the result
        with the raw input in linear-mel space at strength `w^(t)` to
        drive the adaptive suppression from §3.4 — see `streaming.py`.
        """
        return mask * cur_logmel