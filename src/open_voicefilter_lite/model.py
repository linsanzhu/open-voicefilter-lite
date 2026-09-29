"""VoiceFilter-Lite: streaming targeted voice-separation mask predictor.

Architecture (paper §3.2, §3.3, §4.2):
    Linear(d_raw + n_mels*frame_stack → hidden) → LayerNorm
        → LSTM (unidirectional) × num_lstm_layers
        → Linear(hidden → n_mels) → Sigmoid   # mask branch
        → Linear(hidden → 64) → ReLU → Linear(64 → 64) → ReLU → Linear(64 → 1)   # noise-type head

Mask is applied in **linear-mel** space (paper §3.2), not in log-mel — see
`_apply_mask`. The noise-type head emits per-frame logits for an "overlapped
speech?" binary classifier; at inference it drives adaptive suppression
strength `w^(t)` (paper §3.4) — see `streaming.py`.
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

        # Project raw ECAPA (192-d for `spkrec-ecapa-voxceleb`) to hp.d_vector_dim.
        # Attribute name kept as `dvec_proj` to match existing checkpoints.
        self.dvec_proj = nn.Linear(hp.speaker_encoder_raw_dim, hp.d_vector_dim)

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

        self.out_proj = nn.Linear(hp.hidden_dim, hp.n_mels)

        # Per-frame "is this frame overlapped speech?" head (paper §4.2). ~20.6 K params.
        # Attribute name `alpha_head` kept for checkpoint compatibility.
        self.alpha_head = nn.Sequential(
            nn.Linear(hp.hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),             nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Xavier init on every Linear; LSTM defaults are fine.
        for m in (self.dvec_proj, self.in_proj, self.out_proj,
                  *(c for c in self.alpha_head if isinstance(c, nn.Linear))):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # Break the `mask=0.5` saddle of the SI-SDR alpha-projection.
        # Without this, the model initialises at exactly the sigmoid
        # neutral point — every per-bin logit starts at 0, so the mask is
        # uniformly 0.5 from step 1, gradient at that point is zero, and
        # the optimiser sits there indefinitely. The +0.5 bias puts the
        # initial mask at sigmoid(0.5) ≈ 0.62 — close enough to 0.5 that
        # SI-SDR's gradient direction is well-defined, but well inside the
        # sigmoid's high-slope region (d sigmoid/d z at 0.5 ≈ 0.25 vs ≈ 0.1
        # at z=2), so gradient updates actually move the mask. +2.0 looked
        # promising but pushed the sigmoid to its saturation shoulder
        # (gradient ≈ 0.1), where the noisy SI-SDR loss couldn't budge it.
        nn.init.constant_(self.out_proj.bias, 0.5)

    def forward(
        self,
        stacked_logmel: torch.Tensor,
        dvec: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Run the model.

        Args:
            stacked_logmel: (B, T, n_mels * frame_stack) — right-aligned.
            dvec:           (B, hp.speaker_encoder_raw_dim) raw ECAPA embedding.
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

        dvec_tiled = self.dvec_proj(dvec).unsqueeze(1).expand(-1, T, -1)
        x = self.dropout(self.in_norm(self.in_proj(
            torch.cat([stacked_logmel, dvec_tiled], dim=-1)
        )))
        x, hidden = self.lstm(x, hidden)

        mask = torch.sigmoid(self.out_proj(self.dropout(x)))
        frame_logits = self.alpha_head(x).squeeze(-1)

        # Right-aligned stacking: the current frame is the last slot.
        cur_idx = self.hp.frame_stack - 1
        cur = stacked_logmel[..., cur_idx * self.hp.n_mels : (cur_idx + 1) * self.hp.n_mels]
        enhanced = self._apply_mask(mask, cur)
        return enhanced, hidden, frame_logits, mask

    def _apply_mask(self, mask: torch.Tensor, cur_logmel: torch.Tensor) -> torch.Tensor:
        """Mask in linear-mel space (paper §3.2).

        mask=0 on silence ⇒ log(floor)≈-13.8 (not 0!).
        mask=1 is the identity on log-mel.
        Conversion: mel = exp(log_mel) - floor.
        """
        floor = self.hp.log_mel_floor
        mix_mel = (torch.exp(cur_logmel) - floor).clamp(min=0.0)
        return torch.log(mask * mix_mel + floor)