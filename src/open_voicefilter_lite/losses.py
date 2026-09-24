"""Losses for VoiceFilter-Lite.

Two terms (paper §3.3, §3.4):

    1. `asymmetric_l2_loss` — the paper's sign-based asymmetric L2, which
       amplifies the penalty on *over-suppression* errors (model dropped
       target energy) by α² (paper Tables 1 & 2 use α = 10). The asymmetry
       is on the **sign of the residual** `target − pred`, not on the
       target's energy.
    2. `hinge_loss` — per-frame binary classification loss for the
       noise-type predictor (2 FC × 64 + sigmoid head), trained to
       predict "is this frame overlapped speech?". At inference its
       sigmoid drives the adaptive suppression `w^(t)` (§3.4).

The combined objective is:
    L_total = L_asym + hinge_loss_weight · L_hinge
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from open_voicefilter_lite.hparams import HParams


def asymmetric_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 10.0,
) -> torch.Tensor:
    """Asymmetric L2 loss penalising over-suppression (paper §3.3).

    Residual: ``x = S_cln − S_enh = target − pred``.

      - ``x > 0``  (over-suppression: model dropped target energy):
        penalty = ``α·x``  → contribution ``α²·x²``
      - ``x ≤ 0``  (under-suppression: residual interference):
        penalty = ``x``    → contribution ``x²``

    So ``L_asym = mean( g_asym(x, α)² ) = mean( α²·x²  if x>0  else  x² )``.

    The α² amplification on over-suppression errors is the central trick:
    ASR hates dropped speech far more than residual noise, so the model is
    trained to err on the side of *passing through* the input when it is
    uncertain. Paper Tables 1 & 2 use α = 10 (→ 100× amplification on
    over-suppression).

    Args:
        pred:   (B, T, F) predicted enhanced log-mel.
        target: (B, T, F) clean target log-mel (same shape as pred).
        alpha:  asymmetry factor (paper: 10). Effective amplification is α².
    """
    diff = target - pred                                # (B, T, F)  > 0 ⟹ over-suppression
    g = torch.where(diff > 0, alpha * diff, diff)       # (B, T, F)  g_asym
    return (g ** 2).mean()


def hinge_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Binary hinge loss for the per-frame noise-type predictor (§4.2).

    Args:
        logits: (B, T) raw scores from `alpha_head`. Hinge operates on
                pre-sigmoid values.
        labels: (B, T) in {-1, +1}. +1 = frame is overlapped speech,
                -1 = pass-through / no overlap.
        margin: standard hinge margin (default 1.0).

    Returns:
        Scalar loss = mean(max(0, margin - labels · logits))
    """
    if logits.shape != labels.shape:
        raise ValueError(
            f"[hinge_loss] shape mismatch: logits {tuple(logits.shape)} "
            f"vs labels {tuple(labels.shape)}"
        )
    # Clamp labels just in case (e.g., a stray 0 from upstream).
    per_label = labels.to(dtype=logits.dtype, device=logits.device).clamp(min=-1.0, max=1.0)
    elementwise = torch.clamp(margin - per_label * logits, min=0.0)
    return elementwise.mean()


class AsymmetricL2Loss(nn.Module):
    """Combined objective: asymmetric L2 + per-frame hinge (§3.3 + §3.4).

    Total loss = L_asym + hp.hinge_loss_weight · L_hinge

    When `hp.use_noise_type_predictor=False`, the hinge term is skipped
    (the predictor is not trained) and the loss is pure L_asym.
    """

    def __init__(self, hp: HParams):
        super().__init__()
        self.alpha = hp.loss_alpha
        self.use_noise_type_predictor = hp.use_noise_type_predictor
        self.hinge_loss_weight = hp.hinge_loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        frame_logits: Optional[torch.Tensor] = None,
        frame_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute total loss.

        Args:
            pred:          (B, T, F) enhanced log-mel from `VoiceFilterLite.forward_train`.
            target:        (B, T, F) clean target log-mel.
            frame_logits:  (B, T) raw logits from alpha_head (2 FC × 64);
                           required when use_noise_type_predictor=True.
            frame_labels:  (B, T) in {-1, +1}; required when
                           use_noise_type_predictor=True.

        Returns:
            Scalar loss.
        """
        loss_asym = asymmetric_l2_loss(pred, target, alpha=self.alpha)

        loss_hinge = pred.new_zeros(())
        if (
            self.use_noise_type_predictor
                and frame_logits is not None
                and frame_labels is not None
        ):
            loss_hinge = hinge_loss(frame_logits, frame_labels)

        return loss_asym + self.hinge_loss_weight * loss_hinge

    def forward_with_components(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        frame_logits: Optional[torch.Tensor] = None,
        frame_labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Like `forward` but returns each component for logging."""
        loss_asym = asymmetric_l2_loss(pred, target, alpha=self.alpha)

        loss_hinge = pred.new_zeros(())
        if (
            self.use_noise_type_predictor
                and frame_logits is not None
                and frame_labels is not None
        ):
            loss_hinge = hinge_loss(frame_logits, frame_labels)

        total = loss_asym + self.hinge_loss_weight * loss_hinge
        return {
            "loss": total,
            "loss_asym": loss_asym.detach(),
            "loss_hinge": loss_hinge.detach(),
        }