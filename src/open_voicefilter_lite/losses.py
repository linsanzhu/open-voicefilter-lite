"""Losses for VoiceFilter-Lite.

Two terms (paper §3.3, §4.2):

    1. `asymmetric_l2_loss` — the soft mask loss that drives the actual
       suppression output. Penalises over-suppression more than
       under-suppression via the α term on (1-M)·diff².
    2. `hinge_loss` — per-frame binary classification loss for the
       noise-type predictor (alpha_head), trained to predict "is this
       frame overlapped speech?". At inference its sigmoid drives the
       adaptive suppression `w^(t)` (§3.4).

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
    alpha: float = 4.0,
    beta: float = 0.0,
    lam: float = 5.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Asymmetric L2 loss penalising over-suppression (paper §3.3).

    Loss = mean( (M · (Y_pred - Y_target))^2 )
         + α · mean( ((1-M) · (Y_pred - Y_target))^2 )

    where M = sigmoid(lam * (Y_target - beta)) is a soft mask derived from
    the target's energy. The (1-M) term is weighted by `α > 1` so the
    model is *more* penalised when it under-suppresses than when it
    over-suppresses — i.e., it is encouraged to err on the side of
    passing through the input when in doubt. Paper Tables 1 & 2 use α=10.

    Args:
        pred:   (B, T, F) predicted enhanced log-mel.
        target: (B, T, F) clean target log-mel (same shape as pred).
        alpha:  scalar weight on the over-suppression term (paper: 10).
        beta:   threshold for the soft mask. Defaults to 0.0 (log-mel floor ≈ 0).
        lam:    sharpness of the soft mask.
        reduction: "mean" (default) or "none" (debug per-frame loss).
    """
    diff = pred - target                                # (B, T, F)
    soft_mask = torch.sigmoid(lam * (target - beta))    # (B, T, F)

    under = soft_mask * diff                            # (B, T, F)
    over = (1.0 - soft_mask) * diff                     # (B, T, F)

    under_sq = under ** 2                               # (B, T, F)
    over_sq = over ** 2                                 # (B, T, F)

    loss = under_sq.mean() + alpha * over_sq.mean()

    if reduction == "none":
        # Per-frame reconstruction loss for debugging.
        per_frame = ((under + alpha * over) ** 2).mean(dim=-1)
        return per_frame

    return loss


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
    """Combined objective: asymmetric L2 + per-frame hinge (§3.3 + §4.2).

    Total loss = L_asym + hp.hinge_loss_weight · L_hinge

    When `hp.use_noise_type_predictor=False`, the hinge term is skipped
    (the predictor is not trained) and the loss is pure L_asym.
    """

    def __init__(self, hp: HParams):
        super().__init__()
        self.alpha = hp.loss_alpha
        self.beta = hp.loss_beta
        self.lam = hp.loss_lambda
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
            frame_logits:  (B, T) raw logits from alpha_head; required when
                           use_noise_type_predictor=True.
            frame_labels:  (B, T) in {-1, +1}; required when
                           use_noise_type_predictor=True.

        Returns:
            Scalar loss (or dict with 'loss', 'loss_asym', 'loss_hinge' if
            `return_components=True`).
        """
        loss_asym = asymmetric_l2_loss(
            pred, target,
            alpha=self.alpha,
            beta=self.beta,
            lam=self.lam,
        )

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
        loss_asym = asymmetric_l2_loss(
            pred, target,
            alpha=self.alpha,
            beta=self.beta,
            lam=self.lam,
        )

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