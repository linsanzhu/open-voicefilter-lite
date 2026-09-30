"""Losses for VoiceFilter-Lite.

The combined objective is paper §3.3 (asymmetric L2) + §4.2 (per-frame hinge
for the noise-type predictor):

    L_total = L_asym + hp.hinge_loss_weight · L_hinge

`L_asym` is `mean( g_asym(target − pred, α)² )` with
`g_asym(x, α) = α·x if x > 0 else x` and α = 10 (paper Tables 1 & 2).
`L_hinge` is the binary hinge `mean(max(0, margin − y·logits))` over the
per-frame "overlapped speech?" logits, with `y ∈ {−1, +1}`.
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

    Paper §3.3 (Wang et al., 2020) defines the loss as
    ``L = Σ_t Σ_f ( g_asym(S_cln − S_enh, α) )²`` (Eq. 3 — literal sum
    over time and frequency bins, NOT a per-element mean). We honour that
    literally:

      - ``x > 0``  (over-suppression: model dropped target energy):
        penalty = ``α·x``  → contribution ``α²·x²``
      - ``x ≤ 0``  (under-suppression: residual interference):
        penalty = ``x``    → contribution ``x²``

    So ``L_asym = sum( g_asym(x, α)² ) = sum( α²·x²  if x>0  else  x² )``.

    The α² amplification on over-suppression errors is the central trick:
    ASR hates dropped speech far more than residual noise, so the model is
    trained to err on the side of *passing through* the input when it is
    uncertain. Paper Tables 1 & 2 use α = 10 (→ 100× amplification on
    over-suppression).

    NB on the hinge companion: the paper's L_asym is summed over all
    frames × bins, but the hinge loss is per-frame (one label per frame).
    The codebase keeps the hinge as a per-frame mean (so it sits in the
    same [0, margin] band as before). The asymmetry in reduction
    (`sum` vs `mean`) is absorbed by `hp.hinge_loss_weight` — see
    `hparams.py` for the calibrated default (`1e-5` after the switch to
    sum, vs the legacy `0.1` with mean).

    Args:
        pred:   (B, T, F) predicted enhanced log-mel.
        target: (B, T, F) clean target log-mel (same shape as pred).
        alpha:  asymmetry factor (paper: 10). Effective amplification is α².
    """
    diff = target - pred                                # (B, T, F)  > 0 ⟹ over-suppression
    g = torch.where(diff > 0, alpha * diff, diff)       # (B, T, F)  g_asym
    return (g ** 2).sum()


def hinge_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Binary hinge loss for the per-frame noise-type predictor (paper §4.2).

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
    """Paper §3.3 asymmetric L2 + per-frame hinge (§4.2) + mask-saturation
    regulariser.

    Total loss = L_asym + hp.hinge_loss_weight · L_hinge + mask_sat_term

    `L_asym` amplifies over-suppression errors (model's enhanced < target)
    by `α²` so the model errs on the side of passing input through when
    uncertain — see `asymmetric_l2_loss` docstring for the full derivation.
    `L_hinge` trains the noise-type predictor (§4.2): binary classification
    per frame of "is this frame overlapped speech?", `y ∈ {−1, +1}`.

    The optional `mask_sat_term` is a one-sided quadratic penalty activated
    element-wise when mask > `hp.mask_sat_target`. It prevents the
    "mask-collapse to 1" local minimum that the asymmetric L2 has at
    α ≥ 2 — see `hp.mask_sat_*` and `model.py` `out_proj.bias` init for
    the design notes.

    When `hp.use_noise_type_predictor=False`, the hinge term is skipped
    (the predictor is not trained) and the loss reduces to `L_asym +
    mask_sat_term`.
    """

    def __init__(self, hp: HParams):
        super().__init__()
        self.alpha = hp.loss_alpha
        self.use_noise_type_predictor = hp.use_noise_type_predictor
        self.hinge_loss_weight = hp.hinge_loss_weight
        # Mask-saturation regulariser. Element-wise one-sided quadratic:
        # `weight · Σ max(0, mask − target)²`. See `HParams` for the
        # weight/target rationale.
        self.mask_sat_target = hp.mask_sat_target
        self.mask_sat_weight = hp.mask_sat_weight

    def _mask_sat(self, mask: torch.Tensor) -> torch.Tensor:
        """One-sided quadratic penalty for mask values above `mask_sat_target`.

        Element-wise so the per-element gradient (`weight · 2 · (mask −
        target)`) is directly comparable to L_asym's per-element gradient;
        `mean()`-based regularisers would be ~numel× too weak under the
        sum-loss regime.
        """
        over = torch.clamp(mask - self.mask_sat_target, min=0.0)
        return self.mask_sat_weight * over.pow(2).sum()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        frame_logits: Optional[torch.Tensor] = None,
        frame_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute total loss."""
        loss_asym = asymmetric_l2_loss(pred, target, alpha=self.alpha)

        loss_hinge = pred.new_zeros(())
        if (
            self.use_noise_type_predictor
                and frame_logits is not None
                and frame_labels is not None
        ):
            loss_hinge = hinge_loss(frame_logits, frame_labels)

        loss_mask_sat = pred.new_zeros(())
        if mask is not None:
            loss_mask_sat = self._mask_sat(mask)

        return (
            loss_asym
            + self.hinge_loss_weight * loss_hinge
            + loss_mask_sat
        )

    def forward_with_components(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        frame_logits: Optional[torch.Tensor] = None,
        frame_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
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

        loss_mask_sat = pred.new_zeros(())
        if mask is not None:
            loss_mask_sat = self._mask_sat(mask)

        total = (
            loss_asym
            + self.hinge_loss_weight * loss_hinge
            + loss_mask_sat
        )
        return {
            "loss": total,
            "loss_asym": loss_asym.detach(),
            "loss_hinge": loss_hinge.detach(),
            "loss_mask_sat": loss_mask_sat.detach(),
        }


def build_loss(hp: HParams) -> nn.Module:
    """Return the paper-faithful `AsymmetricL2Loss` (paper §3.3 + §4.2).

    The loss family is fixed to the asymmetric L2 the paper uses; we no
    longer carry alternative backends.
    """
    return AsymmetricL2Loss(hp)