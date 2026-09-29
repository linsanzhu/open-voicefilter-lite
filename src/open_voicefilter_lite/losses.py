"""Losses for VoiceFilter-Lite.

Two backends (paper §3.3, §3.4), both combined with the per-frame hinge loss
for the noise-type predictor (paper §4.2):

  A. `asymmetric_l2_loss` — the paper's sign-based asymmetric L2, which
     amplifies the penalty on *over-suppression* errors (model dropped
     target energy) by α² (paper Tables 1 & 2 use α = 10). The asymmetry
     is on the **sign of the residual** `target − pred`, not on the
     target's energy.

  B. `sisdr_loss_on_logmel` — SI-SDR computed in the linear-mel domain
     (after `exp` of the model's log-mel output). Scale-invariant and
     symmetric (no 100× over-suppression penalty), so the gradient at
     mask ≈ 1 is non-zero. That breaks the `AsymmetricL2Loss` local
     minimum where the mask collapses to pass-through (mask ≈ 1, loss
     stuck at ≈ 5) — the failure mode this loss is designed to escape.

The combined objective is:
    L_total = L_main + hinge_loss_weight · L_hinge
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


# ---- SI-SDR on linear-mel (alternative to asymmetric L2) ----


def sisdr_loss_on_logmel(
    enhanced_logmel: torch.Tensor,
    target_logmel: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Scale-invariant SDR computed in the linear-mel domain.

    Converts log-mel → linear-mel via `exp`, then per-sample:

        target_v   = target - mean(target)        # per-sample mean removal
        est_v      = est    - mean(est)
        alpha      = <est_v, target_v> / <target_v, target_v>
        proj       = alpha * target_v             # signal projection
        noise      = est_v - proj                 # distortion residual
        SDR        = 10·log10( ||proj||² / ||noise||² )

    Returns −mean(SDR) so smaller = better (matches the train-loop's
    "loss goes down" convention).

    Why linear-mel and not log-mel? `exp(log_mel)` is positive and gives
    a true energy/power quantity; mean-removal then centres it. SI-SDR on
    raw log-mel has an unbounded negative tail (silence frames log(0+eps)
    dominate the projection) and gives unstable gradients on the floor
    region. Linear-mel avoids both.

    Why this beats asymmetric L2 here: the α²=100 over-suppression penalty
    in `asymmetric_l2_loss` makes `mask=1` a strong attractor (loss ≈ 5
    flat regardless of speech content). SI-SDR is symmetric: any non-zero
    mask < 1 immediately reduces loss, so the gradient at mask ≈ 1 is no
    longer zero. That's what allows the mask to actually move off 1.

    Args:
        enhanced_logmel, target_logmel: (B, T, F) log-mel. Both must be
            on the same device/dtype.
        eps: numerical floor for divisions (default 1e-7).

    Returns:
        Scalar loss (higher = worse; lower = better).
    """
    if enhanced_logmel.shape != target_logmel.shape:
        raise ValueError(
            f"[sisdr_loss_on_logmel] shape mismatch: "
            f"{tuple(enhanced_logmel.shape)} vs {tuple(target_logmel.shape)}"
        )

    # exp(log_mel) is always > 0; the "mean removal" below then centres it.
    enhanced = torch.exp(enhanced_logmel)
    target = torch.exp(target_logmel)

    # Flatten spectral+time so each (B,) sample has a single vector.
    target_v = target.flatten(1)                 # (B, T*F)
    enhanced_v = enhanced.flatten(1)             # (B, T*F)

    target_v = target_v - target_v.mean(dim=-1, keepdim=True)
    enhanced_v = enhanced_v - enhanced_v.mean(dim=-1, keepdim=True)

    target_energy = target_v.pow(2).sum(dim=-1, keepdim=True) + eps
    alpha = (enhanced_v * target_v).sum(dim=-1, keepdim=True) / target_energy
    proj = alpha * target_v
    noise = enhanced_v - proj

    proj_energy = proj.pow(2).sum(dim=-1)
    noise_energy = noise.pow(2).sum(dim=-1)
    sdr = 10.0 * torch.log10((proj_energy + eps) / (noise_energy + eps))
    return -sdr.mean()


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
            pred:          (B, T, F) enhanced log-mel from `VoiceFilterLite.forward`.
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
        mask: Optional[torch.Tensor] = None,   # accepted for symmetry with SISDRMelLoss; unused
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


class SISDRMelLoss(nn.Module):
    """Combined objective: SI-SDR on linear-mel + per-frame hinge (§3.4)
    + optional mask-binary regulariser.

    Total loss = L_sisdr
                 + hp.hinge_loss_weight · L_hinge
                 - hp.mask_binary_reg_weight · L_bimodal

    `L_bimodal` is the per-bin Bernoulli entropy of the mask, in nats:

        H(mask) = -E[ mask·log(mask) + (1-mask)·log(1-mask) ]
        L_bimodal = -H(mask)        # ≤ 0, = 0 iff mask ∈ {0, 1}

    Subtracting `L_bimodal` from the loss (i.e. ADDING `-L_bimodal`) means:
    bimodal masks (close to 0 or 1) REDUCE the loss, while masks stuck at
    0.5 are penalised. Without this term, SI-SDR alone lets the model
    collapse to mask ≈ 0.5 — the alpha-projection at mask=0.5 and mask=1
    give nearly the same SI-SDR when target and interference are similarly
    shaped, so 0.5 is a saddle point the optimiser happily sits in.

    Note: this regulariser doesn't tell the model which direction to go
    (towards 0 or 1); that's the SI-SDR's job. It only forbids sitting at
    0.5 — combined with SI-SDR, the model has to commit to either passing
    target through (mask=1 on active frames) or suppressing interference
    (mask=0 elsewhere), and the gradient picks the right direction.

    Replaces `AsymmetricL2Loss` to escape the `mask ≈ 1` collapse. See
    `sisdr_loss_on_logmel` docstring for why this works where L2 fails.
    """

    def __init__(self, hp: HParams):
        super().__init__()
        self.sisdr_eps = hp.sisdr_eps
        self.mask_binary_reg_weight = hp.mask_binary_reg_weight
        self.use_noise_type_predictor = hp.use_noise_type_predictor
        self.hinge_loss_weight = hp.hinge_loss_weight

    @staticmethod
    def _mask_bimodal_loss(mask: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Return -H(mask), a scalar ≤ 0; 0 iff every bin ∈ {0, 1}."""
        m = mask.clamp(min=eps, max=1.0 - eps)
        # Per-bin Bernoulli entropy, summed then negated.
        entropy = -(m * torch.log(m) + (1.0 - m) * torch.log(1.0 - m))
        # We want to MAXIMISE entropy (push mask away from 0.5). The
        # subtraction in the forward() below encodes that as "decrease
        # loss when entropy is high", so we return -entropy here so the
        # caller can write `loss - weight * (-entropy) = loss + weight * entropy`.
        # Actually we want to MINIMISE -entropy, so the caller subtracts
        # it: `loss - weight * bimodal_term` where bimodal_term = -H ≥ 0
        # (we return -H directly).
        return -entropy.mean()                                       # ≥ 0

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        frame_logits: Optional[torch.Tensor] = None,
        frame_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss_sisdr = sisdr_loss_on_logmel(pred, target, eps=self.sisdr_eps)

        loss_bimodal = pred.new_zeros(())
        if mask is not None and self.mask_binary_reg_weight > 0.0:
            loss_bimodal = self._mask_bimodal_loss(mask, eps=self.sisdr_eps)

        loss_hinge = pred.new_zeros(())
        if (
            self.use_noise_type_predictor
                and frame_logits is not None
                and frame_labels is not None
        ):
            loss_hinge = hinge_loss(frame_logits, frame_labels)

        return (
            loss_sisdr
            + self.hinge_loss_weight * loss_hinge
            - self.mask_binary_reg_weight * loss_bimodal
        )

    def forward_with_components(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        frame_logits: Optional[torch.Tensor] = None,
        frame_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        loss_sisdr = sisdr_loss_on_logmel(pred, target, eps=self.sisdr_eps)

        loss_bimodal = pred.new_zeros(())
        if mask is not None and self.mask_binary_reg_weight > 0.0:
            loss_bimodal = self._mask_bimodal_loss(mask, eps=self.sisdr_eps)

        loss_hinge = pred.new_zeros(())
        if (
            self.use_noise_type_predictor
                and frame_logits is not None
                and frame_labels is not None
        ):
            loss_hinge = hinge_loss(frame_logits, frame_labels)

        total = (
            loss_sisdr
            + self.hinge_loss_weight * loss_hinge
            - self.mask_binary_reg_weight * loss_bimodal
        )
        return {
            "loss": total,
            "loss_sisdr": loss_sisdr.detach(),
            "loss_bimodal": loss_bimodal.detach(),
            "loss_hinge": loss_hinge.detach(),
        }


def build_loss(hp: HParams) -> nn.Module:
    """Pick the loss module based on `hp.loss_type`.

    `"asym_l2"`     → `AsymmetricL2Loss` (paper-faithful, but vulnerable to
                      the mask=1 collapse that motivated this branch).
    `"sisdr_mel"`   → `SISDRMelLoss` (linear-mel SI-SDR; escape hatch for
                      when asymmetric L2 can't escape the attractor).
    """
    if hp.loss_type == "asym_l2":
        return AsymmetricL2Loss(hp)
    if hp.loss_type == "sisdr_mel":
        return SISDRMelLoss(hp)
    raise ValueError(
        f"[losses] unknown loss_type={hp.loss_type!r}; "
        "expected 'asym_l2' or 'sisdr_mel'."
    )