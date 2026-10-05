"""Losses for VoiceFilter-Lite.

The combined objective is paper §3.3 (asymmetric L2) + §4.2 (per-frame
hinge loss for the noise-type predictor):

    L_total = L_asym + hp.noise_type_loss_weight · L_noise_type

    The asymmetric L2 (paper §3.3) is the only mask-supervision term; the
    noise-type predictor is the second term. There is no auxiliary
    regulariser on `M`.

`L_asym` is the literal paper-Eq.3 form `Σ_t Σ_f g_asym(S_cln − S_enh, α)²`
with `g_asym(x, α) = α·x if x > 0 else x` and α = 10 (paper Tables 1 & 2).
`L_noise_type` is the per-frame binary hinge loss for the "overlapped
speech?" predictor that paper §4.2 explicitly specifies as a **hinge** loss.
Labels are in `{−1, +1}`; loss = `mean( max(0, margin − y·logit) )`.

The d-vector speaker encoder is trained with the **GE2E** (Generalized
End-to-End) objective from Wan et al., 2018 (https://arxiv.org/pdf/1710.10467.pdf,
Eq. 6) — see `GE2ELoss`. The implementation mirrors Google's official
`lingvo/sidlingvo/loss_layers.py::GEnd2EndSoftmaxLayer`:

  https://github.com/google/speaker-id/blob/master/lingvo/sidlingvo/loss_layers.py
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from voicefilter_lite.hparams import HParams


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

    So ``L_asym = Σ_t Σ_f g_asym(x, α)² = Σ_t Σ_f ( α²·x²  if x>0  else  x² )``.

    The α² amplification on over-suppression errors is the central trick:
    ASR hates dropped speech far more than residual noise, so the model is
    trained to err on the side of *passing through* the input when it is
    uncertain. Paper Tables 1 & 2 use α = 10 (→ 100× amplification on
    over-suppression).

    Domain note: paper §3.3 / §3.4 work on the **linear filterbank-
    energy** spectrograms (`S_cln`, `S_enh`, `S_in`), not log-mel.
    Callers should pass `pred = M · S_in_linear` and
    `target = S_cln_linear` — see `voicefilter_lite.audio.log_mel_to_linear`.
    The loss itself is domain-agnostic (same algebraic form for any
    positive-definite energy representation); running it on log-mel would
    change the meaning of the asymmetry because the residual direction
    would no longer be the energy difference.

    NB on the noise-type companion: the paper's L_asym is summed over all
    frames × bins, but the hinge term is per-frame (one label per frame).
    The codebase keeps hinge as a per-frame mean (so it sits in the
    ~[0, 1+] band of a working hinge classifier). The asymmetry in
    reduction (`sum` vs `mean`) is absorbed by `hp.noise_type_loss_weight`
    — see `hparams.py` for the calibrated default.

    Args:
        pred:   (B, T, F) predicted enhanced linear-mel (M · S_in_linear).
        target: (B, T, F) clean target linear-mel (same shape as pred).
        alpha:  asymmetry factor (paper: 10). Effective amplification is α².
    """
    diff = target - pred                                # (B, T, F)  > 0 ⟹ over-suppression
    g = torch.where(diff > 0, alpha * diff, diff)       # (B, T, F)  g_asym
    return (g ** 2).sum(dim=(1,2)).mean()


def hinge_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Binary hinge loss for the per-frame noise-type predictor (paper §4.2).

    Paper §4.2 (Wang et al., Interspeech 2020) explicitly specifies the
    noise-type predictor is trained with a **hinge loss** for binary
    classification. Labels are in `{-1, +1}` (delivered as-is by
    `LocalTripletStream`); +1 = frame is overlapped speech, -1 =
    pass-through / no overlap.

    Standard hinge form: `mean( max(0, margin − y · logit) )`. The margin
    defaults to 1.0 (the conventional SVM margin). Per-frame mean so the
    loss sits in roughly the `[0, 1+]` band — well below the asym
    `Σ_t Σ_f` term without heavy re-weighting.

    Args:
        logits: (B, T) raw scores from `alpha_head`. Operates on pre-sigmoid
                values; the inference-time sigmoid→`f_adapt` (§3.4 Eq. 5) is
                independent of this loss.
        labels: (B, T) in {-1, +1}. Labels are clamped to [-1, 1] defensively
                (in case a stray 0 leaks through upstream).
        margin: standard hinge margin (default 1.0).

    Returns:
        Scalar loss = mean(max(0, margin − labels · logits)).
    """
    if logits.shape != labels.shape:
        raise ValueError(
            f"[hinge_loss] shape mismatch: logits {tuple(logits.shape)} "
            f"vs labels {tuple(labels.shape)}"
        )
    per_label = labels.to(dtype=logits.dtype, device=logits.device).clamp(
        min=-1.0, max=1.0,
    )
    elementwise = torch.clamp(margin - per_label * logits, min=0.0)
    return elementwise.mean()


class AsymmetricL2Loss(nn.Module):
    """Paper §3.3 asymmetric L2 + per-frame hinge (§4.2).

    Total loss = L_asym + hp.noise_type_loss_weight · L_noise_type

    `L_asym` amplifies over-suppression errors (model's enhanced < target)
    by `α²` so the model errs on the side of passing input through when
    uncertain — see `asymmetric_l2_loss` docstring for the full derivation.
    Both `pred` and `target` must be in **linear** filterbank-energy
    space (paper §3.3 works on `S_cln`/`S_enh`/`S_in`, not log-mel); the
    training loop converts via `voicefilter_lite.audio.log_mel_to_linear`.
    `L_noise_type` trains the noise-type predictor (§4.2): per-frame
    binary classification of "is this frame overlapped speech?",
    `y ∈ {−1, +1}`. Paper §4.2 explicitly specifies **hinge** for the
    predictor — see `hinge_loss` docstring.

    When `hp.use_noise_type_predictor=False`, the hinge term is skipped
    (the predictor is not trained) and the loss reduces to `L_asym` alone.

    **α annealing** (paper §3.3 trap mitigation). The constant-α form is
    prone to the "always emit M≈1 to dodge α² penalty" collapse (paper
    §3.3 trap). The hyperparameters

      * `loss.alpha_start`     — α at step 0 (default 1.0 → pure MSE)
      * `loss.alpha_ramp_steps`— # steps to ramp from `alpha_start` to
        `alpha` (default 0 = constant α, paper-faithful)

    drive a linear ramp α(t) when `alpha_ramp_steps > 0`:

      * t ≤ 0           → α_start
      * 0 < t < ramp    → α_start + (α - α_start) · t / ramp
      * t ≥ ramp        → α

    `set_step(step)` must be called by the training loop at least once
    per logging window (typically every `hp.log_every` steps) so
    `forward` / `forward_with_components` use the α for the current
    step. The call is cheap (one float assignment); caching it on the
    module avoids recomputing the linear ramp on every forward.
    """

    def __init__(self, hp: HParams):
        super().__init__()
        self.alpha = hp.loss_alpha                  # target α (held after ramp)
        self.alpha_start = hp.loss_alpha_start      # α at step 0
        self.alpha_ramp_steps = hp.loss_alpha_ramp_steps  # 0 = disabled
        self.use_noise_type_predictor = hp.use_noise_type_predictor
        self.noise_type_loss_weight = hp.noise_type_loss_weight
        # Step-aware α. The training loop calls `set_step(step)` per log
        # window; before the first call, α stays at `alpha_start` so
        # pre-training smoke tests / debug forwards behave consistently.
        self._alpha_current: float = float(self.alpha_start)
        self._step_current: int = -1

    # --------------------------------------------------------------- schedule
    def set_step(self, step: int) -> float:
        """Update the step counter and recompute α for the current step.

        Idempotent: calling with the same `step` returns the cached α.
        Cheap enough to call every `log_every` window; no-op beyond.

        Returns the newly-set α (handy for `writer.add_scalar`).
        """
        if step == self._step_current:
            return self._alpha_current
        self._step_current = int(step)
        self._alpha_current = self._alpha_at(int(step))
        return self._alpha_current

    @property
    def alpha_current(self) -> float:
        """α the next forward will use (set via `set_step`)."""
        return self._alpha_current

    def _alpha_at(self, step: int) -> float:
        """Pure-function ramp: α at training step `step`.

        Linear interpolation from `alpha_start` (step 0) to `alpha`
        (step `alpha_ramp_steps`); held constant outside the ramp
        window. When `alpha_ramp_steps <= 0` the ramp is disabled and
        α is held at the target value `alpha` end-to-end.
        """
        if self.alpha_ramp_steps <= 0:
            return float(self.alpha)
        if step <= 0:
            return float(self.alpha_start)
        if step >= self.alpha_ramp_steps:
            return float(self.alpha)
        frac = step / float(self.alpha_ramp_steps)
        return float(self.alpha_start + (self.alpha - self.alpha_start) * frac)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        frame_logits: Optional[torch.Tensor] = None,
        frame_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute total loss."""
        loss_asym = asymmetric_l2_loss(pred, target, alpha=self._alpha_current)

        loss_noise_type = pred.new_zeros(())
        if (
            self.use_noise_type_predictor
                and frame_logits is not None
                and frame_labels is not None
        ):
            loss_noise_type = hinge_loss(frame_logits, frame_labels)

        return loss_asym + self.noise_type_loss_weight * loss_noise_type

    def forward_with_components(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        frame_logits: Optional[torch.Tensor] = None,
        frame_labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Like `forward` but returns each component for logging.

        The asym L2 is decomposed into `loss_asym_over` (over-suppression,
        amplified by α²) and `loss_asym_under` (under-suppression). Watching
        the ratio helps detect when the model collapses to pass-through:
        if `over >> under` for many steps it's because the model is avoiding
        the α² penalty by emitting M≈1 (the trap paper §3.3 warns about).
        """
        alpha = self._alpha_current
        diff = target - pred
        # Match `asymmetric_l2_loss`'s `torch.where(diff > 0, α·diff, diff)`
        # split so the two terms are reported separately.
        over_mask = (diff > 0).to(dtype=diff.dtype)
        loss_asym_over = ((alpha * diff).pow(2) * over_mask).sum(
            dim=(1, 2)
        ).mean()
        loss_asym_under = (diff.pow(2) * (1.0 - over_mask)).sum(
            dim=(1, 2)
        ).mean()
        loss_asym = loss_asym_over + loss_asym_under

        loss_noise_type = pred.new_zeros(())
        if (
            self.use_noise_type_predictor
                and frame_logits is not None
                and frame_labels is not None
        ):
            loss_noise_type = hinge_loss(frame_logits, frame_labels)

        total = loss_asym + self.noise_type_loss_weight * loss_noise_type
        return {
            "loss": total,
            "loss_asym": loss_asym.detach(),
            "loss_asym_over": loss_asym_over.detach(),
            "loss_asym_under": loss_asym_under.detach(),
            "loss_noise_type": loss_noise_type.detach(),
        }


def build_loss(hp: HParams) -> nn.Module:
    """Return the paper-faithful `AsymmetricL2Loss` (paper §3.3 + §4.2).

    The loss family is fixed to the asymmetric L2 the paper uses; we no
    longer carry alternative backends.
    """
    return AsymmetricL2Loss(hp)


# ============================================================================
#  GE2E loss — d-vector speaker encoder training (Wan et al., 2018)
# ============================================================================
#
# Ported from Google's official `lingvo/sidlingvo/loss_layers.py`
# (`GEnd2EndSoftmaxLayer`, https://github.com/google/speaker-id).  See
# https://arxiv.org/pdf/1710.10467.pdf (Eq. 6) for the paper.
#
# This is the canonical Google implementation — no alternative backends:
#     - `ComputeSimilaritySplit` (unbiased centroids; M must be even)
#     - learnable linear transform `score = w · sim + b` (init w=10, b=-5)
#     - L2-normalisation done inside the module, matching
#       `tf.nn.l2_normalize` in the TF source.


def _ge2e_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    """Cosine-similarity score matrix — Google's `ComputeSimilaritySplit`.

    For each speaker, the M utterances are split in half: one half is
    the *enrollment* set whose mean becomes the centroid, the other
    half is the *test* set.  We then swap the roles (so each utterance
    serves as test for half the time) and concatenate.  M must be even;
    for `M = 2` this collapses to strict leave-one-out.

    Returns `(N*M, N)` (rows = utts, cols = speaker centroids).
    """
    if embeddings.dim() != 3:
        raise ValueError(
            f"[_ge2e_similarity] expected (N, M, D) embeddings, got "
            f"{tuple(embeddings.shape)}"
        )
    N, M, D = embeddings.shape
    if M % 2 != 0:
        raise ValueError(
            f"[_ge2e_similarity] ComputeSimilaritySplit needs even M, got M={M}"
        )

    # `F.normalize(..., dim=-1)` is the PyTorch equivalent of
    # `tf.nn.l2_normalize(..., axis=1)`: each row of the (N*M, D) matrix
    # has unit L2 norm.
    flat = F.normalize(embeddings.reshape(N * M, D), dim=-1, eps=1e-8)
    e3d = flat.reshape(N, M, D)

    # Match Google's `_GetCentroidAndData(part)` for part ∈ {0, 1}:
    # `e3d[:, part::2, :]` selects every-other utt.  Half of M is used
    # as test, the other half as enrollment.
    def _half(part: int) -> torch.Tensor:
        test = e3d[:, part::2, :]              # (N, M/2, D) = test part
        enroll = e3d[:, 1 - part::2, :]        # (N, M/2, D) = enrollment
        cen_mean = enroll.mean(dim=1)           # (N, D)
        cen_mean = F.normalize(cen_mean, dim=-1, eps=1e-8)
        # (N, M/2) × (N, D)^T -> (N, M/2, N) then reshape (N*M/2, N)
        scores = torch.matmul(test, cen_mean.transpose(0, 1))
        return scores.reshape(N * (M // 2), N)

    scores_0 = _half(0)
    scores_1 = _half(1)
    # Stack along a new "row" axis and reshape to interleave the halves
    # row-wise — this matches Google's final
    # `tf.reshape(tf.stack([fst, snd], axis=1), [batch, num_spks])`
    # which puts half[0], half[1], half[0], half[1], ... back in row
    # order.
    N_half = N * (M // 2)
    stacked = torch.stack([scores_0, scores_1], dim=1)  # (N*M/2, 2, N)
    return stacked.reshape(N_half * 2, N)               # (N*M, N)


class GE2ELoss(nn.Module):
    """Google's official `GEnd2EndSoftmaxLayer` (GE2E, Wan et al., 2018).

    Reference:
        https://github.com/google/speaker-id/blob/master/lingvo/sidlingvo/loss_layers.py
    Paper:
        https://arxiv.org/pdf/1710.10467.pdf  (Eq. 6)

    Mirrors Google's TF implementation exactly:

      1. `ComputeSimilaritySplit` — M utts per speaker are split in
         half; one half is the enrollment set (centroid), the other
         half is the test set, then the roles are swapped.  M must be
         even.
      2. `TransformScoreAndLabel` — learnable linear transform
         `score = w · sim + b` (init `w = 10.0`, `b = −5.0`) applied
         before cross-entropy.
      3. `CalculateLoss` — softmax cross-entropy with targets
         `[0]*M + [1]*M + ... + [N-1]*M` (each speaker's M utts share
         the same target index).

    L2-normalisation is performed inside the module, matching
    `tf.nn.l2_normalize` in the TF source.  Callers don't need to
    pre-normalise the embeddings (it's idempotent anyway).

    Args:
        init_w: initial value of the linear-transform weight
                (Google: 10.0).
        init_b: initial value of the linear-transform bias
                (Google: -5.0).
    """

    def __init__(self, init_w: float = 10.0, init_b: float = -5.0) -> None:
        super().__init__()
        # Google's `_CreateLayerVariables` registers `theta.w` and
        # `theta.b` as Variables (trainable) — we mirror with
        # `nn.Parameter`.  Both flow into the optimiser.
        self.w = nn.Parameter(torch.tensor(float(init_w)))
        self.b = nn.Parameter(torch.tensor(float(init_b)))

    def extra_repr(self) -> str:
        # `self.w` / `self.b` are `nn.Parameter` with `requires_grad=True`;
        # `float()` on those triggers a UserWarning under recent PyTorch.
        # Detach first so the repr is side-effect-free.
        return f"init_w={float(self.w.detach()):.3f}, init_b={float(self.b.detach()):.3f}"

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute GE2E loss over `(N, M, D)` utterance embeddings.

        The `(N, M, D)` layout means N speakers × M utterances per
        speaker × D embedding dim.  Targets are implicitly
        `[0, 0, ..., 0, 1, 1, ..., 1, ..., N-1, ..., N-1]` with each
        speaker index repeated M times (matches Google's
        `spk_idx = tf.tile(tf.range(num_spks_per_batch)[:, None],
        [1, num_utts_per_spk]).flatten()`).
        """
        if embeddings.dim() != 3:
            raise ValueError(
                f"[GE2ELoss] expected (N, M, D) embeddings, got "
                f"{tuple(embeddings.shape)}"
            )
        N, M, _ = embeddings.shape
        if N < 2 or M < 2:
            raise ValueError(
                f"[GE2ELoss] need N >= 2 speakers and M >= 2 utts, "
                f"got N={N}, M={M}"
            )
        if M % 2 != 0:
            raise ValueError(
                f"[GE2ELoss] ComputeSimilaritySplit needs even M, got M={M}"
            )

        sim = _ge2e_similarity(embeddings)        # (N*M, N)
        score = self.w * sim + self.b             # Google's TransformScoreAndLabel
        targets = (
            torch.arange(N, device=embeddings.device)
            .unsqueeze(1)
            .expand(-1, M)
            .reshape(-1)
        )
        return F.cross_entropy(score, targets)    # Google's CalculateLoss


def build_ge2e_loss(init_w: float = 10.0, init_b: float = -5.0) -> GE2ELoss:
    """Construct a `GE2ELoss` with Google's default initialisation.

    This is the canonical GE2E used by Google's `GEnd2EndSoftmaxLayer`:
    split-batch similarity + learnable `w, b` (init 10 / −5).  No
    alternative backends.
    """
    return GE2ELoss(init_w=init_w, init_b=init_b)