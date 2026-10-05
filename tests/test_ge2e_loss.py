"""Tests for `GE2ELoss` — the d-vector loss.

`GE2ELoss` is a faithful port of Google's official `GEnd2EndSoftmaxLayer`
(https://github.com/google/speaker-id/blob/master/lingvo/sidlingvo/loss_layers.py)
implementing Wan et al. 2018 (https://arxiv.org/pdf/1710.10467.pdf, Eq. 6):

  - `ComputeSimilaritySplit` — M utts per speaker are split in half
    (one half = enrollment, the other = test; swap and concatenate).
    M must be even.
  - `TransformScoreAndLabel` — learnable linear transform
    `score = w · sim + b` (init `w = 10.0`, `b = −5.0`).
  - `CalculateLoss` — softmax cross-entropy with targets
    `[0]*M + [1]*M + ... + [N-1]*M`.

Covers:
  1. The PyTorch port matches a hand-coded numpy re-implementation of
     the TF code path within 1e-7.
  2. Default config forward + backward works; learnable `w, b` are
     `nn.Parameter` and move during optimisation.
  3. Separable embeddings drive the loss down with training.
  4. M < 2 and odd-M cases are rejected.

The reference numbers come from a hand-coded numpy re-implementation of
the Google TensorFlow code path
(`ComputeSimilaritySplit` → `TransformScoreAndLabel` → softmax
cross-entropy) so this test is the executable transcript of that TF file.

Run with:
    python tests/test_ge2e_loss.py
"""

from __future__ import annotations

import numpy as np
import torch

from voicefilter_cli.losses import GE2ELoss, build_ge2e_loss


def _google_ref(emb: torch.Tensor, w: float, b: float) -> tuple[float, list[str]]:
    """Hand-coded reference (numpy) of Google's TF GE2E forward.

    Returns the scalar CE and the list of step names it exercised, so
    test failures can pinpoint which Google code block is diverging.
    """
    log: list[str] = []
    N, M, _D = emb.shape
    e = emb.detach().numpy().astype(np.float64)
    # `tf.nn.l2_normalize(e, axis=-1)` per utterance.
    e = e / np.linalg.norm(e, axis=-1, keepdims=True).clip(min=1e-8)
    log.append("l2_normalize")
    # `_GetCentroidAndData(0)` and `(1)` — Google's split trick.
    half = M // 2
    def part(p: int) -> np.ndarray:
        test = e[:, p::2, :].reshape(N * half, _D)
        enroll = e[:, 1 - p::2, :]
        cent = enroll.mean(axis=1)
        cent = cent / np.linalg.norm(cent, axis=-1, keepdims=True).clip(min=1e-8)
        return test @ cent.T  # (N*half, N)
    s0, s1 = part(0), part(1)
    log.append("ComputeSimilaritySplit")
    sim = np.empty((N * M, N), dtype=np.float64)
    for spk in range(N):
        for utt in range(half):
            sim[spk * M + 2 * utt]     = s0[spk * half + utt]
            sim[spk * M + 2 * utt + 1] = s1[spk * half + utt]
    # `TransformScoreAndLabel`: score = theta.w * sim + theta.b.
    score = w * sim + b
    log.append(f"TransformScoreAndLabel w={w} b={b}")
    # `CalculateLoss`: softmax cross-entropy with targets = repeat(N, M).
    targets = np.repeat(np.arange(N), M)
    m = score.max(axis=1, keepdims=True)
    lse = np.log(np.exp(score - m).sum(axis=1)) + m.squeeze(1)
    log_p = score[np.arange(N * M), targets] - lse
    log.append("CalculateLoss (XentLossFromLogits)")
    return float(-log_p.mean()), log


def test_matches_google_reference() -> None:
    """`GE2ELoss()` ≡ Google `GEnd2EndSoftmaxLayer` (numpy reference)."""
    torch.manual_seed(0)
    emb = torch.randn(4, 6, 8)
    ref, log = _google_ref(emb, w=12.5, b=0.0)
    got = GE2ELoss(init_w=12.5, init_b=0.0)(emb).item()
    print(f"  ref={ref:.6f} got={got:.6f} (steps: {' → '.join(log)})")
    assert abs(ref - got) < 1e-4, "split-batch CE must match Google within 1e-4"


def test_default_init() -> None:
    """Default `w=10, b=-5` and the `w, b` are `nn.Parameter`."""
    loss = GE2ELoss()
    assert isinstance(loss.w, torch.nn.Parameter)
    assert isinstance(loss.b, torch.nn.Parameter)
    assert loss.w.requires_grad and loss.b.requires_grad
    assert abs(float(loss.w.detach()) - 10.0) < 1e-6
    assert abs(float(loss.b.detach()) - (-5.0)) < 1e-6
    print(f"  [init] w={float(loss.w.detach()):.3f} "
          f"b={float(loss.b.detach()):.3f} "
          f"(both nn.Parameter with requires_grad=True)")


def test_forward_backward() -> None:
    """Default config forward + backward succeeds."""
    torch.manual_seed(0)
    emb = torch.randn(4, 6, 8)
    loss = GE2ELoss()(emb)
    loss.backward()
    assert torch.isfinite(loss).item()
    print(f"  [default] loss={loss.item():.4f} backward ok")


def test_learnable_wb_moves_during_training() -> None:
    """On separable embeddings the loss must drop with optimisation and
    `w, b` must move away from their init."""
    torch.manual_seed(1)
    N, M, D = 8, 4, 16
    mu = torch.randn(N, D) * 5.0
    noise = torch.randn(N, M, D) * 0.1
    e0 = (mu.unsqueeze(1) + noise).reshape(N, M, D).detach()
    ge2e = GE2ELoss()
    emb = torch.nn.Parameter(e0.clone())
    opt = torch.optim.Adam(list(ge2e.parameters()) + [emb], lr=0.1)
    init_w = ge2e.w.item()
    init_b = ge2e.b.item()
    l0 = ge2e(emb).item()
    for _ in range(50):
        opt.zero_grad()
        loss = ge2e(emb)
        loss.backward()
        opt.step()
    l1 = ge2e(emb).item()
    print(f"  [train] loss {l0:.3f} -> {l1:.3f}; "
          f"w {init_w:.3f} -> {ge2e.w.item():.3f}; "
          f"b {init_b:.3f} -> {ge2e.b.item():.3f}")
    assert l1 < l0, f"GE2E did not converge: {l0} -> {l1}"
    assert abs(ge2e.w.item() - init_w) > 1e-3, "w should move"
    assert abs(ge2e.b.item() - init_b) > 1e-3, "b should move"


def test_even_m_required() -> None:
    """M must be ≥ 2 and even; both smaller-than-2 and odd M raise."""
    for bad_shape in [(4, 1, 8), (4, 3, 8), (4, 5, 8), (2, 1, 8)]:
        try:
            GE2ELoss()(torch.randn(*bad_shape))
        except ValueError as e:
            print(f"  [shape={bad_shape}] ok raised: {e}")
        else:
            raise AssertionError(f"expected ValueError for {bad_shape}")
    # And the even case works.
    GE2ELoss()(torch.randn(4, 6, 8))
    print("  [shape=(4, 6, 8)] ok forward")


def test_build_factory() -> None:
    """`build_ge2e_loss` honours its `init_w`, `init_b` kwargs."""
    loss = build_ge2e_loss()
    assert isinstance(loss, GE2ELoss)
    assert abs(float(loss.w.detach()) - 10.0) < 1e-6
    assert abs(float(loss.b.detach()) - (-5.0)) < 1e-6
    loss2 = build_ge2e_loss(init_w=12.5, init_b=0.0)
    assert abs(float(loss2.w.detach()) - 12.5) < 1e-6
    assert abs(float(loss2.b.detach()) - 0.0) < 1e-6
    print(f"  [factory] default={loss.extra_repr()}")
    print(f"            custom={loss2.extra_repr()}")


def main() -> None:
    print("[1/6] matches Google TF reference (numpy re-implementation)")
    test_matches_google_reference()
    print("[2/6] default init (w=10, b=-5, both nn.Parameter)")
    test_default_init()
    print("[3/6] forward + backward")
    test_forward_backward()
    print("[4/6] learnable w/b moves during optimisation")
    test_learnable_wb_moves_during_training()
    print("[5/6] even-M enforcement; M<2 / odd-M rejected")
    test_even_m_required()
    print("[6/6] build_ge2e_loss factory")
    test_build_factory()
    print("== ALL GE2E TESTS PASSED ==")


if __name__ == "__main__":
    main()