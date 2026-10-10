#!/usr/bin/env python3
"""Training loop for VoiceFilter-Lite, with optional EMA and the
noise-type predictor (see `configs/paper.yaml`).

Reads pre-baked wav files from `data/processed/` (produced by
`voicefilter_cli.prepare_dataset`). No MUSAN babble synthesis at training
time. D-vectors are filled lazily by `DVectorCache` on first encounter
of each new speaker — see `voicefilter_cli.dataset.DVectorCache`.

Usage
-----
    # Full training on Apple Silicon (default)
    voicefilter-cli train voicefilter-lite \\
        --config configs/paper.yaml --output-dir runs/vflite

    # Smoke test (50 steps, small batch)
    voicefilter-cli train voicefilter-lite --config configs/paper.yaml \\
        --output-dir runs/smoke --total-steps 50 --batch-size 2 --no-tensorboard

    # GPU box
    voicefilter-cli train voicefilter-lite --config configs/paper.yaml \\
        --device cuda --output-dir runs/cuda
"""

from __future__ import annotations

import math
import queue
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Iterable, Optional

import torch
from cyclopts import App, Parameter

# Suppress a benign Python 3.13 + macOS shutdown noise:
# `multiprocessing.resource_tracker` (loaded transitively by torch) scans
# for POSIX semaphores at atexit and flags Python 3.13's `_PyMutex`-backed
# GIL/threading primitives — they show up as `/mp-*` named semaphores that
# the tracker doesn't own. The interpreter manages them fine; the warning is
# cosmetic. See cpython#114347.
warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker.*leaked semaphore.*",
)
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from model.dataset import HfTripletStream, collate_triplets
from voicefilter_cli.hf import apply_hf_endpoint
from model.losses import build_loss, linear_mel_si_snr_per_sample
from model.audio import log_mel_to_linear
from voicefilter_lite.config import load_and_apply
from voicefilter_lite.hparams import HParams
from model.nn import VoiceFilterLite, apply_mask


# ---- Background prefetch ----


class PrefetchIterator:
    """Wraps an iterable so items are produced on a background thread.

    The main thread consumes items via `for x in PrefetchIterator(src)` —
    by the time the GPU finishes step N, batch N+1 is already built and
    sitting in the queue, so the GPU is never idle waiting for soundfile
    decodes / mel extractions / encoder forwards.

    Exceptions raised in the producer thread are re-raised on the consumer
    side at the next `__next__` call (a sentinel `None` would be ambiguous
    because batches can legitimately be `None`-containing dicts).
    """

    _SENTINEL = object()

    def __init__(self, source, max_prefetch: int = 2):
        self.source = source
        self.max_prefetch = max(1, max_prefetch)
        self._q: queue.Queue = queue.Queue(maxsize=self.max_prefetch)
        self._thread: Optional[threading.Thread] = None

    def _producer(self) -> None:
        try:
            for item in self.source:
                self._q.put(item)
        except BaseException as e:  # noqa: BLE001 — propagate to main
            self._q.put(e)
        finally:
            self._q.put(self._SENTINEL)

    def __iter__(self):
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(
                "[prefetch] PrefetchIterator is single-use; create a new one"
            )
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()
        return self

    def __next__(self):
        item = self._q.get()
        if item is self._SENTINEL:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        return item


# ---- LR schedule: linear warmup + cosine decay ----


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)


# ---- Device helpers ----


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


# ---- EMA ----


class ModelEMA:
    """Exponential moving average of a model's trainable parameters.

    Maintains an fp32 shadow copy. Updated after each optimiser step:
        shadow = decay * shadow + (1 - decay) * model_param

    During the first `warmup` steps, shadow is held equal to the live model
    (no averaging) — this avoids a slow ramp-up bias at training start.
    """

    def __init__(self, model: torch.nn.Module, decay: float, warmup: int = 0):
        self.decay = decay
        self.warmup = warmup
        self.step_count = 0
        # Deep-copy the state_dict so EMA is independent of the live model.
        self.shadow = {
            name: param.detach().clone().float()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.step_count += 1
        # Collect (shadow, param) pairs ONCE per call. Iterating
        # `named_parameters()` twice (warmup + steady-state) is the same
        # O(P) cost we used to pay inside the body; hoisting the
        # `requires_grad` filter + shadow/param lookup here lets the
        # `_foreach_*` batched kernels below hit a single launch each
        # instead of one Python-level launch per tensor.
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not named:
            return
        shadow_list = [self.shadow[n] for n, _ in named]
        param_list = [p.detach().float() for _, p in named]
        # Note: shadow lives on the same device as the live model
        # (see `__init__`: `param.detach().clone().float()`), so we
        # can skip the per-step device-alignment dance the old loop
        # did defensively. If `model.to(device)` is called after EMA
        # construction the shadows would diverge — keep the original
        # defensive comment above as the rule of thumb.
        try:
            if self.step_count <= self.warmup:
                # Hold EMA = model. Single batched memcpy.
                torch._foreach_copy_(shadow_list, param_list)
            else:
                d = self.decay
                # `shadow = decay · shadow + (1 − decay) · param` — one
                # fused `_foreach_mul_` then one `_foreach_add_` (≈half
                # the per-tensor Python overhead of the original loop).
                torch._foreach_mul_(shadow_list, d)
                torch._foreach_add_(shadow_list, param_list, alpha=1.0 - d)
        except (AttributeError, RuntimeError):
            # `_foreach_*` may be missing on older PyTorch / some MPS
            # builds; fall back to the per-tensor fused mul_/add_ path.
            # Same numerics, slower Python.
            if self.step_count <= self.warmup:
                for shadow, p in zip(shadow_list, param_list):
                    shadow.copy_(p)
            else:
                d = self.decay
                for shadow, p in zip(shadow_list, param_list):
                    shadow.mul_(d).add_(p, alpha=1.0 - d)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(
        self,
        sd: dict[str, torch.Tensor],
        map_location: Optional[torch.device] = None,
    ) -> None:
        """Restore EMA shadow from a checkpoint.

        `map_location` lets the caller force the loaded tensors onto a
        specific device — important because checkpoints are loaded with
        `torch.load(..., map_location="cpu")` (for portability), but the
        live model may live on MPS/CUDA. Without this, `shadow` would
        end up on CPU while `param` is on MPS and the next `update()`
        crashes with a device-mismatch RuntimeError.
        """
        missing = set(self.shadow.keys()) - set(sd.keys())
        if missing:
            raise ValueError(f"[ema] missing params in checkpoint: {sorted(missing)[:5]}…")
        for k, v in sd.items():
            t = v.clone().float()
            if map_location is not None and t.device != map_location:
                t = t.to(map_location)
            self.shadow[k] = t

    def apply_to(self, model: torch.nn.Module) -> None:
        """Copy EMA shadow weights into the live model's trainable params."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    shadow = self.shadow[name]
                    target = shadow.to(device=param.device, dtype=param.dtype)
                    param.data.copy_(target)


# ---- Core training routine ----


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@torch.no_grad()
def fused_clip_grad_norm_(
    named_params: Iterable[tuple[str, torch.nn.Parameter]],
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """`_foreach_*` re-implementation of `torch.nn.utils.clip_grad_norm_`.

    The reference implementation iterates parameters twice: once to
    `torch.stack([p.grad.norm(...) for p in params])` (one kernel per
    parameter → ~33 launches on a 6.7 M-param model), once to scale.
    On MPS each launch costs 1–2 ms, so the reference eats ~100 ms/step
    just to do the bookkeeping before/after the in-place `mul_`.

    `_foreach_norm` (one launch across all params) + a single
    `clamp_` + a `_foreach_mul_` (one launch for the scale) does the
    same math in ~3 launches.

    Drop-in replacement: returns the same scalar total-norm the
    original would (so callers can log it / clip-coef-opt the same way).
    Float32 numerics are bit-identical to `clip_grad_norm_` for the
    L2 case — see smoke_test.test_grad_clip_foreach_equiv.

    Args:
        named_params: `(name, param)` iterable. Only params whose `.grad`
            is non-None contribute (matches the reference's filter).
        max_norm: threshold. When `total_norm > max_norm`, every grad
            is rescaled by `max_norm / (total_norm + 1e-6)`.
        norm_type: passed by norm_type — only `2.0` is fused here
            (the only norm the reference actually special-cases;
            other norms fall back to the reference path).

    Returns:
        Scalar tensor (`.item()`-safe) holding the pre-rescale total norm.
    """
    grads = [p.grad for _, p in named_params if p.grad is not None]
    if not grads:
        # No grads at all — return a CPU scalar so callers can `.item()`
        # safely without a device check. Matches the reference, which
        # returns a CPU scalar when every `.grad` is None.
        return torch.tensor(0.0)
    if norm_type != 2.0:
        # Fall back to the reference impl for non-L2 norms (inf-norm
        # etc.). Same semantics, slow path.
        params = [p for _, p in named_params if p.grad is not None]
        return torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type)

    # `_foreach_norm` returns one tensor of per-param L2 norms; the
    # total norm is the L2 of that vector — matches the reference's
    # `torch.norm(torch.stack([...]))` exactly.
    per_param_norms = torch._foreach_norm(grads, ord=2)
    # `per_param_norms` is a TensorList — `stack` then `norm` mirrors the
    # reference's two-step reduction.
    stacked = torch.stack(per_param_norms)
    total_norm = torch.norm(stacked)

    # Mirror the reference's early-exit when norm is already ≤ max_norm.
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef.item() < 1.0:
        # Item() forces one sync (matches the reference's behaviour).
        # Single launch for the scale.
        torch._foreach_mul_(grads, clip_coef.item())
    return total_norm


def train(hp: HParams, args: "TrainArgs") -> None:
    # Device is auto-detected at runtime (MPS → CUDA → CPU) unless the
    # user forces one via `--device`. No config knob for it — pick a
    # device once you know what hardware you're on.
    device = select_device(args.device)
    print(f"[train] device={device}")
    print(f"[train] hparams={hp.to_dict()}")

    torch.manual_seed(hp.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hparams.json").write_text(_json_dumps(hp.to_dict()))

    # ---- Model ----
    model = VoiceFilterLite(hp).to(device)
    # `torch.compile` on MPS collapses per-op kernel launches and fuses
    # small ops (LayerNorm + dropout, the masked multiply, etc.) into
    # fewer Metal kernels. LSTM stays eager (inductor's CUDA-focused
    # codegen doesn't beat the MPS LSTM kernel for variable-T inputs);
    # but the surrounding Linear/LayerNorm/Conv1d/sigmoid/where paths
    # see real gains. First step pays ~30 s of `aot_eager` trace + cache;
    # every subsequent step is faster. Disable via `--no-compile` when
    # iterating (each new config recompiles).
    if not args.no_compile:
        try:
            model = torch.compile(model, backend="aot_eager")
            print("[train] torch.compile enabled (first step pays ~30s compile cost)")
        except Exception as e:
            print(f"[train] torch.compile failed ({type(e).__name__}: {e}); "
                  "falling back to eager")
    n_params = count_params(model)
    print(f"[train] trainable params: {n_params:,}")
    # Slides p.28: 3 LSTM × 512 + 1D CNN + 2-FC×64 noise-type predictor
    # is the main-architecture ship-target. After INT8 dynamic-range
    # quantization + adaptive w^(t) training it lands at 2.2 MB (slides
    # p.32 / p.36). FP32 param count scales with `hidden_dim` and
    # `num_lstm_layers` — defaults (256 × 3) → ~1.8 M, the wider
    # "production ship" 512 × 3 used in `configs/paper.yaml` → ~6.7 M.
    # Bounds below guard against *silent* architecture mistakes (forgot
    # to instantiate the LSTM stack, accidentally added an extra
    # predictor branch) without rejecting deliberate hyperparam choices
    # that deviate from the slides.
    assert 1.0e6 <= n_params <= 8.0e6, (
        f"param count {n_params:,} outside expected ~1.5-7.0M range; "
        "double-check the model architecture before training."
    )

    # ---- Loss ----
    loss_fn = build_loss(hp)
    if hp.loss_alpha_ramp_steps > 0:
        print(
            f"[train] loss=asym_l2 alpha_schedule "
            f"{hp.loss_alpha_start:.3f} → {hp.loss_alpha:.3f} "
            f"over {hp.loss_alpha_ramp_steps} steps "
            f"({hp.loss_alpha_ramp_steps / max(1, hp.total_steps):.0%} of total)"
        )
    else:
        print(f"[train] loss=asym_l2 alpha={hp.loss_alpha} (constant)")

    # ---- Optimiser + scheduler ----
    # AdamW (decoupled weight decay) is the standard modern replacement for
    # Adam: weight decay is applied as `θ ← θ - lr · wd · θ` directly to
    # the parameter (not added into the gradient before the adaptive
    # normaliser), so `weight_decay` is no longer "diluted" by the
    # adaptive step size — values like 1e-2 give actual regularisation,
    # where the legacy Adam+L2 form needed 1e-5 to behave like "off".
    # `foreach=True` routes the m/v update + bias-correction + decoupled
    # WD through `torch._foreach_*` — collapses the ~6 kernel-launches-
    # per-tensor into a handful of fused launches across the full
    # parameter list. On MPS, where every launch is meaningfully more
    # expensive than on CUDA, this is the dominant optimisation gain: a
    # 6.7 M-param model has ~33 parameter tensors; without `foreach` the
    # AdamW step is ~33 × 6 ≈ 200 separate kernels, with it ~3–5.
    # PyTorch >= 2.0 supports `foreach` on CPU/CUDA/MPS. Falls back to
    # the per-tensor path automatically when the device rejects the fused
    # kernel (`Optimizer._init_group`).
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp.learning_rate,
        weight_decay=hp.weight_decay,
        foreach=True,
    )
    scheduler = make_warmup_cosine_scheduler(optimizer, hp.warmup_steps, hp.total_steps)

    # ---- EMA ----
    ema: Optional[ModelEMA] = None
    if hp.use_ema:
        ema = ModelEMA(model, decay=hp.ema_decay, warmup=hp.ema_warmup_steps)
        print(f"[train] EMA enabled (decay={hp.ema_decay}, warmup={hp.ema_warmup_steps})")

    # ---- Data ----
    # Shared-memory handle for the curriculum-controlled
    # `multi_type_prob` scalar. Created in the main thread BEFORE the
    # DataLoader forks its workers, so every fork'd worker holds a
    # handle on the same `multiprocessing.Value`; updates via
    # `.value = ...` in the main thread propagate to the workers without
    # any further IPC. `0.0` initially — the loop writes the ramped
    # value each `log_every` window.
    import torch.multiprocessing as _tmp
    multi_type_prob_value = _tmp.Value("f", float(hp.multi_type_start_prob))
    if hp.multi_type_warmup_steps > 0:
        print(
            f"[train] interference curriculum: multi_type_p "
            f"{hp.multi_type_start_prob:.3f} → {hp.multi_type_end_prob:.3f} "
            f"over {hp.multi_type_warmup_steps} steps "
            f"({hp.multi_type_warmup_steps / max(1, hp.total_steps):.0%} of total)"
        )
    else:
        print(
            f"[train] interference curriculum: disabled "
            f"(multi_type_prob fixed at {hp.multi_type_end_prob:.3f})"
        )

    train_ds = HfTripletStream(
        hp=hp,
        worker_id=0,
        num_workers=1,
        seed=hp.seed,
        multi_type_prob_value=multi_type_prob_value,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=hp.batch_size,
        # Number of background prefetcher processes. `0` = main thread
        # pulls batches inline; `>0` = N worker processes each run an
        # independent copy of the dataset and push collated batches
        # onto a shared queue. The dataset detects this via
        # `torch.utils.data.get_worker_info()` inside `__iter__` and
        # re-seeds its RNG by `worker_id`, so each worker emits a
        # distinct stream of samples (no duplicates).
        # See `hp.dataloader_workers` for platform-specific guidance.
        num_workers=hp.dataloader_workers,
        collate_fn=collate_triplets,   # CPU tensors; main thread does .to(device)
        pin_memory=False,              # MPS has no pinned-memory DMA; CUDA could enable but skip for simplicity
    )
    # Background prefetch: decouple CPU data prep from GPU compute so the
    # GPU sees a continuous stream of ready batches instead of waiting on
    # soundfile + mel between every step. Depth is `hp.prefetch_batches_max`
    # (default 2) — on memory-constrained MPS machines the queue holds
    # `prefetch × batch_size × ref_padded_len` of CPU memory plus a copy
    # on MPS after `.to(device)`; deep queues just trade idle-GPU time
    # for OOM kills. PrefetchIterator runs a single producer thread
    # (the GPU is the bottleneck, not decode).
    train_iter: Iterable[dict] = PrefetchIterator(
        train_loader, max_prefetch=hp.prefetch_batches_max
    )

    # ---- Tensorboard ----
    writer = None
    if not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(out_dir / "tb"))
        writer.add_scalar("model/params", n_params, 0)

    # ---- Start from scratch ----
    # Resume-from-checkpoint is intentionally not supported: the dataset
    # iterator's RNG state and `_target_buffer` aren't persisted, so
    # resuming would silently restart epoch-1 — the model re-sees samples
    # 0..104 000 with optimizer momentum already trained on them, which
    # skews the loss curve. If a run dies, restart fresh into a new
    # `--output-dir` and accept the loss; the model's `ckpt_step*.pt`
    # files remain on disk for offline evaluation.
    step = 0

    # ---- Train ----
    model.train()
    last_log = time.time()
    # Accumulate per-step losses as GPU tensors and only `.item()` them
    # once every `hp.log_every` steps. The previous version called
    # `.item()` 3× per step (asym / noise_type / total) — each call
    # forced a GPU→CPU sync that stalled the device pipeline and made
    # the GPU history chart look like a series of bursts with gaps.
    # Pulling the sync out to the log boundary keeps the GPU fed.
    # We start with `zeros((), device=device)` rather than the float 0.0
    # so the running-averaging division at log time stays on-device.
    running_loss = torch.zeros((), device=device)
    running_loss_asym = torch.zeros((), device=device)
    running_loss_asym_over = torch.zeros((), device=device)
    running_loss_asym_under = torch.zeros((), device=device)
    running_loss_noise_type = torch.zeros((), device=device)
    # Opt-in raw-mask penalty on silent-target frames. Always accumulated
    # so the TB curve is stable when the term is turned on; stays at 0
    # when `silence_mask_weight=0` (the loss returns the zero scalar).
    running_loss_silence = torch.zeros((), device=device)
    # Noise-type predictor diagnostics — hinge loss can drop while the
    # classifier cheats (e.g. predicting all-pass when pos_rate ≈ 0).
    # `noise_type_acc` tracks window-mean binary accuracy;
    # `noise_type_pos_rate` tracks the fraction of +1 labels in the window
    # (sanity check: should sit near 0.3-0.7; pinned near 0/1 means the
    # batch stream is class-imbalanced).
    running_noise_type_acc = torch.zeros((), device=device)
    running_noise_type_pos_rate = torch.zeros((), device=device)
    # Mask statistics: mean is accumulated (sum / count → window-mean), min /
    # max are accumulated via running min/max — they catch the moment mask
    # collapses to all-pass (mean → 1) or to all-zero (mean → 0) within the
    # log window even if the mean alone looks uneventful.
    running_mask_mean = torch.zeros((), device=device)
    running_mask_min = torch.full((), float("inf"), device=device)
    running_mask_max = torch.full((), float("-inf"), device=device)
    running_grad_norm = torch.zeros((), device=device)
    # SI-SNR diagnostics — per-bin linear-mel SI-SNR averaged across F,
    # see `losses.linear_mel_si_snr_per_sample`. Tracks "is the model
    # recovering the target's spectral envelope?" through training;
    # pairs with loss_asym so a falling loss with flat SI-SNR flags a
    # collapse (model dodging the α² penalty via M≈1, paper §3.3 trap).
    running_si_snr_in = torch.zeros((), device=device)
    running_si_snr_out = torch.zeros((), device=device)
    running_count = 0

    # First-batch visibility: the iterable dataset pulls a full batch of
    # 16 kHz audio through `soundfile` and STFT, then the main thread runs
    # the frozen SpeakerEncoder on MPS to fill dvecs. First time around
    # this is multi-minute on Apple Silicon because MPS has to compile
    # kernels + warm caches. Print a heartbeat so the user knows the run
    # isn't dead.
    print("[train] building first batch (loading + decoding + mel)…")
    first_batch_t0 = time.time()
    first_batch_done = False

    for batch in train_iter:
        if not first_batch_done:
            dt = time.time() - first_batch_t0
            print(f"[train] first batch built in {dt:.2f}s — training starts")
            first_batch_done = True
            # Reset the log timer so the first `log_every` window doesn't
            # absorb the cold-start cost and report a misleading rate.
            last_log = time.time()
        if step >= hp.total_steps:
            break

        # Move CPU tensors to the training device. Kept out of the collate
        # fn so the background prefetch thread never has to ship MPS
        # tensors through a queue.
        stacked = batch["stacked_mixed"].to(device)
        target = batch["target_logmel"].to(device)
        # D-vectors are filled lazily by `DVectorCache` inside
        # `HfTripletStream`: first sight of a new speaker triggers a
        # single SpeakerEncoder forward + atomic save to
        # `<cache_dir>/voicefilter-dvectors/<spec_id>/<spk>.npy`; every
        # subsequent draw (this run and any future run) is a one-`.npy`-read.
        dvec = batch["ref_dvecs"].to(device)
        overlap_labels = batch["overlap_label"].to(device) if hp.use_noise_type_predictor else None

        # Model returns (mask, hidden). We omit `hidden` deliberately — each
        # training batch resets LSTM state to zeros (the default when
        # `hidden=None`). This is correct for chunked training and matches
        # the streaming wrapper's first-call warmup and the reference
        # project's behaviour. `hidden` exists only for stateful inference.
        mask, _hidden = model(stacked, dvec)
        # Paper §3.2: Y_enh = M · Y in linear filterbank-energy space.
        # The model's log-mel input must be converted to linear-mel
        # before the mask is applied, and the target log-mel must also
        # be in linear-mel for the asymmetric L2 loss (paper §3.3) to
        # see consistent domains — applying M in log-mel would change
        # the meaning of "over-suppression" (target - pred > 0). The
        # log-mel ↔ linear-mel round-trip uses `log_mel_floor` as the
        # shared floor convention (`wav_to_log_mel` does `log(x + ε)`,
        # so the inverse is `exp(y) - ε`, clamped to 0).
        cur_idx = hp.frame_stack - 1
        cur_logmel = stacked[..., cur_idx * hp.n_mels : (cur_idx + 1) * hp.n_mels]
        cur_linear = log_mel_to_linear(cur_logmel, floor=hp.log_mel_floor)
        target_linear = log_mel_to_linear(target, floor=hp.log_mel_floor)
        enhanced_linear = apply_mask(mask, cur_linear)
        # Per-frame noise-type logits (paper §4.2). Two input variants:
        #   * alpha_head_use_current_frame=True  → paper §3.4: S_in^(t).
        #     Input is the current frame's noisy log-mel (matches the
        #     paper literally — S_in is the log-mel feature, not the
        #     linear-mel masked output).
        #   * alpha_head_use_current_frame=False → paper §4.2 alternative:
        #     feed the LSTM output (re-run `lstm_forward` once more — same
        #     weights, no autograd through the mask path).
        frame_logits = None
        if hp.use_noise_type_predictor:
            if hp.alpha_head_use_current_frame:
                head_in = cur_logmel
            else:
                with torch.no_grad():
                    lstm_out, _ = model.lstm_forward(stacked, dvec, None)
                head_in = lstm_out
            frame_logits = model.alpha_head(head_in).squeeze(-1)

        comps = loss_fn.forward_with_components(
            enhanced_linear, target_linear,
            frame_logits=frame_logits,
            frame_labels=overlap_labels,
            mask=mask,
        )
        loss = comps["loss"]

        # Accumulate mask statistics BEFORE backward so the GPU keeps the
        # compute graph alive for the upcoming backward — `.detach()` here
        # only strips the autograd metadata, the underlying storage is
        # shared with the live graph until backward actually frees it.
        with torch.no_grad():
            m = mask.detach()
            running_mask_mean += m.mean()
            running_mask_min = torch.minimum(running_mask_min, m.min())
            running_mask_max = torch.maximum(running_mask_max, m.max())
            # SI-SNR accumulation. Per-sample means are averaged across
            # the window so the TB curve isn't dominated by outlier
            # batches. Done under no_grad because this is a metric —
            # gradients would waste backward work. `enhanced_linear` is
            # already detached inside this no_grad block.
            si_snr_in_t = linear_mel_si_snr_per_sample(cur_linear, target_linear)
            si_snr_out_t = linear_mel_si_snr_per_sample(
                enhanced_linear, target_linear,
            )
            running_si_snr_in += si_snr_in_t.mean()
            running_si_snr_out += si_snr_out_t.mean()
            # Noise-type predictor accuracy + positive-rate. Hinge loss
            # can fall while the classifier cheats; per-window accuracy is
            # the cheap, direct read on whether the predictor is learning
            # anything useful. We skip the reduction when the predictor is
            # disabled (frame_logits is None) — running_* stay zero, the
            # TB writer drops these scalars in that case.
            if frame_logits is not None and overlap_labels is not None:
                preds01 = (frame_logits.detach() > 0).to(dtype=running_noise_type_acc.dtype)
                labels01 = (overlap_labels.detach() > 0).to(dtype=running_noise_type_acc.dtype)
                running_noise_type_acc += (preds01 == labels01).float().mean()
                running_noise_type_pos_rate += labels01.float().mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # `fused_clip_grad_norm_` collapses the per-param norm() stack
        # + the per-param `mul_` scale into two `_foreach_*` launches
        # (one for the norms, one for the conditional rescale) instead
        # of the reference's ~66 launches. Same L2-norm semantics, same
        # bit-level result for float32.
        grad_norm = fused_clip_grad_norm_(
            list(model.named_parameters()), hp.grad_clip_norm,
        )
        # Window-mean of the pre-clip total gradient norm. When this spikes
        # to >> `grad_clip_norm` it's a clear "data problem" signal (one
        # batch drove grads large enough to hit the clip); when it stays well
        # below the clip the constraint is a no-op and we can consider
        # loosening `hp.grad_clip_norm`.
        running_grad_norm += grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else torch.tensor(grad_norm, device=device)
        optimizer.step()
        scheduler.step()

        if ema is not None:
            ema.update(model)

        # GPU-side accumulation. `.detach()` on `comps["loss"]` strips the
        # autograd graph (already consumed by `loss.backward()` above, but
        # detaching keeps the running accumulator cheap to print/scalar-board).
        # `comps["loss_asym"]` / `comps["loss_noise_type"]` are already
        # detached inside `forward_with_components`.
        running_loss += comps["loss"].detach()
        running_loss_asym += comps["loss_asym"]
        running_loss_asym_over += comps["loss_asym_over"]
        running_loss_asym_under += comps["loss_asym_under"]
        running_loss_noise_type += comps["loss_noise_type"]
        running_loss_silence += comps["loss_silence"]
        running_count += 1
        step += 1

        # ---- Periodic allocator cache release ----
        # PyTorch's MPS / CUDA caching allocators keep a high-water mark
        # of every unique tensor shape they ever saw and don't return
        # chunks to the OS until the process ends. Without this, RSS on
        # Apple Silicon climbs steadily even when no Python object is
        # leaking (see hparams.empty_cache_every). Only call on the
        # "big-leak" cadence (default 5000 steps) — empty_cache has
        # measurable per-call cost (~20-50ms on MPS) and shows up as a
        # deep dip in the GPU history chart.
        if (
            hp.empty_cache_every > 0
            and step % hp.empty_cache_every == 0
        ):
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

        if step % hp.log_every == 0:
            now = time.time()
            count = max(1, running_count)
            # Refresh the loss's α from the schedule. Cheap; idempotent
            # if `step` is unchanged. The ramp is over `hp.log_every`
            # resolution (5 steps) which is well below any meaningful
            # α-change granularity (ramp is 10% of `total_steps`).
            alpha_now = loss_fn.set_step(step)
            # Refresh the interference-distribution curriculum knob
            # (multi_type_prob ∈ [start, end] over `multi_type_warmup_steps`).
            # The shared Value is read by every fork'd DataLoader worker
            # at iteration time; updates propagate via shared memory
            # without any further IPC. When `multi_type_warmup_steps == 0`
            # the knob is held at the end value end-to-end.
            if hp.multi_type_warmup_steps > 0 and step <= hp.multi_type_warmup_steps:
                frac = step / float(hp.multi_type_warmup_steps)
                multi_type_prob_value.value = float(
                    hp.multi_type_start_prob
                    + (hp.multi_type_end_prob - hp.multi_type_start_prob) * frac
                )
            else:
                multi_type_prob_value.value = float(hp.multi_type_end_prob)
            # ONE sync point for the whole log window — multiple GPU tensors
            # reduce to floats in sequential `.item()`s, but the kernels
            # before them are dispatched back-to-back so only the first
            # `.item()` waits for the whole pipeline.
            avg_loss_t = running_loss / count
            avg_loss_asym_t = running_loss_asym / count
            avg_loss_asym_over_t = running_loss_asym_over / count
            avg_loss_asym_under_t = running_loss_asym_under / count
            avg_loss_noise_type_t = running_loss_noise_type / count
            avg_loss_silence_t = running_loss_silence / count
            avg_grad_norm_t = running_grad_norm / count
            avg_noise_type_acc_t = running_noise_type_acc / count
            avg_noise_type_pos_rate_t = running_noise_type_pos_rate / count
            avg_si_snr_in_t = running_si_snr_in / count
            avg_si_snr_out_t = running_si_snr_out / count
            avg_loss = avg_loss_t.item()
            avg_loss_asym = avg_loss_asym_t.item()
            avg_loss_asym_over = avg_loss_asym_over_t.item()
            avg_loss_asym_under = avg_loss_asym_under_t.item()
            avg_loss_noise_type = avg_loss_noise_type_t.item()
            avg_loss_silence = avg_loss_silence_t.item()
            avg_grad_norm = avg_grad_norm_t.item()
            avg_noise_type_acc = avg_noise_type_acc_t.item()
            avg_noise_type_pos_rate = avg_noise_type_pos_rate_t.item()
            avg_si_snr_in = avg_si_snr_in_t.item()
            avg_si_snr_out = avg_si_snr_out_t.item()
            # mask stats: mean is window-mean (sum / count). min/max are
            # window-extrema — they catch the instant the mask saturates
            # to all-pass (max → 1) or all-zero (min → 0) within the log
            # window even if the mean alone looks uneventful.
            mask_mean_avg = (running_mask_mean / count).item()
            mask_min_window = running_mask_min.item()
            mask_max_window = running_mask_max.item()
            elapsed = now - last_log
            sps = hp.log_every * hp.batch_size / max(1.0, elapsed)
            cur_lr = scheduler.get_last_lr()[0]
            # Wallclock timestamp + cumulative elapsed since first batch.
            # Useful for tailing the log across long runs (e.g. over
            # SSH where the terminal clock may be off) and for ETA:
            # `elapsed_total / step * (total_steps − step)`. HH:MM:SS
            # rather than full ISO so the line stays one terminal row.
            wallclock = time.strftime("%H:%M:%S", time.localtime(now))
            elapsed_total = now - first_batch_t0
            # Single-line training summary. `asym` / `noise_type` are
            # logged to tensorboard (train/loss_asym, train/loss_noise_type)
            # and are omitted from stdout to keep this line uncluttered —
            # both can dominate the visual width at large magnitudes
            # (e.g. asym≈1e7 early in training).
            print(
                f"[train] {wallclock} step={step}/{hp.total_steps} "
                f"loss={avg_loss:.4f} "
                f"lr={cur_lr:.2e} samples/s={sps:.1f} "
                f"multi_type_p={multi_type_prob_value.value:.2f} "
                f"elapsed={elapsed_total:.1f}s"
            )
            if writer is not None:
                writer.add_scalar("train/loss", avg_loss, step)
                writer.add_scalar("train/loss_asym_over", avg_loss_asym_over, step)
                writer.add_scalar("train/loss_asym_under", avg_loss_asym_under, step)
                writer.add_scalar("train/grad_norm", avg_grad_norm, step)
                writer.add_scalar("train/mask_mean", mask_mean_avg, step)
                writer.add_scalar("train/mask_min", mask_min_window, step)
                writer.add_scalar("train/mask_max", mask_max_window, step)
                writer.add_scalar("train/loss_asym", avg_loss_asym, step)
                writer.add_scalar("train/loss_alpha", alpha_now, step)
                writer.add_scalar("train/lr", cur_lr, step)
                writer.add_scalar("train/si_snr_in", avg_si_snr_in, step)
                writer.add_scalar("train/si_snr_out", avg_si_snr_out, step)
                writer.add_scalar(
                    "train/si_snri", avg_si_snr_out - avg_si_snr_in, step,
                )
                writer.add_scalar(
                    "train/multi_type_prob", multi_type_prob_value.value, step,
                )
                if hp.use_noise_type_predictor:
                    writer.add_scalar("train/loss_noise_type", avg_loss_noise_type, step)
                    writer.add_scalar("train/noise_type_acc", avg_noise_type_acc, step)
                    writer.add_scalar("train/noise_type_pos_rate", avg_noise_type_pos_rate, step)
                # Only log the silence-mask term when it's actually doing
                # work — keeps the TB graph clean in the default config
                # (`silence_mask_weight=0` would otherwise pin a flat
                # zero line forever).
                if hp.silence_mask_weight > 0.0:
                    writer.add_scalar("train/loss_silence", avg_loss_silence, step)
            # In-place zero on the same tensor (no new allocation, unlike
            # `running_loss = torch.zeros(...)`).
            running_loss.zero_()
            running_loss_asym.zero_()
            running_loss_asym_over.zero_()
            running_loss_asym_under.zero_()
            running_loss_noise_type.zero_()
            running_loss_silence.zero_()
            running_mask_mean.zero_()
            # Min/max use ±inf sentinels — `.zero_()` would corrupt them
            # (min-of-zeroes would no longer be a true window-minimum).
            running_mask_min.fill_(float("inf"))
            running_mask_max.fill_(float("-inf"))
            running_grad_norm.zero_()
            running_noise_type_acc.zero_()
            running_noise_type_pos_rate.zero_()
            running_si_snr_in.zero_()
            running_si_snr_out.zero_()
            running_count = 0
            last_log = now

        if step % hp.ckpt_every == 0 or step == hp.total_steps:
            ckpt_path = out_dir / f"ckpt_step{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "ema_state": ema.state_dict() if ema is not None else None,
                    "optim_state": optimizer.state_dict(),
                    "hparams": hp.to_dict(),
                },
                ckpt_path,
            )
            print(f"[train] saved {ckpt_path}")
            # Keep the last 3 ckpts to limit disk.
            ckpts = sorted(out_dir.glob("ckpt_step*.pt"))
            for old in ckpts[:-3]:
                old.unlink()

    # ---- Save final ----
    final_path = out_dir / "ckpt_final.pt"
    final_state = model.state_dict()
    if ema is not None:
        # Swap EMA weights into model_state so downstream scripts (evaluate,
        # export_onnx) automatically use the regularised weights without
        # having to know about EMA.
        ema.apply_to(model)
        final_state = model.state_dict()
        # Restore live weights (so the in-memory model is consistent if reused).
        # Re-read from optimiser snapshot? Simpler: leave EMA applied — caller
        # is expected to re-load from disk for further training.
    torch.save(
        {
            "step": step,
            "model_state": final_state,
            "hparams": hp.to_dict(),
        },
        final_path,
    )
    print(f"[train] done — saved {final_path}" + (" (EMA weights)" if ema is not None else ""))
    if writer is not None:
        writer.close()


# ---- JSON helpers ----


def _json_dumps(obj: object) -> str:
    import json
    return json.dumps(obj, indent=2, ensure_ascii=False)


# ---- CLI ----


@dataclass
class TrainArgs:
    """Minimal CLI surface — everything else lives in the YAML config.

    Training is config-driven. Flags:
    - ``--config PATH``     : YAML overlay (sets all knobs).
    - ``--output-dir PATH`` : where to write ckpts / hparams.json / TB logs.
    - ``--seed INT``        : override ``train.seed`` (smoke-test convenience).
    - ``--device STR``      : force a device (mps / cuda / cpu). Default
                               auto-detects in priority MPS → CUDA → CPU.
    - ``--no-tensorboard``  : skip TensorBoard logging.
    - ``--hf-endpoint``, ``--hf-transfer``, ``--no-hf-transfer``: HF mirror knobs.
    """

    config: Optional[str] = None
    output_dir: str = "runs/exp1"
    seed: Optional[int] = None
    device: str = "auto"
    no_tensorboard: bool = False
    hf_endpoint: Optional[str] = None
    hf_transfer: Optional[bool] = None
    no_compile: bool = False


app = App(name="voicefilter-lite", help_format="plaintext")


@app.default
def cli(
    *,
    config: Annotated[
        Optional[str],
        Parameter(name="--config", help="Path to a YAML config (sets model/dataset/train fields)."),
    ] = None,
    output_dir: Annotated[
        str, Parameter(name="--output-dir", help="Where to write ckpts, hparams.json, and TB logs.")
    ] = "runs/exp1",
    seed: Annotated[
        Optional[int],
        Parameter(name="--seed", help="Override train.seed (smoke-test convenience)."),
    ] = None,
    device: Annotated[
        str,
        Parameter(
            name="--device",
            help="Force a training device (mps / cuda / cpu). Default 'auto' "
                 "picks MPS → CUDA → CPU based on availability.",
        ),
    ] = "auto",
    no_tensorboard: Annotated[
        bool,
        Parameter(negative="", name="--no-tensorboard", help="Skip TensorBoard logging."),
    ] = False,
    hf_endpoint: Annotated[
        Optional[str],
        Parameter(
            name="--hf-endpoint",
            help="Override HF Hub base URL (e.g. https://hf-mirror.com).",
        ),
    ] = None,
    hf_transfer: Annotated[
        Optional[bool],
        Parameter(
            negative="--no-hf-transfer",
            help="Force-enable HF hf_transfer protocol (faster on huggingface.co; "
                 "breaks on most mirrors).",
        ),
    ] = None,
    no_compile: Annotated[
        bool,
        Parameter(
            negative="",
            name="--no-compile",
            help="Disable torch.compile (default: enabled). Useful when iterating "
                 "configs (each new shape recompiles, ~30s cost).",
        ),
    ] = False,
) -> None:
    """Train VoiceFilter-Lite. Config-driven: load all knobs from a YAML.

    See ``configs/*.yaml`` for examples.
    """
    args = TrainArgs(
        config=config,
        output_dir=output_dir,
        seed=seed,
        device=device,
        no_tensorboard=no_tensorboard,
        hf_endpoint=hf_endpoint,
        hf_transfer=hf_transfer,
        no_compile=no_compile,
    )
    hp = HParams()

    # 1. YAML config overlay (sets all knobs in one shot).
    load_and_apply(hp, args.config)

    # 2. Minimal CLI flag overrides (only seed — everything else is
    # config-driven). Device is auto-detected; force via `--device`.
    if args.seed is not None:
        hp.train.seed = args.seed

    # 3. Resolve HF Hub endpoint (relevant for `prepare_dataset.py`; no
    # network calls happen during the training loop itself).
    apply_hf_endpoint(
        hp,
        cli_override=args.hf_endpoint,
        cli_transfer=args.hf_transfer,
    )

    train(hp, args)


if __name__ == "__main__":
    app()
