"""Training loop for VoiceFilter-Lite, with optional EMA, noise-type
predictor, Musan noise and real-RIR augmentation (see `configs/paper.yaml`)."""

from __future__ import annotations

import argparse
import copy
import math
import os
import queue
import threading
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

import torch

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

from open_voicefilter_lite.cli import apply_hf_endpoint, load_and_apply
from open_voicefilter_lite.dataset import LibriSpeechTripletStream, collate_triplets
from open_voicefilter_lite.dvec_cache import DVecCache
from open_voicefilter_lite.hparams import HParams
from open_voicefilter_lite.losses import AsymmetricL2Loss
from open_voicefilter_lite.model import VoiceFilterLite
from open_voicefilter_lite.speaker_encoder import FrozenSpeakerEncoder


# ---- Background prefetch ----


class PrefetchIterator:
    """Wraps an iterable so items are produced on a background thread.

    The main thread consumes items via `for x in PrefetchIterator(src)` —
    by the time the GPU finishes step N, batch N+1 is already built and
    sitting in the queue, so the GPU is never idle waiting for soundfile
    decodes / ECAPA forwards.

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
        if self.step_count <= self.warmup:
            # Hold EMA = model.
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].copy_(param.detach().float())
            return
        d = self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            shadow = self.shadow[name]
            shadow.mul_(d).add_(param.detach().float(), alpha=1.0 - d)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        missing = set(self.shadow.keys()) - set(sd.keys())
        if missing:
            raise ValueError(f"[ema] missing params in checkpoint: {sorted(missing)[:5]}…")
        for k, v in sd.items():
            self.shadow[k] = v.clone().float()

    def apply_to(self, model: torch.nn.Module) -> None:
        """Copy EMA shadow weights into the live model's trainable params."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    param.data.copy_(self.shadow[name].to(param.dtype))


# ---- Core training routine ----


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def train(hp: HParams, args: argparse.Namespace) -> None:
    device = select_device(hp.device)
    print(f"[train] device={device}")
    print(f"[train] hparams={hp.to_dict()}")

    torch.manual_seed(hp.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hparams.json").write_text(_json_dumps(hp.to_dict()))

    # ---- Speaker encoder (frozen) + dvec cache ----
    # ECAPA-TDNN used to live inside `LibriSpeechTripletStream.__iter__`,
    # which is called from `PrefetchIterator`'s background thread. That
    # cross-thread MPS use is what blew up `python3.13` with
    # `BUG IN CLIENT OF LIBMALLOC: memory corruption of free block`
    # (see `~/Library/Logs/DiagnosticReports/python3.13-*.ips`). The
    # encoder is now held by the main process via `DVecCache` and runs
    # only on the main thread inside the training loop.
    speaker_encoder = FrozenSpeakerEncoder(hp).to(device)
    dvec_cache = DVecCache(hp, speaker_encoder, device=device)

    # ---- Model ----
    model = VoiceFilterLite(hp).to(device)
    n_params = count_params(model)
    print(f"[train] trainable params: {n_params:,}")
    # Paper §4.4: 3 LSTM × 256 + 2-FC×64 noise-type predictor ≈ 2.2 M
    # params at 8-bit quantization (2.2 MB). Our FP32 model is ~1.82 M,
    # ~17% smaller than the paper number — same architecture, the gap
    # is within quantization-overhead tolerance (paper §3.5 FlatBuffer
    # metadata not counted here). Allow 1.5–2.5 M to catch obvious drift
    # (e.g., wrong `hidden_dim` or extra linear layers).
    assert 1.5e6 <= n_params <= 2.5e6, (
        f"param count {n_params:,} outside expected ~1.8-2.2M range; "
        "double-check the model architecture before training."
    )

    # ---- Loss ----
    loss_fn = AsymmetricL2Loss(hp)

    # ---- Optimiser + scheduler ----
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp.learning_rate,
        weight_decay=hp.weight_decay,
    )
    scheduler = make_warmup_cosine_scheduler(optimizer, hp.warmup_steps, hp.total_steps)

    # ---- EMA ----
    ema: Optional[ModelEMA] = None
    if hp.use_ema:
        ema = ModelEMA(model, decay=hp.ema_decay, warmup=hp.ema_warmup_steps)
        print(f"[train] EMA enabled (decay={hp.ema_decay}, warmup={hp.ema_warmup_steps})")

    # ---- Data ----
    train_ds = LibriSpeechTripletStream(
        hp=hp,
        worker_id=0,
        num_workers=1,
        seed=hp.seed,
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
    # soundfile + ECAPA between every step. Depth is `hp.prefetch_batches_max`
    # (default 2) — on memory-constrained MPS machines the queue holds
    # `prefetch × batch_size × ref_padded_len` of CPU memory plus a copy
    # on MPS after `.to(device)`; deep queues just trade idle-GPU time
    # for OOM kills. 2 keeps the GPU fed given the dataset's own
    # `decode_prefetch` jobs running in parallel.
    train_iter: Iterable[dict] = PrefetchIterator(
        train_loader, max_prefetch=hp.prefetch_batches_max
    )

    # ---- Tensorboard ----
    writer = None
    if not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(out_dir / "tb"))
        writer.add_scalar("model/params", n_params, 0)

    # ---- Train ----
    model.train()
    step = 0
    last_log = time.time()
    running_loss = 0.0
    running_loss_asym = 0.0
    running_loss_hinge = 0.0
    running_count = 0

    # First-batch visibility: the iterable dataset pulls a full batch of
    # 16 kHz audio through `soundfile` and STFT, then the main thread runs
    # the frozen ECAPA encoder on MPS to fill dvecs. First time around this
    # is multi-minute on Apple Silicon because MPS has to compile kernels +
    # warm caches. Print a heartbeat so the user knows the run isn't dead.
    print("[train] building first batch (loading + decoding + mel + ECAPA)…")
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
        # Compute dvecs on the main thread (the only thread that touches
        # MPS). `dvec_cache` does cache lookup → ECAPA forward → cache
        # write-back; this is the moved-out version of what used to be
        # `_flush_buffer` inside the dataset's iterator.
        dvec = dvec_cache.get_dvecs(
            batch["ref_wavs"].to(device),
            batch["ref_lens"].to(device),
            batch["ref_spk_ids"],
        )
        overlap_labels = batch["overlap_label"].to(device) if hp.use_noise_type_predictor else None

        # forward_train returns (enhanced, hidden, frame_logits) — the
        # last is the per-frame logit from the 2-FC×64 noise-type predictor
        # (paper §4.2). Combined loss = L_asym + hinge_loss_weight · L_hinge.
        enhanced, _hidden, frame_logits = model.forward_train(stacked, dvec)
        comps = loss_fn.forward_with_components(
            enhanced, target,
            frame_logits=frame_logits,
            frame_labels=overlap_labels,
        )
        loss = comps["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        if ema is not None:
            ema.update(model)

        running_loss += comps["loss"].item()
        running_loss_asym += comps["loss_asym"].item()
        running_loss_hinge += comps["loss_hinge"].item()
        running_count += 1
        step += 1

        if step % hp.log_every == 0:
            now = time.time()
            avg_loss = running_loss / max(1, running_count)
            avg_loss_asym = running_loss_asym / max(1, running_count)
            avg_loss_hinge = running_loss_hinge / max(1, running_count)
            elapsed = now - last_log
            sps = hp.log_every * hp.batch_size / max(1.0, elapsed)
            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"[train] step={step}/{hp.total_steps} "
                f"loss={avg_loss:.4f} asym={avg_loss_asym:.4f} hinge={avg_loss_hinge:.4f} "
                f"lr={cur_lr:.2e} samples/s={sps:.1f}"
            )
            if writer is not None:
                writer.add_scalar("train/loss", avg_loss, step)
                writer.add_scalar("train/loss_asym", avg_loss_asym, step)
                writer.add_scalar("train/lr", cur_lr, step)
                if hp.use_noise_type_predictor:
                    writer.add_scalar("train/loss_hinge", avg_loss_hinge, step)
            running_loss = 0.0
            running_loss_asym = 0.0
            running_loss_hinge = 0.0
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


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train VoiceFilter-Lite")
    p.add_argument("--output-dir", type=str, default="runs/exp1")
    p.add_argument("--device", type=str, default=None,
                   help="Override hp.device (mps/cuda/cpu/auto).")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-tensorboard", action="store_true")
    # YAML config overlay (applied before CLI flag overrides).
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML file with HParams overrides.")
    # Convenience toggles that flip the new flags off (matching the YAML).
    p.add_argument("--no-ema", action="store_true",
                   help="Disable EMA (overrides config).")
    p.add_argument("--no-noise-type-predictor", action="store_true",
                   help="Disable noise-type predictor / per-sample α (overrides config).")

    # Musan / RIR are OFF by default (paper.yaml opts in). Use these to enable
    # from the CLI when not loading a YAML config.
    p.add_argument("--use-musan", action="store_true",
                   help="Enable Musan noise augmentation (requires pre-downloaded assets).")
    p.add_argument("--use-rir", action="store_true",
                   help="Enable real-RIR reverberation (requires pre-downloaded assets).")

    p.add_argument("--ema-decay", type=float, default=None,
                   help="Override hp.ema_decay (e.g. 0.999).")

    # HF Hub endpoint override (highest-priority in the precedence chain —
    # see `cli.apply_hf_endpoint`). Useful for users behind firewalls or in
    # regions where `huggingface.co` is blocked; e.g. China: `--hf-endpoint
    # https://hf-mirror.com`.
    p.add_argument("--hf-endpoint", type=str, default=None,
                   help="Override HF Hub base URL (e.g. https://hf-mirror.com).")
    # HF hf_transfer protocol toggle. Most mirrors don't implement it
    # (will 502). Default is whatever the env says (regular HTTPS).
    xf = p.add_mutually_exclusive_group()
    xf.add_argument("--hf-transfer", dest="hf_transfer", action="store_true",
                    default=None,
                    help="Force-enable HF hf_transfer protocol (faster on "
                         "huggingface.co; breaks on most mirrors).")
    xf.add_argument("--no-hf-transfer", dest="hf_transfer", action="store_false",
                    help="Force-disable HF hf_transfer protocol (safe on mirrors).")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    hp = HParams()

    # 1. YAML config overlay (sets most hparams in one shot).
    load_and_apply(hp, args.config)

    # 2. CLI flag overrides (always beat the config).
    if args.device is not None:
        hp.device = args.device
    if args.batch_size is not None:
        hp.batch_size = args.batch_size
    if args.total_steps is not None:
        hp.total_steps = args.total_steps
    if args.learning_rate is not None:
        hp.learning_rate = args.learning_rate
    if args.seed is not None:
        hp.seed = args.seed
    if args.no_ema:
        hp.use_ema = False
    if args.no_noise_type_predictor:
        hp.use_noise_type_predictor = False
    # `--use-X` flips the corresponding flag ON (default is OFF).
    if args.use_musan:
        hp.use_musan = True
    if args.use_rir:
        hp.use_real_rir = True
    if args.ema_decay is not None:
        hp.ema_decay = args.ema_decay

    # Resolve HF Hub endpoint as early as possible — before any
    # `datasets.load_dataset` / `huggingface_hub` / speechbrain call.
    # Precedence: --hf-endpoint > $HF_ENDPOINT > hp.hf_endpoint > default.
    # Same precedence chain for the hf_transfer toggle (CLI > hp > env).
    apply_hf_endpoint(
        hp,
        cli_override=args.hf_endpoint,
        cli_transfer=args.hf_transfer,
    )

    train(hp, args)


if __name__ == "__main__":
    main()