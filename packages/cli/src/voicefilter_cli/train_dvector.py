#!/usr/bin/env python3
"""Train the d-vector speaker encoder (paper §3.1, SpeechEmbedder).

The d-vector is the speaker-embedding branch that conditions the
VoiceFilter-Lite mask predictor. The bundled `models/embedder.pt`
already ships a pretrained d-vector; this script lets you re-train it
from scratch (or fine-tune) on LibriSpeech using the standard
**GE2E (Generalized End-to-End) objective**.

Layout per step
-----------------------
- Sample N speakers (default 32) × M utterances (default 6) per speaker
  → batch size = N × M
- Each utterance is cropped to `ref_max_sec` (default 6 s) @ `sample_rate`,
  converted to a 40-mel log filterbank with `wav_to_dvec_mel`, and pushed
  through `SpeakerEncoder` to a single 256-d vector.
- The embedding is L2-normalised per utterance (`extract_dvec` aggregation
  step) and consumed by the GE2E loss.

Loss
-----------------------
GE2E (Wan et al., 2018 — "Generalized End-to-End Loss for Speaker
Verification"):

    For each utterance e_ij (speaker i, j-th utt), the per-speaker centre
    is c_j = mean of the *other half* of speaker j's M utterances
    (Google's `ComputeSimilaritySplit` — enrollment / test halves swap
    so each utterance serves as test for half the time).
    Softmax over the N speaker centres:
        L_i = -log exp(w · sim(e_ij, c_j) + b)
                       / Σ_k exp(w · sim(e_ij, c_k) + b)
    Loss is the mean of L_i over all (i, j).

    `sim` is cosine similarity (embeddings are L2-normalised inside the
    loss module, so a dot product). `w` and `b` are learnable scalar
    weights (Google's `theta.w`/`theta.b`, default init 10.0 / −5.0).

The implementation is a faithful port of Google's official
`lingvo/sidlingvo/loss_layers.py::GEnd2EndSoftmaxLayer`:
    https://github.com/google/speaker-id/blob/master/lingvo/sidlingvo/loss_layers.py

M (= batch-size / speakers-per-batch) must be even.

Usage
-----
    # Default (paper-friendly: 32 speakers × 6 utts, 10k steps, Adam 1e-4)
    `voicefilter-cli train dvector` --output-dir runs/dvec

    # Smaller smoke run on Apple Silicon (4 speakers × 4 utts, 200 steps)
    `voicefilter-cli train dvector` \\
        --output-dir runs/dvec_smoke \\
        --speakers-per-batch 4 --batch-size 16 --steps 200

    # Fine-tune from bundled weights instead of starting from scratch
    `voicefilter-cli train dvector` --init-from models/embedder.pt
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import torch
from cyclopts import App, Parameter

from voicefilter_lite.audio import pad_or_trim, wav_to_dvec_mel
from voicefilter_cli.hf import apply_hf_endpoint
from voicefilter_cli.losses import GE2ELoss
from voicefilter_lite.hparams import HParams
from voicefilter_lite.speaker_encoder import SpeakerEncoder

# ---- GE2E loss -------------------------------------------------------------
#
# The actual loss lives in `voicefilter_cli.losses.GE2ELoss` — a faithful
# port of Google's official `GEnd2EndSoftmaxLayer`
# (https://github.com/google/speaker-id/blob/master/lingvo/sidlingvo/loss_layers.py)
# implementing Wan et al. 2018 (https://arxiv.org/pdf/1710.10467.pdf,
# Eq. 6).  Uses `ComputeSimilaritySplit` (unbiased centroids; M must
# be even) + learnable `w, b` linear transform on the cosine logits
# (init 10.0 / −5.0).

# ---- Data: LibriSpeech streaming ---------------------------------------------

def _build_librispeech_iterator(
    hp: HParams,
    hf_endpoint: str | None,
    hf_transfer: bool | None,
):
    """Yield `{"speaker_id": str, "audio": {"array": ..., "sampling_rate": ...}}`
    examples from LibriSpeech train.100.  We deliberately avoid HF's
    audio-decode hook (`_disable_hf_audio_decode`) so we can stream the
    bytes and decode with `soundfile` on the main thread — keeps the
    dataset stream factory dependency-light per lib."""
    from datasets import load_dataset
    apply_hf_endpoint(
        hp,
        cli_override=hf_endpoint,
        cli_transfer=hf_transfer,
        silent=True,
    )
    ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="train.100",
        streaming=True,
    )
    for ex in ds:
        yield ex

class _SpeakerPool:
    """Sliding buffer: keeps up to `max_utts_per_speaker` utterances per
    speaker so we can sample `(speaker, utt)` pairs without re-decoding
    audio.  Decoded lazily on first sample."""

    def __init__(self, hp: HParams, max_utts_per_speaker: int = 32, hf_endpoint=None, hf_transfer=None):
        self.hp = hp
        self.max_utts_per_speaker = max_utts_per_speaker
        self._it = _build_librispeech_iterator(hp, hf_endpoint, hf_transfer)
        self._by_speaker: dict[str, list] = defaultdict(list)
        self._spk_ids: list[str] = []
        self._rng = random.Random(0)

    def _refill(self, target_speakers: int, min_utts: int) -> None:
        """Stream until we have ≥ `target_speakers` distinct speakers each
        with ≥ `min_utts` decoded utterances."""
        while True:
            enough = (
                len(self._spk_ids) >= target_speakers
                and all(
                    len(self._by_speaker[s]) >= min_utts
                    for s in self._spk_ids[:target_speakers]
                )
            )
            if enough:
                return
            try:
                ex = next(self._it)
            except StopIteration:
                return
            spk = str(ex.get("speaker_id", "?"))
            utts = self._by_speaker[spk]
            if len(utts) >= self.max_utts_per_speaker:
                continue
            utts.append(ex)
            if spk not in self._spk_ids:
                self._spk_ids.append(spk)

    def sample_batch(
        self, speakers_per_batch: int, utts_per_speaker: int, rng: random.Random
    ) -> tuple[list[str], list]:
        """Returns (spk_per_list, examples_list)."""
        self._refill(speakers_per_batch, utts_per_speaker)
        # Snapshot of distinct speakers with enough buffered utts.
        eligible = [
            s for s in self._spk_ids
            if len(self._by_speaker[s]) >= utts_per_speaker
        ]
        if len(eligible) < speakers_per_batch:
            raise RuntimeError(
                f"[dvec] only {len(eligible)} eligible speakers available "
                f"({speakers_per_batch} required)"
            )
        chosen = rng.sample(eligible, speakers_per_batch)
        spk_per_list: list[str] = []
        examples: list = []
        for s in chosen:
            picked = rng.sample(self._by_speaker[s], utts_per_speaker)
            for ex in picked:
                spk_per_list.append(s)
                examples.append(ex)
        return spk_per_list, examples

def _decode(ex, hp: HParams) -> torch.Tensor | None:
    """HF audio field (bytes or array) → 1-D float32 tensor @ hp.sample_rate."""
    import io
    import soundfile as sf
    audio_field = ex.get("audio")
    if isinstance(audio_field, dict):
        if audio_field.get("bytes") is not None:
            data, sr = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32")
        elif audio_field.get("array") is not None:
            data = np.asarray(audio_field["array"], dtype="float32")
            sr = int(audio_field.get("sampling_rate", hp.sample_rate))
        else:
            return None
    else:
        return None
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.from_numpy(data)
    if sr != hp.sample_rate:
        import torchaudio
        wav = torchaudio.functional.resample(
            wav.unsqueeze(0), sr, hp.sample_rate
        ).squeeze(0)
    peak = wav.abs().max().clamp(min=1e-8)
    if peak > 5.0:
        wav = wav / peak
    return wav

# ---- Training loop ---------------------------------------------------------

app = App(name="dvector", help_format="plaintext")


@app.default
def main(
    *,
    output_dir: Annotated[
        str,
        Parameter(name="--output-dir", help="Where to write ckpt_final.pt + hparams.json."),
    ] = "runs/dvec",
    steps: Annotated[int, Parameter(help="Total optimisation steps.")] = 10_000,
    batch_size: Annotated[
        int,
        Parameter(name="--batch-size", help="Utterances per step (= speakers-per-batch × utts-per-speaker)."),
    ] = 192,
    speakers_per_batch: Annotated[
        int,
        Parameter(
            name="--speakers-per-batch",
            help="N: speakers sampled per step. utts-per-speaker = "
                 "batch-size / speakers-per-batch and must be ≥ 2 for the "
                 "leave-one-out centre to be well-defined.",
        ),
    ] = 32,
    learning_rate: Annotated[
        float,
        Parameter(name="--learning-rate", help="Adam learning rate (1e-4 matches the upstream GE2E setup)."),
    ] = 1.0e-4,
    warmup_steps: Annotated[
        int,
        Parameter(name="--warmup-steps", help="Linear warmup before the cosine decay kicks in."),
    ] = 500,
    weight_decay: Annotated[float, Parameter(name="--weight-decay")] = 1.0e-5,
    grad_clip_norm: Annotated[float, Parameter(name="--grad-clip-norm")] = 5.0,
    log_every: Annotated[int, Parameter(name="--log-every")] = 50,
    ckpt_every: Annotated[int, Parameter(name="--ckpt-every")] = 1000,
    device: Annotated[
        str,
        Parameter(name="--device", help="Force a training device (mps / cuda / cpu)."),
    ] = "auto",
    seed: int = 1234,
    init_from: Annotated[
        Optional[str],
        Parameter(
            name="--init-from",
            help="Optional path to a state_dict to warm-start from (e.g. `models/embedder.pt`).",
        ),
    ] = None,
    hf_endpoint: Annotated[Optional[str], Parameter(name="--hf-endpoint")] = None,
    hf_transfer: Annotated[
        Optional[bool],
        Parameter(negative="--no-hf-transfer", help="Force-enable HF hf_transfer protocol."),
    ] = None,
) -> None:
    """Train the d-vector speaker encoder (SpeechEmbedder, paper §3.1).

    Uses Google's GE2E loss (split-batch + learnable w/b) on
    LibriSpeech train.100.
    """
    hp = HParams()
    if batch_size % speakers_per_batch != 0:
        raise ValueError(
            f"--batch-size ({batch_size}) must be a multiple of "
            f"--speakers-per-batch ({speakers_per_batch})."
        )
    utts_per_speaker = batch_size // speakers_per_batch
    if utts_per_speaker < 2:
        raise ValueError(
            f"utts-per-speaker = batch-size / speakers-per-batch = "
            f"{utts_per_speaker}; need ≥ 2 for GE2E leave-one-out."
        )

    torch.manual_seed(seed)
    rng = random.Random(seed)

    device_obj = select_device(device)
    print(f"[dvec] device={device_obj}")
    print(
        f"[dvec] N(speakers)={speakers_per_batch} "
        f"M(utts/speaker)={utts_per_speaker} "
        f"batch={batch_size} steps={steps} lr={learning_rate}"
    )

    # Model
    model = SpeakerEncoder(hp).to(device_obj)
    if init_from is not None and Path(init_from).is_file():
        sd = torch.load(init_from, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"[dvec] WARNING unexpected keys from {init_from}: {unexpected[:3]}…")
        if missing:
            print(f"[dvec] WARNING missing keys from {init_from}: {missing[:3]}…")
        print(f"[dvec] warm-started from {init_from}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[dvec] trainable params: {n_params:,}")

    # GE2E loss — Google's official `GEnd2EndSoftmaxLayer` (split-batch +
    # learnable w/b).  Lives in `voicefilter_cli.losses.GE2ELoss`; L2-norm
    # is done inside the module.  M must be even (Google's split-batch
    # requirement, asserted at `GE2ELoss.forward`).
    if utts_per_speaker % 2 != 0:
        raise ValueError(
            f"Google's GE2E requires even M (M = batch_size / "
            f"speakers_per_batch = {utts_per_speaker}); pick a "
            f"--batch-size that's an even multiple of --speakers-per-batch."
        )
    ge2e = GE2ELoss().to(device_obj)
    print(f"[dvec] GE2E loss: {ge2e.extra_repr()}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(ge2e.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.5 * (1.0 + float(np.cos(np.pi * min(1.0, progress))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data pool
    pool = _SpeakerPool(
        hp,
        max_utts_per_speaker=max(utts_per_speaker * 4, 32),
        hf_endpoint=hf_endpoint,
        hf_transfer=hf_transfer,
    )

    # Ref-length (samples) used for crop / pad.
    ref_len = int(hp.ref_max_sec * hp.sample_rate)

    model.train()
    running_loss = 0.0
    running_count = 0
    last_log = time.time()
    _ = log_every  # alias kept for clarity inside the loop

    for step in range(1, steps + 1):
        spk_per_list, examples = pool.sample_batch(
            speakers_per_batch, utts_per_speaker, rng
        )

        # Decode → crop → 40-mel log filterbank.
        mels: list[torch.Tensor] = []
        for ex in examples:
            wav = _decode(ex, hp)
            if wav is None or wav.numel() < int(0.3 * hp.sample_rate):
                # Drop too-short: re-sample a replacement utt from the same
                # speaker when possible.  For now pad a tiny wav with
                # zeros so the batch shape stays consistent — GE2E on a
                # near-silent utt is a no-op (L2-norm of zeros is ill-
                # defined; handle that below).
                wav = torch.zeros(int(0.3 * hp.sample_rate), dtype=torch.float32)
            wav = pad_or_trim(wav, ref_len)
            mel = wav_to_dvec_mel(wav.unsqueeze(0), hp).squeeze(0)  # (T_frames, 40)
            mels.append(mel)
        # Pad T_frames to the longest entry.  The encoder reads the last
        # time-step only, but `wav_to_dvec_mel` already pads T_frames;
        # we re-stack into a single (B, T_max, 40) tensor.
        T_max = max(m.shape[0] for m in mels)
        D = mels[0].shape[-1]
        batch_mel = torch.zeros(len(mels), T_max, D, dtype=torch.float32)
        for i, m in enumerate(mels):
            batch_mel[i, : m.shape[0]] = m

        batch_mel = batch_mel.to(device_obj)

        # Forward: (B, T, 40) → (B, 256) raw embeddings.
        emb = model(batch_mel)
        # L2-normalise (per the upstream embedder contract).  `GE2ELoss`
        # also L2-normalises internally when `normalize=True` (default);
        # the double-L2 is idempotent and keeps the safe path for any future
        # `normalize=False` use.
        norm = emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        emb = emb / norm
        emb = emb.view(speakers_per_batch, utts_per_speaker, -1)

        loss = ge2e(emb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()

        running_loss += float(loss.item())
        running_count += 1

        if step % log_every == 0:
            now = time.time()
            elapsed = now - last_log
            avg = running_loss / max(1, running_count)
            sps = log_every / max(1e-6, elapsed)
            print(
                f"[dvec] step={step}/{steps} loss={avg:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} steps/s={sps:.2f}"
            )
            running_loss = 0.0
            running_count = 0
            last_log = now

        if step % ckpt_every == 0 or step == steps:
            ckpt_path = _save_step(out_dir, hp, step, model)
            print(f"[dvec] saved {ckpt_path}")

    final_path = _save(out_dir, hp, steps, model)
    print(f"[dvec] done — saved {final_path}")


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)

def _save(out_dir: Path, hp: HParams, step: int, model: SpeakerEncoder) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ckpt_final.pt"
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "hparams": hp.to_dict(),
        },
        path,
    )
    (out_dir / "hparams.json").write_text(
        __import__("json").dumps(hp.to_dict(), indent=2, ensure_ascii=False)
    )
    return path

def _save_step(out_dir: Path, hp: HParams, step: int, model: SpeakerEncoder) -> Path:
    path = out_dir / f"ckpt_step{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "hparams": hp.to_dict(),
        },
        path,
    )
    # Keep last 3 ckpts.
    ckpts = sorted(out_dir.glob("ckpt_step*.pt"))
    for old in ckpts[:-3]:
        old.unlink()
    return path


if __name__ == "__main__":
    app()