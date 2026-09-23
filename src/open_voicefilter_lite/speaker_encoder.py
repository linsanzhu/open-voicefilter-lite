"""Frozen speaker encoder (d-vector) wrapper.

Wraps the pretrained `speechbrain/spkrec-ecapa-voxceleb` ECAPA-TDNN to
produce a fixed-dim embedding for the target speaker. The wrapper is
designed to be `eval()`-only and have its parameters frozen.

The ECAPA-TDNN emits `hp.speaker_encoder_raw_dim`-dim embeddings; we
project them down to `hp.d_vector_dim` inside VoiceFilterLite (a single
Linear whose params are trained).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from open_voicefilter_lite.hparams import HParams


class FrozenSpeakerEncoder(nn.Module):
    """Wraps `speechbrain` ECAPA-TDNN; extracts `hp.speaker_encoder_raw_dim`-dim
    embeddings.

    Lazy-loads the underlying speechbrain model on first `forward`. We do not
    import speechbrain at module load time so that unit-testing can import
    this file without the heavy speechbrain dependency.
    """

    def __init__(self, hp: HParams):
        super().__init__()
        self.hp = hp
        self._encoder = None  # Lazy: speechbrain.inference.speaker.EncoderClassifier
        self._raw_dim = hp.speaker_encoder_raw_dim  # ECAPA-TDNN output dim

    @property
    def raw_dim(self) -> int:
        return self._raw_dim

    def _load(self, device: torch.device) -> None:
        if self._encoder is not None:
            return

        # Lazy import so the package is only required when actually used.
        from speechbrain.inference.speaker import EncoderClassifier

        cache_dir = Path(self.hp.speaker_encoder_cache).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)

        self._encoder = EncoderClassifier.from_hparams(
            source=self.hp.speaker_encoder_source,
            savedir=str(cache_dir),
            run_opts={"device": str(device)},
        )
        # Freeze and eval.
        for p in self._encoder.parameters():
            p.requires_grad_(False)
        self._encoder.eval()

    @torch.no_grad()
    def forward(
        self,
        wav: torch.Tensor,
        wav_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract speaker embedding.

        Args:
            wav: (B, T) waveform in 16 kHz mono, in roughly [-1, 1].
            wav_lens: optional (B,) relative lengths in [0, 1] (longest in the
                batch = 1.0). When the batch is variable-length padded with
                zeros, pass `len(wav_i) / max_len` for each row so ECAPA can
                mask out the padding in its TDNN/global-pool layers.

        Returns:
            (B, raw_dim=hp.speaker_encoder_raw_dim) L2-normalised speaker embedding.
        """
        device = wav.device
        self._load(device)
        # ECAPA expects (B, T) float on the same device as the encoder.
        emb = self._encoder.encode_batch(wav, wav_lens)
        # L2-normalise for cosine-similarity-friendly comparison; main model
        # can absorb the scale change via its Linear projection.
        emb = torch.nn.functional.normalize(emb, dim=-1)
        # SpeechBrain 1.x returns (B, 1, raw_dim) with a singleton channel
        # dim; the wrapper's contract is (B, raw_dim) so flatten it. Earlier
        # versions returned (B, raw_dim) directly — be defensive.
        if emb.dim() == 3 and emb.shape[1] == 1:
            emb = emb.squeeze(1)
        return emb

    def to(self, *args, **kwargs):  # type: ignore[override]
        # speechbrain internals move with the parent module; just super().
        return super().to(*args, **kwargs)
