"""VoiceFilter d-vector speaker encoder (paper §3.1).

Pure model — no audio preprocessing. A faithful PyTorch reimplementation
of `SpeechEmbedder` from
https://github.com/maum-ai/voicefilter/blob/master/model/embedder.py.

Architecture (identical to upstream `SpeechEmbedder`):
    LSTM (3 × 768) → LinearNorm (768 → 256)
    input  (B, T, F=40) one mel window
        → LSTM over T frames
        → take last time-step
        → LinearNorm(768 → 256)
    output (B, 256) raw embedding

The forward signature `forward(mel_window) → (B, 256)` is the upstream
`SpeechEmbedder.forward` literally. All audio preprocessing (fbank
extraction, sliding-window aggregation, L2-normalisation, per-window
mean) lives in `open_voicefilter_lite.audio` — see `wav_to_dvec_mel`
and `extract_dvec`. This mirrors the upstream VoiceFilter repo's
separation (`model/embedder.py` is just the model; `utils/audio.py`
owns the audio side).

Pretrained weights
------------------
The original VoiceFilter d-vector is bundled as
`models/embedder.pt` (a PyTorch state_dict) at the repo root. Load with:

    SpeakerEncoder(hp).load_pretrained(hp.dvector_pt_path)   # default: models/embedder.pt

State_dict keys (matching `model/embedder.py`):
    lstm.weight_ih_l{0,1,2}      (3072, 40 | 768 | 768)
    lstm.weight_hh_l{0,1,2}      (3072, 768)
    lstm.bias_ih_l{0,1,2}        (3072,)
    lstm.bias_hh_l{0,1,2}        (3072,)
    proj.linear_layer.weight     (256, 768)
    proj.linear_layer.bias       (256,)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from open_voicefilter_lite.hparams import HParams


class LinearNorm(nn.Module):
    """Thin `nn.Linear` wrapper used by the upstream VoiceFilter embedder.

    The upstream `model/embedder.py` declares:

        class LinearNorm(nn.Module):
            def __init__(self, hp):
                super().__init__()
                self.linear_layer = nn.Linear(hp.embedder.lstm_hidden,
                                              hp.embedder.emb_dim)

    We mirror that exactly so the pretrained state_dict keys
    (`proj.linear_layer.weight/bias`) line up without any remapping.
    """

    def __init__(self, hp: HParams):
        super().__init__()
        self.linear_layer = nn.Linear(hp.dvector_lstm_hidden, hp.dvector_emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


class SpeakerEncoder(nn.Module):
    """VoiceFilter paper §3.1 d-vector (3-layer LSTM × 768 → LinearNorm → 256).

    Mirrors the upstream `SpeechEmbedder` layer-by-layer so the pretrained
    `models/embedder.pt` (also from the upstream project) loads with zero
    key remapping.

    Forward contract:
        mel_window: (B, T=80, F=40) mel-spectrogram window
            → (B, 256)  raw embedding (NOT L2-normalised).

    The model itself is windowed — i.e. it consumes a single 80-frame
    mel window. Aggregation over multiple windows (L2-norm per window →
    mean → final L2-norm) is the caller's job and lives in
    `open_voicefilter_lite.audio.extract_dvec`. Pretrained weights load
    with `load_pretrained(...)`; `freeze()` after loading when used as a
    frozen feature extractor.
    """

    def __init__(self, hp: HParams):
        super().__init__()
        self.hp = hp

        # ---- Core network (matches upstream `SpeechEmbedder` literally) ----
        # `nn.LSTM` with `batch_first=True` is the same forward direction as
        # the TF `BasicLSTMCell` chain in the original VoiceFilter repo; the
        # upstream PyTorch implementation also uses `batch_first=True`.
        self.lstm = nn.LSTM(
            input_size=hp.dvector_n_mels,        # 40
            hidden_size=hp.dvector_lstm_hidden,  # 768
            num_layers=hp.dvector_lstm_layers,   # 3
            batch_first=True,
            bidirectional=False,
        )
        # Upstream wraps the projection Linear in `LinearNorm`; the state_dict
        # key prefix is therefore `proj.linear_layer.*`, matching the
        # pretrained `models/embedder.pt` shipped in this repo.
        self.proj = LinearNorm(hp)               # 768 → 256

    def forward(self, mel_window: torch.Tensor) -> torch.Tensor:
        """(B, T=80, F=40) mel window → (B, 256) raw embedding (NOT L2).

        Matches upstream `SpeechEmbedder.forward` literally:
            lstm_out, _ = self.lstm(mel_window)
            return self.proj(lstm_out[:, -1, :])

        Aggregation (per-window L2 → mean → final L2) and audio
        preprocessing (fbank extraction) are the caller's job — see
        `open_voicefilter_lite.audio.wav_to_dvec_mel` and
        `open_voicefilter_lite.audio.extract_dvec`.
        """
        # `flatten_parameters()` packs the LSTM weights into contiguous
        # memory for faster cuDNN dispatch; the original VoiceFilter repo
        # calls it too.
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(mel_window)         # (B, T, 768)
        emb = self.proj(lstm_out[:, -1, :])         # (B, 256) — last step
        return emb

    # ---------------------------------------------------------------- #
    # Weights loading + freezing
    # ---------------------------------------------------------------- #
    def load_pretrained(self, path: str | Path) -> None:
        """Load a `.pt` state_dict and freeze parameters. The path defaults
        to `hp.dvector_pt_path` (`models/embedder.pt`), the pretrained
        `SpeechEmbedder` weights shipped with this repo.

        Expected state_dict keys (matches upstream `model/embedder.py`):
            `lstm.weight_ih_l{0,1,2}`, `lstm.weight_hh_l{0,1,2}`,
            `lstm.bias_ih_l{0,1,2}`, `lstm.bias_hh_l{0,1,2}`,
            `proj.linear_layer.weight`, `proj.linear_layer.bias`.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"[SpeakerEncoder] weights not found at {path.resolve()}. "
                f"The pretrained embedder (`models/embedder.pt`) ships with "
                f"this repo (from https://github.com/maum-ai/voicefilter)."
            )
        state = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not unexpected, (
            f"[SpeakerEncoder.load_pretrained] unexpected keys in {path}: "
            f"{unexpected[:5]}… (expected upstream embedder.py keys)."
        )
        if missing:
            print(f"[SpeakerEncoder] WARNING missing keys while loading {path}: {missing}")
        self.freeze()

    def freeze(self) -> None:
        """Disable grad on all parameters (frozen feature extractor mode)."""
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)