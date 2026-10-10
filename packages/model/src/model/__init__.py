"""VoiceFilter-Lite training library.

Workspace-only package (NOT published to PyPI). Provides the
training-side building blocks:

    model.nn                  # VoiceFilterLite (mask predictor), apply_mask
    model.speaker_encoder     # SpeakerEncoder (d-vector LSTM), LinearNorm
    model.audio               # DSP helpers (mel, STFT, OLA, SNR mix, ...)
    model.losses              # AsymmetricL2Loss, GE2ELoss, SI-SNR metric
    model.dataset             # HF / local triplet stream + interference synthesis

The public inference surface (`StreamingVoiceFilter` + bundled ONNX)
lives in the separate `open-voicefilter-lite` package, which `model`
depends on (for `HParams`, `config`, ONNX metadata parsing).
"""

__all__: list[str] = []
