"""open-voicefilter-lite — streaming targeted voice separation.

The library exposes a single turnkey class: `StreamingVoiceFilter`. The
bundled ONNX models (mask predictor + d-vector encoder) carry their
own hyperparameters in `metadata_props`, so callers never see log-mel,
frame stacking, d-vector extraction, or the LSTM state machine — just
PCM in, PCM out:

    from voicefilter_lite import StreamingVoiceFilter
    sf = StreamingVoiceFilter()               # bundled ONNX
    sf.set_enrollment("enroll.wav")           # target speaker
    for chunk in mic_stream:                  # each chunk is hp.hop samples
        enhanced = sf.step(chunk)             # same shape, enhanced PCM

The internal modules (`hparams.py`, `audio.py`, `model.py`,
`speaker_encoder.py`, `config.py`) remain importable by their full
path for advanced / training-side use, but they're no longer part of
the public surface.
"""

from voicefilter_lite.streaming import StreamingVoiceFilter

__all__ = ["StreamingVoiceFilter"]
