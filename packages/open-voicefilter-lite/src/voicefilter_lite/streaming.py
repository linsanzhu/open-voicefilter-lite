"""Turnkey streaming targeted-voice separation (paper §3).

`StreamingVoiceFilter` is the entire user-facing surface of this library.
Callers hand it PCM audio and a target-speaker enrollment recording;
it hands back enhanced PCM, one chunk at a time:

    from voicefilter_lite import StreamingVoiceFilter
    import soundfile as sf
    import numpy as np
    sf_vfl = StreamingVoiceFilter()               # uses bundled ONNX
    enroll, sr = sf.read("target_speaker.wav", dtype="float32")
    if sr != sf_vfl.hp.sample_rate:
        # resample externally to hp.sample_rate
        ...
    sf_vfl.set_enrollment(enroll)                 # numpy float32 1-D PCM
    for chunk in microphone_stream:               # each chunk is hp.hop_length samples
        enhanced = sf_vfl.step(chunk)             # numpy float32 1-D, same length

The bundled ONNX models carry their own hyperparameters in
`metadata_props` (see `voicefilter_cli._onnx_metadata`), so the
hyperparameter details (n_fft, win_length, adaptive β, ...) are read
out of the model at construction time and never need to be set by the
caller. Advanced users can override via the optional `mask_model` /
`dvector_model` constructor kwargs.

This module is **torch-free at runtime** — it uses numpy, scipy
implicitly via librosa, and onnxruntime. The training-side DSP
(`wav_to_log_mel`, `extract_dvec`, `reconstruct_wav_from_logmel`, ...)
lives in the workspace-only `model` package, which is the only place
torch / torchaudio is allowed to appear.

Per-step pipeline
-----------------

1. Slide an audio buffer of length `(L-1)·hop + n_fft` (where
   `L = ⌈n_fft / hop⌉`), dropping the oldest `hop` samples and
   appending the new chunk.
2. STFT the last `n_fft` samples of the buffer (Hann window, `center=False`,
   matches training). Compute log-mel via the cached mel basis.
3. Append the new log-mel to a left-context buffer (right-aligned;
   left-padded with zeros for the first `frame_stack-1` calls — same
   warm-up pattern the training pipeline uses).
4. Run the mask-predictor ONNX. Get the (n_mels,) mask and the
   noise-type head's `f_adapt ∈ [0, 1]`.
5. Apply the mask in **linear-mel** (paper §3.2: `M·S_in`), then
   blend with the raw input via the paper §3.4 adaptive
   suppression `w^(t)` smoother (using β / a / b from the ONNX
   metadata — note: the shipped models were exported with
   β=0.8 / a=0.8 / b=0.1, not the older default β=0.4 / a=0.7 / b=0.2).
6. Lift the enhanced linear-mel back to linear-spectrum via the cached
   mel pseudo-inverse. Combine with the input STFT's phase (the
   microphone phase the model never sees). Push the resulting complex
   spectrum into a rolling buffer of size L.
7. Reconstruct audio via a hand-rolled overlap-add over the L-frame
   spec buffer, Hann window, and a precomputed window²-sum normaliser.
   Return the last `hop` samples — the "newly determined" hop, which
   is the only region covered by all L frames' windows.

PCM reconstruction uses the hand-rolled OLA loop (matching the
training-side `reconstruct_wav_from_logmel`) because the shipped mask
ONNX has `n_fft=1200, win_length=640` and `torch.istft(center=False)`
trips on Hann window zero-endpoints ("window overlap add min: 1" —
known limitation). Using the hand-rolled loop is also a single-file,
no-iteration, no-Griffin-Lim path: it borrows the input phase and
re-uses the Hann window the forward STFT used, so the OLA lines up
exactly with the training inference.

State
-----
`StreamingVoiceFilter` holds the LSTM hidden state (`_h`, `_c` in
numpy — onnxruntime's IO contract), the left-context log-mel buffer
(`_context`), the audio buffer (`_audio_buf`), the enhanced spectrum
buffer (`_spec_buf`), and the adaptive-suppression scalar (`_w`).
`reset_state()` zeros all of them; `set_enrollment(audio)` calls it
implicitly (a new speaker invalidates the LSTM state).
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Optional

import numpy as np

from voicefilter_lite.hparams import HParams


# ---------------------------------------------------------- bundled model paths

def _bundled_path(filename: str) -> str:
    """Resolve a bundled ONNX filename to an absolute path on disk.

    The package ships `models/voicefilter_lite.int8.onnx` and
    `models/embedder.int8.onnx` inside the wheel. We read them via
    `importlib.resources` so the resolution works both in editable
    installs (the package source is on `sys.path`) and in installed
    wheels (the files live inside the package's `models/` directory).
    """
    try:
        # Python 3.9+: `files()` returns a Traversable.
        with resources.as_file(
            resources.files("voicefilter_lite").joinpath("models", filename)
        ) as p:
            return str(p)
    except (ModuleNotFoundError, FileNotFoundError):
        # Fallback: walk up from this file to find `models/<filename>`.
        # Helpful in dev / when the package is loaded from a source
        # checkout that's been symlinked or copied.
        here = Path(__file__).resolve().parent
        for cand in (here / "models" / filename, here.parent / "models" / filename):
            if cand.is_file():
                return str(cand)
        raise FileNotFoundError(
            f"Could not locate bundled model file {filename!r}. "
            f"Tried: {here/'models'/filename}, {here.parent/'models'/filename}."
        )


# ---------------------------------------------------------- per-step machinery

def _to_pcm_array(audio: np.ndarray) -> np.ndarray:
    """Coerce a wav-like input to a 1-D float32 numpy array.

    Accepts a 1-D `(T,)` array (returned as-is) or a 2-D `(T, C)` /
    `(C, T)` array (mean-pooled to mono along the last axis). Any other
    type or rank raises.

    Why we don't duck-type torch tensors: the published package is
    torch-free. If you have a torch tensor, call `.numpy()` (and
    `.cpu()` if it's on GPU) yourself before passing it in.
    """
    if not isinstance(audio, np.ndarray):
        raise TypeError(
            f"audio must be a numpy array, got {type(audio).__name__}"
        )
    a = np.ascontiguousarray(audio, dtype=np.float32)
    if a.ndim == 2:
        # Mean-pool multi-channel down to mono. Channels-last (T, C)
        # is the convention for `soundfile.read`; (C, T) is the
        # torchaudio convention — either way mean over the last dim
        # collapses channels correctly.
        a = a.mean(axis=-1)
    elif a.ndim != 1:
        raise ValueError(
            f"audio must be 1-D (T,) or 2-D (T, C) / (C, T); got shape {a.shape}"
        )
    return a


# ---------------------------------------------------------- d-vector ONNX setup

def _extract_dvec(
    wav: np.ndarray,
    hp: HParams,
    dvec_session,                   # onnxruntime.InferenceSession (duck-typed)
) -> np.ndarray:
    """(T,) @ hp.sample_rate → (hp.d_vector_dim,) L2-normalised d-vector.

    Mirrors `model.audio.extract_dvec` (paper §3.1) exactly, with the
    `encoder.forward` call replaced by an onnxruntime session.run
    call. The recipe is locked to match the pretrained embedder:

      1) 40-mel log10 fbank       (librosa, hamming window)
      2) 80-frame sliding windows (40-frame stride)
      3) per window: embedder ONNX → L2
      4) mean over windows
      5) final L2
    """
    import librosa

    wav_np = np.ascontiguousarray(wav, dtype=np.float32)
    # 1) Log-mel fbank (librosa, hamming window, no center pad).
    S = librosa.feature.melspectrogram(
        y=wav_np,
        sr=hp.sample_rate,
        n_fft=hp.dvector_n_fft,
        win_length=hp.dvector_win_length,
        hop_length=hp.dvector_hop_length,
        n_mels=hp.dvector_n_mels,
        window="hamming",
        center=False,
        power=2.0,
    )                                              # (n_mels, T_frames)
    feat = np.log10(S + 1e-10).T                  # (T_frames, n_mels)

    win = hp.dvector_window                         # 80
    hop = hp.dvector_stride                         # 40
    n_mels = hp.dvector_n_mels
    T_frames = feat.shape[0]

    # Pad short refs up to one window so the LSTM still produces a vec.
    if T_frames < win:
        pad = np.zeros((win - T_frames, n_mels), dtype=feat.dtype)
        feat = np.concatenate([feat, pad], axis=0)
        T_frames = win

    n_win = 1 + (T_frames - win) // hop

    # 2) Sliding windows: (n_win, win, n_mels).
    windows = np.zeros((n_win, win, n_mels), dtype=np.float32)
    for i in range(n_win):
        start = i * hop
        windows[i] = feat[start:start + win]

    # 3) ONNX encoder forward.
    emb = dvec_session.run(None, {"mel_window": windows})[0]   # (n_win, emb_dim)

    # 4) per-window L2 → mean → final L2.
    norms = np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-12
    emb_n = emb / norms
    mean = emb_n.mean(axis=0)
    return (mean / (np.linalg.norm(mean, axis=-1, keepdims=True) + 1e-12)).astype(np.float32)


# ---------------------------------------------------------- main class

class StreamingVoiceFilter:
    """Turnkey PCM-in / PCM-out streaming voice separator.

    Args:
        mask_model: optional path or `ort.InferenceSession` for the
            mask-predictor ONNX. Defaults to the bundled INT8
            `voicefilter_lite.int8.onnx`.
        dvector_model: optional path or session for the embedder
            ONNX. Defaults to the bundled INT8 `embedder.int8.onnx`.

    Typical use::

        import soundfile as sf
        import numpy as np
        enroll, sr = sf.read("enroll.wav", dtype="float32")   # load yourself
        if sr != 16000:
            ...                                               # resample
        sf = StreamingVoiceFilter()                           # bundled models
        sf.set_enrollment(enroll)                              # numpy float32 1-D PCM
        for chunk in mic_stream:                              # each chunk is hp.hop samples
            out = sf.step(chunk)                              # enhanced PCM, same shape
    """

    def __init__(
        self,
        mask_model: str | Path | object | None = None,
        speaker_encoder_model: str | Path | object = None,
    ) -> None:
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "StreamingVoiceFilter requires `onnxruntime`. "
                "Install with `pip install onnxruntime`."
            ) from e
        from onnxruntime import InferenceSession

        # Resolve `mask_model` to one of: (path to ONNX) | (ORT session) | None.
        # Anything else (e.g. a torch nn.Module) is rejected at the
        # duck-type check below — the new API only supports ONNX.
        mask_path: Optional[str]
        if mask_model is None:
            mask_path = _bundled_path("voicefilter_lite.int8.onnx")
        elif isinstance(mask_model, (str, Path)):
            mask_path = str(mask_model)
        elif _is_ort_session(mask_model):
            mask_path = None
        else:
            raise TypeError(
                f"mask_model must be a path (str / Path), an "
                f"onnxruntime.InferenceSession, or None; got {type(mask_model).__name__}."
            )

        dvec_path: str
        if speaker_encoder_model is None:
            dvec_path = _bundled_path("embedder.int8.onnx")
        elif isinstance(speaker_encoder_model, (str, Path)):
            dvec_path = str(speaker_encoder_model)
        else:
            # Could be an ORT session, in which case dvec_path is unused.
            if not _is_ort_session(speaker_encoder_model):
                raise TypeError(
                    f"dvector_model must be a path (str / Path), an "
                    f"onnxruntime.InferenceSession, or None; got "
                    f"{type(speaker_encoder_model).__name__}."
                )
            dvec_path = ""  # sentinel — won't be opened below

        # Build HParams from the mask ONNX, then overlay d-vector
        # fields from the embedder ONNX. The two files are independent
        # because the embedder can be re-trained / swapped without
        # touching the mask predictor.
        if mask_path is not None:
            self.hp = HParams.from_onnx(mask_path)
        else:
            self.hp = HParams()
        if dvec_path:
            self.hp.apply_onnx_dvector_overrides(dvec_path)

        # Open ONNX sessions. Use the CPU provider — this is the
        # turnkey inference path; advanced users who want GPU can
        # pass a pre-built `ort.InferenceSession(..., providers=["CUDAExecutionProvider"])`.
        if mask_path is not None:
            self._mask_sess = InferenceSession(mask_path, providers=["CPUExecutionProvider"])
        else:
            self._mask_sess = mask_model                       # already a session
        if dvec_path:
            self._dvec_sess = InferenceSession(dvec_path, providers=["CPUExecutionProvider"])
        else:
            self._dvec_sess = speaker_encoder_model                    # already a session

        # ---- Cached mel basis (paper §3.2 forward) and its pseudo-inverse (for §3.4 inverse).
        # Built once per (sample_rate, n_fft, n_mels). The basis itself
        # is (n_freq, n_mels). The pseudo-inverse of `fb.T` (shape
        # (n_mels, n_freq)) is (n_freq, n_mels) — `np.linalg.pinv` swaps
        # dims — so the inverse projection `mel_pinv @ linear_mel` maps
        # (n_mels,) → (n_freq,) and `fb.T @ mag2` does the forward
        # projection (n_freq,) → (n_mels,).
        from voicefilter_lite.mel_basis import make_mel_basis
        self._mel = make_mel_basis(self.hp)
        self._mel_fb = self._mel.fb                            # (n_freq, n_mels)
        self._mel_pinv = np.linalg.pinv(
            self._mel_fb.T, rcond=0.1
        )                                                      # (n_freq, n_mels)

        # ---- Cached Hann window (paper §3.2 forward STFT).
        # We use the SAME window the training pipeline's torchaudio
        # MelSpectrogram used internally (`np.hanning` matches
        # `torch.hann_window(periodic=True)`), so the forward STFT in
        # `step` and the inverse OLA over `_spec_buf` are perfectly
        # aligned with the training-time features the ONNX weights
        # were learned on.
        self._hann = self._mel.hann                            # (win_length,)

        # ---- Streaming state.
        # Number of frames that fit in the audio buffer.
        self._n_freq = self.hp.n_fft // 2 + 1
        self._L = (self.hp.n_fft + self.hp.hop_length - 1) // self.hp.hop_length
        self._audio_buf_len = (self._L - 1) * self.hp.hop_length + self.hp.n_fft
        self._wss = self._compute_wss()                    # (audio_buf_len,)

        self._h: Optional[np.ndarray] = None
        self._c: Optional[np.ndarray] = None
        self._context: list[np.ndarray] = []
        self._audio_buf: Optional[np.ndarray] = None
        self._spec_buf: Optional[np.ndarray] = None
        self._w: float = 0.0
        self._dvec: Optional[np.ndarray] = None

        self.reset_state()

    # ----------------------------------------------------------------- public
    def set_enrollment(self, audio: Optional[np.ndarray] = None) -> None:
        """Register the target speaker from PCM audio (or clear enrollment).

        Args:
            audio: 1-D numpy float32 array of PCM samples at
                `hp.sample_rate` (16 kHz by default). Multi-channel
                (T, C) or (C, T) inputs are mean-pooled to mono.

                Pass `None` (or omit the argument) to **clear** the
                enrollment. Subsequent `step()` calls will pass the
                input through unchanged — no mask, no OLA, no d-vector.
                This is useful for a "speaker-detected?" gate, or for
                VAD-style "only enhance when a known speaker is around"
                flows. You can call `set_enrollment(pcm)` again at any
                time to switch target speakers mid-stream; the streaming
                state is reset on every call (clear and re-enroll alike).

        Note: file I/O is the caller's responsibility. To enroll from
        a wav file, load it yourself first (e.g. with `soundfile`) and
        resample to `hp.sample_rate` if needed. Pass the resulting
        numpy array here.
        """
        if audio is None:
            # Caller is explicitly clearing — drop the d-vector and
            # zero the streaming state so the next `step()` is a clean
            # passthrough. The audio/spec buffers are also reset so
            # re-enrolling later doesn't blend old enhanced frames into
            # the new mask's output.
            self._dvec = None
            self.reset_state()
            return
        wav = _to_pcm_array(audio)
        if wav.size < int(0.1 * self.hp.sample_rate):
            # 100 ms of audio is the practical minimum the
            # d-vector pipeline needs to produce a stable speaker
            # embedding (4 sliding windows at default settings).
            raise ValueError(
                f"Enrollment audio too short: {wav.size} samples "
                f"({wav.size / self.hp.sample_rate:.2f}s); need at "
                f"least 0.1s for a stable d-vector."
            )
        self._dvec = _extract_dvec(wav, self.hp, self._dvec_sess)
        self.reset_state()

    def clear_enrollment(self) -> None:
        """Drop the registered d-vector; `step()` becomes a passthrough.

        Equivalent to `set_enrollment(None)`. Provided as a
        self-documenting no-argument alternative for code that reads
        like a state machine.
        """
        self.set_enrollment(None)

    @property
    def hop_length(self) -> int:
        return self.hp.hop_length

    def reset_state(self) -> None:
        """Zero LSTM hidden, audio buffer, spec buffer, and adaptive state.

        Safe to call any time. Wipes everything except the registered
        d-vector — `set_enrollment` calls this internally.
        """
        H = self.hp.hidden_dim
        L = self.hp.num_lstm_layers
        self._h = np.zeros((L, 1, H), dtype=np.float32)
        self._c = np.zeros((L, 1, H), dtype=np.float32)
        self._context = []
        self._audio_buf = np.zeros(self._audio_buf_len, dtype=np.float32)
        self._spec_buf = np.zeros((self._n_freq, self._L), dtype=np.complex64)
        self._w = 0.0

    def step(self, pcm_frame: np.ndarray) -> np.ndarray:
        """Process one hop of PCM, return one hop of enhanced PCM.

        Args:
            pcm_frame: 1-D numpy float32 array of length exactly
                `hp.hop_length` samples (16 kHz). Any deviation raises
                `ValueError`.

        Returns:
            1-D numpy float32 array of length `hp.hop_length`.

        If no enrollment is currently set (i.e. `set_enrollment` was
        never called, or was called with `None`), the input is passed
        through **unchanged** — no STFT, no mask, no OLA, no LSTM. This
        is the safe default before a target speaker is known; the same
        model instance can be used to "only enhance when the right
        person is talking" by toggling enrollment with
        `set_enrollment(pcm)` / `set_enrollment(None)` between calls.

        The first `frame_stack + L` calls after enrolling are warmup;
        their outputs are mathematically valid but lack the full left
        context.
        """
        # Coerce input once for both the passthrough and the
        # enhanced paths so the hop-length check fires regardless of
        # the enrollment state.
        t = _to_pcm_array(pcm_frame)
        if t.shape[-1] != self.hp.hop_length:
            raise ValueError(
                f"pcm_frame length {t.shape[-1]} != hp.hop_length "
                f"({self.hp.hop_length}); pass exactly one hop per call."
            )
        if self._dvec is None:
            # Passthrough: no d-vector registered, so there's nothing
            # to extract the target speaker for. Return the input
            # chunk as-is in float32. We deliberately do NOT touch the
            # audio / spec / LSTM buffers here — `reset_state` already
            # wiped them at clear-time, and we want the next
            # `set_enrollment(pcm)` to start from a clean slate, not
            # from a buffer half-filled with passthrough samples that
            # would then bleed into the first enhanced frames.
            return t.astype(np.float32, copy=False)
        hp = self.hp

        # 1) Slide audio buffer.
        t_f32 = t.astype(np.float32, copy=False)
        self._audio_buf = np.concatenate(
            [self._audio_buf[hp.hop_length:], t_f32]
        )

        # 2) STFT the new frame (Hann window, center=False, no zero pad).
        # `make_mel_basis` builds the same Hann window the training
        # pipeline's `torchaudio.MelSpectrogram` used, so the forward
        # STFT here matches the features the ONNX weights were learned
        # on bit-for-bit. We replicate the forward STFT explicitly
        # here (rather than calling a one-liner from `audio.py`) to
        # keep the buffer / window / phase relationship crystal-clear
        # in this single file.
        win_padded = np.zeros(hp.n_fft, dtype=np.float32)
        win_padded[: hp.win_length] = self._hann
        frame_audio = self._audio_buf[-hp.n_fft:]
        spec = np.fft.rfft(frame_audio * win_padded)     # (n_freq,) complex128
        spec = spec.astype(np.complex64)                 # match _spec_buf dtype

        # 3) Linear-mel via the cached basis, then log.
        mag2 = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)  # (n_freq,)
        linear_mel = self._mel_fb.T @ mag2                              # (n_mels,)
        log_mel = np.log(linear_mel + hp.log_mel_floor)                 # (n_mels,)

        # 4) Frame stacking with left zero-padding (matches training).
        stacked = self._stack_context(log_mel)               # (1, frame_stack, n_mels)

        # 5) Run mask ONNX.
        feed = {
            "stacked_logmel": stacked.reshape(1, 1, -1).astype(np.float32),
            "dvec":           self._dvec.reshape(1, -1) if self._dvec.shape[0] != 1 else self._dvec,
            "h0":             self._h,
            "c0":             self._c,
        }
        mask_np, fa_np, self._h, self._c = self._mask_sess.run(None, feed)
        mask = np.asarray(mask_np).reshape(-1).astype(np.float32)
        f_adapt = float(np.asarray(fa_np).reshape(-1)[0])

        # 6) Mask + adaptive suppression in linear-mel (paper §3.2 / §3.4).
        raw_lin = np.maximum(np.exp(log_mel) - hp.log_mel_floor, 0.0)
        filtered_lin = np.maximum(mask * raw_lin, 0.0)
        a, b, beta = hp.adaptive_w_a, hp.adaptive_w_b, hp.adaptive_w_beta
        target = a * f_adapt + b
        self._w = max(0.0, min(1.0, beta * self._w + (1.0 - beta) * target))
        enhanced_lin = self._w * filtered_lin + (1.0 - self._w) * raw_lin

        # 7) Lift enhanced linear-mel back to linear-spectrum, then paint
        #    the input's phase on top (the mic phase the model never
        #    sees; combining it with the model's enhanced magnitude
        #    is the "re-skin" trick — see `model.audio.reconstruct_wav_from_logmel`).
        enh_mag2 = np.maximum(self._mel_pinv @ enhanced_lin, 0.0)
        enh_spec = (
            np.sqrt(enh_mag2) * np.exp(1j * np.angle(spec))
        ).astype(np.complex64)

        # 8) Push to spec buffer (drop oldest, append newest).
        self._spec_buf = np.concatenate(
            [self._spec_buf[:, 1:], enh_spec.reshape(-1, 1)],
            axis=-1,
        )

        # 9) Reconstruct audio via hand-rolled OLA over the spec buffer.
        #    `wss` is precomputed in __init__ (constant — depends only on
        #    L, hop, n_fft, win_length, hann). See
        #    `model.audio.reconstruct_wav_from_logmel` for the same recipe.
        ola = np.zeros(self._audio_buf_len, dtype=np.float32)
        for t in range(self._L):
            frame_time = np.fft.irfft(self._spec_buf[:, t], n=hp.n_fft)
            if hp.n_fft != hp.win_length:
                frame_time = frame_time[..., : hp.win_length]
            start = t * hp.hop_length
            ola[start:start + hp.win_length] += frame_time * self._hann
        # Normalise by per-sample window² sum. We precomputed `_wss`
        # once in __init__; it's constant for the lifetime of the
        # instance. Avoid the per-step recompute of the existing
        # batch function.
        nz = self._wss > 1e-10
        ola[nz] = ola[nz] / self._wss[nz]
        # Return the "newest hop" — the hop corresponding to the audio
        # that was just received, located at positions
        # [(L-1)·hop, (L-1)·hop + hop) in the OLA buffer. This is the
        # fully overlap-added region (covered by all L frames' windows)
        # and corresponds to the most recent input chunk. The very end
        # of the OLA buffer (positions [start + win_length, end]) has
        # zero wss by construction (Hann window zero endpoints), so we
        # explicitly slice into the "fully covered" region.
        newest_hop_start = (self._L - 1) * hp.hop_length
        newest_hop_end = newest_hop_start + hp.hop_length
        return ola[newest_hop_start:newest_hop_end]

    # ---------------------------------------------------------------- helpers
    def _stack_context(self, frame_logmel: np.ndarray) -> np.ndarray:
        """Right-align the new log-mel frame into a `(1, frame_stack, n_mels)` array.

        Left-padded with zeros for the first `frame_stack-1` calls —
        same warm-up pattern as the training pipeline (`frame_stack_left`).
        """
        stack = self.hp.frame_stack
        n_mels = self.hp.n_mels
        self._context.append(frame_logmel)
        if len(self._context) > stack:
            self._context = self._context[-stack:]
        stacked = np.zeros((1, stack, n_mels), dtype=np.float32)
        offset = stack - len(self._context)
        for i, f in enumerate(self._context):
            stacked[0, offset + i] = f
        return stacked

    def _compute_wss(self) -> np.ndarray:
        """Precompute the per-sample window² sum for the OLA normaliser.

        Constant for the lifetime of the instance: depends only on
        L, hop, n_fft, win_length, and the Hann window. Shape
        `(audio_buf_len,)`, same as the OLA accumulator. The Hann
        window has zero endpoints, so a few samples at the very
        start / end of this buffer are zero — `step()` skips the
        division at those positions (`nz = wss > 1e-10`), which
        mirrors the forward STFT's silent-edge behaviour exactly.
        """
        win = self._hann
        n_fft = self.hp.n_fft
        win_length = self.hp.win_length
        hop = self.hp.hop_length
        L = self._L
        # Place a `win`-length slice of `win^2` at every frame position,
        # then sum overlaps. This is `reconstruct_wav_from_logmel`'s
        # `wss` accumulation, but precomputed once.
        win_padded = np.zeros(n_fft, dtype=np.float32)
        win_padded[:win_length] = win
        wss = np.zeros(self._audio_buf_len, dtype=np.float32)
        for t in range(L):
            start = t * hop
            wss[start:start + win_length] += win_padded[:win_length] ** 2
        return wss


# ---------------------------------------------------------- session helpers

def _is_ort_session(obj: object) -> bool:
    """True if `obj` looks like an `onnxruntime.InferenceSession`."""
    return (
        callable(getattr(obj, "run", None))
        and callable(getattr(obj, "get_inputs", None))
        and callable(getattr(obj, "get_outputs", None))
    )
