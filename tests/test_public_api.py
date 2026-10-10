"""Smoke test for the public `StreamingVoiceFilter` API.

Tests the turnkey PCM-in / PCM-out interface against the bundled ONNX
models. No external data, no HuggingFace download, no checkpoint — the
ONNX graphs in the wheel carry their own weights and metadata.

Run with:
    python tests/test_public_api.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import soundfile as sf

# Make the lib importable when running from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "packages", "open-voicefilter-lite", "src"))


def _import_public_api():
    """Import only the public surface — should be just `StreamingVoiceFilter`."""
    import voicefilter_lite  # noqa: F401
    assert hasattr(voicefilter_lite, "StreamingVoiceFilter"), (
        "voicefilter_lite must expose StreamingVoiceFilter"
    )
    public = [n for n in voicefilter_lite.__all__]
    assert public == ["StreamingVoiceFilter"], (
        f"public surface must be exactly [StreamingVoiceFilter], got {public}"
    )


def _test_construction_with_bundled_onnx() -> None:
    """`StreamingVoiceFilter()` loads the bundled ONNX graphs."""
    from voicefilter_lite import StreamingVoiceFilter

    sf = StreamingVoiceFilter()
    # The shipped mask ONNX was exported with these non-default values —
    # they're a regression test that `HParams.from_onnx` actually reads
    # the metadata, not just the dataclass defaults.
    assert sf.hp.n_fft == 1200, f"n_fft={sf.hp.n_fft} (expected 1200 from ONNX)"
    assert sf.hp.win_length == 640, f"win_length={sf.hp.win_length} (expected 640 from ONNX)"
    assert sf.hp.adaptive_w_beta == 0.8, (
        f"adaptive_w_beta={sf.hp.adaptive_w_beta} (expected 0.8 from ONNX)"
    )
    # Dvector overrides
    assert sf.hp.dvector_n_fft == 512, f"dvector_n_fft={sf.hp.dvector_n_fft}"
    assert sf.hp.dvector_n_mels == 40, f"dvector_n_mels={sf.hp.dvector_n_mels}"
    assert sf.hp.dvector_lstm_hidden == 768, f"dvector_lstm_hidden={sf.hp.dvector_lstm_hidden}"
    # Hop length is the contract for `step(pcm_frame)`.
    assert sf.hop_length == 160, f"hop_length={sf.hop_length}"
    print(
        f"  bundled-ONNX construction OK — "
        f"n_fft={sf.hp.n_fft}, win_length={sf.hp.win_length}, "
        f"β={sf.hp.adaptive_w_beta}, hop={sf.hop_length}"
    )


def _test_set_enrollment_pcm_array() -> None:
    """`set_enrollment(pcm)` takes a numpy array and extracts a d-vector."""
    from voicefilter_lite import StreamingVoiceFilter

    sr = 16000
    np.random.seed(0)
    enroll = (np.random.randn(2 * sr) * 0.3).astype(np.float32)

    sfm = StreamingVoiceFilter()
    sfm.set_enrollment(enroll)
    dvec = sfm._dvec
    assert dvec.shape == (256,), f"dvec shape {dvec.shape} != (256,)"
    norm = float(np.linalg.norm(dvec))
    assert abs(norm - 1.0) < 1e-3, f"dvec not L2-normalised: |dvec|={norm}"
    print(f"  set_enrollment(pcm array) OK — dvec norm={norm:.4f}")


def _test_set_enrollment_rejects_path() -> None:
    """`set_enrollment(path)` is no longer supported — must raise TypeError."""
    from voicefilter_lite import StreamingVoiceFilter

    sfm = StreamingVoiceFilter()
    # The user is responsible for loading the wav themselves.
    try:
        sfm.set_enrollment("enroll.wav")
    except TypeError:
        pass
    else:
        raise AssertionError(
            "set_enrollment(path) should raise TypeError; "
            "load the wav yourself and pass the PCM array."
        )
    print("  set_enrollment rejects path strings (caller loads wav)")


def _test_set_enrollment_array_tensor() -> None:
    """`set_enrollment(array)` accepts 1-D and 2-D numpy float32 arrays."""
    from voicefilter_lite import StreamingVoiceFilter

    sfm = StreamingVoiceFilter()
    sfm.set_enrollment(np.random.randn(16000 * 2).astype(np.float32))
    sfm.set_enrollment(np.random.randn(16000 * 2, 2).astype(np.float32))  # multi-channel
    print("  set_enrollment(numpy 1-D / 2-D) OK")


def _test_step_pcm_loop() -> None:
    """`step(pcm_chunk)` returns one hop of enhanced PCM, finite, correct shape."""
    from voicefilter_lite import StreamingVoiceFilter

    sfm = StreamingVoiceFilter()
    sfm.set_enrollment(np.random.randn(16000 * 2).astype(np.float32))

    hop = sfm.hop_length
    n_iters = 200  # 200 * 10ms = 2s of audio
    np.random.seed(42)
    in_chunks = [np.random.randn(hop).astype(np.float32) * 0.3 for _ in range(n_iters)]
    out_chunks = [sfm.step(c) for c in in_chunks]
    out = np.concatenate(out_chunks)
    assert out.shape == (hop * n_iters,), f"out shape {out.shape} != ({hop * n_iters},)"
    assert out.dtype == np.float32, f"out dtype {out.dtype} != float32"
    assert bool(np.isfinite(out).all()), "out has non-finite values"
    # Some real audio must come out (otherwise the model is dead).
    assert float(np.abs(out).max()) > 0.0, "out is all zeros — model not running"
    print(
        f"  step(pcm) loop OK — {n_iters} hops, output mean abs={float(np.abs(out).mean()):.4f}"
    )


def _test_input_validation() -> None:
    """`step` and `set_*` raise on bad input.

    `step()` BEFORE `set_enrollment()` is NOT a validation error any
    more — it's a passthrough. We assert that here, and add coverage
    for the new `set_enrollment(None)` clear path.
    """
    from voicefilter_lite import StreamingVoiceFilter

    sfm = StreamingVoiceFilter()
    # step() before set_enrollment() must be a passthrough, not an error.
    pcm_in = np.random.randn(160).astype(np.float32) * 0.5
    out = sfm.step(pcm_in)
    assert out.shape == (160,), f"passthrough shape {tuple(out.shape)} != (160,)"
    assert np.allclose(out, pcm_in, atol=1e-7), (
        "step() without enrollment must return the input unchanged"
    )
    # clear_enrollment() / set_enrollment(None) on a never-enrolled
    # model must also be a no-op (idempotent).
    sfm.clear_enrollment()
    sfm.set_enrollment(None)

    sfm.set_enrollment(np.random.randn(16000).astype(np.float32))
    out_enhanced = sfm.step(np.zeros(160, dtype=np.float32))
    assert out_enhanced.shape == (160,)

    # wrong chunk length
    try:
        sfm.step(np.zeros(80, dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("step() should raise on wrong-length chunk")

    # too-short enrollment
    try:
        sfm.set_enrollment(np.zeros(1000, dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("set_enrollment() should raise on too-short audio")

    print("  input validation OK (passthrough, clear-idempotent, "
          "wrong-chunk, short-enroll)")


def _test_enrollment_toggle() -> None:
    """`set_enrollment(None)` and re-enrollment mid-stream work cleanly.

    The same model instance should be usable as: passthrough → enhance
    speaker A → passthrough → enhance speaker B. After each toggle the
    streaming state must be reset so old enhanced frames don't bleed
    into the new mode's first outputs.
    """
    from voicefilter_lite import StreamingVoiceFilter

    sfm = StreamingVoiceFilter()
    hop = sfm.hop_length

    # 1) Passthrough phase (no enrollment yet).
    pcm_a = np.random.randn(hop).astype(np.float32) * 0.4
    out_a = sfm.step(pcm_a)
    assert np.allclose(out_a, pcm_a, atol=1e-7), (
        "step() before any enrollment must passthrough"
    )

    # 2) Enroll speaker A and run a few enhanced hops.
    sfm.set_enrollment(np.random.randn(16000).astype(np.float32))
    enhanced_a = [sfm.step(np.random.randn(hop).astype(np.float32) * 0.3)
                  for _ in range(8)]
    assert all(o.shape == (hop,) for o in enhanced_a)

    # 3) Clear enrollment → passthrough again. The very next call must
    #    echo its input exactly, with no LSTM / OLA leakage from the
    #    enhanced-A phase (state was zeroed by clear).
    sfm.clear_enrollment()
    pcm_b = np.random.randn(hop).astype(np.float32) * 0.4
    out_b = sfm.step(pcm_b)
    assert np.allclose(out_b, pcm_b, atol=1e-7), (
        "step() after clear_enrollment() must passthrough with no leakage"
    )

    # 4) Re-enroll speaker B and confirm the d-vector differs from A.
    sfm.set_enrollment(np.random.randn(16000).astype(np.float32))
    dvec_b = sfm._dvec.copy()
    assert dvec_b is not sfm._dvec
    sfm.set_enrollment(np.random.randn(16000).astype(np.float32))   # speaker B again
    dvec_b2 = sfm._dvec.copy()
    # Different enrollment wavs → different d-vectors (in general; the
    # PRNG draws here are independent, so cos_sim should be small).
    cos = float(np.dot(dvec_b, dvec_b2) /
                (np.linalg.norm(dvec_b) * np.linalg.norm(dvec_b2) + 1e-12))
    assert abs(cos) < 0.95, f"re-enrollment should refresh the dvec, cos={cos:.3f}"

    print(f"  enrollment toggle OK (passthrough↔enhance↔passthrough↔enhance, "
          f"dvec refresh cos={cos:.3f})")


def _test_reset_state() -> None:
    """`reset_state()` wipes LSTM hidden + buffers, keeps dvec."""
    from voicefilter_lite import StreamingVoiceFilter

    sfm = StreamingVoiceFilter()
    sfm.set_enrollment(np.random.randn(16000).astype(np.float32))

    # Build up some state
    for _ in range(20):
        sfm.step(np.random.randn(160).astype(np.float32) * 0.3)
    # Save the dvec
    dvec_before = sfm._dvec.copy()

    sfm.reset_state()
    # Dvec is preserved, hidden is zeroed
    assert np.array_equal(sfm._dvec, dvec_before), "reset_state() must preserve dvec"
    assert (sfm._h == 0).all() and (sfm._c == 0).all(), "reset_state() must zero LSTM hidden"
    print("  reset_state() OK (dvec preserved, hidden zeroed)")


def main() -> None:
    print("== open-voicefilter-lite public API smoke test ==")
    print("[1/7] public surface only exports StreamingVoiceFilter")
    _import_public_api()
    print("[2/7] bundled-ONNX construction reads metadata")
    _test_construction_with_bundled_onnx()
    print("[3/7] set_enrollment(pcm array)")
    _test_set_enrollment_pcm_array()
    print("[4/7] set_enrollment rejects wav path")
    _test_set_enrollment_rejects_path()
    print("[5/7] set_enrollment(array / tensor / multi-channel)")
    _test_set_enrollment_array_tensor()
    print("[6/7] step(pcm) streaming loop")
    _test_step_pcm_loop()
    print("[7/8] input validation + reset_state")
    _test_input_validation()
    _test_reset_state()
    print("[8/8] enrollment toggle (passthrough ↔ enhance mid-stream)")
    _test_enrollment_toggle()
    print("== ALL PUBLIC-API SMOKE TESTS PASSED ==")


if __name__ == "__main__":
    main()
