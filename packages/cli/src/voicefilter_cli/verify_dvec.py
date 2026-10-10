"""Verify whether the VoiceFilter d-vector is consistent across utterances of
the same LibriSpeech speaker, and discriminable between different speakers.

Hypothesis: if the dvec quality is fine, within-speaker cosine similarity should
be much higher than cross-speaker. If they're similar, the dvec isn't doing its
job — the mask predictor would have no reliable cue to pick the right speaker.

Targets (paper d-vector, 256-d):
    within-speaker mean cos sim ≥ 0.7
    cross-speaker mean cos sim  ≤ 0.3
    margin (within − cross) > 0.4 → GOOD

Usage
-----
    `voicefilter-cli evaluate dvector`
    `voicefilter-cli evaluate dvector` \\
        --num-speakers 16 --num-utts-per-speaker 8 \\
        --device mps --results-json runs/dvec_verify.json
"""
from __future__ import annotations

import io
import json
import os
import random
import sys
from pathlib import Path
from typing import Annotated, Optional

import torch
import numpy as np
import soundfile as sf
from cyclopts import App, Parameter

from datasets import load_dataset
from voicefilter_cli.hf import apply_hf_endpoint
from model.dataset import (
    _disable_hf_audio_decode,
    _cast_audio_no_decode,
)
from voicefilter_lite.hparams import HParams, resolve_source
from model.audio import extract_dvec, pad_or_trim
from model.speaker_encoder import SpeakerEncoder

def decode_audio(audio_field, target_sr):
    if isinstance(audio_field, dict):
        if "bytes" in audio_field and audio_field["bytes"] is not None:
            data, sr = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32")
        elif "array" in audio_field and audio_field["array"] is not None:
            data = np.asarray(audio_field["array"], dtype=np.float32)
            sr = int(audio_field.get("sampling_rate", target_sr))
        else:
            return None
        wav = torch.from_numpy(data)
        if sr != target_sr:
            import torchaudio
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
    else:
        data = np.asarray(audio_field, dtype=np.float32)
        sr = target_sr
        wav = torch.from_numpy(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
        wav = torch.from_numpy(data)
    peak = wav.abs().max().clamp(min=1e-8)
    if peak > 5.0:
        wav = wav / peak
    return wav

def select_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(arg)

def run(num_speakers: int, num_utts_per_speaker: int, seed: int, device: str,
        results_json: Optional[str], hf_endpoint: Optional[str], hf_transfer: Optional[bool]) -> None:
    hp = HParams()
    apply_hf_endpoint(
        hp,
        cli_override=hf_endpoint,
        cli_transfer=hf_transfer,
    )

    target_per_speaker = num_utts_per_speaker
    n_speakers = num_speakers
    by_spk: dict[str, list] = {}
    rng = random.Random(seed)

    # ---- Stream from HF directly ----
    # Read the first declared target source (`hp.target_sources[0]`).
    # Resolve via `resolve_source` so the cache_dir + repo come from
    # `HfSourceSpec` (paper.yaml → openslr/librispeech_asr → data/librispeech).
    if not hp.target_sources:
        raise RuntimeError("[verify-dvec] hp.target_sources is empty")
    resolved = resolve_source(hp, hp.target_sources[0])
    assert resolved.is_librispeech(), (
        f"verify-dvec target must be a LibriSpeech source; got {resolved}"
    )

    _disable_hf_audio_decode()

    def _load_kwargs(cache_dir: str) -> dict:
        kw: dict = {"streaming": hp.streaming}
        if not hp.streaming:
            kw["cache_dir"] = cache_dir or "data/librispeech"
        return kw

    target_iter = iter(
        _cast_audio_no_decode(
            load_dataset(
                resolved.repo, "all",
                split=resolved.splits[0],
                **_load_kwargs(resolved.cache_dir),
            )
        )
    )

    while len(by_spk) < n_speakers:
        try:
            ex = next(target_iter)
        except StopIteration:
            break
        spk = str(ex.get("speaker_id", "?"))
        by_spk.setdefault(spk, [])
        if len(by_spk[spk]) < target_per_speaker:
            by_spk[spk].append(ex)

    speakers = list(by_spk.keys())[:n_speakers]
    print(f"[verify] streaming from HF (target: {resolved.repo} split={resolved.splits[0]})")
    print(f"Speakers: {speakers}\n")

    device = select_device(device)
    enc = SpeakerEncoder(hp).to(device).eval()
    if Path(hp.dvector_pt_path).is_file():
        enc.load_pretrained(hp.dvector_pt_path)
        print(f"[verify] SpeakerEncoder loaded from {hp.dvector_pt_path}")
    else:
        print(f"[verify] WARNING: {hp.dvector_pt_path} not found — encoder uses RANDOM weights.")
        print(f"[verify] The pretrained embedder (`models/embedder.pt`) ships with this repo.")
        enc.freeze()

    # Compute dvec for each utterance
    dvecs: dict[str, list[torch.Tensor]] = {}
    for spk in speakers:
        dvecs[spk] = []
        for ex in by_spk[spk]:
            wav = decode_audio(ex["audio"], hp.sample_rate)
            if wav is None or wav.numel() < int(0.3 * hp.sample_rate):
                continue
            # VoiceFilter d-vector recipe expects arbitrary-length refs;
            # cap at hp.ref_max_sec (default 6 s) to match what the
            # training pipeline hands the encoder.
            wav = pad_or_trim(wav, int(hp.ref_max_sec * hp.sample_rate))
            with torch.no_grad():
                d = extract_dvec(wav, hp, enc, device=device).squeeze(0).cpu()
            dvecs[spk].append(d)

    print(f"Per-speaker dvec counts: {[(s, len(dvecs[s])) for s in speakers]}\n")

    # Within-speaker cosine
    print("WITHIN-speaker cosine (each row = speaker, avg of all pairs):")
    within_per_spk = []
    for spk in speakers:
        vs = dvecs[spk]
        if len(vs) < 2:
            continue
        cos_vals = []
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                c = torch.nn.functional.cosine_similarity(vs[i].unsqueeze(0), vs[j].unsqueeze(0)).item()
                cos_vals.append(c)
        avg = float(np.mean(cos_vals))
        within_per_spk.append(avg)
        print(f"  {spk}: avg={avg:.4f}  min={min(cos_vals):.4f}  max={max(cos_vals):.4f}  n={len(cos_vals)}")

    # Cross-speaker cosine
    print("\nCROSS-speaker cosine (each row = speaker, avg to all other-speaker dvecs):")
    cross_per_spk = []
    for spk in speakers:
        vs = dvecs[spk]
        cos_vals = []
        for other in speakers:
            if other == spk:
                continue
            for v1 in vs:
                for v2 in dvecs[other]:
                    c = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                    cos_vals.append(c)
        avg = float(np.mean(cos_vals))
        cross_per_spk.append(avg)
        print(f"  {spk}: avg={avg:.4f}")

    overall_within = float(np.mean(within_per_spk))
    overall_cross = float(np.mean(cross_per_spk))
    margin = overall_within - overall_cross
    print(f"\n=== Summary ===")
    print(f"Overall within-speaker cosine: {overall_within:.4f}")
    print(f"Overall cross-speaker cosine : {overall_cross:.4f}")
    print(f"Margin (within - cross)      : {margin:+.4f}")
    if margin > 0.4:
        verdict = "GOOD"
    elif margin > 0.2:
        verdict = "MARGINAL"
    else:
        verdict = "BAD"
    print(f"→ d-vector is {verdict}: speaker identity is "
          f"{'clearly encoded' if verdict == 'GOOD' else 'weakly encoded' if verdict == 'MARGINAL' else 'NOT encoded'}.")
    if verdict == "BAD":
        print("  Verify that `models/embedder.pt` matches the upstream "
              "`SpeechEmbedder` weights from "
              "https://github.com/maum-ai/voicefilter/blob/master/model/"
              "embedder.py (key layout: `lstm.*` and `proj.linear_layer.*`).")

    if results_json:
        Path(results_json).write_text(json.dumps({
            "num_speakers": len(speakers),
            "num_utts_per_speaker": target_per_speaker,
            "overall_within_cosine": overall_within,
            "overall_cross_cosine": overall_cross,
            "margin": margin,
            "verdict": verdict,
            "within_per_spk": {spk: avg for spk, avg in zip(speakers, within_per_spk)},
            "cross_per_spk": {spk: avg for spk, avg in zip(speakers, cross_per_spk)},
        }, indent=2, ensure_ascii=False))
        print(f"[verify] wrote {results_json}")

app = App(name="dvector", help_format="plaintext")


@app.default
def main(
    *,
    num_speakers: Annotated[
        int, Parameter(name="--num-speakers", help="Randomly sample this many distinct speakers.")
    ] = 8,
    num_utts_per_speaker: Annotated[
        int,
        Parameter(
            name="--num-utts-per-speaker",
            help="Number of utterances to sample per speaker.",
        ),
    ] = 6,
    seed: int = 0,
    device: Annotated[
        str,
        Parameter(
            name="--device",
            help="mps / cuda / cpu (default 'auto' picks MPS → CUDA → CPU).",
        ),
    ] = "auto",
    results_json: Annotated[
        Optional[str], Parameter(name="--results-json", help="Write the verification results to this JSON file path.")
    ] = None,
    hf_endpoint: Annotated[
        Optional[str], Parameter(name="--hf-endpoint", help="Override HF Hub base URL (e.g. https://hf-mirror.com).")
    ] = None,
    hf_transfer: Annotated[
        Optional[bool], Parameter(negative="--no-hf-transfer", help="Force-enable HF hf_transfer protocol.")
    ] = None,
) -> None:
    """Verify d-vector speaker consistency.

    Within-speaker cosine should be much higher than cross-speaker. If not,
    the embedder isn't encoding speaker identity and the mask predictor has
    no useful cue.
    """
    run(
        num_speakers=num_speakers,
        num_utts_per_speaker=num_utts_per_speaker,
        seed=seed,
        device=device,
        results_json=results_json,
        hf_endpoint=hf_endpoint,
        hf_transfer=hf_transfer,
    )


if __name__ == "__main__":
    app()