#!/usr/bin/env python3
"""Verify whether the VoiceFilter d-vector is consistent across utterances of
the same LibriSpeech speaker, and discriminable between different speakers.

Hypothesis: if the dvec quality is fine, within-speaker cosine similarity should
be much higher than cross-speaker. If they're similar, the dvec isn't doing its
job — the mask predictor would have no reliable cue to pick the right speaker.

Targets (paper d-vector, 256-d):
    within-speaker mean cos sim ≥ 0.7
    cross-speaker mean cos sim  ≤ 0.3
    margin (within − cross) > 0.4 → GOOD
"""
from __future__ import annotations

import os
import sys
import io
import random
import torch
import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from datasets import load_dataset
from open_voicefilter_lite.cli import apply_hf_endpoint
from open_voicefilter_lite.dataset import (
    _disable_hf_audio_decode,
    _cast_audio_no_decode,
)
from open_voicefilter_lite.hparams import HParams
from open_voicefilter_lite.audio import extract_dvec, pad_or_trim
from open_voicefilter_lite.speaker_encoder import SpeakerEncoder


def decode_audio(audio_field, target_sr):
    if isinstance(audio_field, dict):
        if "bytes" in audio_field and audio_field["bytes"] is not None:
            data, sr = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32")
        elif "array" in audio_field and audio_field["array"] is not None:
            data = np.asarray(audio_field["array"], dtype="float32")
            sr = int(audio_field.get("sampling_rate", target_sr))
        else:
            return None
    else:
        data = np.asarray(audio_field, dtype="float32")
        sr = target_sr
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.from_numpy(data)
    if sr != target_sr:
        import torchaudio
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
    peak = wav.abs().max().clamp(min=1e-8)
    if peak > 5.0:
        wav = wav / peak
    return wav


def main():
    hp = HParams()
    apply_hf_endpoint(hp)

    _disable_hf_audio_decode()
    ds = load_dataset("openslr/librispeech_asr", "clean", split="train.360", streaming=True)
    ds = _cast_audio_no_decode(ds)

    # Collect up to 6 utterances from each of 8 speakers
    target_per_speaker = 6
    n_speakers = 8
    by_spk: dict[str, list] = {}
    rng = random.Random(0)
    it = iter(ds)
    while len(by_spk) < n_speakers:
        try:
            ex = next(it)
        except StopIteration:
            break
        spk = str(ex.get("speaker_id", "?"))
        by_spk.setdefault(spk, [])
        if len(by_spk[spk]) < target_per_speaker:
            by_spk[spk].append(ex)

    speakers = list(by_spk.keys())[:n_speakers]
    print(f"Speakers: {speakers}\n")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    enc = SpeakerEncoder(hp).to(device).eval()
    from pathlib import Path
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
        print("→ d-vector is GOOD: speaker identity is clearly encoded.")
    elif margin > 0.2:
        print("→ d-vector is MARGINAL: signal is present but weaker than expected.")
    else:
        print("→ d-vector is BAD: speaker identity is NOT encoded. The "
              "model has no useful signal. Verify that "
              "`models/embedder.pt` matches the upstream `SpeechEmbedder` "
              "weights from "
              "https://github.com/maum-ai/voicefilter/blob/master/model/"
              "embedder.py (key layout: `lstm.*` and `proj.linear_layer.*`).")


if __name__ == "__main__":
    main()