#!/usr/bin/env python3
"""Pre-download assets for VoiceFilter-Lite training.

By default this only pre-caches LibriSpeech (which `scripts/train.py` would
otherwise stream on the fly). It can also pre-download:

    - musan    : music / noise / speech noise recordings from
                 `corypaik/musan` (~10 GB total — paper §3.4
                 augmentation source; `babble` is synthesised at training
                 time from the `speech` subset)
    - rir      : real room impulse responses — primary loader
                 `confit/rir-noise` (pulls OpenSLR28 from openslr.org),
                 fallback `quinnlue/rirs`
    - vctk     : paper §3.2 target clean-speech source (CSTR-Edinburgh/vctk)

The speaker encoder (paper §3.1 d-vector) ships pretrained as
`models/embedder.pt` in the repo root and is **not** fetched from any
external source.

Each asset is independently optional; a failure on one does not stop the others.
All downloads land in `models/` by default (which is gitignored).

Usage
-----
    # Just LibriSpeech `train.100` (default — only ~6 GB):
    python scripts/download_data.py

    # LibriSpeech `train.100` AND `train.360` (~50 GB combined):
    python scripts/download_data.py --train-360

    # Just `train.360`, skip `train.100`:
    python scripts/download_data.py --no-train-100 --train-360

    # The paper's training set: `train.360` + `train.500` (~120 hours combined
    # — note paper §3.4 uses these, not train.100):
    python scripts/download_data.py --no-train-100 --train-360 --train-500

    # Everything (LibriSpeech + Musan + RIR + VCTK):
    python scripts/download_data.py --assets all

    # A specific subset:
    python scripts/download_data.py --assets musan rir
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the `src/` layout importable without `pip install -e .` so this
# script can call `cli.apply_hf_endpoint` to honour HF_ENDPOINT / mirror
# config before any `load_dataset` round-trip.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

from open_voicefilter_lite.cli import apply_hf_endpoint  # noqa: E402


# Splits that actually exist in each config of `openslr/librispeech_asr`.
# (HF calls the validation set just `validation` — not `validation.clean`.)
CONFIG_SPLITS: dict[str, list[str]] = {
    "clean": ["train.100", "train.360", "validation", "test"],
    "other": ["train.500", "validation", "test"],
}

# Musan configs on `corypaik/musan`. `babble` is not an HF config —
# it is synthesised at training time from `speech` (see `_synthesise_babble`
# in `dataset.py`). We only need to pre-download the three real subsets.
MUSAN_SUBSETS: list[str] = ["music", "noise", "speech"]


# ---- Per-asset downloaders ----


def download_librispeech(args) -> None:
    from datasets import load_dataset

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    requested = set(args.splits)
    if args.train_100:
        requested.add("train.100")
    if args.train_360:
        requested.add("train.360")
    if args.train_500:
        requested.add("train.500")
    for config in args.configs:
        valid = CONFIG_SPLITS.get(config, [])
        splits = [s for s in requested if s in valid]
        for split in splits:
            print(f"[download] librispeech/{config}/{split} → {out / config / split}")
            ds = load_dataset(
                "openslr/librispeech_asr",
                config,
                split=split,
                cache_dir=str(out / "hf_cache"),
            )
            print(f"[download]   cached {len(ds):,} utterances")


def download_musan(args) -> None:
    """Pre-download Musan music/noise/speech from `corypaik/musan`.

    Each subset is loaded non-streaming into a separate cache directory under
    `models/musan/`. Total size on disk is ~10 GB.
    """
    from datasets import load_dataset

    root = Path(args.musan_cache)
    root.mkdir(parents=True, exist_ok=True)

    for subset in MUSAN_SUBSETS:
        target = root / subset
        target.mkdir(parents=True, exist_ok=True)
        print(f"[download] musan/{subset} → {target}")
        try:
            ds = load_dataset(
                args.musan_source,
                subset,
                split="train",
                cache_dir=str(target),
            )
            print(f"[download]   cached {len(ds):,} files")
        except Exception as e:
            print(f"[download]   FAILED musan/{subset}: {e!r}")


def download_vctk(args) -> None:
    """Pre-download VCTK (paper §3.2 target clean-speech source).

    `CSTR-Edinburgh/vctk` exposes a single `train` split (with `speaker_id`
    + `audio` per row). We download non-streaming into
    `models/vctk/` so a paper-faithful run with `target_source: vctk` can
    train without per-batch network fetches. Total size on disk is ~10 GB.
    """
    from datasets import load_dataset

    root = Path(args.vctk_cache)
    root.mkdir(parents=True, exist_ok=True)
    print(f"[download] vctk (CSTR-Edinburgh/vctk) → {root}")
    try:
        ds = load_dataset(
            args.vctk_source,
            split="train",
            cache_dir=str(root),
        )
        print(f"[download]   cached {len(ds):,} files (VCTK target clean speech)")
    except Exception as e:
        print(f"[download]   FAILED vctk: {e!r}")


def download_rir(args) -> None:
    """Pre-download real room impulse responses.

    Tries `args.rir_source` first, then any `--rir-source-fallbacks`, then
    each with a `simulated_rirs` config then the default config. First
    (source, config) that yields a usable dataset wins.

    If every option fails, training degrades to procedural exponential-decay
    RIRs — see `dataset._ensure_rir_open`.
    """
    from datasets import load_dataset

    savedir = Path(args.rir_cache)
    savedir.mkdir(parents=True, exist_ok=True)
    sources: list[str] = [args.rir_source, *args.rir_source_fallbacks]

    config_candidates = [
        ({"name": "simulated_rirs"}, "config=simulated_rirs"),
        ({}, "default config"),
    ]

    for source in sources:
        for load_kwargs, label in config_candidates:
            try:
                print(f"[download] rir {source} ({label}) → {savedir}")
                ds = load_dataset(
                    source,
                    split="train",
                    cache_dir=str(savedir),
                    **load_kwargs,
                )
                print(f"[download]   cached {len(ds):,} RIRs from {source}")
                return
            except Exception as e:
                print(f"[download]   FAILED {source} ({label}): {e!r}")
                continue

    print(f"[download]   all rir sources exhausted")
    print("[download]   training will fall back to procedural RIRs")


# ---- CLI ----


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download VoiceFilter-Lite assets")

    # Asset selection.
    p.add_argument(
        "--assets", nargs="+",
        default=["librispeech"],
        choices=["librispeech", "musan", "rir", "vctk", "all"],
        help=(
            "Which assets to download. `all` is shorthand for "
            "`librispeech vctk musan rir`. "
            "`vctk` is the paper-faithful target-speech source (paper §3.2: "
            "target=VCTK, interference=LibriSpeech). The d-vector embedder "
            "(paper §3.1) ships pretrained as `models/embedder.pt` and is "
            "NOT fetched from any external source. Multiple allowed."
        ),
    )

    # LibriSpeech options.
    p.add_argument("--output-dir", default="data/librispeech",
                   help="Output directory for LibriSpeech cache.")
    p.add_argument("--configs", nargs="+", default=["clean", "other"])
    # Each major training split is opt-in. Defaults to just `train.100`; the
    # paper (§3.4) trains on `train.360` + `train.500` instead — pass those
    # two flags to reproduce.
    p.add_argument(
        "--train-100", dest="train_100", action="store_true", default=True,
        help="Download librispeech train.100 split (default: on).",
    )
    p.add_argument(
        "--no-train-100", dest="train_100", action="store_false",
        help="Skip librispeech train.100 split.",
    )
    p.add_argument(
        "--train-360", dest="train_360", action="store_true", default=False,
        help="Also download librispeech train.360 split (~44 GB).",
    )
    p.add_argument(
        "--train-500", dest="train_500", action="store_true", default=False,
        help="Also download librispeech train.500 split (~54 GB, lives in `other` config).",
    )
    p.add_argument(
        "--splits", nargs="+",
        default=["validation"],
        help=(
            "Other LibriSpeech splits to download in addition to the train "
            "splits above (e.g. `validation`, `test`). Default: validation."
        ),
    )

    # Musan options.
    p.add_argument("--musan-source", default="corypaik/musan")
    p.add_argument("--musan-cache", default="models/musan")

    # VCTK options. Paper §3.2 target clean-speech source. Only used when
    # `--assets vctk` is requested and `hp.target_source="vctk"`.
    p.add_argument(
        "--vctk-source", default="CSTR-Edinburgh/vctk",
        help="HF dataset id for the VCTK target clean-speech source.",
    )
    p.add_argument(
        "--vctk-cache", default="models/vctk",
        help="Where the VCTK arrow cache is unpacked.",
    )

    # RIR options.
    # Default `confit/rir-noise` is an OpenSLR28 loader — pulls the canonical
    # rirs_noises.zip from openslr.org at load time. `quinnlue/rirs` is a
    # HF-resident mirror with the same `simulated_rirs` config layout as a
    # backup if the OpenSLR pull is unreachable.
    p.add_argument(
        "--rir-source", default="confit/rir-noise",
        help="Primary HF dataset id (loader) for real-room impulse responses.",
    )
    p.add_argument(
        "--rir-source-fallbacks", nargs="*",
        default=["quinnlue/rirs"],
        help="Fallback HF dataset ids tried in order after the primary source.",
    )
    p.add_argument("--rir-cache", default="models/rir")

    # HF Hub endpoint override (highest priority — see `cli.apply_hf_endpoint`).
    # Useful when `huggingface.co` is unreachable, e.g. mainland China:
    # `--hf-endpoint https://hf-mirror.com`.
    p.add_argument("--hf-endpoint", default=None,
                   help="Override HF Hub base URL (e.g. https://hf-mirror.com).")
    # HF hf_transfer protocol toggle. Use --no-hf-transfer when pulling
    # big assets from a mirror that doesn't implement the protocol.
    xf = p.add_mutually_exclusive_group()
    xf.add_argument("--hf-transfer", dest="hf_transfer", action="store_true",
                    default=None,
                    help="Force-enable HF hf_transfer protocol.")
    xf.add_argument("--no-hf-transfer", dest="hf_transfer", action="store_false",
                    help="Force-disable HF hf_transfer protocol.")

    # Back-compat: --all is an alias for --assets all.
    p.add_argument("--all", dest="all_flag", action="store_true",
                   help="Shortcut for --assets all.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.all_flag:
        args.assets = ["librispeech", "vctk", "musan", "rir"]

    # Resolve HF endpoint before any `datasets.load_dataset` /
    # `hf_hub_download` call. This script doesn't have an
    # HParams instance — so the config branch is skipped, but CLI flag
    # and `HF_ENDPOINT` env var are still honoured.
    apply_hf_endpoint(
        hp=None,
        cli_override=args.hf_endpoint,
        cli_transfer=args.hf_transfer,
    )

    requested = list(args.assets)
    if "all" in requested:
        requested = ["librispeech", "vctk", "musan", "rir"]

    print(f"[download] assets requested: {requested}")
    for asset in requested:
        try:
            if asset == "librispeech":
                download_librispeech(args)
            elif asset == "vctk":
                download_vctk(args)
            elif asset == "musan":
                download_musan(args)
            elif asset == "rir":
                download_rir(args)
            else:
                # Unknown choices are rejected by argparse; this branch
                # is unreachable unless the choices list is extended.
                print(f"[download] unknown asset: {asset}")
        except Exception as e:
            # Each downloader already prints its own error; this is a safety net.
            print(f"[download] unexpected error downloading {asset}: {e!r}")

    print(f"[download] all done.")


if __name__ == "__main__":
    main()
