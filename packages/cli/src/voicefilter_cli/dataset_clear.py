#!/usr/bin/env python3
"""Delete baked dataset products and HF arrow caches.

`voicefilter-cli dataset prepare` writes three kinds of artefacts. The
*paths below are the conventional defaults baked into the CLI flags*;
your `*.yaml` may have set `cache_dir:` on any `HfSourceSpec`, in
which case that source's HF cache lives wherever the spec points (e.g.
`paper.yaml` pins LibriSpeech to `data/librispeech` and MUSAN to
`data/musan` — see `--librispeech-cache` / `--musan-cache` to learn
what *this run* actually used).

    data/processed/...                          baked wavs + manifest.json
                                               (the `processed_dir` flag)
    data/librispeech/                            LibriSpeech HF arrow cache
      openslr___librispeech_asr/<config>/        (root path is the
        <version>/<hash>/...                      `--librispeech-cache`
                                                   default; HF's
                                                   `datasets` library
                                                   nests the actual
                                                   arrows under
                                                   `<repo_sanitized>/`)
    data/musan/                                 MUSAN HF arrow cache
                                                (root path is the
                                                 `--musan-cache`
                                                 default — matches
                                                 `paper.yaml`'s
                                                 `cache_dir: data/musan`;
                                                 HF's `datasets`
                                                 library nests the
                                                 actual arrows under
                                                 `<repo_sanitized>/`)

This script wipes those directories so the next `dataset prepare` run
re-downloads / re-decodes from scratch — useful when the manifest is
stale, when an HF mirror serves corrupted arrows, or when you just want
to reclaim disk space.

By default the script asks for an interactive confirmation. Pass
`--yes` (`-y`) to skip the prompt (handy for CI).

Usage
-----
    # Interactive (asks for confirmation)
    python scripts/dataset_clear.py

    # Non-interactive
    python scripts/dataset_clear.py --yes

    # Skip the baked wav dir, only clear HF caches
    python scripts/dataset_clear.py --no-processed --yes

    # Custom cache locations (overrides defaults)
    python scripts/dataset_clear.py \\
        --processed-dir /tmp/vflite \\
        --musan-cache /tmp/musan_hf \\
        --yes
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter


def _rm(path: Path) -> tuple[str, int]:
    """Recursively delete `path`.  Returns the path (display) and the size
    that was freed (bytes).  Missing directories count as 0 freed."""
    if not path.exists():
        return str(path), 0

    total = 0
    if path.is_dir():
        for root, _dirs, files in path.walk() if hasattr(path, "walk") else _walk(path):
            for f in files:
                fp = Path(root) / f
                try:
                    total += fp.stat().st_size
                except OSError:
                    pass
    else:
        try:
            total += path.stat().st_size
        except OSError:
            pass

    shutil_rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)
    return str(path), total


def _walk(path: Path):
    """`Path.walk` was added in 3.12; provide a fallback for 3.11."""
    import os as _os
    for root, dirs, files in _os.walk(path):
        yield root, dirs, files


def shutil_rmtree(path: Path) -> None:
    import shutil
    shutil.rmtree(path)


app = App(name="clear", help_format="plaintext")


@app.default
def main(
    *,
    processed_dir: Annotated[
        str,
        Parameter(name="--processed-dir", help="Baked wav / npy / manifest output of `prepare_dataset.py`."),
    ] = "data/processed",
    librispeech_cache: Annotated[
        str,
        Parameter(
            name="--librispeech-cache",
            help=(
                "HF arrow cache for LibriSpeech. Default matches "
                "`configs/paper.yaml` (`cache_dir: data/librispeech`); "
                "override if your spec points elsewhere."
            ),
        ),
    ] = "data/librispeech",
    musan_cache: Annotated[
        str,
        Parameter(
            name="--musan-cache",
            help=(
                "HF arrow cache for MUSAN. Default matches "
                "`configs/paper.yaml` (`cache_dir: data/musan`); "
                "override if your spec points elsewhere."
            ),
        ),
    ] = "data/musan",
    # Toggle flags (default = delete; pass the negative flag to skip).
    clear_processed: Annotated[
        bool,
        Parameter(name="--processed", negative="--no-processed", help="Clear --processed-dir (default: delete). Pass --no-processed to skip."),
    ] = True,
    clear_librispeech: Annotated[
        bool,
        Parameter(name="--librispeech", negative="--no-librispeech", help="Clear --librispeech-cache (default: delete). Pass --no-librispeech to skip."),
    ] = True,
    clear_musan: Annotated[
        bool,
        Parameter(name="--musan", negative="--no-musan", help="Clear --musan-cache (default: delete). Pass --no-musan to skip."),
    ] = True,
    yes: Annotated[
        bool,
        Parameter(alias=("-y",), help="Skip the interactive confirmation prompt."),
    ] = False,
    dry_run: Annotated[
        bool,
        Parameter(name="--dry-run", help="Print what would be deleted without actually removing anything."),
    ] = False,
) -> None:
    """Delete baked dataset products and HF arrow caches.

    Pass ``--yes`` (``-y``) to skip the interactive confirmation.
    """
    targets: list[tuple[str, str, bool]] = [
        ("processed",     processed_dir,     clear_processed),
        ("librispeech",   librispeech_cache, clear_librispeech),
        ("musan",         musan_cache,       clear_musan),
    ]
    planned = [(name, p_str) for name, p_str, on in targets if on]

    if not planned:
        print("[clear] nothing selected — pass at least one target (or drop --no-* flags).")
        return

    print("[clear] plan:")
    total_size = 0
    rows: list[tuple[str, Path, bool, int]] = []  # (label, path, exists, bytes)
    for label, p_str in planned:
        p = Path(p_str).resolve()
        exists = p.exists()
        size = 0
        if exists:
            if p.is_dir():
                for root, _dirs, files in _walk(p):
                    for f in files:
                        fp = Path(root) / f
                        try:
                            size += fp.stat().st_size
                        except OSError:
                            pass
            else:
                try:
                    size = p.stat().st_size
                except OSError:
                    pass
        rows.append((label, p, exists, size))
        total_size += size
        marker = "[ok]" if exists else "[missing]"
        print(f"  {marker} {label:<14s} {p}  ({_fmt_size(size)})")

    print(f"[clear] total to free: {_fmt_size(total_size)}")

    if dry_run:
        print("[clear] dry-run — nothing was deleted.")
        return

    if not yes:
        try:
            answer = input("[clear] type 'yes' to confirm deletion: ").lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[clear] aborted (no input).")
            return
        if answer not in ("yes", "y"):
            print("[clear] aborted.")
            return

    # Delete.
    import shutil
    for label, p, exists, size in rows:
        if not exists:
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            print(f"[clear] removed {label}  ({_fmt_size(size)})")
        except OSError as e:
            print(f"[clear] FAILED to remove {label} ({p}): {e}", file=sys.stderr)


def _fmt_size(n: int) -> str:
    if n <= 0:
        return "0 B"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


if __name__ == "__main__":
    app()