#!/usr/bin/env python3
"""DEPRECATED — this script has been renamed to `voicefilter-cli dataset prepare`.

The old `download_data.py` only pre-cached arrow files; the new
`prepare_dataset.py` does the *full* preprocess (HF download → decode →
write 16 kHz float32 wav → pre-bake babble pool → compute per-speaker
d-vectors) in one pass, so the training loop can read local files only.

This shim forwards all CLI args to `prepare_dataset.py` for backward
compatibility — anything you ran with `download_data.py` works
identically with `prepare_dataset.py`. A deprecation warning is emitted
on every invocation.

Migration:

    # before
    python scripts/download_data.py --assets musan

    # after (recommended)
    `voicefilter-cli dataset prepare` --musan
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.warn(
    "scripts/download_data.py is deprecated; "
    "use scripts/prepare_dataset.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the new entry point so existing shell pipelines
# (`python scripts/download_data.py --assets X`) keep working.
from voicefilter_cli.prepare_dataset import main  # noqa: E402

if __name__ == "__main__":
    main()
