#!/usr/bin/env python3
"""Wrapper to run training with sensible defaults.

Examples
--------
    # Full training on Apple Silicon (default)
    python scripts/train.py --output-dir runs/vflite

    # Smoke test (50 steps, small batch)
    python scripts/train.py --output-dir runs/smoke --total-steps 50 --batch-size 2 --no-tensorboard

    # GPU box
    python scripts/train.py --device cuda --output-dir runs/cuda
"""

from __future__ import annotations

import os
import sys

# Make the `src/` layout importable without `pip install -e .`
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from open_voicefilter_lite.train import main  # noqa: E402

if __name__ == "__main__":
    main()
