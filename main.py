#!/usr/bin/env python3
"""Top-level convenience wrapper.

Equivalent to the `voicefilter-cli` console script installed by the
`open-voicefilter-lite-cli` package. Lets you keep the existing
`python main.py <cmd> ...` workflow even before `uv sync` has installed
the console entry script.

    python main.py train voicefilter-lite --config configs/paper.yaml
    python main.py dataset clear --dry-run
    python main.py export onnx --checkpoint runs/vflite/ckpt_final.pt
"""

from __future__ import annotations

from voicefilter_cli.main import main

if __name__ == "__main__":
    main()