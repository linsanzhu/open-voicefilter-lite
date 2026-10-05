#!/usr/bin/env python3
"""open-voicefilter-lite-cli — unified CLI entry point.

Aggregates the per-task scripts under 4 top-level subcommands:

  train voicefilter-lite    Train the mask-predictor (VoiceFilter-Lite).
  train dvector             Train the d-vector speaker encoder
                            (SpeechEmbedder, GE2E loss on train.100).
  dataset prepare           Bake training data (LibriSpeech + MUSAN +
                            babble) into `data/processed/`.
  dataset clear             Delete baked products + HF arrow caches.
  evaluate voicefilter-lite Evaluate a VoiceFilter-Lite checkpoint on
                            LibriSpeech test-clean/other (SI-SNR /
                            SDR / log-mel SNR improvements).
  evaluate dvector          Verify d-vector speaker consistency
                            (within-speaker vs cross-speaker cosine).
  export onnx               Export VoiceFilter-Lite to ONNX (FP32 or
                            INT8 post-training dynamic quantisation).

Each per-task module owns its own cyclopts ``App`` with a typed-flag
surface (see ``voicefilter_cli.<task>``); the root CLI wires them
together via cyclopts' lazy-import ``App.command("module:attr")``
syntax — heavy deps (torch, ``datasets``, …) only load when the user
actually runs that command.

Examples
--------
    # Train VoiceFilter-Lite with the paper-faithful config
    voicefilter-cli train voicefilter-lite \\
        --config configs/paper.yaml --output-dir runs/vflite

    # Train the d-vector encoder
    voicefilter-cli train dvector --output-dir runs/dvec --steps 10000

    # Prepare training data
    voicefilter-cli dataset prepare \\
        --config configs/paper.yaml --output-dir data/processed

    # Wipe baked data + caches
    voicefilter-cli dataset clear --yes

    # Evaluate a trained VoiceFilter-Lite checkpoint
    voicefilter-cli evaluate voicefilter-lite \\
        --checkpoint runs/vflite/ckpt_final.pt

    # Verify the bundled (or freshly trained) d-vector embedder
    voicefilter-cli evaluate dvector --num-speakers 16

    # Export ONNX (paper §3.5 INT8 quantisation variant)
    voicefilter-cli export onnx \\
        --checkpoint runs/vflite/ckpt_final.pt \\
        --output voicefilter_lite.onnx --quantize int8
"""

from __future__ import annotations

from cyclopts import App


# ---- Top-level root -------------------------------------------------------


root = App(
    name="voicefilter-cli",
    help="open-voicefilter-lite-cli unified CLI.",
    help_format="plaintext",
)


# ---- Subcommand shells ----------------------------------------------------
#
# Each shell exists only to group per-task apps under a parent name.
# Cyclopts accepts ``App.command("module:attr")`` for lazy loading, so
# the per-task ``App`` is only imported when the user actually invokes
# that subcommand.

# `train <target>` — voicefilter-lite or dvector.
train_app = App(name="train", help="Train a model.")
train_app.command(
    "voicefilter_cli.train:app",
    name="voicefilter-lite",
    help="Train the mask-predictor (VoiceFilter-Lite).",
)
train_app.command(
    "voicefilter_cli.train_dvector:app",
    name="dvector",
    help="Train the d-vector speaker encoder (SpeechEmbedder) using "
         "the GE2E loss on LibriSpeech.",
)
root.command(train_app)

# `dataset <action>` — prepare or clear.
dataset_app = App(name="dataset", help="Manage the baked dataset.")
dataset_app.command(
    "voicefilter_cli.prepare_dataset:app",
    name="prepare",
    help="Bake audio into data/processed/.",
)
dataset_app.command(
    "voicefilter_cli.dataset_clear:app",
    name="clear",
    help="Delete baked dataset products + HF arrow caches.",
)
root.command(dataset_app)

# `evaluate <target>` — voicefilter-lite or dvector.
evaluate_app = App(name="evaluate", help="Evaluate a trained model.")
evaluate_app.command(
    "voicefilter_cli.evaluate:app",
    name="voicefilter-lite",
    help="Evaluate a VoiceFilter-Lite checkpoint on LibriSpeech "
         "test-clean/other (SI-SNR / SDR / log-mel SNR).",
)
evaluate_app.command(
    "voicefilter_cli.verify_dvec:app",
    name="dvector",
    help="Verify d-vector speaker consistency (within-speaker vs "
         "cross-speaker cosine).",
)
root.command(evaluate_app)

# `export <target>` — currently only `onnx`.
export_app = App(name="export", help="Export a trained model.")
export_app.command(
    "voicefilter_cli.export_onnx:app",
    name="onnx",
    help="Export VoiceFilter-Lite to ONNX.",
)
root.command(export_app)


# ---- Entry point ----------------------------------------------------------


def cli(argv: list[str] | None = None) -> None:
    """Unified CLI entry.  ``argv`` lets external callers (e.g. tests)
    inject tokens without mutating ``sys.argv``."""
    root(argv)


# Backwards-compatible alias (the package's __init__ exports ``main``).
main = cli


if __name__ == "__main__":
    cli()
