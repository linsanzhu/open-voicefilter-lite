"""open-voicefilter-lite-cli — training / data-prep / ONNX export CLI.

This package owns everything needed to retrain or fine-tune the
VoiceFilter-Lite model from raw LibriSpeech / MUSAN assets:

* `losses.py`, `dataset.py`, `train.py`, `train_dvector.py` — training
* `prepare_dataset.py`, `dataset_clear.py`, `verify_dvec.py` — data
* `export_onnx.py`, `evaluate.py`, `enhance_mix_dump.py`,
  `diag_mask.py`, `download_data.py` — evaluation / deployment
* `hf.py` — HF Hub endpoint resolution (only relevant during data
  download / training, so it lives here rather than in the lib)
* `main.py` — unified CLI entry: `train` / `dataset` / `export` subcommands
"""

from voicefilter_cli.main import main

__all__ = ["main"]