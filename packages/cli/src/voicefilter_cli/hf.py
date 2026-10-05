"""HF Hub endpoint resolution — CLI-only.

Lives in the CLI package rather than the library because:

1. It depends on `huggingface_hub` for `HF_ENDPOINT` env handling — a
   dependency that lib users (who just want to load a pretrained
   `model.pt`) shouldn't have to install.
2. The resolution is only meaningful for the HF datasets the CLI
   downloads (`openslr/librispeech_asr`, `corypaik/musan`, etc.) — the
   library's runtime path doesn't make any HF calls.

`apply_hf_endpoint` is exported as `voicefilter_cli.hf.apply_hf_endpoint`
and is the single entry point used by `train.py`, `prepare_dataset.py`,
`verify_dvec.py`, `evaluate.py`, `diag_mask.py`, etc.
"""

from __future__ import annotations

import os
from typing import Optional

from voicefilter_lite.hparams import HParams


def apply_hf_endpoint(
    hp: Optional[HParams] = None,
    *,
    cli_override: Optional[str] = None,
    cli_transfer: Optional[bool] = None,
    silent: bool = False,
) -> Optional[str]:
    """Resolve and apply the HF Hub endpoint for this process.

    Precedence (first non-empty wins):
      1. ``cli_override`` — typically the ``--hf-endpoint`` flag on each
         entry-point script (train / evaluate / prepare_dataset).
      2. ``HF_ENDPOINT`` env var — the convention understood by
         ``huggingface_hub`` / ``datasets``.
      3. ``hp.hf_endpoint`` — the YAML config field (or its CLI override
         applied via ``voicefilter_lite.config.load_and_apply``).
      4. ``None`` → leave the default ``https://huggingface.co`` in place.

    Side effects when a non-empty endpoint is chosen:
      * ``os.environ["HF_ENDPOINT"]`` is set to the resolved value so all
        downstream HF calls (made lazily by ``datasets.load_dataset``,
        ``EncoderClassifier.from_hparams``, etc.) pick it up.

    ``HF_HUB_ENABLE_HF_TRANSFER`` is resolved in parallel:
      * Explicit ``cli_transfer`` (from ``--hf-transfer`` / ``--no-hf-transfer``)
        or ``hp.hf_transfer`` (YAML) wins → env var is force-set.
      * Otherwise, when an endpoint is chosen and the env var is unset,
        it's auto-disabled as a mirror-safety net (most public mirrors
        don't speak ``hf_transfer``).
      * Otherwise the env var is left alone (regular HTTPS default).

    Returns the resolved endpoint (or ``None`` if using the default).
    Pass ``silent=True`` to skip the one-line status print.
    """
    chosen: Optional[str] = None
    if cli_override:
        chosen = cli_override
    elif os.environ.get("HF_ENDPOINT"):
        chosen = os.environ["HF_ENDPOINT"]
    elif hp is not None and hp.hf_endpoint:
        chosen = hp.hf_endpoint

    if chosen:
        os.environ["HF_ENDPOINT"] = chosen

    # ---- hf_transfer resolution ----
    # Explicit knob wins (CLI > YAML). When neither side specifies, only
    # auto-flip if an endpoint was chosen AND the env var is unset — this
    # preserves any deliberate user setting (e.g. `HF_HUB_ENABLE_HF_TRANSFER=1`
    # on the original HF.co host).
    transfer = cli_transfer
    if transfer is None and hp is not None:
        transfer = hp.hf_transfer

    if transfer is not None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" if transfer else "0"
    elif chosen and "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
        # Mirror safety net: default to regular HTTPS so users who point
        # at hf-mirror.com (or any mirror that doesn't speak hf_transfer)
        # don't silently 502.
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    if not silent:
        if chosen is None:
            print("[hf] endpoint=<default huggingface.co> "
                  "(set hf_endpoint in config or HF_ENDPOINT env / --hf-endpoint to use a mirror)")
        else:
            print(f"[hf] endpoint={chosen}")
        # Show the effective transfer mode so users see what was applied.
        eff = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1"
        marker = "explicit" if transfer is not None else "auto"
        print(f"[hf] transfer={'on' if eff else 'off'} ({marker})")

    return chosen