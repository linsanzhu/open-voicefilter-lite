"""CLI helpers — top-level `open-voicefilter-lite` entry + YAML config loader.

This module is intentionally tiny and side-effect free so it can be
imported by tests without touching the heavy HF / torch / speechbrain
dependencies.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

from open_voicefilter_lite.hparams import HParams


def main() -> None:
    """Top-level console-script entry point."""
    print("open-voicefilter-lite")
    print("Use scripts/train.py, scripts/evaluate.py, scripts/export_onnx.py.")


# ---- YAML config support ----


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file as a flat dict.

    Requires `pyyaml`. A missing file raises with a clear message.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"[cli] config file not found: {p.resolve()}")
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "[cli] PyYAML is required for --config. "
            "Install it: pip install pyyaml (or `pip install -e .` to pick up "
            "the new pyproject.toml dependency)."
        ) from e

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"[cli] config root must be a mapping (got {type(data).__name__}): {p}"
        )
    return data


def apply_overrides(
    hp: HParams,
    overrides: Mapping[str, Any],
    *,
    strict: bool = True,
) -> list[str]:
    """Overlay `overrides` onto the dataclass `hp` in place.

    Only keys that exist in `HParams.__dataclass_fields__` are applied.
    Returns the list of keys that were applied. In strict mode (default),
    unknown keys raise a ValueError; otherwise they are silently ignored.
    """
    valid_keys = set(HParams.__dataclass_fields__.keys())
    applied: list[str] = []
    unknown: list[str] = []

    for key, value in overrides.items():
        if key not in valid_keys:
            unknown.append(key)
            continue
        # Tuples round-trip through lists in JSON/YAML.
        if key == "target_splits" and isinstance(value, list):
            value = tuple(value)
        setattr(hp, key, value)
        applied.append(key)

    if unknown:
        msg = f"[cli] unknown hparam(s) in config: {', '.join(sorted(unknown))}"
        if strict:
            raise ValueError(msg)
        print(msg, file=sys.stderr)

    return applied


def load_and_apply(hp: HParams, config_path: str | Path | None) -> None:
    """Convenience: load config (if given) and overlay onto `hp`."""
    if config_path is None:
        return
    overrides = load_config(config_path)
    applied = apply_overrides(hp, overrides, strict=True)
    print(f"[cli] loaded config: {config_path} ({len(applied)} field(s) overridden)")


# ---- HF Hub endpoint resolution ----


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
         entry-point script (train / evaluate / download_data).
      2. ``HF_ENDPOINT`` env var — the convention understood by
         ``huggingface_hub`` / ``datasets`` / speechbrain.
      3. ``hp.hf_endpoint`` — the YAML config field (or its CLI override
         applied via ``apply_overrides``).
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


if __name__ == "__main__":
    main()