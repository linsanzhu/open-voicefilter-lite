"""YAML config loader for `HParams`.

Loaded by the training CLI (`open-voicefilter-lite-cli`) but kept in
the library because:

1. Third-party users that fine-tune from a released `model.pt` /
   `hparams.json` may want to overlay a YAML on top of the dataclass
   to flip a few knobs (e.g. device, EMA decay).
2. It's a pure-function helper — no HF / torch / tensorflow heavy
   imports — so it doesn't bloat the install footprint.

The companion `hf.py` (HF Hub endpoint resolution) lives in
`open-voicefilter-lite-cli` because it depends on `huggingface_hub` and
is only relevant during dataset preparation / training.
"""

from __future__ import annotations

import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from voicefilter_lite.hparams import HParams


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

    Accepts three key shapes:
      * Nested dict (the new YAML form):
            model: { audio: { sample_rate: 16000 } }
        Recurses into each block via `_flatten_overrides` so each leaf
        becomes a dotted path that `_route_nested` can dispatch.
      * Dotted key:
            model.audio.sample_rate: 16000
        Routed directly via `_route_nested`.
      * Flat key (legacy `hparams.json` shape):
            sample_rate: 16000
        Routed to the matching top-level field / property shim.

    Unknown keys raise in strict mode (default), are silently ignored
    otherwise. `streaming` and `d_vector_dim` are accepted but no-ops
    (streaming mode removed; d-vector dim is fixed).

    Returns the list of dotted key paths that were applied.
    """
    applied: list[str] = []
    unknown: list[str] = []
    noop_keys = {"d_vector_dim", "streaming"}

    flat_overrides = _flatten_overrides(overrides)

    for key, value in flat_overrides.items():
        if key in noop_keys:
            applied.append(key)
            continue
        # Dotted path: route through the sub-dataclass hierarchy.
        if "." in key:
            ok = _route_nested(hp, key, value)
            if ok:
                applied.append(key)
            else:
                unknown.append(key)
            continue
        # Flat key: must be a dataclass field or a property shim on HParams.
        flat_valid = (
            key in HParams.__dataclass_fields__
            or hasattr(HParams, key)
        )
        if not flat_valid:
            unknown.append(key)
            continue
        if key == "target_splits" and isinstance(value, list):
            value = tuple(value)
        try:
            setattr(hp, key, value)
        except AttributeError:
            unknown.append(key)
            continue
        applied.append(key)

    hp.__post_init__()

    if unknown:
        msg = f"[cli] unknown hparam(s) in config: {', '.join(sorted(unknown))}"
        if strict:
            raise ValueError(msg)
        print(msg, file=sys.stderr)

    return applied


def _flatten_overrides(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a (possibly nested) mapping into dotted-path keys.

    Example:
        {"model": {"audio": {"sample_rate": 16000}}}
        → {"model.audio.sample_rate": 16000}

    Leaves that are themselves mappings are recursed; any other value
    (including lists, scalars, None) is kept as-is. The caller routes
    each dotted key through `_route_nested`, which already handles
    `list → tuple` coercion for tuple-typed fields like `target_splits`.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, Mapping):
            out.update(_flatten_overrides(v, path))
        else:
            out[path] = v
    return out


def _route_nested(hp: HParams, dotted_key: str, value: Any) -> bool:
    """Set `hp.<block>.<sub>.<field>` via dotted key path.

    Returns True on success, False if any segment is invalid. The
    router walks dataclass fields, but also accepts `dict[str, X]`
    and `tuple[X, ...]` / `list[X]` segments so function-oriented
    configs (e.g. `dataset.interference.babble.prob`,
    `dataset.target.sources.0.splits`) can be overlaid.
    """
    parts = dotted_key.split(".")
    obj: Any = hp
    for part in parts[:-1]:
        if isinstance(obj, dict):
            if part not in obj:
                return False
            obj = obj[part]
            continue
        if isinstance(obj, (list, tuple)):
            try:
                idx = int(part)
            except ValueError:
                return False
            if idx < 0 or idx >= len(obj):
                return False
            obj = obj[idx]
            continue
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
        if not is_dataclass(obj) and not isinstance(obj, (dict, list, tuple)):
            return False
    leaf = parts[-1]
    # Final container: dict (assign key) vs dataclass (setattr leaf).
    if isinstance(obj, dict):
        if leaf not in obj and not isinstance(obj.get(leaf), object):
            # Allow insertion of new keys only if container is non-strict;
            # we treat all dicts as strict for safety.
            return leaf in obj
        obj[leaf] = _coerce_for_assignment(obj.get(leaf), value)
        return True
    if is_dataclass(obj):
        if leaf not in {f.name for f in fields(obj)} and not hasattr(type(obj), leaf):
            return False
        setattr(obj, leaf, _coerce_for_assignment(getattr(obj, leaf, None), value))
        return True
    return False


def _coerce_for_assignment(existing: Any, value: Any) -> Any:
    """Coerce `value` so that `existing = value` matches the old type.

    Handles:
      * `list → tuple` for tuple-typed fields (e.g. `splits`).
      * `dict → dataclass` when the field's existing value is a
        dataclass instance (e.g. replacing `TargetSourceConfig` /
        `InterferenceTypeConfig` from a YAML mapping).
      * `list[dict] → tuple[dataclass]` when the field is a tuple of
        dataclass instances (e.g. `sources`).
    """
    if existing is None:
        return value
    if is_dataclass(existing) and isinstance(value, dict):
        # Re-instantiate the existing dataclass with the dict's keys.
        return type(existing)(**value)
    if isinstance(existing, tuple) and isinstance(value, list):
        elem_type = type(existing[0]) if existing and is_dataclass(existing[0]) else None
        coerced = [
            elem_type(**item) if elem_type and isinstance(item, dict) else item
            for item in value
        ]
        return tuple(coerced)
    return value


def load_and_apply(hp: HParams, config_path: str | Path | None) -> None:
    """Convenience: load config (if given) and overlay onto `hp`."""
    if config_path is None:
        return
    overrides = load_config(config_path)
    applied = apply_overrides(hp, overrides, strict=True)
    print(f"[cli] loaded config: {config_path} ({len(applied)} field(s) overridden)")