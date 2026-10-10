"""Write/load VoiceFilter-Lite hparam metadata into exported ONNX files.

Both ``export_onnx.py`` (mask predictor) and ``export_embedder_onnx.py``
(d-vector) call :func:`embed_hp_metadata` after the graph is finalised
(post ``torch.onnx.export`` and post ``quantize_dynamic``). The
metadata is written as ONNX ``metadata_props`` so a third party can
recover the configuration without a PyTorch checkpoint::

    from voicefilter_cli._onnx_metadata import load_hp_metadata
    meta = load_hp_metadata("voicefilter_lite.int8.onnx")
    # meta = {"schema_version": "1", "model_kind": "mask_predictor",
    #         "config": {...}}

Schema is versioned (``schema_version``) so consumers can guard against
silent evolution. ``config`` is a flat JSON dict for trivial parsing.

Re-saving a model with new metadata overwrites the existing
serialisation, so this helper also re-inlines external weights
(``save_as_external_data=False``); otherwise the file would silently
become a thin graph plus a sibling ``.data`` and break consumers that
move or skip the ``.data`` sibling.
"""

from __future__ import annotations

import json
import os
from typing import Any

import onnx


# Bump when the structure of `config` below changes in a breaking way.
SCHEMA_VERSION = "1"

# Our keys — stripped on re-export so rewriting onto an existing file
# doesn't accumulate stale fields from the previous run.
_OURS = ("schema_version", "model_kind", "voicefilter_lite_config")


def _json_safe(value: Any) -> Any:
    """Coerce a nested config value to JSON-safe types.

    Tuples → lists, ``Path`` and unknown objects → str()s, dicts recurse.
    Used to serialise ``HParams`` slices (which use tuples for things like
    ``HfSourceSpec.configs``) into the JSON blob.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(x) for x in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def embed_hp_metadata(path: str, model_kind: str, hp_dict: dict) -> None:
    """Write hparam metadata into ``path`` and re-inline external weights.

    Idempotent: strips any prior run of our own keys before writing the
    new ones, so re-exporting onto the same file doesn't accumulate
    stale fields. Preserves any other vendor's keys (e.g. ORT's
    ``onnx.infer`` / ``onnx.quant.pre_process`` left over from
    ``quantize_dynamic``).

    Args:
        path: ONNX file to mutate (in place).
        model_kind: short tag written into ``model_kind`` so they can
            filter without parsing JSON — one of ``"dvector"`` or
            ``"mask_predictor"``.
        hp_dict: nested dict of hparams; serialised as JSON into
            ``voicefilter_lite_config``.
    """
    model = onnx.load(path)

    # Rebuild metadata_props preserving non-our keys.
    preserved = [p for p in model.metadata_props if p.key not in _OURS]
    del model.metadata_props[:]
    for p in preserved:
        new = model.metadata_props.add()
        new.key = p.key
        new.value = p.value

    # Append our three keys. `schema_version` is the version guard;
    # `model_kind` is a cheap filter; `voicefilter_lite_config` carries
    # the full JSON blob (always use json.loads to read it).
    for k, v in (
        ("schema_version", SCHEMA_VERSION),
        ("model_kind", model_kind),
        ("voicefilter_lite_config", json.dumps(_json_safe(hp_dict))),
    ):
        prop = model.metadata_props.add()
        prop.key = k
        prop.value = v

    # Re-serialise without external-data siblings (otherwise we'd ship
    # a thin `.onnx` plus a `.data` file).
    onnx.save(model, path, save_as_external_data=False)
    try:
        os.remove(path + ".data")
    except FileNotFoundError:
        pass


def load_hp_metadata(path: str) -> dict:
    """Read metadata embedded by :func:`embed_hp_metadata`.

    Returns ``{"schema_version": str, "model_kind": str | None,
    "config": dict}``. Raises :class:`ValueError` if no
    ``schema_version`` is present — the caller should re-export with
    the current CLI in that case.
    """
    model = onnx.load(path)
    props = {p.key: p.value for p in model.metadata_props}
    if "schema_version" not in props:
        raise ValueError(
            f"{path} has no voicefilter_lite metadata (no `schema_version`); "
            f"the file was exported with an older CLI — re-run "
            f"`voicefilter-cli export ...` against the current source."
        )
    return {
        "schema_version": props["schema_version"],
        "model_kind": props.get("model_kind"),
        "config": json.loads(props["voicefilter_lite_config"]),
    }