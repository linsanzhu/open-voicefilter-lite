"""Unit tests for ``prepare_dataset._yaml_to_prepare_defaults``.

Verifies that:
    1. The function-oriented ``configs/paper.yaml`` yields the expected
       dataset-baking defaults (target sources, interference types,
       paths, augmentation).
    2. CLI-flag-style overrides from a YAML config block layer correctly
       (CLI > YAML > built-in default).
    3. A bare ``--config some.yaml`` with no ``dataset.*`` block is a
       no-op for dataset-baking fields (built-in defaults still apply).

Run with:
    python tests/test_prepare_dataset_yaml.py

Exit code 0 = all tests passed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the cli source dir importable when running this file directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "cli" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "open-voicefilter-lite" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "model" / "src"))

from voicefilter_cli.prepare_dataset import _yaml_to_prepare_defaults  # noqa: E402
from voicefilter_lite.config import load_config  # noqa: E402


def test_paper_yaml_yields_function_oriented_defaults() -> None:
    cfg = load_config(ROOT / "configs" / "paper.yaml")
    out = _yaml_to_prepare_defaults(cfg)

    # Target sources (function-oriented list of SourceRefs that point
    # at `dataset.sources[*]` by id). The librispeech ref doesn't carry
    # `subset:` — the role tag is encoded in the split name
    # (`train.clean.360` → `clean`).
    assert "target_sources" in out, "target_sources should be derived"
    assert out["target_sources"] == [
        {"source": "librispeech", "splits": ["train.clean.360"]},
    ], out["target_sources"]

    # LibriSpeech role tags to bake: target (clean) + interference speech
    # (other) — derived from split name prefixes, not from a ref field.
    assert set(out["librispeech_subsets"]) == {"clean", "other"}, (
        out["librispeech_subsets"]
    )

    # Splits: target.train.clean.360 (ref override) + interference.
    # train.other.500 (ref override) → train_clean_360 / train_other_500.
    assert out["train_clean_360"] is True, out.get("train_clean_360")
    assert out["train_other_500"] is True, out.get("train_other_500")
    # paper.yaml omits train.{clean,other}.{100} — must NOT default to True.
    assert out.get("train_clean_100", False) is False, (
        "YAML-with-splits should disable train_clean_100, got "
        f"{out.get('train_clean_100')!r}"
    )

    # Interference types — direct fields on `interference` (no `types`
    # wrapper). Built-in keys: babble / noise / music.
    types = out["interference_types"]
    assert set(types.keys()) == {"babble", "noise", "music"}, (
        f"expected 3 types, got {set(types.keys())}"
    )
    # Babble is config-driven — the runtime babble pool is exactly
    # `babble.sources[*]` (no hardcoded fallback). paper.yaml keeps
    # babble.sources = [musan(speech)] only, so runtime babble uses
    # ONLY MUSAN speech. The librispeech(other) pool lives under
    # `noise.sources` (below) so `prepare_dataset` still bakes it and
    # `first_librispeech_other_source` stays discoverable for
    # evaluation / audition harnesses.
    assert types["babble"]["enabled"] is True
    assert types["babble"]["prob"] == 0.8
    assert types["babble"]["n_speakers_min"] == 1
    assert types["babble"]["n_speakers_max"] == 3
    assert len(types["babble"]["sources"]) == 1, (
        f"paper.yaml babble.sources should contain only musan(speech); "
        f"got {types['babble']['sources']!r}"
    )
    assert types["babble"]["sources"][0]["source"] == "musan"
    assert types["babble"]["sources"][0]["subset"] == "speech"
    # Babble synthesis happens at training time; the relevant knobs are
    # per-speaker segment length (1-4 s per paper.yaml).
    assert types["babble"]["babble_seg_min_sec"] == 1
    assert types["babble"]["babble_seg_max_sec"] == 4.0
    # `pool_size` no longer exists — was a pre-bake pool count, removed
    # when babble moved to runtime synthesis.
    assert "pool_size" not in types["babble"], (
        f"pool_size should be removed; got {types['babble']!r}"
    )
    # `weight` no longer exists — replaced by `prob` (independent
    # Bernoulli inclusion probability, no longer sums to 1).
    assert "weight" not in types["babble"], (
        f"`weight` field removed; got {types['babble']!r}"
    )
    # noise/music have their own independent `prob`s. librispeech(other)
    # moved from babble.sources to noise.sources so it gets baked + is
    # discoverable, but the runtime noise mixer still only consumes
    # pool/musan/noise/ (config-driven behaviour matches code).
    assert types["noise"]["prob"] == 0.5
    assert types["music"]["prob"] == 0.5
    assert len(types["noise"]["sources"]) == 2, (
        f"paper.yaml noise.sources should contain musan(noise) + "
        f"librispeech(other); got {types['noise']['sources']!r}"
    )
    assert types["noise"]["sources"][0]["source"] == "musan"
    assert types["noise"]["sources"][0]["subset"] == "noise"
    assert types["noise"]["sources"][1]["source"] == "librispeech"
    assert "subset" not in types["noise"]["sources"][1], (
        "librispeech refs must not carry `subset:` (use split names)"
    )
    assert types["music"]["sources"][0]["source"] == "musan"
    assert types["music"]["sources"][0]["subset"] == "music"

    # processed_dir is fully retired — no `output_dir` key in defaults.
    # Output is per-cache-dir `<cache_dir>/voicefilter-manifest.json`
    # alongside each HF arrow cache, not a single central `data/processed`.
    assert "output_dir" not in out, (
        f"processed_dir is retired; got output_dir={out.get('output_dir')!r}"
    )
    # sample_rate from model.audio
    assert out["sample_rate"] == 16000, out.get("sample_rate")

    # No VCTK leak anywhere.
    for k in out:
        assert "vctk" not in k.lower(), f"unexpected vctk key: {k}"
    print("[ok] paper.yaml yields function-oriented defaults")


def test_yaml_without_dataset_block_is_empty() -> None:
    cfg = {"model": {"audio": {"sample_rate": 16000}}}
    out = _yaml_to_prepare_defaults(cfg)
    assert out == {"sample_rate": 16000}, out
    print("[ok] model-only config yields only audio fields")


def test_explicit_splits_disable_unmentioned_train_flags() -> None:
    """If the user enumerates target.splits, train.{role}.100 must NOT default to True."""
    cfg = {
        "dataset": {
            "sources": [
                {"id": "librispeech", "repo": "openslr/librispeech_asr",
                 "configs": ["all"]},
            ],
            "target": {
                "sources": [
                    {"source": "librispeech", "splits": ["train.clean.500"]},
                ]
            },
            "interference": {
                "babble": {
                    "sources": [
                        {"source": "librispeech", "splits": ["train.other.500"]},
                    ]
                }
            },
        }
    }
    out = _yaml_to_prepare_defaults(cfg)
    assert out["train_clean_500"] is True
    assert "train_clean_100" not in out, (
        f"Enumerating only train_clean.500 should not enable train_clean.100; "
        f"got {out.get('train_clean_100')!r}"
    )
    print("[ok] enumerating splits disables unmentioned train_* flags")


def test_target_and_interference_libripeech_subsets_unioned() -> None:
    """librispeech_subsets is the union of every referenced LibriSpeech role tag."""
    cfg = {
        "dataset": {
            "sources": [
                {"id": "librispeech", "repo": "openslr/librispeech_asr",
                 "configs": ["all"]},
            ],
            "target": {
                "sources": [
                    {"source": "librispeech", "splits": ["train.clean.360"]},
                ],
            },
            "interference": {
                "babble": {
                    "sources": [
                        {"source": "librispeech", "splits": ["train.other.500"]},
                    ],
                },
            },
        }
    }
    out = _yaml_to_prepare_defaults(cfg)
    assert set(out["librispeech_subsets"]) == {"clean", "other"}
    print("[ok] target + interference LibriSpeech subsets are unioned")


def test_interference_type_subset_field_works() -> None:
    """A YAML that only sets some interference fields still resolves."""
    cfg = {
        "dataset": {
            "interference": {
                "babble": {"n_speakers_max": 7, "prob": 0.10},
                "noise": {"enabled": False},
            }
        }
    }
    out = _yaml_to_prepare_defaults(cfg)
    assert out["interference_types"]["babble"]["n_speakers_max"] == 7
    assert out["interference_types"]["babble"]["prob"] == 0.10
    assert out["interference_types"]["noise"]["enabled"] is False
    print("[ok] partial interference-type YAML is preserved verbatim")


if __name__ == "__main__":
    test_paper_yaml_yields_function_oriented_defaults()
    test_yaml_without_dataset_block_is_empty()
    test_explicit_splits_disable_unmentioned_train_flags()
    test_target_and_interference_libripeech_subsets_unioned()
    test_interference_type_subset_field_works()
    print("\n[done] all yaml-config tests passed")
