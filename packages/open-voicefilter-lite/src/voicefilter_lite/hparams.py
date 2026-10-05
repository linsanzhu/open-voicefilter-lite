"""Centralised hyperparameters for VoiceFilter-Lite.

Only one speaker encoder is supported: the paper §3.1 d-vector
(`SpeakerEncoder`, 3-layer LSTM × 768 → Linear → 256). 256 is the only
d-vector dimension in the codebase — no `hp.speaker_encoder` selection,
no factory dispatch.

Configuration is split into three blocks (`model:` / `dataset:` / `train:`)
plus two top-level HF-mirror fields. The flat property accessors
(`hp.batch_size`, `hp.hidden_dim`, ...) are kept as a backward-compat
shim so the rest of the codebase (`model.py`, `losses.py`, ...) can
keep reading `hp.<name>` while the dataclass hierarchy lives underneath.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Optional, get_type_hints as _get_type_hints


# d-vector output dim — matches `SpeakerEncoder` (paper §3.1, 256-d).
# Hardcoded here so the model's `in_proj` input shape is always 320+256.
D_VECTOR_DIM = 256


# ============================================================ #
# Model block                                                    #
# ============================================================ #


@dataclass
class AudioConfig:
    """Main-model STFT/mel parameters."""

    # Target sample rate (Hz). Shared with the d-vector pipeline
    # (model.dvector.*) — must match for consistent feature scales.
    sample_rate: int = 16000
    # FFT window size (samples). 400 @ 16 kHz = 25 ms.
    n_fft: int = 400
    # STFT hop (samples). 160 @ 16 kHz = 10 ms — one mel-frame per hop
    # (paper §3.2).
    hop_length: int = 160
    # Window length (samples). Equal to n_fft → no edge padding.
    win_length: int = 400
    # Number of mel filterbanks. 128 is the paper §4.2 main-experiment I/O
    # ("a 128-dim log Mel-filterbank energies, stacked-by-4 as inputs to
    # the model" — the 2.2 MB ship-target row in Table 2). The 513-dim FFT
    # magnitude feature variant mentioned in §4.2 is a separate ablation
    # (Table 1's "FFT magnitude" row); the ship-target 2.2 MB uses the
    # stacked 128-dim mel path. 80 was an ASR-convention shortcut.
    n_mels: int = 128
    # Floor added before log() to avoid log(0). 1e-6 ≈ -13.8 nats.
    log_mel_floor: float = 1e-6
    # Past frames concatenated with the current frame (paper §4.5).
    # 4 → 128×4=512 stacked-mel input + 256-d d-vector = input_dim 768.
    frame_stack: int = 4


@dataclass
class DVectorConfig:
    """VoiceFilter d-vector encoder (paper §3.1).

    All fields are tied to the bundled pretrained `models/embedder.pt`
    weights; changing them silently breaks weight loading. Exposed as
    config so the architecture is explicit, but the dataset never
    overwrites them.
    """

    # Path to the pretrained d-vector encoder state dict. Bundled at
    # models/embedder.pt.
    pt_path: str = "models/embedder.pt"
    # LSTM hidden width (matches upstream VoiceFilter / paper §3.1).
    lstm_hidden: int = 768
    # Number of stacked LSTM layers (paper §3.1).
    lstm_layers: int = 3
    # Projection dimension (paper §3.1, 256-d). Fixed at D_VECTOR_DIM;
    # any YAML override is silently clamped back to 256 by
    # __post_init__ / HParams.__init__.
    emb_dim: int = 256
    # Mel filterbanks for the d-vector pipeline (40, upstream default).
    n_mels: int = 40
    # FFT size for the d-vector STFT (512 @ 16 kHz = 32 ms).
    n_fft: int = 512
    # Window length for the d-vector STFT (25 ms @ 16 kHz).
    win_length: int = 400
    # Hop length for the d-vector STFT (10 ms @ 16 kHz).
    hop_length: int = 160
    # Sliding-window width (frames) for per-window encoder forward.
    window: int = 80
    # Sliding-window stride (frames). 80/40 = 50% overlap.
    stride: int = 40


@dataclass
class ArchConfig:
    """Mask-predictor LSTM + optional pre-LSTM CNN."""

    # LSTM hidden width. Slides p.28 lists "3 × LSTM 512" as the
    # main-architecture width (the model that, after INT8 quantization
    # + adaptive w^(t), lands at the 2.2 MB ship target). 256 is a
    # generic library default that older code paths may rely on; the
    # canonical training config (`configs/paper.yaml`) overrides this
    # to 512.
    hidden_dim: int = 256
    # Number of stacked LSTM layers. Slides p.28: 3 × LSTM.
    num_lstm_layers: int = 3
    # Dropout probability on LSTM input/output. 0.1 is a paper-style
    # default; raise to fight overfitting on small data.
    dropout: float = 0.1
    # Insert num_cnn_layers stacked Conv1d before the LSTM (paper
    # §3.2 Figure 2; slides p.17 Fig. 1b draws the 1D CNN explicitly).
    # The paper admits "most of our models actually remove these CNN
    # layers and purely consist of LSTM layers" — default off matches
    # that LSTM-only topology.
    use_cnn: bool = False
    # Number of Conv1d layers when use_cnn=True. Paper Figure 2: 2.
    num_cnn_layers: int = 2
    # Conv1d kernel size. 3 matches paper Figure 2.
    cnn_kernel_size: int = 3
    # Conv1d intermediate channel count. Last layer always projects to
    # hidden_dim so it feeds the LSTM directly.
    cnn_channels_first: int = 8


@dataclass
class PredictorConfig:
    """Noise-type predictor (paper §4.2 / §3.4).

    Per-frame binary classifier ("is this frame overlapped speech?").
    At inference, its sigmoid drives the adaptive suppression w^(t)
    per §3.4 — the paper's "always harmless and sometimes helpful"
    production behaviour.
    """

    # Train the hinge-loss predictor branch. Flip False for the
    # minimal ablation (no §4.2 branch; loss drops the hinge term).
    use_noise_type_predictor: bool = True
    # Paper §3.4: f_adapt reads the *current frame's* noisy filterbank
    # (80-d). False = legacy LSTM-hidden-state variant, kept for
    # backward compat with already-trained checkpoints.
    alpha_head_use_current_frame: bool = True


@dataclass
class LossConfig:
    """Asymmetric L2 + hinge (paper §3.3 + §4.2).

    The asymmetric L2 term is paper §3.3 Eq. 3 (literal Σ_t Σ_f sum,
    not per-element mean) with α=10; the hinge term is the
    noise-type predictor's loss (paper §4.2).
    """

    # g_asym(x, α) = α·x if x > 0 else x. Paper §3.3 α=10 → over-
    # suppression errors (S_enh < S_cln) amplified by α²=100.
    alpha: float = 10.0
    # α schedule start value (step 0). When `alpha_ramp_steps > 0` the
    # asymmetry factor ramps linearly from `alpha_start` at step 0 to
    # `alpha` at step `alpha_ramp_steps`, then holds at `alpha`. The
    # intent is to break the "always emit M≈1 to dodge the α² penalty"
    # trap the paper's α=10 form is prone to: starting at α=1.0 (pure
    # MSE, no over-suppression preference) lets the model first learn
    # which bins to discriminate, then the ramp turns on the over-
    # suppression penalty in the latter 90% of training to push the
    # model toward the paper's "err on the side of pass-through"
    # production behaviour. Set `alpha_ramp_steps=0` (default) to keep
    # the legacy constant-α form (full asymmetry from step 0).
    alpha_start: float = 1.0
    # Number of steps over which α ramps from `alpha_start` to `alpha`.
    # `0` (default) disables annealing and uses constant `alpha`
    # end-to-end (paper-faithful behaviour). Recommended: ~10% of
    # `total_steps` (e.g. 20 000 for `total_steps=200 000`).
    alpha_ramp_steps: int = 0
    # Weight on the noise-type predictor hinge loss. Paper §4.2
    # explicitly specifies hinge for the predictor's binary
    # classification. Calibrated against the Σ_t Σ_f asym term
    # (~T·F larger than legacy mean-asym); sample_length=3s, hop=10ms,
    # n_mels=128 → 1/(T·F) ≈ 3e-5.
    noise_type_loss_weight: float = 1e-5
    # Paper §4.4 last-row description (the 2.2 MB Table 2 entry):
    # "If we use adaptive suppression strength with a moving average
    # coefficient of 0.4, ... we observe almost no WER degradation on
    # clean and non-speech noise conditions, and still a large improvement
    # on speech noise conditions." β=0.4 is the value the paper uses for
    # its "always harmless, sometimes helpful" production setting.
    adaptive_w_beta: float = 0.4
    # Paper §3.4: linear transform slope "a > 0" applied to f_adapt.
    # a=0.7 pairs with b=0.2 below to map f_adapt ∈ [0,1] onto
    # w_raw ∈ [0.2, 0.9] — see configs/paper.yaml for the full
    # derivation (β=0.4 + always-on minimum compensation).
    adaptive_w_a: float = 0.7
    # Paper §3.4: linear transform intercept "b ≥ 0" applied to
    # f_adapt. b=0.2 sets w_min; see adaptive_w_a above.
    adaptive_w_b: float = 0.2


@dataclass
class EMAConfig:
    """Exponential moving average of model weights (paper best practice)."""

    # Enable EMA. Typically gives +0.5 dB SDR on validation; the
    # final ckpt saved by train.py uses EMA weights (see train.py).
    use: bool = True
    # Decay rate: shadow = decay · shadow + (1-decay) · model.
    # 0.999 ≈ 1000-step effective memory. Raise to 0.9999 for longer.
    decay: float = 0.999
    # Steps during which EMA is forced equal to model (no averaging).
    # Avoids a slow ramp-up bias at training start.
    warmup_steps: int = 100


@dataclass
class ModelConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    dvector: DVectorConfig = field(default_factory=DVectorConfig)
    arch: ArchConfig = field(default_factory=ArchConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)


# ============================================================ #
# Dataset block                                                  #
# ============================================================ #


@dataclass
class HfSourceSpec:
    """One upstream HF dataset declaration (top-level `dataset.sources[*]`).

    Both `target.sources[*]` and
    `interference.{babble,noise,music}.sources[*]` reference these by
    `id` via `SourceRef`. The indirection lets the same upstream dataset
    be referenced by multiple TYPEs without duplicating the HF repo,
    configs, splits, and cache_dir.

    `configs` is the list of HF `config` names this spec exposes. On HF
    datasets, a `config` is the top-level structural slice (e.g. MUSAN's
    `speech` / `noise` / `music`; LibriSpeech's `clean` / `other` /
    `all`). Listing them on the spec makes them visible to refs without
    repeating the HF repo:

      * LibriSpeech (`configs: ["all"]`): the `all` config exposes every
        `train.<role>.<size>` split under one arrow cache; the role tag
        (`clean` / `other`) lives in the split name. `SourceRef.subset`
        then picks `"clean"` / `"other"` for the per-role pool dir.
      * MUSAN (`configs: ["speech", "noise", "music"]`): each config is
        a separate HF config; `SourceRef.subset` picks the one to load.

    `cache_dir` is the on-disk HF arrow cache; empty string = resolve to
    the dataset-appropriate default at use time
    (`data/librispeech` for LibriSpeech, `data/musan` for MUSAN — values
    match `configs/paper.yaml`).
    """

    # Unique alias referenced by `SourceRef.source`. Must be unique within
    # `dataset.sources`.
    id: str
    # HF repo id, e.g. "openslr/librispeech_asr", "corypaik/musan".
    repo: str
    # HF `config` names exposed by this spec. Empty for sources without an
    # HF config layer (e.g. single-config datasets); `SourceRef.subset`
    # must pick one when this list has more than one entry.
    configs: tuple[str, ...] = ()
    # LibriSpeech splits this spec exposes (e.g. `train.clean.360`,
    # `train.other.500`). Empty for MUSAN (one `train` split per config).
    splits: tuple[str, ...] = ()
    # HF arrow cache for this source. Empty → resolve at use site.
    cache_dir: str = ""


@dataclass
class SourceRef:
    """A reference into `DatasetConfig.sources` (an `HfSourceSpec` by id).

    Used in both `target.sources[*]` and
    `interference.{babble,noise,music}.sources[*]`. The full upstream definition
    (HF repo, configs, splits, cache) lives on the referenced
    `HfSourceSpec`; this struct only carries the per-usage overrides:

      * `subset`  — narrows the ref to one entry of `HfSourceSpec.configs`
                     (the HF config name to load). Defaults to the only
                     entry when the spec has exactly one config; required
                     otherwise (e.g. MUSAN refs must pick `speech` /
                     `noise` / `music`).
      * `splits`  — overrides the spec's `splits` for this usage. For
                     LibriSpeech under the `all` config the role tag
                     (`clean` / `other`) is parsed from the split name
                     (`train.clean.360` etc.) — refs don't carry it
                     separately.
    """

    # `HfSourceSpec.id` to reference.
    source: str
    # One entry of `HfSourceSpec.configs` (i.e. an HF config name). None
    # is auto-filled by `resolve_source` when the spec has exactly one
    # config.
    subset: Optional[str] = None
    # Override the spec's `splits`. Empty tuple means "use the spec's
    # default".
    splits: tuple[str, ...] = ()


@dataclass
class DatasetTargetConfig:
    """Target clean-speech block. A list of `SourceRef`s pointing at
    `DatasetConfig.sources` entries — the dataset bakes each referenced
    upstream source and the iterator pools them together.
    """

    sources: tuple[SourceRef, ...] = field(
        default_factory=lambda: (
            SourceRef(source="librispeech", splits=("train.clean.360",)),
        )
    )


# ---- Interference (function-oriented) ----


@dataclass
class InterferenceTypeConfig:
    """One interference TYPE (`babble` / `noise` / `music`).

    Per-sample composition is **independent inclusion + sum**: each TYPE
    has its own `prob` (Bernoulli inclusion probability, ∈ [0, 1]) and
    is either synthesised and added into the sample's interference or
    skipped. Multiple TYPEs can be included in the same sample — their
    waveforms are summed. The three `prob`s are INDEPENDENT (no longer
    sum to 1), so any combination is possible: pure noise, pure music,
    noise+music, babble+noise, babble+music, all three, or even all
    three excluded (silent sample — combined with `silent_interference_prob`
    that already covers the "do-nothing" pass-through training).

    Each type:
      - has an independent `prob` (Bernoulli inclusion probability)
        controlling how often it is mixed into a sample's interference.
        0 disables the type even if `enabled` is True.
      - pulls clips from one or more `SourceRef`s pointing at
        `DatasetConfig.sources` entries.
      - synthesises its interference waveform at TRAINING time per type:
          * `babble`          : overlay N ∈ [n_speakers_min, n_speakers_max]
                                DISTINCT speakers drawn from MUSAN
                                `speech` only. Each speaker contributes a
                                segment of length ∈ [babble_seg_min_sec,
                                babble_seg_max_sec] placed at a random
                                offset within the sample, with per-speaker
                                gain drawn from [gain_min, gain_max].
                                `n_speakers_min=1` gives the
                                single-speaker case; `n_speakers_min≥2`
                                gives true multi-talker babble. The
                                librispeech `other` pool is NOT used for
                                babble at runtime (kept as a separate
                                "inter" pool for evaluation harnesses).
          * `noise` / `music` : pick 1 source clip from the matching MUSAN
                                pool, `random_crop` to sample_len.

    Mixing happens at training time (not at `prepare_dataset` time) so the
    baked dataset is a flat per-source pool under `pool/<spec.id>/...`
    and is reused across all types that reference it.
    """

    # Toggle for this type. Disabled types are skipped at bake time
    # and at runtime (equivalent to prob=0).
    enabled: bool = True
    # INDEPENDENT Bernoulli inclusion probability ∈ [0, 1] for this TYPE.
    # Per sample, `rng.random() < prob` decides whether the TYPE is mixed
    # into the sample's interference. The three TYPES' probs are
    # independent (no longer sum to 1) so any combination is possible.
    # 0 disables the type even if `enabled` is True.
    prob: float = 0.0
    # `SourceRef` list this TYPE pools over. Empty → TYPE skipped.
    # Multiple TYPEs may reference the same upstream `HfSourceSpec`;
    # `prepare_dataset` dedupes by spec so each upstream pool is baked once.
    sources: tuple[SourceRef, ...] = ()
    # ---- babble-only knobs (no-op for other types) ----
    # Lower bound on overlaid speaker count per babble clip. `1` makes
    # the picker fall back to single-speaker; `≥2` gives true babble.
    n_speakers_min: int = 1
    # Upper bound on overlaid speaker count per babble clip.
    n_speakers_max: int = 5
    # Per-speaker segment length lower bound (s). Each babble speaker
    # contributes a segment of length uniform in [min, max].
    babble_seg_min_sec: float = 0.5
    # Per-speaker segment length upper bound (s).
    babble_seg_max_sec: float = 2.0
    # Per-speaker gain lower bound (uniform per clip in [min, max]).
    gain_min: float = 0.3
    # Per-speaker gain upper bound.
    gain_max: float = 1.0


def _default_sources() -> tuple[HfSourceSpec, ...]:
    """Built-in default upstream HF dataset declarations.

    Paper-faithful defaults: LibriSpeech (read via the `all` config so a
    single arrow cache holds every `train.<role>.<size>` split) + MUSAN
    (all three configs). YAML may override / extend by setting
    `dataset.sources[*]` explicitly.
    """
    return (
        # One LibriSpeech spec covers both `clean` (target) and `other`
        # (interference speech). The `all` config exposes every split
        # under one arrow cache; the role tag lives in the split name
        # (`train.clean.<n>` / `train.other.<n>`) and is selected per-ref
        # via `SourceRef.subset`.
        HfSourceSpec(
            id="librispeech",
            repo="openslr/librispeech_asr",
            configs=("all",),
            splits=("train.clean.360", "train.other.500"),
            cache_dir="data/librispeech",
        ),
        HfSourceSpec(
            id="musan",
            repo="corypaik/musan",
            configs=("speech", "noise", "music"),
            cache_dir="data/musan",
        ),
    )


def _default_speech_sources() -> tuple[SourceRef, ...]:
    """Speech-pool refs for the `babble` TYPE — MUSAN `speech` only.

    Per design, babble at runtime uses ONLY MUSAN speech (treating each
    long monologue clip as one identity). The librispeech `other` pool
    is no longer mixed into babble at training time; it remains a
    separate "inter" pool consumed by evaluation / audition harnesses
    (`evaluate.py`, `examples/04_integration_mix_and_enhance.py`,
    `diag_mask.py`).
    """
    return (
        SourceRef(source="musan", subset="speech"),
    )


def _default_babble() -> InterferenceTypeConfig:
    """Built-in default for the `babble` TYPE.

    `n_speakers_min=1` covers both single-speaker (most common phone
    interference) and true multi-talker babble; raise to `≥2` for a
    babble-only deployment.

    `prob=0.5` is the INDEPENDENT inclusion probability per sample —
    half the samples will include a babble component in their
    interference mix (combined with noise/music probs).
    """
    return InterferenceTypeConfig(
        enabled=True, prob=0.5, sources=_default_speech_sources(),
        n_speakers_min=1, n_speakers_max=5,
        babble_seg_min_sec=0.5, babble_seg_max_sec=2.0,
        gain_min=0.3, gain_max=1.0,
    )


def _default_noise() -> InterferenceTypeConfig:
    """Built-in default for the `noise` TYPE (MUSAN `noise` subset).

    `prob=0.6` — noise is the most common real-world interference
    component, included in 60% of samples independently of babble/music.
    """
    return InterferenceTypeConfig(
        enabled=True, prob=0.6,
        sources=(SourceRef(source="musan", subset="noise"),),
    )


def _default_music() -> InterferenceTypeConfig:
    """Built-in default for the `music` TYPE (MUSAN `music` subset).

    `prob=0.3` — music is the least common real-world component;
    included in 30% of samples independently of babble/noise.
    """
    return InterferenceTypeConfig(
        enabled=True, prob=0.3,
        sources=(SourceRef(source="musan", subset="music"),),
    )


# ============================================================ #
# Resolution helpers                                            #
# ============================================================ #


@dataclass(frozen=True)
class ResolvedSource:
    """A `SourceRef` resolved against its `HfSourceSpec`.

    All downstream code (baking, dispatch, evaluation) reads from this
    instead of chasing `SourceRef → HfSourceSpec` itself. The `subset`
    and `splits` fields carry the effective (per-usage) values, not the
    raw `SourceRef` fields — `subset` defaults to the only entry in
    `HfSourceSpec.configs` when the ref didn't specify one, and `splits`
    falls back to `HfSourceSpec.splits`.

    Construct via `resolve_source(hp, ref)` (it validates the ref and
    raises `KeyError` on an unknown `id`).
    """

    spec: HfSourceSpec
    subset: Optional[str] = None
    splits: tuple[str, ...] = ()

    # Convenience accessors that just forward to `spec` so callers don't
    # have to write `resolved.spec.repo` everywhere.
    @property
    def repo(self) -> str:
        return self.spec.repo

    @property
    def cache_dir(self) -> str:
        return self.spec.cache_dir

    @property
    def id(self) -> str:
        return self.spec.id

    def is_librispeech(self) -> bool:
        """True when this spec is a LibriSpeech source."""
        return self.spec.repo == "openslr/librispeech_asr"

    def is_musan(self) -> bool:
        """True when this spec is a MUSAN source."""
        return self.spec.repo == "corypaik/musan"

    def librispeech_role(self) -> Optional[str]:
        """Extract the LibriSpeech role tag (`clean` / `other`) from the
        split name prefix (e.g. `train.clean.360` → `clean`).

        Returns None when this isn't a LibriSpeech source, or when no
        split carries a recognised role tag.
        """
        if not self.is_librispeech():
            return None
        for s in self.splits:
            parts = s.split(".")
            if (
                len(parts) == 3
                and parts[0] == "train"
                and parts[1] in ("clean", "other")
            ):
                return parts[1]
        return None


@dataclass
class DatasetInterferenceConfig:
    """Interference block. Three built-in TYPES, each a direct field:

      * `babble` — N speakers (n_speakers_min..n_speakers_max) per sample,
                   with `n_speakers_min=1` covering single-speaker too.
                   Drawn from MUSAN `speech` only at runtime.
      * `noise`  — MUSAN `noise` subset, passthrough with `random_crop`.
      * `music`  — MUSAN `music` subset, passthrough with `random_crop`.

    Each TYPE has its own INDEPENDENT inclusion `prob` (Bernoulli ∈ [0, 1],
    no longer summing to 1 across types) and `sources` list of `SourceRef`s.
    Per sample the dataset rolls each TYPE's prob independently and sums
    the included components into one interference waveform. Mixing
    happens at TRAINING time; `prepare_dataset` only decodes the upstream
    pools and the iterator mixes per sample.
    """

    babble: InterferenceTypeConfig = field(default_factory=_default_babble)
    noise: InterferenceTypeConfig = field(default_factory=_default_noise)
    music: InterferenceTypeConfig = field(default_factory=_default_music)

    def entries(self) -> dict[str, InterferenceTypeConfig]:
        """`name → InterferenceTypeConfig` view for code that iterates
        all TYPES (weighted sampling, pool dedup, manifests)."""
        return {"babble": self.babble, "noise": self.noise, "music": self.music}


@dataclass
class DatasetMixtureConfig:
    """SNR / silence / segment-length knobs for the mixture stage."""

    # Training segment length (s). Paper Fig 2: 3.0 s. Halve to
    # 1.5-2.0 s for memory-constrained configs (smaller T → smaller
    # per-sample mel + stacked tensors).
    sample_length_sec: float = 3.0
    # Lower bound on training SNR (dB). Paper §3.2 / slides p.29: 1 dB.
    snr_min_db: float = 1.0
    # Upper bound on training SNR (dB). Paper §3.2: 10 dB. SNR is
    # drawn uniformly from [snr_min, snr_max] each sample.
    snr_max_db: float = 10.0
    # Probability that the target wav is replaced by silence in a
    # sample. Trains the "do nothing" pass-through mode (paper §3.2).
    silent_target_prob: float = 0.10
    # Probability that the interference wav is replaced by silence.
    # Trains the silent-side branches of the noise-type predictor.
    silent_interference_prob: float = 0.10


@dataclass
class DatasetActivityConfig:
    """Per-frame activity burst parameters (paper §4.2 / deviation 5)."""

    # Lower bound on burst duration (s). Bursts shorter than this
    # would not contribute a full STFT window's worth of frames.
    burst_min_sec: float = 0.5
    # Upper bound on burst duration (s). Capped at sample_length_sec
    # inside _sample_activity_burst; reduce proportionally when
    # shrinking sample_length_sec.
    burst_max_sec: float = 3.0
    # Fade-in / fade-out length (ms) at burst edges. Avoids click
    # artefacts that would leak into the noise-type predictor's
    # per-frame labels.
    fade_ms: float = 50.0


@dataclass
class DatasetReferenceConfig:
    """Reference utterance caps (used to build the d-vector input)."""

    # Maximum reference-wav length (s). Caps the d-vector pipeline at
    # ref_max_sec × sample_rate samples, so collate_triplets pads to
    # a bounded max instead of the longest batch utterance (some
    # LibriSpeech speakers are ~30 s — caps prevent OOM).
    max_sec: float = 6.0
    # Reserved for future streaming buffers; not currently used by
    # LocalTripletStream (d-vectors are pre-baked).
    buffer_max_per_spk: int = 4
    # Reserved for future streaming buffers; not currently used by
    # LocalTripletStream (d-vectors are pre-baked).
    buffer_max_speakers: int = 512


@dataclass
class DatasetConfig:
    """Top-level dataset config — composes `sources` (the upstream HF
    dataset declarations) plus `target` / `interference` / `mixture` /
    `activity` / `reference` sub-blocks and the single shared output
    directory.

    Layout:

      * `sources`            — top-level list of upstream HF dataset
                               declarations (`HfSourceSpec`). Both
                               `target.sources[*]` and
                               `interference.{babble,noise,music}.sources[*]`
                               reference these by `id` via `SourceRef`.
      * `target.sources[*]`   — `SourceRef`s pointing at the source(s)
                               used as target clean speech.
      * `interference`        — `DatasetInterferenceConfig` with direct
                               `babble` / `noise` / `music` fields, each
                               pooling over a list of `SourceRef`s.

    Per-source HF arrow caches live on each `HfSourceSpec.cache_dir` —
    no separate `dataset.paths` block. The only shared path is
    `processed_dir` (the output of `voicefilter-cli dataset prepare`).
    """

    # Top-level upstream HF dataset declarations; both target.sources
    # and interference.{babble,noise,music}.sources[*] reference these
    # by `id` via `SourceRef`.
    sources: tuple[HfSourceSpec, ...] = field(default_factory=_default_sources)
    # Target clean-speech source (VCTK or LibriSpeech clean).
    target: DatasetTargetConfig = field(default_factory=DatasetTargetConfig)
    # Non-target speaker interference (typically LibriSpeech `other`).
    interference: DatasetInterferenceConfig = field(
        default_factory=DatasetInterferenceConfig,
    )
    # Per-sample SNR, silence, and segment-length knobs.
    mixture: DatasetMixtureConfig = field(default_factory=DatasetMixtureConfig)
    # Per-frame activity-burst window for the noise-type predictor.
    activity: DatasetActivityConfig = field(default_factory=DatasetActivityConfig)
    # Reference-utterance length caps (d-vector pipeline input).
    reference: DatasetReferenceConfig = field(
        default_factory=DatasetReferenceConfig,
    )
    # Output directory of `voicefilter-cli dataset prepare`. Contains
    # the target/ interference/ dvectors/ subtrees + manifest.json.
    # This is the ONLY shared output path; per-source HF arrow caches
    # live on each source spec's `cache_dir` instead.
    processed_dir: str = "data/processed"

    # Background-prefetch queue depth on the main thread. Higher
    # values keep the GPU fed at the cost of more CPU/MPS memory;
    # LocalTripletStream reads local wavs (~5 ms/sample) so depth 1-2
    # is usually enough.
    prefetch_batches_max: int = 2
    # Vestigial: legacy thread-pool size for parallel FLAC decode.
    # LocalTripletStream does not use a decode pool (local wavs only),
    # so this field is ignored. Kept so old YAMLs don't error.
    decode_workers: int = 0
    # Vestigial: legacy per-thread decode-pipeline depth. Ignored;
    # kept for backward YAML compatibility.
    decode_prefetch: int = 0


# ============================================================ #
# Train block                                                    #
# ============================================================ #


@dataclass
class TrainConfig:
    """Optimisation + I/O cadence knobs for the training loop."""

    # Mini-batch size. Paper-faithful 32 for CUDA; halve on MPS to
    # fit RAM (16 on M-Pro, 8 on base M5).
    batch_size: int = 6
    # Adam learning rate. Paper convention at batch=32: 2e-3; linear-
    # scale when halving batch_size (lr ∝ batch).
    learning_rate: float = 1e-3
    # Adam weight_decay (L2 regularisation on params). 1e-5 is a
    # paper-style small value that does not dominate Adam.
    weight_decay: float = 1e-5
    # Total optimisation steps. Paper: 200 000; halve for MPS.
    total_steps: int = 50_000
    # Linear warmup duration before cosine decay starts.
    warmup_steps: int = 1_000
    # Max global gradient norm (clip_grad_norm_).
    grad_clip_norm: float = 5.0
    # Print training heartbeat every N steps.
    log_every: int = 5
    # Run evaluation every N steps (0 = disabled).
    eval_every: int = 1_000
    # Save a checkpoint every N steps. Last 3 ckpts are kept; older
    # ones are auto-deleted to bound disk usage.
    ckpt_every: int = 1_000
    # Call torch.{mps,cuda}.empty_cache() every N steps to release
    # the allocator's high-water-mark back to the OS. 0 = disabled.
    empty_cache_every: int = 2000
    # PRNG seed for dataset / model init / training.
    seed: int = 1234
    # Number of background DataLoader worker processes. Each runs
    # an independent copy of LocalTripletStream with its own RNG
    # seed (seed + worker_id). 0 = single-process (safest on MPS).
    dataloader_workers: int = 0


# ============================================================ #
# Top-level HParams (composes the three blocks)                  #
# ============================================================ #


@dataclass
class HParams:
    """Top-level hyperparameter container.

    Composes three blocks (`model` / `dataset` / `train`) plus two
    top-level HF-mirror fields. The flat property accessors
    (`hp.batch_size`, `hp.hidden_dim`, ...) are kept as a backward-compat
    shim so the rest of the codebase (`model.py`, `losses.py`, ...)
    can keep reading `hp.<name>` while the dataclass hierarchy lives
    underneath.
    """

    # Model block — audio + dvector + arch + predictor + loss + ema.
    model: ModelConfig = field(default_factory=ModelConfig)
    # Dataset block — target/interference/mixture/activity/reference/
    # paths + vestigial decode knobs.
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    # Train block — optimisation + I/O cadence.
    train: TrainConfig = field(default_factory=TrainConfig)
    # HF Hub base URL (e.g. https://hf-mirror.com). Overrides
    # HF_ENDPOINT env var when set; CLI flag --hf-endpoint wins over
    # both. Applied via `apply_hf_endpoint` at startup.
    hf_endpoint: Optional[str] = None
    # Whether to opt into hf_transfer (faster on HF.co; breaks on
    # most mirrors). None = auto (off when an endpoint is chosen).
    # CLI flag --hf-transfer / --no-hf-transfer wins over this.
    hf_transfer: Optional[bool] = None

    # Tolerant `__init__`: legacy callers (and the smoke test) pass
    # flat fields like `use_cnn=True` / `cnn_kernel_size=3` / `streaming=False`
    # / `d_vector_dim=192` even though those are now property shims or
    # nested-dataclass fields. The auto-generated dataclass `__init__`
    # rejects unknown kwargs, so we wrap it to route extras through
    # `setattr` (which honours the property shims) and silently drop
    # truly unknown keys.
    def __init__(
        self,
        model: ModelConfig | None = None,
        dataset: DatasetConfig | None = None,
        train: TrainConfig | None = None,
        hf_endpoint: Optional[str] = None,
        hf_transfer: Optional[bool] = None,
        **extras: Any,
    ) -> None:
        if model is None:
            model = ModelConfig()
        if dataset is None:
            dataset = DatasetConfig()
        if train is None:
            train = TrainConfig()
        self.model = model
        self.dataset = dataset
        self.train = train
        self.hf_endpoint = hf_endpoint
        self.hf_transfer = hf_transfer
        # Route extras through `setattr`. Property shims (`use_cnn`,
        # `cnn_kernel_size`, etc.) write to the nested dataclass fields.
        # `streaming` and `d_vector_dim` are read-only / no-op properties
        # whose setters silently drop the write.
        for k, v in extras.items():
            try:
                setattr(self, k, v)
            except AttributeError:
                pass  # truly unknown — silently drop, matches old behaviour

    # ---- Derived (no derivation needed — d_vector_dim is fixed) ----
    def __post_init__(self) -> None:
        # Re-affirm the fixed d-vector dim even if a YAML overrides
        # the nested field to a stale value.
        self.model.dvector.emb_dim = D_VECTOR_DIM

    # ---- Flat @property shims (backward-compat for hp.<name> reads) ----
    # These let the rest of the codebase keep reading `hp.batch_size`,
    # `hp.hidden_dim`, etc. while the actual data lives in
    # `hp.train.batch_size`, `hp.model.arch.hidden_dim`, ...
    # New code should read from the nested form; the flat accessors are
    # kept so existing modules (`model.py`, `losses.py`, `audio.py`,
    # `streaming.py`) work without a sweeping rename.
    #
    # Pattern: each property is a pair of (getter, setter) that delegates
    # to the nested dataclass. Setter exists so CLI flag overrides
    # (`hp.batch_size = 32`) and YAML `apply_overrides` (`setattr`) keep
    # working.

    # Audio / d-vector
    @property
    def sample_rate(self) -> int: return self.model.audio.sample_rate
    @sample_rate.setter
    def sample_rate(self, v: int) -> None: self.model.audio.sample_rate = v

    @property
    def n_fft(self) -> int: return self.model.audio.n_fft
    @n_fft.setter
    def n_fft(self, v: int) -> None: self.model.audio.n_fft = v

    @property
    def hop_length(self) -> int: return self.model.audio.hop_length
    @hop_length.setter
    def hop_length(self, v: int) -> None: self.model.audio.hop_length = v

    @property
    def win_length(self) -> int: return self.model.audio.win_length
    @win_length.setter
    def win_length(self, v: int) -> None: self.model.audio.win_length = v

    @property
    def n_mels(self) -> int: return self.model.audio.n_mels
    @n_mels.setter
    def n_mels(self, v: int) -> None: self.model.audio.n_mels = v

    @property
    def log_mel_floor(self) -> float: return self.model.audio.log_mel_floor
    @log_mel_floor.setter
    def log_mel_floor(self, v: float) -> None: self.model.audio.log_mel_floor = v

    @property
    def frame_stack(self) -> int: return self.model.audio.frame_stack
    @frame_stack.setter
    def frame_stack(self, v: int) -> None: self.model.audio.frame_stack = v

    # d-vector
    @property
    def dvector_pt_path(self) -> str: return self.model.dvector.pt_path
    @dvector_pt_path.setter
    def dvector_pt_path(self, v: str) -> None: self.model.dvector.pt_path = v

    @property
    def dvector_lstm_hidden(self) -> int: return self.model.dvector.lstm_hidden
    @dvector_lstm_hidden.setter
    def dvector_lstm_hidden(self, v: int) -> None: self.model.dvector.lstm_hidden = v

    @property
    def dvector_lstm_layers(self) -> int: return self.model.dvector.lstm_layers
    @dvector_lstm_layers.setter
    def dvector_lstm_layers(self, v: int) -> None: self.model.dvector.lstm_layers = v

    @property
    def dvector_emb_dim(self) -> int: return self.model.dvector.emb_dim
    @dvector_emb_dim.setter
    def dvector_emb_dim(self, v: int) -> None: self.model.dvector.emb_dim = v

    @property
    def dvector_n_mels(self) -> int: return self.model.dvector.n_mels
    @dvector_n_mels.setter
    def dvector_n_mels(self, v: int) -> None: self.model.dvector.n_mels = v

    @property
    def dvector_n_fft(self) -> int: return self.model.dvector.n_fft
    @dvector_n_fft.setter
    def dvector_n_fft(self, v: int) -> None: self.model.dvector.n_fft = v

    @property
    def dvector_win_length(self) -> int: return self.model.dvector.win_length
    @dvector_win_length.setter
    def dvector_win_length(self, v: int) -> None: self.model.dvector.win_length = v

    @property
    def dvector_hop_length(self) -> int: return self.model.dvector.hop_length
    @dvector_hop_length.setter
    def dvector_hop_length(self, v: int) -> None: self.model.dvector.hop_length = v

    @property
    def dvector_window(self) -> int: return self.model.dvector.window
    @dvector_window.setter
    def dvector_window(self, v: int) -> None: self.model.dvector.window = v

    @property
    def dvector_stride(self) -> int: return self.model.dvector.stride
    @dvector_stride.setter
    def dvector_stride(self, v: int) -> None: self.model.dvector.stride = v

    @property
    def d_vector_dim(self) -> int: return D_VECTOR_DIM
    @d_vector_dim.setter
    def d_vector_dim(self, v: int) -> None:
        # Always reset to fixed dim; ignore the write.
        pass

    # Architecture
    @property
    def hidden_dim(self) -> int: return self.model.arch.hidden_dim
    @hidden_dim.setter
    def hidden_dim(self, v: int) -> None: self.model.arch.hidden_dim = v

    @property
    def num_lstm_layers(self) -> int: return self.model.arch.num_lstm_layers
    @num_lstm_layers.setter
    def num_lstm_layers(self, v: int) -> None: self.model.arch.num_lstm_layers = v

    @property
    def dropout(self) -> float: return self.model.arch.dropout
    @dropout.setter
    def dropout(self, v: float) -> None: self.model.arch.dropout = v

    @property
    def use_cnn(self) -> bool: return self.model.arch.use_cnn
    @use_cnn.setter
    def use_cnn(self, v: bool) -> None: self.model.arch.use_cnn = v

    @property
    def num_cnn_layers(self) -> int: return self.model.arch.num_cnn_layers
    @num_cnn_layers.setter
    def num_cnn_layers(self, v: int) -> None: self.model.arch.num_cnn_layers = v

    @property
    def cnn_kernel_size(self) -> int: return self.model.arch.cnn_kernel_size
    @cnn_kernel_size.setter
    def cnn_kernel_size(self, v: int) -> None: self.model.arch.cnn_kernel_size = v

    @property
    def cnn_channels_first(self) -> int: return self.model.arch.cnn_channels_first
    @cnn_channels_first.setter
    def cnn_channels_first(self, v: int) -> None: self.model.arch.cnn_channels_first = v

    # Predictor
    @property
    def use_noise_type_predictor(self) -> bool:
        return self.model.predictor.use_noise_type_predictor
    @use_noise_type_predictor.setter
    def use_noise_type_predictor(self, v: bool) -> None:
        self.model.predictor.use_noise_type_predictor = v

    @property
    def alpha_head_use_current_frame(self) -> bool:
        return self.model.predictor.alpha_head_use_current_frame
    @alpha_head_use_current_frame.setter
    def alpha_head_use_current_frame(self, v: bool) -> None:
        self.model.predictor.alpha_head_use_current_frame = v

    # Loss
    @property
    def loss_alpha(self) -> float: return self.model.loss.alpha
    @loss_alpha.setter
    def loss_alpha(self, v: float) -> None: self.model.loss.alpha = v

    @property
    def loss_alpha_start(self) -> float: return self.model.loss.alpha_start
    @loss_alpha_start.setter
    def loss_alpha_start(self, v: float) -> None: self.model.loss.alpha_start = v

    @property
    def loss_alpha_ramp_steps(self) -> int: return self.model.loss.alpha_ramp_steps
    @loss_alpha_ramp_steps.setter
    def loss_alpha_ramp_steps(self, v: int) -> None: self.model.loss.alpha_ramp_steps = v

    @property
    def noise_type_loss_weight(self) -> float: return self.model.loss.noise_type_loss_weight
    @noise_type_loss_weight.setter
    def noise_type_loss_weight(self, v: float) -> None: self.model.loss.noise_type_loss_weight = v

    @property
    def adaptive_w_beta(self) -> float: return self.model.loss.adaptive_w_beta
    @adaptive_w_beta.setter
    def adaptive_w_beta(self, v: float) -> None: self.model.loss.adaptive_w_beta = v

    @property
    def adaptive_w_a(self) -> float: return self.model.loss.adaptive_w_a
    @adaptive_w_a.setter
    def adaptive_w_a(self, v: float) -> None: self.model.loss.adaptive_w_a = v

    @property
    def adaptive_w_b(self) -> float: return self.model.loss.adaptive_w_b
    @adaptive_w_b.setter
    def adaptive_w_b(self, v: float) -> None: self.model.loss.adaptive_w_b = v

    @property
    def adaptive_w_b(self) -> float: return self.model.loss.adaptive_w_b
    @adaptive_w_b.setter
    def adaptive_w_b(self, v: float) -> None: self.model.loss.adaptive_w_b = v

    # EMA
    @property
    def use_ema(self) -> bool: return self.model.ema.use
    @use_ema.setter
    def use_ema(self, v: bool) -> None: self.model.ema.use = v

    @property
    def ema_decay(self) -> float: return self.model.ema.decay
    @ema_decay.setter
    def ema_decay(self, v: float) -> None: self.model.ema.decay = v

    @property
    def ema_warmup_steps(self) -> int: return self.model.ema.warmup_steps
    @ema_warmup_steps.setter
    def ema_warmup_steps(self, v: int) -> None: self.model.ema.warmup_steps = v

    # Dataset.target / interference (function-oriented)
    @property
    def target_sources(self) -> tuple[SourceRef, ...]:
        return self.dataset.target.sources
    @target_sources.setter
    def target_sources(self, v: tuple[SourceRef, ...]) -> None:
        self.dataset.target.sources = v

    @property
    def interference_types(self) -> dict[str, InterferenceTypeConfig]:
        return self.dataset.interference.entries()
    @interference_types.setter
    def interference_types(self, v: dict[str, InterferenceTypeConfig]) -> None:
        # Replace each known TYPE field from the dict; unknown keys are
        # ignored (the dataclass no longer accepts custom TYPE names).
        for name in ("babble", "noise", "music"):
            if name in v:
                setattr(self.dataset.interference, name, v[name])

    # Dataset.sources (top-level upstream HF dataset declarations)
    @property
    def dataset_sources(self) -> tuple[HfSourceSpec, ...]:
        return self.dataset.sources
    @dataset_sources.setter
    def dataset_sources(self, v: tuple[HfSourceSpec, ...]) -> None:
        self.dataset.sources = v

    @property
    def sources_by_id(self) -> dict[str, HfSourceSpec]:
        """`HfSourceSpec.id → HfSourceSpec` map (built every call; cheap)."""
        return {s.id: s for s in self.dataset.sources}

    # Dataset.mixture
    @property
    def sample_length_sec(self) -> float: return self.dataset.mixture.sample_length_sec
    @sample_length_sec.setter
    def sample_length_sec(self, v: float) -> None: self.dataset.mixture.sample_length_sec = v

    @property
    def snr_min_db(self) -> float: return self.dataset.mixture.snr_min_db
    @snr_min_db.setter
    def snr_min_db(self, v: float) -> None: self.dataset.mixture.snr_min_db = v

    @property
    def snr_max_db(self) -> float: return self.dataset.mixture.snr_max_db
    @snr_max_db.setter
    def snr_max_db(self, v: float) -> None: self.dataset.mixture.snr_max_db = v

    @property
    def silent_target_prob(self) -> float:
        return self.dataset.mixture.silent_target_prob
    @silent_target_prob.setter
    def silent_target_prob(self, v: float) -> None:
        self.dataset.mixture.silent_target_prob = v

    @property
    def silent_interference_prob(self) -> float:
        return self.dataset.mixture.silent_interference_prob
    @silent_interference_prob.setter
    def silent_interference_prob(self, v: float) -> None:
        self.dataset.mixture.silent_interference_prob = v

    # Dataset.activity
    @property
    def activity_burst_min_sec(self) -> float:
        return self.dataset.activity.burst_min_sec
    @activity_burst_min_sec.setter
    def activity_burst_min_sec(self, v: float) -> None:
        self.dataset.activity.burst_min_sec = v

    @property
    def activity_burst_max_sec(self) -> float:
        return self.dataset.activity.burst_max_sec
    @activity_burst_max_sec.setter
    def activity_burst_max_sec(self, v: float) -> None:
        self.dataset.activity.burst_max_sec = v

    @property
    def activity_fade_ms(self) -> float: return self.dataset.activity.fade_ms
    @activity_fade_ms.setter
    def activity_fade_ms(self, v: float) -> None: self.dataset.activity.fade_ms = v

    # Dataset.reference
    @property
    def ref_max_sec(self) -> float: return self.dataset.reference.max_sec
    @ref_max_sec.setter
    def ref_max_sec(self, v: float) -> None: self.dataset.reference.max_sec = v

    @property
    def ref_buffer_max_per_spk(self) -> int:
        return self.dataset.reference.buffer_max_per_spk
    @ref_buffer_max_per_spk.setter
    def ref_buffer_max_per_spk(self, v: int) -> None:
        self.dataset.reference.buffer_max_per_spk = v

    @property
    def ref_buffer_max_speakers(self) -> int:
        return self.dataset.reference.buffer_max_speakers
    @ref_buffer_max_speakers.setter
    def ref_buffer_max_speakers(self, v: int) -> None:
        self.dataset.reference.buffer_max_speakers = v

    # Dataset.processed_dir (output of `voicefilter-cli dataset prepare`)
    @property
    def processed_dir(self) -> str: return self.dataset.processed_dir
    @processed_dir.setter
    def processed_dir(self, v: str) -> None: self.dataset.processed_dir = v

    # Train
    @property
    def batch_size(self) -> int: return self.train.batch_size
    @batch_size.setter
    def batch_size(self, v: int) -> None: self.train.batch_size = v

    @property
    def learning_rate(self) -> float: return self.train.learning_rate
    @learning_rate.setter
    def learning_rate(self, v: float) -> None: self.train.learning_rate = v

    @property
    def weight_decay(self) -> float: return self.train.weight_decay
    @weight_decay.setter
    def weight_decay(self, v: float) -> None: self.train.weight_decay = v

    @property
    def total_steps(self) -> int: return self.train.total_steps
    @total_steps.setter
    def total_steps(self, v: int) -> None: self.train.total_steps = v

    @property
    def warmup_steps(self) -> int: return self.train.warmup_steps
    @warmup_steps.setter
    def warmup_steps(self, v: int) -> None: self.train.warmup_steps = v

    @property
    def grad_clip_norm(self) -> float: return self.train.grad_clip_norm
    @grad_clip_norm.setter
    def grad_clip_norm(self, v: float) -> None: self.train.grad_clip_norm = v

    @property
    def log_every(self) -> int: return self.train.log_every
    @log_every.setter
    def log_every(self, v: int) -> None: self.train.log_every = v

    @property
    def eval_every(self) -> int: return self.train.eval_every
    @eval_every.setter
    def eval_every(self, v: int) -> None: self.train.eval_every = v

    @property
    def ckpt_every(self) -> int: return self.train.ckpt_every
    @ckpt_every.setter
    def ckpt_every(self, v: int) -> None: self.train.ckpt_every = v

    @property
    def empty_cache_every(self) -> int: return self.train.empty_cache_every
    @empty_cache_every.setter
    def empty_cache_every(self, v: int) -> None: self.train.empty_cache_every = v

    @property
    def seed(self) -> int: return self.train.seed
    @seed.setter
    def seed(self, v: int) -> None: self.train.seed = v

    @property
    def dataloader_workers(self) -> int: return self.train.dataloader_workers
    @dataloader_workers.setter
    def dataloader_workers(self, v: int) -> None: self.train.dataloader_workers = v

    @property
    def streaming(self) -> bool:
        """Streaming mode is permanently removed. Always False."""
        return False

    @property
    def decode_workers(self) -> int: return self.dataset.decode_workers
    @decode_workers.setter
    def decode_workers(self, v: int) -> None: self.dataset.decode_workers = v

    @property
    def decode_prefetch(self) -> int: return self.dataset.decode_prefetch
    @decode_prefetch.setter
    def decode_prefetch(self, v: int) -> None: self.dataset.decode_prefetch = v

    @property
    def prefetch_batches_max(self) -> int: return self.dataset.prefetch_batches_max
    @prefetch_batches_max.setter
    def prefetch_batches_max(self, v: int) -> None: self.dataset.prefetch_batches_max = v

    # ---- Derived properties ----
    @property
    def input_dim(self) -> int:
        return self.n_mels * self.frame_stack + self.d_vector_dim

    @property
    def sample_length_samples(self) -> int:
        return int(self.sample_length_sec * self.sample_rate)

    @property
    def num_frames(self) -> int:
        # frames = 1 + (n - n_fft) // hop
        return 1 + (self.sample_length_samples - self.n_fft) // self.hop_length

    # ---- Serialisation ----
    def to_dict(self) -> dict[str, Any]:
        """Nested dict representation.

        New code should use this. The flat `from_dict` shim handles
        legacy `hparams.json` files written before the restructure.
        """
        return _to_dict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HParams":
        """Build an HParams from either nested or flat key/value dicts.

        Nested form: `{"model": {"audio": {"sample_rate": 16000, ...}}, ...}`
        Flat form (legacy hparams.json): `{"sample_rate": 16000, ...}`
        Flat keys with a known `block.field` prefix (e.g.
        `model.audio.sample_rate`) are routed to the nested field.

        The new `dataset.sources[*].cache_dir` schema has no flat-form
        cache keys — there's nothing to migrate from `dataset.paths.*`
        anymore. Old `hparams.json` checkpoints written before the
        `HfSourceSpec` / `SourceRef` refactor will fail to load
        (re-train to refresh).
        """
        if not d:
            return cls()
        nested, _legacy_cache_overrides = _coerce_to_nested(d, cls)
        return _from_dict(cls, nested)


# ============================================================ #
# (De)serialisation helpers                                       #
# ============================================================ #


def _to_dict(obj: Any) -> Any:
    """Recursive dataclass → dict. Tuples become lists for JSON."""
    if is_dataclass(obj):
        return {k: _to_dict(v) for k, v in ((f.name, getattr(obj, f.name)) for f in fields(obj))}
    if isinstance(obj, tuple):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, list):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _from_dict(cls: type, d: dict[str, Any]) -> Any:
    """Recursive dict → dataclass. Unknown keys are silently dropped.

    For nested dataclass fields we look up the actual sub-dataclass type
    via `_NESTED_TYPES` (registered at import time), since
    `fld.type` is unreliable when annotations involve string forward refs
    or PEP 563 deferred evaluation. Tuple-typed fields are coerced from
    JSON lists back to tuples (JSON has no tuple type). For fields
    declared as `dict[str, X]` or `tuple[X, ...]` where X is a dataclass,
    each element is recursively converted (see
    `_CONTAINER_ELEMENT_TYPES`).
    """
    nested_specs = _NESTED_TYPES.get(cls, {})
    cont_specs = _CONTAINER_ELEMENT_TYPES.get(cls, {})
    valid_keys = {f.name for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for k, v in d.items():
        if k not in valid_keys:
            continue
        inner_cls = nested_specs.get(k)
        if inner_cls is not None and isinstance(v, dict):
            kwargs[k] = _from_dict(inner_cls, v)
            continue
        cont_cls = cont_specs.get(k)
        if cont_cls is not None:
            if isinstance(v, dict):
                kwargs[k] = {
                    key: _from_dict(cont_cls, val)
                    for key, val in v.items()
                }
                continue
            if isinstance(v, list):
                kwargs[k] = tuple(
                    _from_dict(cont_cls, item) if isinstance(item, dict) else item
                    for item in v
                )
                continue
        kwargs[k] = v
    # Tuple annotations without element dataclass: dataclass doesn't
    # auto-convert list → tuple on assignment, so re-wrap lists for any
    # `tuple[...]` field. Detected via the field's annotation.
    try:
        hints = _get_type_hints(cls)
    except Exception:
        hints = {}
    for fname, val in list(kwargs.items()):
        if fname in cont_specs:
            continue  # already converted above
        hint = hints.get(fname)
        if hint is not None and _is_tuple_annotation(hint) and isinstance(val, list):
            kwargs[fname] = tuple(val)
    return cls(**kwargs)


def _is_tuple_annotation(hint: Any) -> bool:
    """True if `hint` is `tuple[...]` / `Tuple[...]` (any arity)."""
    import typing
    origin = getattr(hint, "__origin__", None)
    if origin is tuple:
        return True
    if isinstance(hint, str):
        return hint.startswith("tuple[") or hint.startswith("Tuple[")
    return False


# Sub-dataclass registry: maps each parent dataclass to {field_name: sub_cls}.
# Built once at import time below the sub-class definitions.
_NESTED_TYPES: dict[type, dict[str, type]] = {}
# Per-class registry of container-typed fields whose elements are dataclasses:
# `_from_dict` walks each element of the value and converts it from dict to
# the dataclass so the in-memory representation matches the annotation.
#   - `dict[str, X]` (e.g. `types: dict[str, InterferenceTypeConfig]`)
#   - `tuple[X, ...]` (e.g. `sources: tuple[InterferenceSourceConfig, ...]`)
_CONTAINER_ELEMENT_TYPES: dict[type, dict[str, type]] = {}


def _register_nested_types() -> None:
    """Populate `_NESTED_TYPES` for every dataclass with sub-dataclass fields.

    Called at module bottom so all sub-classes exist by the time we run.
    """
    from voicefilter_lite import hparams as _self  # noqa: F401

    def _resolve(tp: Any) -> Any:
        if isinstance(tp, str):
            return globals().get(tp) or getattr(_self, tp, None)
        return tp

    def _extract_container_element_type(tp: Any) -> Optional[type]:
        """Pull the element type out of `dict[K, V]` or `tuple[X, ...]`.

        Returns the element dataclass type (V for dict, X for tuple), or
        None if `tp` isn't a container-of-dataclass. Strings (PEP 563)
        are not introspectable here — caller must resolve via
        `typing.get_type_hints`.
        """
        if isinstance(tp, str):
            return None
        origin = getattr(tp, "__origin__", None)
        if origin is None:
            return None
        args = getattr(tp, "__args__", ())
        if origin is dict and len(args) == 2:
            return _resolve(args[1])
        if origin is tuple and len(args) >= 1:
            # `tuple[X, ...]` → args == (X, Ellipsis); `tuple[X, Y]` → (X, Y).
            # We treat any homogeneous dataclass-element tuple as a candidate.
            elem = _resolve(args[0])
            if isinstance(elem, type) and is_dataclass(elem):
                return elem
            return None
        return None

    for cls in (
        HParams, ModelConfig, DatasetConfig,
        DatasetTargetConfig, HfSourceSpec, SourceRef,
        DatasetInterferenceConfig, InterferenceTypeConfig,
        DatasetMixtureConfig, DatasetActivityConfig,
        DatasetReferenceConfig,
        AudioConfig, DVectorConfig, ArchConfig, PredictorConfig,
        LossConfig, EMAConfig, TrainConfig,
    ):
        if not is_dataclass(cls):
            continue
        # Resolve type hints up front (PEP 563 / `from __future__ import
        # annotations` leaves `f.type` as a string).
        try:
            hints = _get_type_hints(cls)
        except Exception:
            hints = {}
        spec: dict[str, type] = {}
        cont_spec: dict[str, type] = {}
        for f in fields(cls):
            resolved = hints.get(f.name) or _resolve(f.type)
            if isinstance(resolved, type) and is_dataclass(resolved):
                spec[f.name] = resolved
                continue
            elem_t = _extract_container_element_type(resolved)
            if isinstance(elem_t, type) and is_dataclass(elem_t):
                cont_spec[f.name] = elem_t
        if spec:
            _NESTED_TYPES[cls] = spec
        if cont_spec:
            _CONTAINER_ELEMENT_TYPES[cls] = cont_spec


# Mapping: flat key → (block attr, sub-block attr, field attr).
# Used by `_coerce_to_nested` to upgrade a flat dict into the nested form.
_LEGACY_FLAT_TO_NESTED: dict[str, tuple[str, str, str]] = {
    # model.audio
    "sample_rate": ("model", "audio", "sample_rate"),
    "n_fft": ("model", "audio", "n_fft"),
    "hop_length": ("model", "audio", "hop_length"),
    "win_length": ("model", "audio", "win_length"),
    "n_mels": ("model", "audio", "n_mels"),
    "log_mel_floor": ("model", "audio", "log_mel_floor"),
    "frame_stack": ("model", "audio", "frame_stack"),
    # model.dvector
    "dvector_pt_path": ("model", "dvector", "pt_path"),
    "dvector_lstm_hidden": ("model", "dvector", "lstm_hidden"),
    "dvector_lstm_layers": ("model", "dvector", "lstm_layers"),
    "dvector_emb_dim": ("model", "dvector", "emb_dim"),
    "dvector_n_mels": ("model", "dvector", "n_mels"),
    "dvector_n_fft": ("model", "dvector", "n_fft"),
    "dvector_win_length": ("model", "dvector", "win_length"),
    "dvector_hop_length": ("model", "dvector", "hop_length"),
    "dvector_window": ("model", "dvector", "window"),
    "dvector_stride": ("model", "dvector", "stride"),
    # model.arch
    "hidden_dim": ("model", "arch", "hidden_dim"),
    "num_lstm_layers": ("model", "arch", "num_lstm_layers"),
    "dropout": ("model", "arch", "dropout"),
    "use_cnn": ("model", "arch", "use_cnn"),
    "num_cnn_layers": ("model", "arch", "num_cnn_layers"),
    "cnn_kernel_size": ("model", "arch", "cnn_kernel_size"),
    "cnn_channels_first": ("model", "arch", "cnn_channels_first"),
    # model.predictor
    "use_noise_type_predictor": ("model", "predictor", "use_noise_type_predictor"),
    "alpha_head_use_current_frame": (
        "model", "predictor", "alpha_head_use_current_frame",
    ),
    # model.loss
    "loss_alpha": ("model", "loss", "alpha"),
    "loss_alpha_start": ("model", "loss", "alpha_start"),
    "loss_alpha_ramp_steps": ("model", "loss", "alpha_ramp_steps"),
    "noise_type_loss_weight": ("model", "loss", "noise_type_loss_weight"),
    "adaptive_w_beta": ("model", "loss", "adaptive_w_beta"),
    "adaptive_w_a": ("model", "loss", "adaptive_w_a"),
    "adaptive_w_b": ("model", "loss", "adaptive_w_b"),
    # model.ema
    "use_ema": ("model", "ema", "use"),
    "ema_decay": ("model", "ema", "decay"),
    "ema_warmup_steps": ("model", "ema", "warmup_steps"),
    # dataset.target / interference (function-oriented with `sources` refs;
    # no flat legacy keys — see YAML form)
    # dataset.sources (top-level upstream declarations; no flat legacy keys)
    # dataset.mixture
    "sample_length_sec": ("dataset", "mixture", "sample_length_sec"),
    "snr_min_db": ("dataset", "mixture", "snr_min_db"),
    "snr_max_db": ("dataset", "mixture", "snr_max_db"),
    "silent_target_prob": ("dataset", "mixture", "silent_target_prob"),
    "silent_interference_prob": ("dataset", "mixture", "silent_interference_prob"),
    # dataset.activity
    "activity_burst_min_sec": ("dataset", "activity", "burst_min_sec"),
    "activity_burst_max_sec": ("dataset", "activity", "burst_max_sec"),
    "activity_fade_ms": ("dataset", "activity", "fade_ms"),
    # dataset.reference
    "ref_max_sec": ("dataset", "reference", "max_sec"),
    "ref_buffer_max_per_spk": ("dataset", "reference", "buffer_max_per_spk"),
    "ref_buffer_max_speakers": ("dataset", "reference", "buffer_max_speakers"),
    # dataset (top: processed_dir + decode_*)
    "processed_dir": ("dataset", None, "processed_dir"),
    "decode_workers": ("dataset", None, "decode_workers"),
    "decode_prefetch": ("dataset", None, "decode_prefetch"),
    "prefetch_batches_max": ("dataset", None, "prefetch_batches_max"),
    # train
    "batch_size": ("train", None, "batch_size"),
    "learning_rate": ("train", None, "learning_rate"),
    "weight_decay": ("train", None, "weight_decay"),
    "total_steps": ("train", None, "total_steps"),
    "warmup_steps": ("train", None, "warmup_steps"),
    "grad_clip_norm": ("train", None, "grad_clip_norm"),
    "log_every": ("train", None, "log_every"),
    "eval_every": ("train", None, "eval_every"),
    "ckpt_every": ("train", None, "ckpt_every"),
    "empty_cache_every": ("train", None, "empty_cache_every"),
    "seed": ("train", None, "seed"),
    "dataloader_workers": ("train", None, "dataloader_workers"),
    # top-level
    "hf_endpoint": (None, None, "hf_endpoint"),
    "hf_transfer": (None, None, "hf_transfer"),
}


def _coerce_to_nested(d: dict[str, Any], cls: type) -> tuple[dict[str, Any], dict[str, str]]:
    """Convert a possibly-flat dict to the nested form expected by `_from_dict`.

    Heuristic: if `d` already has any of the top-level keys (`model`,
    `dataset`, `train`), it's nested — return as-is. Otherwise, treat
    it as flat and route each known key through `_LEGACY_FLAT_TO_NESTED`.

    Returns the (possibly-rewritten) dict. A second tuple element is
    reserved for legacy-cache overrides; the new `dataset.sources` /
    `HfSourceSpec.cache_dir` schema has no flat-form cache keys anymore,
    so this is currently always an empty dict (kept as a tuple so
    `HParams.from_dict` doesn't need a structural change to the
    deserialization contract).
    """
    nested_top = {"model", "dataset", "train"}
    legacy_cache_overrides: dict[str, str] = {}

    if nested_top & set(d.keys()):
        # Already nested. Old `dataset.paths.*` blocks (legacy schema
        # where librispeech_cache / musan_cache lived under a shared
        # `dataset.paths` block) are no longer recognised; the new
        # `dataset.sources[*].cache_dir` field is the only place HF
        # arrow caches live.
        return dict(d), legacy_cache_overrides

    out: dict[str, Any] = {"model": {}, "dataset": {}, "train": {}}
    for k, v in d.items():
        if k in {"d_vector_dim", "streaming"}:
            # No-op fields; fixed dim and removed streaming mode.
            continue
        spec = _LEGACY_FLAT_TO_NESTED.get(k)
        if spec is None:
            # Unknown flat key — drop (matches the previous dataclass
            # filter behaviour in `HParams.from_dict`).
            continue
        block, sub, field = spec
        if sub is None:
            out[block][field] = v
        else:
            out[block].setdefault(sub, {})[field] = v
    # Drop empty blocks so `_from_dict` doesn't trip on missing inner
    # dicts and instantiate default sub-dataclasses with overrides.
    out = {k: v for k, v in out.items() if v}
    return out, legacy_cache_overrides


def resolve_source(hp: "HParams", ref: "SourceRef") -> "ResolvedSource":
    """Resolve a `SourceRef` against `hp.dataset.sources` (an `HfSourceSpec`).

    Returns a `ResolvedSource` that captures the effective (per-usage)
    `subset` and `splits`: `subset` defaults to the only entry in
    `HfSourceSpec.configs` when the ref didn't specify one (so a
    single-config spec can be referenced with just `source: <id>`), and
    `splits` falls back to `HfSourceSpec.splits`.

    Raises `KeyError` if `ref.source` is not in `dataset.sources` —
    this is a config bug (typo / forgotten declaration) and should
    surface at startup, not silently produce an empty pool.
    """
    spec = hp.sources_by_id.get(ref.source)
    if spec is None:
        raise KeyError(
            f"SourceRef.source={ref.source!r} not found in "
            f"dataset.sources (declared ids: {sorted(hp.sources_by_id)})"
        )
    subset: Optional[str] = ref.subset
    if subset is None and len(spec.configs) == 1:
        subset = spec.configs[0]
    splits = ref.splits or spec.splits
    return ResolvedSource(spec=spec, subset=subset, splits=splits)


# Populate the sub-dataclass registry at import time so `from_dict` can
# route nested fields to the right class.
_register_nested_types()
