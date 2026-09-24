"""Centralised hyperparameters for VoiceFilter-Lite."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class HParams:
    # ---- Audio ----
    sample_rate: int = 16000
    n_fft: int = 400           # 25 ms window @ 16 kHz
    hop_length: int = 160      # 10 ms hop @ 16 kHz
    win_length: int = 400
    n_mels: int = 80
    log_mel_floor: float = 1e-6

    # ---- Streaming / frame stacking ----
    # Stack `frame_stack` past frames (left context only) with the current frame.
    # Real-time telephony cannot use future context.
    # Paper §4.5: "stacked-by-4 log Mel-filterbank energies" — model I/O is
    # 80*4 = 320-dim stacked filterbanks.
    frame_stack: int = 4
    right_context: int = 0

    # ---- d-vector ----
    # Raw ECAPA-TDNN from speechbrain (`spkrec-ecapa-voxceleb`) emits 192-dim
    # embeddings; we project to d_vector_dim inside the model. The raw dim
    # lives in HParams so model.py and speaker_encoder.py stay in sync.
    speaker_encoder_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    speaker_encoder_raw_dim: int = 192
    d_vector_dim: int = 256
    speaker_encoder_cache: str = "models/spkrec-ecapa-voxceleb"

    # ---- Model ----
    hidden_dim: int = 384
    num_lstm_layers: int = 2
    dropout: float = 0.1

    # ---- Training ----
    sample_length_sec: float = 4.0
    # Paper §3.2: "random within 1 dB to 10 dB". Negative SNRs (target
    # quieter than interference) push the mask network into a regime the
    # paper doesn't evaluate and tend to destabilise the asymmetric loss
    # — keep both bounds ≥ 1.
    snr_min_db: float = 1.0
    snr_max_db: float = 10.0
    silent_target_prob: float = 0.10    # P(target is silence) — train the "do nothing" mode
    silent_interference_prob: float = 0.10  # P(interference is silence)
    batch_size: int = 6
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    total_steps: int = 50_000
    warmup_steps: int = 1_000
    grad_clip_norm: float = 5.0
    log_every: int = 5
    eval_every: int = 1_000
    ckpt_every: int = 1_000
    num_eval_samples: int = 32

    # How often (in optimizer steps) to call `torch.<device>.empty_cache()`
    # mid-training. PyTorch's MPS / CUDA caching allocators keep a high-water
    # mark of every unique tensor shape they ever saw; on long runs that
    # cache grows monotonically and shows up as a slowly climbing RSS even
    # when nothing is leaking in the Python sense. Calling `empty_cache()`
    # periodically returns unused chunks to the OS — small per-call overhead,
    # but it keeps the resident set flat. Set to 0 to disable.
    #   - MPS (Apple Silicon): every 2000 steps is a good starting point
    #     (the cache grows fast on the first few thousand steps, then
    #      plateaus once unique shapes are exhausted).
    #   - CUDA: every 5000 steps is usually sufficient on a single GPU;
    #     larger models on multi-GPU may prefer every 1000.
    empty_cache_every: int = 2000

    # ---- Loss (asymmetric L2, paper §3.3) ----
    # `alpha` is the asymmetry factor: over-suppression errors are amplified
    # by α² in the loss (g_asym(x,α) = α·x for x>0). Paper Tables 1 & 2 use
    # α = 10 → 100× amplification on dropped speech. Going much lower kills
    # the "err on the side of passing through" behaviour; going much higher
    # makes gradients unstable at training start.
    loss_alpha: float = 10.0

    # ---- Noise-type predictor (paper §4.2 / §3.4) ----
    # Two feedforward layers (256 → 64 → 64 → 1) emit **per-frame** logits
    # indicating "is this frame overlapped speech?". Trained with a hinge
    # loss (binary classification). At inference, the sigmoid of the logit
    # drives the adaptive suppression strength `w^(t)` per §3.4.
    # Adds ~20.6 K trainable params on top of the 3-LSTM backbone; total
    # sits at ~2.2 M params (the paper's 2.2 MB post-quantization target).
    use_noise_type_predictor: bool = True
    hinge_loss_weight: float = 0.1  # weight on hinge loss in the total objective
    adaptive_w_beta: float = 0.8    # paper §4.4 "moving average coefficient β = 0.8"
    adaptive_w_a: float = 1.0       # paper §3.4 linear transform "a > 0"
    adaptive_w_b: float = 0.0       # paper §3.4 linear transform "b ≥ 0"

    # ---- Deprecated (kept for backward-compat with older checkpoints) ----
    # The per-sample α regression that used to live here is replaced by the
    # per-frame hinge-loss classifier above. These fields are no longer
    # consumed by the model or loss; they remain so old checkpoints can be
    # round-tripped through `HParams.from_dict`.
    alpha_min: float = 1.0
    alpha_max: float = 10.0
    # Old soft-mask L2 (loss_beta / loss_lambda) — superseded by the paper's
    # sign-based g_asym(x, α). Silently ignored by the current loss; kept so
    # configs/checkpoints that still mention them don't fail HParams.from_dict.
    loss_beta: float = 0.0
    loss_lambda: float = 5.0

    # ---- EMA (exponential moving average of model weights) ----
    # EMA weights are what gets saved as ckpt_final.pt so downstream
    # evaluate.py / export_onnx.py automatically use the regularised weights.
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_warmup_steps: int = 100  # before this step, EMA == model (no averaging)

    # ---- Data ----
    target_config: str = "clean"
    target_splits: tuple[str, ...] = ("train.360")
    interference_config: str = "other"
    interference_split: str = "train.500"
    streaming: bool = True     # HF datasets streaming=True (saves disk)
    # When streaming=False, `load_dataset` reads from this on-disk cache (HF
    # datasets arrow format). The repo's `scripts/download_data.py` writes here.
    librispeech_cache: str = "data/librispeech/hf_cache"

    # Reference-waveform length for ECAPA d-vector extraction. ECAPA-TDNN
    # needs ~3-6 s to produce a stable speaker embedding; longer refs only
    # inflate memory. The dataset caps `ref_wav` to this length before
    # yielding, so `collate_triplets` pads to at most `int(ref_max_sec *
    # sample_rate)` samples instead of the longest utterance in the batch
    # (which can be ~30 s for some LibriSpeech speakers). This is the
    # single biggest memory win on MPS — at batch_size=16 the padded tensor
    # drops from ~30 MB/eye to ~2 MB/eye per batch (8x). Applies to both
    # the in-flight CPU batch *and* every batch sitting in
    # `PrefetchIterator`'s queue.
    ref_max_sec: float = 6.0

    # Same-speaker reference cache, in `LibriSpeechTripletStream`.
    # The iterator decodes every target utterance once, then caches a
    # ref_max_len-capped slice so 70% of the next samples from the same
    # speaker can reuse it without re-running soundfile. Two caps apply:
    #   - `ref_buffer_max_per_spk`: max cached utterances per speaker.
    #     2 is the floor (the same-speaker pick gate is `len(buf) >= 2`);
    #     4–8 gives a touch more d-vector coverage at negligible cost.
    #   - `ref_buffer_max_speakers`: max distinct speaker keys held.
    #     LibriSpeech-500h has ~1172 speakers; without this cap every
    #     speaker ever seen stays in the dict until process exit, and
    #     with ref_max_sec=6 the in-flight buffer can grow past 20 GB
    #     on MPS. Old default was 64 × ∞; this fixes the unbounded leak.
    ref_buffer_max_per_spk: int = 4
    ref_buffer_max_speakers: int = 512

    # ---- Data-loading parallelism ----
    # `LibriSpeechTripletStream` decodes 3 audio files per training sample
    # (target, interference, reference). Single-threaded decoding is the
    # bottleneck on MPS — GPU compute is sub-second per batch but each batch
    # takes ~50 s of audio I/O + soundfile decode, leaving the GPU idle.
    # `decode_workers` runs that decode in a thread pool; `decode_prefetch`
    # pipelines that many decode jobs ahead of the consumer so the GPU
    # always has a finished batch waiting. 4×4 is a good default on M-series;
    # bump `decode_workers` if `samples/s` in the heartbeat still looks low.
    decode_workers: int = 4
    decode_prefetch: int = 4

    # Number of complete batches `PrefetchIterator` queues on the consumer
    # side. Each queued batch sits in CPU memory (with a copy on MPS once
    # the main thread has done `.to(device)`); on memory-constrained
    # machines (16 GB Apple Silicon) 4 batches per queue is too many —
    # especially because each batch carries padded ref waveforms. The
    # deep queue only buys ~1-step GPU idle-time elimination once the
    # data pipeline itself has its own decode prefetch; 2 is a good
    # default on MPS, drop to 1 if you're still OOMing.
    prefetch_batches_max: int = 2

    # ---- Speaker d-vector cache ----
    # ECAPA-TDNN on the reference waveform dominates per-batch CPU cost
    # (~50 ms × batch_size per flush). The d-vector is a function of the
    # *speaker*, not the utterance, so once we've computed one for a given
    # speaker_id we never need to re-run ECAPA for them. In-memory cache
    # always-on; set `dvec_cache_dir` to also persist to disk so subsequent
    # training runs (or checkpoint resumes) don't pay the ECAPA warm-up
    # cost again. `None` = in-memory only (cache lost on process exit).
    dvec_cache_dir: Optional[str] = "models/dvec_cache"

    # ---- DataLoader workers ----
    # Number of background processes PyTorch spawns to prefetch training
    # batches from the dataset. With `0` (default), the main training
    # thread pulls each batch inline — simplest and lowest overhead, but
    # the GPU sits idle while it waits for audio decode + ECAPA + mel.
    # With `N > 0`, N worker processes each load + augment + collate
    # batches in parallel; the main thread consumes from a queue.
    #
    #   CUDA (single GPU) : 2-4 is the sweet spot; each worker holds
    #                        ~200-500 MB and burns one CPU core.
    #   MPS (Apple Silicon): 0-2; unified-memory bandwidth is shared with
    #                        the GPU, so over-spawning workers starves the
    #                        GPU. M2 Pro has only 4-6 perf cores.
    #   CUDA + DDP         : 2 per rank — DDP is already multi-process,
    #                        stacking dataloader workers can blow the
    #                        core budget.
    #
    # Caveat when `streaming=True`: each worker opens its own HF stream,
    # so N workers = N× the HF Hub egress bandwidth on your network.
    dataloader_workers: int = 0

    # ---- Interference augmentation (Musan + real RIR) ----
    # Both default to OFF so a fresh checkout can train without downloading
    # any extra datasets. Opt-in via configs/paper.yaml, --use-musan, or by
    # editing these fields. If a dataset fails to load, training degrades
    # gracefully (Musan → all-LibriSpeech; RIR → procedural exponential decay).
    use_musan: bool = False
    musan_prob: float = 0.7            # P(interference = Musan noise vs LibriSpeech speaker)
    musan_source: str = "corypaik/musan"
    musan_cache: str = "models/musan"
    # Real MUSAN has only {music, noise, speech}. There's no `babble` subset;
    # we synthesise it on the fly by overlaying N `speech` clips with random
    # gains (see Kaldi/ESPnet spec-aug recipe — Snyder et al. 2015, App. A).
    musan_subset: str = "babble,music,noise"  # comma-separated subset names
    musan_babble: bool = True           # synthesise `babble` from `speech` subset
    musan_babble_n_speakers: int = 3    # clips to overlay per babble sample
    musan_babble_gain_min: float = 0.3  # per-clip uniform gain range
    musan_babble_gain_max: float = 1.0
    use_real_rir: bool = False
    rir_prob: float = 0.4               # P(target is convolved with a random RIR)
    # Primary source is `confit/rir-noise` — a loader that pulls SLR28
    # `rirs_noises.zip` directly from openslr.org. `rir_source_fallbacks` is
    # a small list of mirrors to try in order after the primary fails.
    # `dataset._ensure_rir_open` iterates (source × config). Last resort is
    # the procedural exponential-decay RIR in `_synthetic_rir` — always works.
    rir_source: str = "confit/rir-noise"
    rir_source_fallbacks: tuple[str, ...] = (
        "quinnlue/rirs",
    )
    rir_cache: str = "models/rir"

    # ---- Misc ----
    seed: int = 1234
    device: str = "mps"

    # ---- HF Hub endpoint (network mirror) ----
    # Override the HuggingFace Hub base URL — useful for users behind
    # firewalls or in regions where `huggingface.co` is unreachable
    # (e.g. China → `https://hf-mirror.com`). Anything `datasets` /
    # `huggingface_hub` / speechbrain's `from_hparams` would normally
    # fetch from HF is redirected through this endpoint instead.
    #
    # Precedence (first non-empty wins) is applied by
    # `open_voicefilter_lite.cli.apply_hf_endpoint`:
    #   1. `--hf-endpoint` CLI flag on each entry-point script
    #   2. `HF_ENDPOINT` env var (the standard HF library convention)
    #   3. this `hf_endpoint` field (YAML config / CLI override)
    #   4. None → default `https://huggingface.co`
    hf_endpoint: Optional[str] = None

    # ---- HF Hub transfer mode (hf_transfer protocol) ----
    # `huggingface_hub` supports a faster, resumable download protocol
    # (`hf_transfer`) that's *opt-in* via the `HF_HUB_ENABLE_HF_TRANSFER`
    # env var. On `huggingface.co` it's noticeably faster for big pulls
    # (LibriSpeech train.500 = ~54 GB) but on most public mirrors
    # (e.g. `hf-mirror.com`) it isn't implemented and silently 502s.
    #
    # `None` (default): honour whatever's in `$HF_HUB_ENABLE_HF_TRANSFER`
    #     (which is unset/false by default — regular HTTPS, works on both
    #     HF.co and mirrors). When `hf_endpoint` is *also* set and the
    #     env var is unset, `apply_hf_endpoint` auto-disables it as a
    #     mirror-safety net.
    # `True`  : force `HF_HUB_ENABLE_HF_TRANSFER=1` (fast HF.co pulls,
    #     will likely break on mirrors — caller is responsible).
    # `False` : force `HF_HUB_ENABLE_HF_TRANSFER=0` (safe everywhere).
    #
    # Equivalent CLI: `--hf-transfer` / `--no-hf-transfer`.
    hf_transfer: Optional[bool] = None

    # ---- Derived ----
    @property
    def input_dim(self) -> int:
        return self.n_mels * self.frame_stack + self.d_vector_dim

    @property
    def output_dim(self) -> int:
        return self.n_mels

    @property
    def sample_length_samples(self) -> int:
        return int(self.sample_length_sec * self.sample_rate)

    @property
    def num_frames(self) -> int:
        # Number of frames in a sample after STFT.
        # frames = 1 + (n - n_fft) // hop
        return 1 + (self.sample_length_samples - self.n_fft) // self.hop_length

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HParams":
        if "target_splits" in d and isinstance(d["target_splits"], list):
            d["target_splits"] = tuple(d["target_splits"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
