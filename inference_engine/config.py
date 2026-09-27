"""
config.py — Central configuration dataclass for the inference engine.

All tuneable parameters live here. Import Config from this module everywhere else
so there is exactly one source of truth.

Every field can be set with an environment variable of the same name in
UPPER_CASE (e.g. ``MAX_BATCH_SIZE=64``).  Values passed explicitly to the
constructor take precedence over the environment.  Fields documented as "0 = auto" are
resolved from the detected device (and, for the KV pool, from a memory
profile of the loaded model — see ``model_runner.profile_num_blocks``).
"""

from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field, fields

# Every engine step allocates temporaries of a different shape; expandable
# segments stop PyTorch's CUDA caching allocator from fragmenting (and slowly
# growing its reserve) under that pattern.  Must be set before the first CUDA
# allocation; ignored on non-CUDA builds.  Respect a user-provided value.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _auto_detect_device() -> str:
    """
    Return the best available device string.

    Priority: cuda > mps > cpu.
    MPS is available on Apple Silicon with PyTorch >= 2.0.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"Cannot parse boolean from {value!r}")


@dataclass
class Config:
    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2-0.5B"
    # "auto" → bfloat16 on Ampere+ GPUs, float16 on older GPUs (T4) and MPS,
    # float32 on CPU.  Or one of: float16 | bfloat16 | float32.
    dtype: str = "auto"

    # ── Inference defaults ────────────────────────────────────────────────────
    max_new_tokens: int = 50
    # Longest sequence (prompt + generated) the engine will accept.
    # 0 = auto: min(model max_position_embeddings, 4096).
    max_model_len: int = 0
    # Seed for the sampler's RNG (temperature > 0 requests).
    seed: int = 0

    # ── Hardware ──────────────────────────────────────────────────────────────
    # Resolved once at startup; downstream code reads config.device, never
    # calls _auto_detect_device() again.
    device: str = field(default_factory=_auto_detect_device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics_output_path: str = "baseline_metrics.json"
    # Maximum number of per-request results kept in memory for percentiles.
    metrics_history_size: int = 1000

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "info"

    # ── Phase 2: Continuous Batching Scheduler ────────────────────────────────
    # Maximum number of sequences resident in the running batch.
    # 0 = auto: 64 on CUDA, 8 elsewhere.
    max_batch_size: int = 0
    # Sleep interval (ms) for the scheduler idle loop when there is no work.
    scheduler_poll_interval_ms: float = 1.0
    # Maximum time (ms) a request may wait in the RequestQueue before being
    # expired with asyncio.TimeoutError.
    request_timeout_ms: float = 60_000.0
    # Maximum number of requests waiting in the queue before HTTP 503.
    # 0 = auto: 16 × max_batch_size.
    max_queue_size: int = 0

    # ── Phase 4: Prefill / Decode Separation ─────────────────────────────────
    # Maximum number of prompt tokens processed per scheduler step (summed
    # over all sequences).  Decode tokens are not counted against it.
    # 0 = auto: 2048 on CUDA, 512 elsewhere.
    prefill_budget_tokens: int = 0
    # Maximum number of sequences resident in the decode stage.
    # 0 = same as max_batch_size.
    decode_batch_limit: int = 0
    # Maximum prompt tokens a single sequence processes per step (chunked
    # prefill).  Smaller values interleave decode more finely (lower ITL for
    # running requests) at the cost of more steps per long prompt.
    # 0 = auto: 512 on CUDA, 128 elsewhere.
    prefill_chunk_size: int = 0

    # ── Phase 5: KV Cache Tracking ────────────────────────────────────────────
    # On CPU/MPS this is also the size of the KV pool when kv_num_blocks=0.
    kv_cache_max_memory_mb: float = 1024.0

    # ── Phase 6: Block Allocator ──────────────────────────────────────────────
    # Number of token slots per KV cache block.
    kv_block_size: int = 16
    # Total number of blocks in the device pool.  0 = auto: profile the model
    # on CUDA and use gpu_memory_utilization; kv_cache_max_memory_mb elsewhere.
    kv_num_blocks: int = 0
    # Fraction of total GPU memory the engine may use (weights + activations
    # + KV cache).  Only used when kv_num_blocks=0 on CUDA.
    gpu_memory_utilization: float = 0.85
    # Enable automatic prefix caching: full KV blocks are content-hashed and
    # shared between requests with a common prompt prefix.
    enable_prefix_caching: bool = True

    # ── Phase 9: CPU Swap Pool ────────────────────────────────────────────────
    # Size of the CPU-side KV staging pool, in blocks.  0 = auto from
    # swap_space_gb.
    kv_num_cpu_blocks: int = 0
    swap_space_gb: float = 2.0
    # What to do with a running sequence when the device pool is exhausted:
    #   "swap"      — copy its blocks to the CPU pool (falls back to
    #                 recompute when the CPU pool is full)
    #   "recompute" — drop its blocks and re-prefill it later (cheap with
    #                 prefix caching, since its blocks usually stay cached)
    preemption_mode: str = "swap"

    # ── Speculative decoding ──────────────────────────────────────────────────
    # "" (off) | "ngram" (prompt lookup) | "draft" (small draft model)
    speculative_method: str = ""
    num_speculative_tokens: int = 4
    # n-gram proposer: try suffix lengths from max down to min.
    ngram_max: int = 4
    ngram_min: int = 2
    # Draft model for speculative_method="draft".  Must share the target
    # model's tokenizer (e.g. Qwen2.5-0.5B-Instruct for Qwen2.5-7B-Instruct).
    draft_model_name: str = ""

    def __post_init__(self) -> None:
        self._apply_env_overrides()
        self._resolve_auto_values()
        self._validate()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _apply_env_overrides(self) -> None:
        """Fill fields left at their default from UPPER_CASE environment variables."""
        for f in fields(self):
            raw = os.environ.get(f.name.upper())
            if raw is None or raw == "":
                continue
            current = getattr(self, f.name)
            default = f.default if f.default is not MISSING else (
                f.default_factory() if f.default_factory is not MISSING else MISSING
            )
            if current != default:
                continue  # explicitly passed to the constructor
            if isinstance(current, bool):
                setattr(self, f.name, _parse_bool(raw))
            elif isinstance(current, int):
                setattr(self, f.name, int(raw))
            elif isinstance(current, float):
                setattr(self, f.name, float(raw))
            else:
                setattr(self, f.name, raw)

    def _resolve_auto_values(self) -> None:
        on_cuda = self.device == "cuda"
        if self.max_batch_size == 0:
            self.max_batch_size = 64 if on_cuda else 8
        if self.decode_batch_limit == 0:
            self.decode_batch_limit = self.max_batch_size
        if self.prefill_budget_tokens == 0:
            self.prefill_budget_tokens = 2048 if on_cuda else 512
        if self.prefill_chunk_size == 0:
            self.prefill_chunk_size = 512 if on_cuda else 128
        if self.max_queue_size == 0:
            self.max_queue_size = 16 * self.max_batch_size

    def _validate(self) -> None:
        positive_int_fields = (
            "max_new_tokens",
            "metrics_history_size",
            "max_batch_size",
            "max_queue_size",
            "prefill_budget_tokens",
            "decode_batch_limit",
            "prefill_chunk_size",
            "kv_block_size",
        )
        for field_name in positive_int_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be greater than zero")

        non_negative_int_fields = (
            "max_model_len",
            "kv_num_blocks",
            "kv_num_cpu_blocks",
            "num_speculative_tokens",
        )
        for field_name in non_negative_int_fields:
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be >= 0")

        positive_float_fields = (
            "scheduler_poll_interval_ms",
            "request_timeout_ms",
            "kv_cache_max_memory_mb",
        )
        for field_name in positive_float_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be greater than zero")

        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.swap_space_gb < 0:
            raise ValueError("swap_space_gb must be >= 0")
        if not 1 <= self.port <= 65_535:
            raise ValueError("port must be between 1 and 65535")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps")
        if self.dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be one of: auto, float16, bfloat16, float32")
        if self.preemption_mode not in {"swap", "recompute"}:
            raise ValueError("preemption_mode must be 'swap' or 'recompute'")
        if self.speculative_method not in {"", "ngram", "draft"}:
            raise ValueError("speculative_method must be '', 'ngram' or 'draft'")
        if self.speculative_method == "draft" and not self.draft_model_name:
            raise ValueError("speculative_method='draft' requires draft_model_name")
        if self.speculative_method and self.num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1 when speculating")
        if not 1 <= self.ngram_min <= self.ngram_max:
            raise ValueError("require 1 <= ngram_min <= ngram_max")


# Module-level singleton — import and use directly when you don't need
# customisation.  The server creates its own instance from scratch so it can
# pick up environment variables set before startup.
default_config = Config()
