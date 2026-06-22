"""
config.py — Central configuration dataclass for the inference engine.

All tuneable parameters live here. Import Config from this module everywhere else
so there is exactly one source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


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
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@dataclass
class Config:
    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2-0.5B"

    # ── Inference defaults ────────────────────────────────────────────────────
    max_new_tokens: int = 50

    # ── Hardware ──────────────────────────────────────────────────────────────
    # Resolved once at startup; downstream code reads config.device, never
    # calls _auto_detect_device() again.
    device: str = field(default_factory=_auto_detect_device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics_output_path: str = "baseline_metrics.json"
    # Maximum number of GenerationResult objects kept in memory.
    metrics_history_size: int = 100

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "info"

    # ── Phase 2: Continuous Batching Scheduler ────────────────────────────────
    # Maximum number of sequences processed concurrently by the scheduler.
    max_batch_size: int = 4
    # Sleep interval (ms) for the scheduler idle loop when both the waiting
    # queue and running list are empty.  Also used as the /generate poll
    # interval when awaiting sequence completion (app_v2.py uses 5× this).
    scheduler_poll_interval_ms: float = 1.0
    # Maximum time (ms) a request may wait in the RequestQueue before being
    # expired with asyncio.TimeoutError.  Applies only to Phase 3+ queue.
    request_timeout_ms: float = 30_000.0

    # ── Phase 4: Prefill / Decode Separation ─────────────────────────────────
    # Maximum number of prompt tokens admitted for prefill per iteration.
    prefill_budget_tokens: int = 512
    # Maximum number of sequences resident in the decode stage.
    decode_batch_limit: int = 8

    # ── Phase 5: KV Cache Tracking ────────────────────────────────────────────
    kv_cache_max_memory_mb: float = 1024.0

    # ── Phase 6: Block Allocator ──────────────────────────────────────────────
    # Number of token slots per KV cache block.
    kv_block_size: int = 16
    # Total number of blocks in the pool.
    kv_num_blocks: int = 256

    def __post_init__(self) -> None:
        # Allow environment variable overrides for common settings so the
        # server can be configured without code changes.
        if env_model := os.environ.get("MODEL_NAME"):
            self.model_name = env_model
        if env_device := os.environ.get("DEVICE"):
            self.device = env_device
        if env_port := os.environ.get("PORT"):
            self.port = int(env_port)
        if env_out := os.environ.get("METRICS_OUTPUT_PATH"):
            self.metrics_output_path = env_out
        # Phase 2 env overrides
        if env_bs := os.environ.get("MAX_BATCH_SIZE"):
            self.max_batch_size = int(env_bs)
        if env_poll := os.environ.get("SCHEDULER_POLL_INTERVAL_MS"):
            self.scheduler_poll_interval_ms = float(env_poll)
        if env_timeout := os.environ.get("REQUEST_TIMEOUT_MS"):
            self.request_timeout_ms = float(env_timeout)
        if env_prefill_budget := os.environ.get("PREFILL_BUDGET_TOKENS"):
            self.prefill_budget_tokens = int(env_prefill_budget)
        if env_decode_limit := os.environ.get("DECODE_BATCH_LIMIT"):
            self.decode_batch_limit = int(env_decode_limit)
        if env_kv_memory := os.environ.get("KV_CACHE_MAX_MEMORY_MB"):
            self.kv_cache_max_memory_mb = float(env_kv_memory)
        # Phase 6 env overrides
        if env_block_size := os.environ.get("KV_BLOCK_SIZE"):
            self.kv_block_size = int(env_block_size)
        if env_num_blocks := os.environ.get("KV_NUM_BLOCKS"):
            self.kv_num_blocks = int(env_num_blocks)


# Module-level singleton — import and use directly when you don't need
# customisation.  The server creates its own instance from scratch so it can
# pick up environment variables set before startup.
default_config = Config()
