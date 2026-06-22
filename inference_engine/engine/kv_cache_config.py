"""Analytical KV-cache sizing from HuggingFace model metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class KVCacheConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: str
    bytes_per_token: int = field(init=False)
    bytes_per_token_mb: float = field(init=False)

    def __post_init__(self) -> None:
        dtype_bytes = {
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float32: 4,
        }.get(self.dtype, 2)
        self.bytes_per_token = (
            2 * self.num_layers * self.num_kv_heads * self.head_dim * dtype_bytes
        )
        self.bytes_per_token_mb = self.bytes_per_token / (1024 * 1024)


def compute_kv_cache_config(model, config) -> KVCacheConfig:
    """Build cache sizing metadata from a decoder-only HuggingFace model."""
    model_config = model.config
    num_attention_heads = model_config.num_attention_heads
    num_kv_heads = getattr(
        model_config, "num_key_value_heads", num_attention_heads
    )
    return KVCacheConfig(
        num_layers=model_config.num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=model_config.hidden_size // num_attention_heads,
        dtype=next(model.parameters()).dtype,
        device=config.device,
    )


def estimate_max_sequences(
    kv_cache_config: KVCacheConfig,
    available_memory_mb: float,
    avg_sequence_length: int = 512,
) -> int:
    """Estimate how many average-length sequence caches fit in memory."""
    memory_per_sequence_mb = (
        kv_cache_config.bytes_per_token_mb * avg_sequence_length
    )
    if memory_per_sequence_mb <= 0:
        return 1
    return max(1, math.floor(available_memory_mb / memory_per_sequence_mb))


def format_kv_cache_report(kv_cache_config: KVCacheConfig) -> str:
    """Return a human-readable summary of KV-cache sizing metadata."""
    return "\n".join(
        [
            "KV Cache Configuration",
            f"  Layers: {kv_cache_config.num_layers}",
            f"  KV heads: {kv_cache_config.num_kv_heads}",
            f"  Head dimension: {kv_cache_config.head_dim}",
            f"  Dtype: {kv_cache_config.dtype}",
            f"  Bytes per token: {kv_cache_config.bytes_per_token / 1024:.2f} KB",
            f"  Memory per token: {kv_cache_config.bytes_per_token_mb:.6f} MB",
        ]
    )
