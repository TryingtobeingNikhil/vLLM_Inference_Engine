"""Analytical KV-cache sizing from HuggingFace model metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


_DTYPE_BYTES = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
}


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
        dtype_bytes = _DTYPE_BYTES.get(self.dtype, 2)
        # K and V, for every layer.
        self.bytes_per_token = (
            2 * self.num_layers * self.num_kv_heads * self.head_dim * dtype_bytes
        )
        self.bytes_per_token_mb = self.bytes_per_token / (1024 * 1024)

    def bytes_per_block(self, block_size: int) -> int:
        """Bytes of device memory one KV block occupies across all layers."""
        return self.bytes_per_token * block_size


def compute_kv_cache_config(model, config) -> KVCacheConfig:
    """Build cache sizing metadata from a decoder-only HuggingFace model.

    ``head_dim`` is read from the model config when present: several model
    families (Qwen3, Gemma, some Mistral variants) decouple it from
    ``hidden_size // num_attention_heads``, and using the derived value would
    allocate a pool of the wrong shape.
    """
    model_config = model.config
    num_attention_heads = model_config.num_attention_heads
    num_kv_heads = getattr(model_config, "num_key_value_heads", None) or num_attention_heads
    head_dim = getattr(model_config, "head_dim", None) or (
        model_config.hidden_size // num_attention_heads
    )
    return KVCacheConfig(
        num_layers=model_config.num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
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
