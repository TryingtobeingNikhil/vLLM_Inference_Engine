"""
tests/test_attention_wrapper.py — Phase 8 unit tests for attention_wrapper.py.

7 synchronous pytest tests — no async, no model loading.
Uses torch on CPU only.
"""

from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache

from inference_engine.engine.attention_wrapper import (
    build_past_key_values,
    extract_new_token_kv,
    reconstruct_dynamic_cache,
    reconstruct_past_key_values,
)
from inference_engine.engine.block_allocator import BlockAllocator
from inference_engine.engine.kv_cache_config import KVCacheConfig
from inference_engine.engine.paged_kv_cache import PagedKVCacheManager


# ── Minimal config stub ───────────────────────────────────────────────────────

class _Cfg:
    kv_block_size = 4
    kv_num_blocks = 8
    device = "cpu"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def kv_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_layers=2,
        num_kv_heads=4,
        head_dim=16,
        dtype=torch.float32,   # float32 for easier numerical comparison
        device="cpu",
    )


@pytest.fixture
def allocator() -> BlockAllocator:
    return BlockAllocator(num_blocks=8, block_size=4)


@pytest.fixture
def manager(kv_config: KVCacheConfig, allocator: BlockAllocator) -> PagedKVCacheManager:
    return PagedKVCacheManager(kv_config, allocator, _Cfg())


@pytest.fixture
def populated_manager(manager: PagedKVCacheManager, allocator: BlockAllocator) -> PagedKVCacheManager:
    """Allocate 1 block for seq1, write 3 tokens across 2 layers."""
    allocator.allocate("seq1", 1)
    bid = allocator.get_blocks("seq1")[0]
    for token_pos in range(3):
        for layer_idx in range(2):
            key = torch.full(
                (4, 16), float(token_pos + layer_idx), dtype=torch.float32
            )
            val = torch.full(
                (4, 16), float(token_pos + layer_idx) * 2.0, dtype=torch.float32
            )
            manager.write_kv("seq1", layer_idx, token_pos, key, val)
        # Update tokens_used after each token write
        allocator._blocks[bid].tokens_used = token_pos + 1
    return manager


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_reconstruct_past_key_values_shape(populated_manager: PagedKVCacheManager) -> None:
    """reconstruct_past_key_values returns correct shape for each layer."""
    result = reconstruct_past_key_values(
        "seq1", populated_manager, num_layers=2, device="cpu"
    )
    assert len(result) == 2               # one per layer
    assert result[0][0].shape == (1, 4, 3, 16)  # [batch, heads, tokens, head_dim]
    assert result[0][1].shape == (1, 4, 3, 16)
    assert result[1][0].shape == (1, 4, 3, 16)


def test_reconstruct_values_correct(populated_manager: PagedKVCacheManager) -> None:
    """Layer 0 token 0 key should be all 0.0 (token_pos=0 + layer_idx=0)."""
    result = reconstruct_past_key_values(
        "seq1", populated_manager, num_layers=2, device="cpu"
    )
    # Layer 0, token 0: value written was float(0 + 0) = 0.0
    assert torch.allclose(
        result[0][0][0, :, 0, :],     # [heads, head_dim]
        torch.zeros(4, 16, dtype=torch.float32),
    )
    # Layer 1, token 1: value written was float(1 + 1) = 2.0
    assert torch.allclose(
        result[1][0][0, :, 1, :],
        torch.full((4, 16), 2.0, dtype=torch.float32),
    )


def test_reconstruct_dynamic_cache_type(populated_manager: PagedKVCacheManager) -> None:
    """reconstruct_dynamic_cache returns a DynamicCache instance."""
    result = reconstruct_dynamic_cache(
        "seq1", populated_manager, num_layers=2, device="cpu"
    )
    assert isinstance(result, DynamicCache)


def test_build_past_key_values_dynamic_true(populated_manager: PagedKVCacheManager) -> None:
    """build_past_key_values(use_dynamic_cache=True) returns DynamicCache."""
    result = build_past_key_values(
        "seq1", populated_manager, 2, "cpu", use_dynamic_cache=True
    )
    assert isinstance(result, DynamicCache)


def test_build_past_key_values_dynamic_false(populated_manager: PagedKVCacheManager) -> None:
    """build_past_key_values(use_dynamic_cache=False) returns a tuple."""
    result = build_past_key_values(
        "seq1", populated_manager, 2, "cpu", use_dynamic_cache=False
    )
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_extract_new_token_kv_legacy_tuple() -> None:
    """extract_new_token_kv handles the legacy tuple-of-tuples format."""
    # Build fake legacy past_key_values:
    # batch=1, heads=4, seq_len=5, head_dim=16
    # Each position i has value float(i)
    keys = (
        torch.arange(5, dtype=torch.float32)
        .view(1, 1, 5, 1)
        .expand(1, 4, 5, 16)
    )
    fake_pkv = tuple(
        (keys.clone(), keys.clone() * 2)
        for _ in range(2)
    )

    key_slice, val_slice = extract_new_token_kv(fake_pkv, layer_idx=0, token_position=4)

    assert key_slice.shape == (4, 16)
    # The last position in seq_len=5 is index 4 → value 4.0
    assert torch.allclose(key_slice, torch.full((4, 16), 4.0))
    assert torch.allclose(val_slice, torch.full((4, 16), 8.0))  # 4.0 * 2


def test_extract_new_token_kv_dynamic_cache() -> None:
    """extract_new_token_kv handles the DynamicCache format."""
    # Build a DynamicCache with 2 layers, 3 tokens
    # Each token position gets value float(token_pos)
    cache = DynamicCache()
    for layer_idx in range(2):
        # shape: [1, num_kv_heads=4, seq_len=3, head_dim=16]
        key_data = torch.zeros(1, 4, 3, 16, dtype=torch.float32)
        for pos in range(3):
            key_data[0, :, pos, :] = float(pos)
        val_data = key_data.clone() * 2.0
        cache.update(key_data, val_data, layer_idx)

    key_slice, val_slice = extract_new_token_kv(cache, layer_idx=0, token_position=2)

    assert key_slice.shape == (4, 16)
    # Last position in the 3-token sequence is index 2 → value 2.0
    assert torch.allclose(key_slice, torch.full((4, 16), 2.0))
    assert torch.allclose(val_slice, torch.full((4, 16), 4.0))  # 2.0 * 2
