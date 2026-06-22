"""
tests/test_paged_kv_cache.py — Phase 7 unit tests for PagedKVCacheManager.

9 synchronous pytest tests — no async, no model loading.
Uses torch on CPU only.
"""

from __future__ import annotations

import pytest
import torch

from inference_engine.engine.block_allocator import BlockAllocator
from inference_engine.engine.kv_cache_config import KVCacheConfig
from inference_engine.engine.paged_kv_cache import PagedKVCacheManager


# ── Minimal config stub ───────────────────────────────────────────────────────

class _Cfg:
    """Minimal config-like object accepted by PagedKVCacheManager."""
    kv_block_size = 8
    kv_num_blocks = 16
    device = "cpu"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def kv_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_layers=4,
        num_kv_heads=4,
        head_dim=32,
        dtype=torch.float16,
        device="cpu",
    )


@pytest.fixture
def allocator() -> BlockAllocator:
    return BlockAllocator(num_blocks=16, block_size=8)


@pytest.fixture
def manager(kv_config: KVCacheConfig, allocator: BlockAllocator) -> PagedKVCacheManager:
    return PagedKVCacheManager(kv_config, allocator, _Cfg())


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_pool_allocated(manager: PagedKVCacheManager) -> None:
    """Verify pool tensors have the expected shape and dtype."""
    # [num_blocks, block_size, num_layers, num_kv_heads, head_dim]
    assert manager.key_pool.shape == (16, 8, 4, 4, 32)
    assert manager.value_pool.shape == (16, 8, 4, 4, 32)
    assert manager.key_pool.dtype == torch.float16


def test_pool_size_mb_positive(manager: PagedKVCacheManager) -> None:
    """Pool size must be positive (tensors were allocated)."""
    assert manager.stats()["pool_size_mb"] > 0.0


def test_write_and_read_single_token(
    manager: PagedKVCacheManager,
    allocator: BlockAllocator,
) -> None:
    """Write one token, read it back — shapes and values must match."""
    allocator.allocate("seq1", 1)
    key = torch.ones(4, 32, dtype=torch.float16)
    val = torch.full((4, 32), 2.0, dtype=torch.float16)

    manager.write_kv("seq1", layer_idx=0, token_position=0, key_tensor=key, value_tensor=val)

    # Set tokens_used so read_kv_sequence knows how many slots are filled
    bid = allocator.get_blocks("seq1")[0]
    allocator._blocks[bid].tokens_used = 1

    keys, values = manager.read_kv_sequence("seq1", layer_idx=0)
    assert keys.shape == (1, 4, 32)
    assert torch.allclose(keys[0], torch.ones(4, 32, dtype=torch.float16))
    assert torch.allclose(values[0], torch.full((4, 32), 2.0, dtype=torch.float16))


def test_write_across_two_blocks(
    manager: PagedKVCacheManager,
    allocator: BlockAllocator,
) -> None:
    """Writes that span two blocks are assembled correctly by read_kv_sequence."""
    allocator.allocate("seq1", 2)
    bids = allocator.get_blocks("seq1")

    # Write 9 tokens: 0..7 in block 0, token 8 in block 1
    for token_position in range(9):
        key = torch.full((4, 32), float(token_position), dtype=torch.float16)
        val = torch.zeros(4, 32, dtype=torch.float16)
        manager.write_kv("seq1", layer_idx=0, token_position=token_position,
                         key_tensor=key, value_tensor=val)

    # Set tokens_used: block 0 is full (8), block 1 has 1
    allocator._blocks[bids[0]].tokens_used = 8
    allocator._blocks[bids[1]].tokens_used = 1

    keys, _ = manager.read_kv_sequence("seq1", layer_idx=0)
    assert keys.shape[0] == 9
    # Token at position 8 should have value 8.0
    assert float(keys[8, 0, 0]) == pytest.approx(8.0, rel=1e-3)


def test_read_kv_block_raw(
    manager: PagedKVCacheManager,
    allocator: BlockAllocator,
) -> None:
    """read_kv_block returns the full block_size, unfiltered."""
    allocator.allocate("seq1", 1)
    bid = allocator.get_blocks("seq1")[0]

    manager.write_kv(
        "seq1", 0, 0,
        torch.ones(4, 32, dtype=torch.float16),
        torch.ones(4, 32, dtype=torch.float16),
    )

    keys, values = manager.read_kv_block(bid, layer_idx=0)
    # Full block_size rows — empty slots are zeros
    assert keys.shape == (8, 4, 32)
    assert values.shape == (8, 4, 32)


def test_clear_sequence_zeros_pool(
    manager: PagedKVCacheManager,
    allocator: BlockAllocator,
) -> None:
    """clear_sequence() zeroes all slots in the pool for that sequence."""
    allocator.allocate("seq1", 1)
    bid = allocator.get_blocks("seq1")[0]

    manager.write_kv(
        "seq1", 0, 0,
        torch.ones(4, 32, dtype=torch.float16),
        torch.ones(4, 32, dtype=torch.float16),
    )

    manager.clear_sequence("seq1")
    assert float(manager.key_pool[bid].sum()) == pytest.approx(0.0)


def test_copy_blocks_all_layers(
    manager: PagedKVCacheManager,
    allocator: BlockAllocator,
) -> None:
    """copy_blocks(layer_idx=None) copies all layers from src to dst."""
    allocator.allocate("seq1", 1)
    allocator.allocate("seq2", 1)
    src_id = allocator.get_blocks("seq1")[0]
    dst_id = allocator.get_blocks("seq2")[0]

    # Write distinct values into every layer of seq1's block
    for layer in range(4):
        for slot in range(4):
            key = torch.full((4, 32), float(layer + slot + 1), dtype=torch.float16)
            val = torch.full((4, 32), float(layer + slot + 1), dtype=torch.float16)
            manager.write_kv("seq1", layer, slot, key, val)

    manager.copy_blocks(src_id, dst_id, layer_idx=None)

    assert torch.equal(manager.key_pool[dst_id], manager.key_pool[src_id])
    assert torch.equal(manager.value_pool[dst_id], manager.value_pool[src_id])


def test_copy_blocks_single_layer(
    manager: PagedKVCacheManager,
    allocator: BlockAllocator,
) -> None:
    """copy_blocks(layer_idx=0) copies only the specified layer."""
    allocator.allocate("seq1", 1)
    allocator.allocate("seq2", 1)
    src_id = allocator.get_blocks("seq1")[0]
    dst_id = allocator.get_blocks("seq2")[0]

    # Write non-zero values to layer 0 only
    manager.write_kv(
        "seq1", layer_idx=0, token_position=0,
        key_tensor=torch.ones(4, 32, dtype=torch.float16),
        value_tensor=torch.ones(4, 32, dtype=torch.float16),
    )

    manager.copy_blocks(src_id, dst_id, layer_idx=0)

    # Layer 0 slot 0 of dst should now equal src
    assert torch.equal(
        manager.key_pool[dst_id, :, 0],
        manager.key_pool[src_id, :, 0],
    )
    # Layer 1 of dst should remain all zeros
    assert float(manager.key_pool[dst_id, :, 1].sum()) == pytest.approx(0.0)


def test_stats_structure(
    manager: PagedKVCacheManager,
    allocator: BlockAllocator,
) -> None:
    """stats() has all required keys; active_sequences reflects write cursors."""
    allocator.allocate("seq1", 2)
    s = manager.stats()

    assert "pool_shape" in s
    assert "pool_size_mb" in s
    assert "block_allocator" in s
    assert "num_layers" in s
    assert "dtype" in s
    # No write_kv called yet → _seq_write_cursors is empty
    assert s["active_sequences"] == 0
