"""
tests/test_cpu_swap_manager.py — Unit tests for Phase 9: CPUSwapManager.

8 synchronous pytest tests.  No model loading, no async.
All tensor operations use CPU device.

Run with:
    cd /Users/nikhilmourya/Desktop/PageServe
    pytest inference_engine/tests/test_cpu_swap_manager.py -v
"""

from __future__ import annotations

import pytest
import torch

from inference_engine.engine.block_allocator import BlockAllocator
from inference_engine.engine.cpu_swap_manager import CPUSwapError, CPUSwapManager
from inference_engine.engine.kv_cache_config import KVCacheConfig
from inference_engine.engine.paged_kv_cache import PagedKVCacheManager


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def kv_config() -> KVCacheConfig:
    """Minimal KVCacheConfig wired for CPU-only testing."""
    return KVCacheConfig(
        num_layers=2,
        num_kv_heads=4,
        head_dim=16,
        dtype=torch.float32,
        device="cpu",
    )


@pytest.fixture
def allocator() -> BlockAllocator:
    """Small block pool: 8 blocks of size 4."""
    return BlockAllocator(num_blocks=8, block_size=4)


@pytest.fixture
def paged_manager(kv_config: KVCacheConfig, allocator: BlockAllocator) -> PagedKVCacheManager:
    """PagedKVCacheManager backed by the CPU allocator fixture."""

    class Cfg:
        kv_block_size = 4
        kv_num_blocks = 8
        device = "cpu"

    return PagedKVCacheManager(kv_config, allocator, Cfg())


@pytest.fixture
def swap_manager(kv_config: KVCacheConfig) -> CPUSwapManager:
    """CPUSwapManager with 8 CPU blocks of size 4."""
    return CPUSwapManager(kv_config, block_size=4, num_cpu_blocks=8)


# ── Test 1: CPU pool allocated with correct shape and device ─────────────────


def test_cpu_pool_allocated(swap_manager: CPUSwapManager) -> None:
    """cpu_key_pool must have shape (num_cpu_blocks, block_size, num_layers, num_kv_heads, head_dim)."""
    # 8 blocks × 4 slots × 2 layers × 4 heads × 16 head_dim
    assert swap_manager.cpu_key_pool.shape == (8, 4, 2, 4, 16), (
        f"Unexpected cpu_key_pool shape: {swap_manager.cpu_key_pool.shape}"
    )
    assert swap_manager.cpu_key_pool.device.type == "cpu", (
        "cpu_key_pool must be on CPU regardless of model device"
    )


# ── Test 2: swap_out frees device blocks ─────────────────────────────────────


def test_swap_out_frees_device_blocks(
    swap_manager: CPUSwapManager,
    paged_manager: PagedKVCacheManager,
    allocator: BlockAllocator,
) -> None:
    """After swap_out the sequence must hold 0 device blocks and be marked swapped."""
    allocator.allocate("seq1", 2)

    # Write dummy data so the pool slots are non-zero
    for layer in range(2):
        for tok_pos in range(4):  # 2 blocks × 2 tokens each
            key = torch.randn(4, 16)
            val = torch.randn(4, 16)
            paged_manager.write_kv("seq1", layer, tok_pos, key, val)

    # Set tokens_used on both blocks
    device_blocks = allocator.get_blocks("seq1")
    for bid in device_blocks:
        allocator._blocks[bid].tokens_used = 2

    swap_manager.swap_out("seq1", device_blocks, paged_manager, allocator)

    assert allocator.num_blocks_for_seq("seq1") == 0, (
        "Device blocks must be freed after swap_out"
    )
    assert swap_manager.is_swapped("seq1") is True, (
        "Sequence must be marked as swapped in CPUSwapManager"
    )


# ── Test 3: swap_out preserves data ──────────────────────────────────────────


def test_swap_out_preserves_data(
    swap_manager: CPUSwapManager,
    paged_manager: PagedKVCacheManager,
    allocator: BlockAllocator,
) -> None:
    """Data written to the device pool must appear verbatim in CPU staging memory."""
    allocator.allocate("seq1", 1)
    device_block_id = allocator.get_blocks("seq1")[0]

    # Write all-ones into layer 0, token position 0
    key_data = torch.ones(4, 16)
    val_data = torch.ones(4, 16) * 2.0
    paged_manager.write_kv("seq1", 0, 0, key_data, val_data)

    # Set tokens_used so swap_out counts the token
    allocator._blocks[device_block_id].tokens_used = 1

    device_blocks = allocator.get_blocks("seq1")
    swapped = swap_manager.swap_out("seq1", device_blocks, paged_manager, allocator)

    cpu_bid = swapped.cpu_block_ids[0]
    # cpu_key_pool[cpu_bid, slot=0, layer=0] should equal key_data
    assert torch.allclose(
        swap_manager.cpu_key_pool[cpu_bid, 0, 0], torch.ones(4, 16)
    ), "Key data not preserved in CPU pool"
    assert torch.allclose(
        swap_manager.cpu_value_pool[cpu_bid, 0, 0], torch.ones(4, 16) * 2.0
    ), "Value data not preserved in CPU pool"


# ── Test 4: swap_out raises CPUSwapError when CPU pool is full ────────────────


def test_swap_out_raises_when_cpu_pool_full(
    kv_config: KVCacheConfig,
    paged_manager: PagedKVCacheManager,
    allocator: BlockAllocator,
) -> None:
    """CPUSwapError must be raised when CPU pool cannot hold all device blocks."""
    tiny_swap = CPUSwapManager(kv_config, block_size=4, num_cpu_blocks=1)

    # Allocate 2 device blocks — swap-out needs 2 CPU blocks but pool has only 1
    allocator.allocate("seq1", 2)
    device_blocks = allocator.get_blocks("seq1")
    for bid in device_blocks:
        allocator._blocks[bid].tokens_used = 1

    with pytest.raises(CPUSwapError):
        tiny_swap.swap_out("seq1", device_blocks, paged_manager, allocator)


# ── Test 5: swap_in restores blocks ──────────────────────────────────────────


def test_swap_in_restores_blocks(
    swap_manager: CPUSwapManager,
    paged_manager: PagedKVCacheManager,
    allocator: BlockAllocator,
) -> None:
    """After swap_in the sequence must have exactly 1 device block again."""
    allocator.allocate("seq1", 1)
    device_block = allocator.get_blocks("seq1")[0]
    paged_manager.write_kv("seq1", 0, 0, torch.ones(4, 16), torch.ones(4, 16))
    allocator._blocks[device_block].tokens_used = 1

    device_blocks = allocator.get_blocks("seq1")
    swap_manager.swap_out("seq1", device_blocks, paged_manager, allocator)

    new_block_ids = swap_manager.swap_in("seq1", paged_manager, allocator)

    assert len(new_block_ids) == 1, (
        f"Expected 1 device block after swap_in, got {len(new_block_ids)}"
    )
    assert allocator.num_blocks_for_seq("seq1") == 1, (
        "Sequence must hold exactly 1 device block after swap_in"
    )
    assert swap_manager.is_swapped("seq1") is False, (
        "Sequence must no longer be marked as swapped after swap_in"
    )


# ── Test 6: swap_in restores data ────────────────────────────────────────────


def test_swap_in_restores_data(
    swap_manager: CPUSwapManager,
    paged_manager: PagedKVCacheManager,
    allocator: BlockAllocator,
) -> None:
    """Data must survive a full swap-out / swap-in round-trip."""
    sentinel = torch.full((4, 16), 7.0)

    allocator.allocate("seq1", 1)
    device_block = allocator.get_blocks("seq1")[0]
    paged_manager.write_kv("seq1", 0, 0, sentinel, sentinel)
    allocator._blocks[device_block].tokens_used = 1

    device_blocks = allocator.get_blocks("seq1")
    swap_manager.swap_out("seq1", device_blocks, paged_manager, allocator)
    swap_manager.swap_in("seq1", paged_manager, allocator)

    new_block_id = allocator.get_blocks("seq1")[0]
    # paged_manager.key_pool[new_block_id, slot=0, layer=0] should be all-7
    assert torch.allclose(
        paged_manager.key_pool[new_block_id, 0, 0],
        sentinel,
    ), "Key data not restored after swap-in"
    assert torch.allclose(
        paged_manager.value_pool[new_block_id, 0, 0],
        sentinel,
    ), "Value data not restored after swap-in"


# ── Test 7: swap_in raises KeyError if sequence not swapped ──────────────────


def test_swap_in_raises_keyerror_if_not_swapped(
    swap_manager: CPUSwapManager,
    paged_manager: PagedKVCacheManager,
    allocator: BlockAllocator,
) -> None:
    """swap_in on an unknown seq_id must raise KeyError."""
    with pytest.raises(KeyError):
        swap_manager.swap_in("nonexistent", paged_manager, allocator)


# ── Test 8: stats structure ───────────────────────────────────────────────────


def test_stats_structure(
    swap_manager: CPUSwapManager,
    paged_manager: PagedKVCacheManager,
    allocator: BlockAllocator,
) -> None:
    """stats() must reflect one swapped-out sequence correctly."""
    allocator.allocate("seq1", 1)
    device_block = allocator.get_blocks("seq1")[0]
    paged_manager.write_kv("seq1", 0, 0, torch.ones(4, 16), torch.ones(4, 16))
    allocator._blocks[device_block].tokens_used = 1

    device_blocks = allocator.get_blocks("seq1")
    swap_manager.swap_out("seq1", device_blocks, paged_manager, allocator)

    s = swap_manager.stats()

    assert s["swapped_sequences"] == 1, (
        f"Expected 1 swapped sequence, got {s['swapped_sequences']}"
    )
    assert s["total_swap_outs"] == 1, (
        f"Expected total_swap_outs=1, got {s['total_swap_outs']}"
    )
    assert "seq1" in s["swapped_seq_ids"], (
        f"'seq1' must appear in swapped_seq_ids, got {s['swapped_seq_ids']}"
    )
    # Verify all required keys are present
    required_keys = {
        "num_cpu_blocks", "free_cpu_blocks", "swapped_sequences",
        "total_swap_outs", "total_swap_ins",
        "total_swap_out_tokens", "total_swap_in_tokens", "swapped_seq_ids",
    }
    assert required_keys.issubset(s.keys()), (
        f"Missing keys in stats(): {required_keys - s.keys()}"
    )
