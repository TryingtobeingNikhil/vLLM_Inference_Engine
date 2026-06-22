"""
tests/test_block_allocator.py — Phase 6 unit tests for BlockAllocator.

10 synchronous pytest tests — no async, no model, no torch.
"""

from __future__ import annotations

import pytest

from inference_engine.engine.block_allocator import (
    BlockAllocator,
    BlockAllocatorError,
    OutOfBlocksError,
)


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def allocator() -> BlockAllocator:
    """Small 16-block pool with 16-token blocks for fast, predictable tests."""
    return BlockAllocator(num_blocks=16, block_size=16)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_initial_state(allocator: BlockAllocator) -> None:
    """Fresh allocator: all blocks free, no active sequences."""
    assert allocator.num_free_blocks() == 16
    assert allocator.num_used_blocks() == 0
    assert allocator.active_sequences() == []


def test_allocate_single_sequence(allocator: BlockAllocator) -> None:
    """Allocating 2 blocks for one sequence reduces free pool and is retrievable."""
    ids = allocator.allocate("seq1", 2)
    assert len(ids) == 2
    assert allocator.num_free_blocks() == 14
    assert allocator.num_used_blocks() == 2
    assert set(allocator.get_blocks("seq1")) == set(ids)


def test_allocate_out_of_blocks_raises(allocator: BlockAllocator) -> None:
    """Allocating beyond pool capacity raises OutOfBlocksError."""
    allocator.allocate("seq1", 16)  # takes all blocks
    with pytest.raises(OutOfBlocksError):
        allocator.allocate("seq2", 1)


def test_free_returns_blocks_to_pool(allocator: BlockAllocator) -> None:
    """free() releases all blocks owned by a sequence back to the pool."""
    allocator.allocate("seq1", 4)
    freed = allocator.free("seq1")
    assert freed == 4
    assert allocator.num_free_blocks() == 16


def test_write_token_fills_block(allocator: BlockAllocator) -> None:
    """write_token() accumulates tokens_used; block reports is_full when filled."""
    allocator.allocate("seq1", 1)
    allocator.write_token("seq1", 16)  # fill the block exactly
    bid = allocator.get_blocks("seq1")[0]
    blk = allocator.get_block(bid)
    assert blk.tokens_used == 16
    assert blk.is_full() is True
    assert blk.is_dirty is True


def test_write_token_auto_allocates_new_block(allocator: BlockAllocator) -> None:
    """write_token() auto-allocates a new block when the current one overflows."""
    allocator.allocate("seq1", 1)
    allocator.write_token("seq1", 16)       # fills first block exactly
    new_blocks = allocator.write_token("seq1", 1)  # must spill into a new block
    assert len(new_blocks) == 1
    assert allocator.num_blocks_for_seq("seq1") == 2


def test_evict_lru(allocator: BlockAllocator) -> None:
    """evict_lru() removes the sequence whose last block was allocated earliest."""
    allocator.allocate("seq1", 2)
    # Small sleep to guarantee different allocated_at timestamps
    import time; time.sleep(0.001)
    allocator.allocate("seq2", 2)

    evicted = allocator.evict_lru(1)
    assert evicted == ["seq1"]
    assert "seq1" not in allocator.active_sequences()
    # seq1's 2 blocks freed; seq2 still holds 2 → 14 free
    assert allocator.num_free_blocks() == 14


def test_evict_largest(allocator: BlockAllocator) -> None:
    """evict_largest() removes the sequence holding the most blocks."""
    allocator.allocate("seq1", 1)
    allocator.allocate("seq2", 4)

    evicted = allocator.evict_largest(1)
    assert evicted == ["seq2"]
    assert "seq2" not in allocator.active_sequences()
    # seq2's 4 blocks freed; seq1 still holds 1 → 15 free
    assert allocator.num_free_blocks() == 15


def test_stats_structure(allocator: BlockAllocator) -> None:
    """stats() returns correctly shaped dict with accurate counts."""
    allocator.allocate("seq1", 3)
    allocator.write_token("seq1", 5)

    s = allocator.stats()
    assert s["free_blocks"] == 13
    assert s["used_blocks"] == 3
    assert s["utilization"] == pytest.approx(3 / 16, rel=1e-5)
    assert "seq1" in s["per_sequence"]
    # tokens_used should be 5 (written to first block)
    assert s["per_sequence"]["seq1"]["tokens_used"] == 5
    assert s["per_sequence"]["seq1"]["num_blocks"] == 3


def test_total_evictions_counter(allocator: BlockAllocator) -> None:
    """total_evictions counter increments once per evicted sequence."""
    allocator.allocate("seq1", 2)
    allocator.allocate("seq2", 2)
    allocator.evict_lru(2)
    assert allocator.stats()["total_evictions"] == 2
