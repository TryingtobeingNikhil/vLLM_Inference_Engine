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


def test_write_token_fills_preallocated_blocks_in_order(allocator: BlockAllocator) -> None:
    """Token accounting must not leave zero-filled holes before used blocks."""
    block_ids = allocator.allocate("seq1", 3)
    allocator.write_token("seq1", 20)

    assert [allocator.get_block(bid).tokens_used for bid in block_ids] == [16, 4, 0]
    assert allocator.stats()["total_allocated"] == 3


def test_free_block_is_idempotent(allocator: BlockAllocator) -> None:
    """Freeing an already-free block must not duplicate it in the free pool."""
    block_id = allocator.allocate("seq1", 1)[0]
    allocator.free_block(block_id)
    allocator.free_block(block_id)

    assert allocator.num_free_blocks() == 16


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


# ── Prefix caching ────────────────────────────────────────────────────────────


def _prefix_allocator(num_blocks: int = 8) -> BlockAllocator:
    return BlockAllocator(num_blocks=num_blocks, block_size=4, enable_prefix_caching=True)


def test_allocate_for_tokens_without_cache_hit() -> None:
    a = _prefix_allocator()
    cached = a.allocate_for_tokens("s1", list(range(10)))
    assert cached == 0
    assert a.num_blocks_for_seq("s1") == 3          # ceil(10 / 4)


def test_prefix_hit_shares_full_blocks() -> None:
    a = _prefix_allocator()
    prompt = list(range(10))
    a.allocate_for_tokens("s1", prompt)
    a.register_computed_blocks("s1", prompt, num_computed_tokens=10)

    # Same first 8 tokens (two full blocks), different tail.
    cached = a.allocate_for_tokens("s2", prompt[:8] + [99, 98])
    assert cached == 8
    t1, t2 = a.get_blocks("s1"), a.get_blocks("s2")
    assert t2[:2] == t1[:2], "full prefix blocks must be shared"
    assert t2[2] != t1[2], "the partial tail block stays private"
    assert a.get_block(t1[0]).ref_count == 2
    assert a.stats()["prefix_cache_hit_tokens"] == 8


def test_prefix_match_always_leaves_one_token_to_compute() -> None:
    a = _prefix_allocator()
    prompt = list(range(8))                          # exactly two full blocks
    a.allocate_for_tokens("s1", prompt)
    a.register_computed_blocks("s1", prompt, 8)
    # An identical prompt may reuse only the first block: the last token must
    # run through the model to produce next-token logits.
    assert a.allocate_for_tokens("s2", prompt) == 4


def test_hash_chain_depends_on_whole_prefix() -> None:
    a = _prefix_allocator()
    p1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    a.allocate_for_tokens("s1", p1)
    a.register_computed_blocks("s1", p1, 9)
    # Block 2 has the same tokens as p1's block 2, but a different block 1:
    p2 = [9, 9, 9, 9, 5, 6, 7, 8, 9]
    assert a.allocate_for_tokens("s2", p2) == 0


def test_freed_blocks_stay_cached_until_evicted() -> None:
    a = _prefix_allocator(num_blocks=4)
    prompt = list(range(9))
    a.allocate_for_tokens("s1", prompt)
    a.register_computed_blocks("s1", prompt, 9)
    a.free("s1")
    assert a.num_free_blocks() == 4, "cached blocks still count as free capacity"
    assert a.num_cached_blocks() == 2

    assert a.allocate_for_tokens("s2", prompt) == 8   # hit after free
    a.free("s2")

    # Fill the pool with unrelated work → cached blocks get recycled (LRU).
    a.allocate("s3", 4)
    assert a.num_cached_blocks() == 0
    a.free("s3")
    assert a.allocate_for_tokens("s4", prompt) == 0


def test_allocate_for_tokens_is_all_or_nothing() -> None:
    a = _prefix_allocator(num_blocks=4)
    a.allocate("busy", 2)
    with pytest.raises(OutOfBlocksError):
        a.allocate_for_tokens("s1", list(range(12)))  # needs 3, only 2 free
    assert a.num_blocks_for_seq("s1") == 0
    assert a.num_free_blocks() == 2
    with pytest.raises(OutOfBlocksError):             # watermark counts too
        a.allocate_for_tokens("s1", list(range(8)), watermark_blocks=1)


def test_ensure_capacity_grows_table() -> None:
    a = _prefix_allocator()
    a.allocate_for_tokens("s1", list(range(3)))
    assert a.ensure_capacity("s1", 4) == []
    new = a.ensure_capacity("s1", 9)
    assert len(new) == 2 and a.num_blocks_for_seq("s1") == 3
    with pytest.raises(OutOfBlocksError):
        a.ensure_capacity("s1", 100)
    assert a.num_blocks_for_seq("s1") == 3


def test_shared_block_survives_one_owner_freeing() -> None:
    a = _prefix_allocator()
    prompt = list(range(9))
    a.allocate_for_tokens("s1", prompt)
    a.register_computed_blocks("s1", prompt, 9)
    a.allocate_for_tokens("s2", prompt)
    shared = a.get_blocks("s1")[0]
    a.free("s1")
    blk = a.get_block(shared)
    assert blk.ref_count == 1 and not blk.is_free()
    a.free("s2")
    assert a.num_free_blocks() == 8
