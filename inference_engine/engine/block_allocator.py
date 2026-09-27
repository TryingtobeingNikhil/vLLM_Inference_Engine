"""
engine/block_allocator.py — Phase 6: Block Allocator (+ automatic prefix caching).

Manages a fixed pool of logical KV-cache blocks.  Each block holds
``block_size`` token slots.  Sequences are assigned blocks as they grow; the
list of block ids a sequence owns is its *block table* — the indirection that
lets a sequence's KV live in non-contiguous physical memory.

Prefix caching
--------------
When ``enable_prefix_caching`` is on, every *full* block whose K/V has been
computed is registered under a content hash::

    hash(block_i) = hash((hash(block_{i-1}), tokens_in_block_i))

Chaining the parent hash means a hash identifies the entire token prefix up
to and including that block — equivalent to a node in a radix tree whose
edges are blocks of ``block_size`` tokens.  A new request walks its prompt
block by block; every hash hit is a block it can share instead of
recomputing.

Sharing is safe without copy-on-write because only *full* blocks are ever
shared, and a sequence only ever writes into its last, partially-filled
block (which is private to it).

Block life cycle::

    free ──allocate──► in use (ref_count ≥ 1) ──free (ref→0)──┬──► free       (no hash)
                          ▲                                    └──► evictable (hashed:
                          └──────── prefix-cache hit ◄─────────┘     contents still valid)

Evictable blocks count as free capacity; they are recycled in LRU order only
when no truly free block is left, so popular prefixes (system prompts,
few-shot examples) stay cached for as long as memory allows.

No torch dependency.  No async.  Thread-safe via threading.Lock.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Optional, Sequence as Seq


# ── Exceptions ────────────────────────────────────────────────────────────────

class BlockAllocatorError(Exception):
    """Base exception for all BlockAllocator errors."""


class OutOfBlocksError(BlockAllocatorError):
    """Raised when a block allocation request cannot be satisfied."""

    def __init__(self, requested: int, available: int) -> None:
        super().__init__(
            f"Cannot allocate {requested} blocks — only {available} free"
        )
        self.requested = requested
        self.available = available


class BlockNotFoundError(BlockAllocatorError):
    """Raised when a block_id is not present in the pool."""

    def __init__(self, block_id: int) -> None:
        super().__init__(f"Block {block_id} not found in allocator")
        self.block_id = block_id


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class Block:
    """A single logical KV-cache block."""

    block_id: int
    block_size: int                           # max tokens this block can hold
    tokens_used: int = 0                      # how many token slots are filled
    ref_count: int = 0                        # sequences referencing this block
    seq_id: Optional[str] = None              # first owner (informational)
    is_dirty: bool = False                    # True once written to
    block_hash: Optional[int] = None          # prefix-cache key (full blocks only)
    allocated_at: float = field(default_factory=time.perf_counter)

    def is_free(self) -> bool:
        """Return True when no sequence references the block."""
        return self.ref_count == 0

    def is_full(self) -> bool:
        """Return True when all token slots are occupied."""
        return self.tokens_used >= self.block_size

    def remaining_slots(self) -> int:
        """Number of free token slots in this block."""
        return self.block_size - self.tokens_used


def hash_block_tokens(parent_hash: Optional[int], token_ids: Seq[int]) -> int:
    """Content hash of one full block, chained to its prefix."""
    return hash((parent_hash, tuple(token_ids)))


# ── BlockAllocator ────────────────────────────────────────────────────────────

class BlockAllocator:
    """Thread-safe fixed-size block pool manager.

    Parameters
    ----------
    num_blocks:
        Total number of blocks in the pool (e.g. 256).
    block_size:
        Number of token slots per block (e.g. 16).
    enable_prefix_caching:
        Share full, content-identical blocks between sequences.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_prefix_caching: bool = False,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than zero")
        if block_size <= 0:
            raise ValueError("block_size must be greater than zero")

        self._num_blocks = num_blocks
        self._block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching

        # block_id → Block
        self._blocks: dict[int, Block] = {
            i: Block(block_id=i, block_size=block_size)
            for i in range(num_blocks)
        }
        # Blocks with no useful content, FIFO.
        self._free_block_ids: deque[int] = deque(range(num_blocks))
        # ref_count == 0 but still holding hashed (reusable) KV, LRU order.
        self._evictable: OrderedDict[int, None] = OrderedDict()
        self._hash_to_block: dict[int, int] = {}

        # seq_id → [block_id, ...] in logical order (the block table)
        self._seq_to_blocks: dict[str, list[int]] = {}
        # seq_id → hash chain of the seq's leading full blocks
        self._seq_hashes: dict[str, list[int]] = {}

        # Cumulative counters
        self._total_allocated: int = 0
        self._total_freed: int = 0
        self._total_evictions: int = 0
        self._cache_query_tokens: int = 0
        self._cache_hit_tokens: int = 0
        self._cache_evicted_blocks: int = 0

        self._lock = threading.Lock()

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self._block_size - 1) // self._block_size

    # ── Internal block recycling (caller holds the lock) ──────────────────────

    def _num_available_locked(self) -> int:
        return len(self._free_block_ids) + len(self._evictable)

    def _take_block_locked(self, seq_id: str, t_now: float) -> int:
        if self._free_block_ids:
            block_id = self._free_block_ids.popleft()
        else:
            # Recycle the least-recently-used cached block.
            block_id, _ = self._evictable.popitem(last=False)
            blk = self._blocks[block_id]
            if blk.block_hash is not None:
                self._hash_to_block.pop(blk.block_hash, None)
                blk.block_hash = None
            self._cache_evicted_blocks += 1
        blk = self._blocks[block_id]
        blk.seq_id = seq_id
        blk.ref_count = 1
        blk.is_dirty = False
        blk.tokens_used = 0
        blk.allocated_at = t_now
        self._total_allocated += 1
        return block_id

    def _release_block_locked(self, block_id: int) -> None:
        blk = self._blocks[block_id]
        blk.ref_count -= 1
        if blk.ref_count > 0:
            return
        blk.ref_count = 0
        blk.seq_id = None
        self._total_freed += 1
        if blk.block_hash is not None and self.enable_prefix_caching:
            # Keep the contents: a future request may hit this prefix.
            self._evictable[block_id] = None
        else:
            blk.block_hash = None
            blk.tokens_used = 0
            blk.is_dirty = False
            self._free_block_ids.append(block_id)

    # ── Core allocation ───────────────────────────────────────────────────────

    def allocate(self, seq_id: str, num_blocks: int = 1) -> list[int]:
        """Append *num_blocks* fresh blocks to *seq_id*'s block table.

        Raises
        ------
        OutOfBlocksError
            If the pool does not have *num_blocks* free blocks.
        """
        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than zero")

        with self._lock:
            available = self._num_available_locked()
            if num_blocks > available:
                raise OutOfBlocksError(requested=num_blocks, available=available)

            t_now = time.perf_counter()
            allocated_ids = [self._take_block_locked(seq_id, t_now) for _ in range(num_blocks)]
            self._seq_to_blocks.setdefault(seq_id, []).extend(allocated_ids)
            return allocated_ids

    def allocate_for_tokens(
        self,
        seq_id: str,
        token_ids: Seq[int],
        watermark_blocks: int = 0,
        count_stats: bool = True,
    ) -> int:
        """Give a new sequence enough blocks for all of *token_ids*.

        Leading full blocks already in the prefix cache are shared rather than
        allocated.  At most ``(len(token_ids) - 1) // block_size`` blocks are
        matched so at least one token is always computed — the model needs a
        forward pass over the final token to produce next-token logits.

        The operation is all-or-nothing: if fewer than
        ``needed + watermark_blocks`` blocks are available nothing changes and
        OutOfBlocksError is raised.  The watermark keeps a little headroom so
        running sequences can grow without immediately preempting.

        Returns the number of tokens whose KV is already cached.
        """
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        bs = self._block_size

        with self._lock:
            if seq_id in self._seq_to_blocks:
                raise BlockAllocatorError(f"seq_id={seq_id!r} already has blocks")

            hit_blocks: list[int] = []
            hashes: list[int] = []
            if self.enable_prefix_caching:
                parent: Optional[int] = None
                max_hits = (len(token_ids) - 1) // bs
                for i in range(max_hits):
                    h = hash_block_tokens(parent, token_ids[i * bs:(i + 1) * bs])
                    block_id = self._hash_to_block.get(h)
                    if block_id is None:
                        break
                    hit_blocks.append(block_id)
                    hashes.append(h)
                    parent = h

            num_new = self.blocks_needed(len(token_ids)) - len(hit_blocks)
            # Hits currently sitting in the evictable pool stop being available.
            evictable_hits = sum(1 for b in hit_blocks if b in self._evictable)
            available = self._num_available_locked() - evictable_hits
            if num_new + watermark_blocks > available:
                raise OutOfBlocksError(
                    requested=num_new + watermark_blocks, available=available
                )

            table: list[int] = []
            for block_id in hit_blocks:
                blk = self._blocks[block_id]
                if blk.ref_count == 0:
                    self._evictable.pop(block_id, None)
                    blk.seq_id = seq_id
                blk.ref_count += 1
                table.append(block_id)
            t_now = time.perf_counter()
            table.extend(self._take_block_locked(seq_id, t_now) for _ in range(num_new))

            self._seq_to_blocks[seq_id] = table
            self._seq_hashes[seq_id] = hashes
            num_cached = len(hit_blocks) * bs
            if self.enable_prefix_caching and count_stats:
                self._cache_query_tokens += len(token_ids)
                self._cache_hit_tokens += num_cached
            self._set_token_count_locked(table, num_cached)
            return num_cached

    def ensure_capacity(self, seq_id: str, total_tokens: int) -> list[int]:
        """Grow *seq_id*'s block table so it can hold *total_tokens* tokens.

        All-or-nothing.  Returns the newly allocated block ids.
        """
        with self._lock:
            table = self._seq_to_blocks.get(seq_id)
            if table is None:
                raise BlockAllocatorError(f"No blocks allocated for seq_id={seq_id!r}")
            extra = self.blocks_needed(total_tokens) - len(table)
            if extra <= 0:
                return []
            available = self._num_available_locked()
            if extra > available:
                raise OutOfBlocksError(requested=extra, available=available)
            t_now = time.perf_counter()
            new_ids = [self._take_block_locked(seq_id, t_now) for _ in range(extra)]
            table.extend(new_ids)
            return new_ids

    def register_computed_blocks(
        self, seq_id: str, token_ids: Seq[int], num_computed_tokens: int
    ) -> None:
        """Publish newly *full and computed* blocks to the prefix cache."""
        if not self.enable_prefix_caching:
            return
        bs = self._block_size
        with self._lock:
            table = self._seq_to_blocks.get(seq_id)
            if table is None:
                return
            chain = self._seq_hashes.setdefault(seq_id, [])
            num_full = min(num_computed_tokens // bs, len(table))
            for i in range(len(chain), num_full):
                parent = chain[-1] if chain else None
                h = hash_block_tokens(parent, token_ids[i * bs:(i + 1) * bs])
                chain.append(h)
                blk = self._blocks[table[i]]
                if blk.block_hash is None and h not in self._hash_to_block:
                    blk.block_hash = h
                    self._hash_to_block[h] = blk.block_id

    def free(self, seq_id: str) -> int:
        """Release ALL blocks referenced by *seq_id*.

        Shared blocks stay alive for their other users.  Blocks are released
        tail-first so that, among cached blocks, a sequence's *prefix* is the
        last thing the LRU evicts.

        Returns the number of blocks the sequence referenced (0 if none).
        """
        with self._lock:
            block_ids = self._seq_to_blocks.pop(seq_id, [])
            self._seq_hashes.pop(seq_id, None)
            for block_id in reversed(block_ids):
                self._release_block_locked(block_id)
            return len(block_ids)

    def free_block(self, block_id: int) -> None:
        """Forcibly free a single block by *block_id* (drops cached contents).

        Raises
        ------
        BlockNotFoundError
            If *block_id* is not a valid pool block.
        """
        with self._lock:
            if block_id not in self._blocks:
                raise BlockNotFoundError(block_id)

            blk = self._blocks[block_id]
            if blk.is_free():
                return

            # Remove from every table that references it.
            for sid in list(self._seq_to_blocks):
                table = self._seq_to_blocks[sid]
                if block_id in table:
                    table.remove(block_id)
                    if not table:
                        del self._seq_to_blocks[sid]
                        self._seq_hashes.pop(sid, None)

            if blk.block_hash is not None:
                self._hash_to_block.pop(blk.block_hash, None)
                blk.block_hash = None
            blk.ref_count = 0
            blk.seq_id = None
            blk.tokens_used = 0
            blk.is_dirty = False
            self._free_block_ids.append(block_id)
            self._total_freed += 1

    def write_token(self, seq_id: str, count: int = 1) -> list[int]:
        """Record that *count* tokens were written to *seq_id*'s current block.

        Handles overflow: when the current (last) block fills up, a new block
        is automatically allocated.

        Returns
        -------
        list[int]
            Block ids that were newly allocated during this call (empty if no
            new block was needed).

        Raises
        ------
        BlockAllocatorError
            If *seq_id* has no blocks allocated yet.
        OutOfBlocksError
            If a new block is needed but the pool is exhausted.
        """
        if count < 0:
            raise ValueError("count must be non-negative")
        if count == 0:
            return []

        with self._lock:
            block_ids = self._seq_to_blocks.get(seq_id)
            if not block_ids:
                raise BlockAllocatorError(
                    f"No blocks allocated for seq_id={seq_id!r}"
                )

            current_tokens = sum(self._blocks[bid].tokens_used for bid in block_ids)
            target_tokens = current_tokens + count
            extra_blocks = max(0, self.blocks_needed(target_tokens) - len(block_ids))
            available = self._num_available_locked()
            if extra_blocks > available:
                raise OutOfBlocksError(requested=extra_blocks, available=available)

            t_now = time.perf_counter()
            newly_allocated = [
                self._take_block_locked(seq_id, t_now) for _ in range(extra_blocks)
            ]
            block_ids.extend(newly_allocated)
            self._set_token_count_locked(block_ids, target_tokens)
            return newly_allocated

    def set_token_count(self, seq_id: str, count: int) -> None:
        """Set exact filled-token accounting across a sequence's blocks."""
        if count < 0:
            raise ValueError("count must be non-negative")

        with self._lock:
            block_ids = self._seq_to_blocks.get(seq_id)
            if not block_ids:
                raise BlockAllocatorError(
                    f"No blocks allocated for seq_id={seq_id!r}"
                )
            capacity = len(block_ids) * self._block_size
            if count > capacity:
                raise ValueError(
                    f"Token count {count} exceeds sequence capacity {capacity}"
                )
            self._set_token_count_locked(block_ids, count)

    def _set_token_count_locked(self, block_ids: list[int], count: int) -> None:
        """Distribute *count* filled slots in block order; caller holds lock."""
        remaining = count
        for block_id in block_ids:
            tokens_used = min(remaining, self._block_size)
            blk = self._blocks[block_id]
            blk.tokens_used = tokens_used
            blk.is_dirty = tokens_used > 0
            remaining -= tokens_used

    def get_blocks(self, seq_id: str) -> list[int]:
        """Return the block table of *seq_id* (logical order)."""
        with self._lock:
            return list(self._seq_to_blocks.get(seq_id, []))

    def get_block(self, block_id: int) -> Block:
        """Return the Block object for *block_id*.

        Raises
        ------
        BlockNotFoundError
            If *block_id* is not in the pool.
        """
        with self._lock:
            if block_id not in self._blocks:
                raise BlockNotFoundError(block_id)
            return self._blocks[block_id]

    # ── Eviction ──────────────────────────────────────────────────────────────

    def evict_lru(self, n: int = 1) -> list[str]:
        """Evict the *n* sequences whose last block was allocated earliest (LRU).

        Uses the existing ``free()`` method internally.  Returns the list of
        evicted seq_ids.  If fewer than *n* sequences are active, all are
        evicted.
        """
        with self._lock:
            candidates = sorted(
                self._seq_to_blocks,
                key=lambda sid: self._blocks[self._seq_to_blocks[sid][-1]].allocated_at,
            )
        evicted: list[str] = []
        for sid in candidates[:n]:
            self.free(sid)
            with self._lock:
                self._total_evictions += 1
            evicted.append(sid)
        return evicted

    def evict_largest(self, n: int = 1) -> list[str]:
        """Evict the *n* sequences holding the most blocks.

        Tie-break: oldest last-block allocated_at wins (evicted first).
        Returns the list of evicted seq_ids.
        """
        with self._lock:
            def _sort_key(sid: str):
                ids = self._seq_to_blocks[sid]
                return (-len(ids), self._blocks[ids[-1]].allocated_at)

            candidates = sorted(self._seq_to_blocks, key=_sort_key)
        evicted: list[str] = []
        for sid in candidates[:n]:
            self.free(sid)
            with self._lock:
                self._total_evictions += 1
            evicted.append(sid)
        return evicted

    # ── Query ──────────────────────────────────────────────────────────────────

    def num_free_blocks(self) -> int:
        """Blocks available for allocation (truly free + evictable cached)."""
        with self._lock:
            return self._num_available_locked()

    def num_used_blocks(self) -> int:
        """Number of blocks referenced by at least one sequence."""
        with self._lock:
            return self._num_blocks - self._num_available_locked()

    def num_cached_blocks(self) -> int:
        """Unreferenced blocks whose contents are kept for prefix hits."""
        with self._lock:
            return len(self._evictable)

    def num_blocks_for_seq(self, seq_id: str) -> int:
        """Number of blocks in *seq_id*'s block table."""
        with self._lock:
            return len(self._seq_to_blocks.get(seq_id, []))

    def num_tokens_for_seq(self, seq_id: str) -> int:
        """Number of filled token slots recorded for *seq_id*."""
        with self._lock:
            return sum(
                self._blocks[block_id].tokens_used
                for block_id in self._seq_to_blocks.get(seq_id, [])
            )

    def active_sequences(self) -> list[str]:
        """List of seq_ids that currently hold at least one block."""
        with self._lock:
            return list(self._seq_to_blocks.keys())

    def prefix_cache_hit_rate(self) -> float:
        with self._lock:
            if self._cache_query_tokens == 0:
                return 0.0
            return self._cache_hit_tokens / self._cache_query_tokens

    def stats(self, include_per_sequence: bool = True) -> dict:
        """Return a snapshot of allocator state.

        Returns
        -------
        dict with keys:
            num_blocks, block_size, free_blocks, used_blocks,
            active_sequences, total_allocated, total_freed,
            total_evictions, utilization, per_sequence, and prefix-cache
            counters (cached_blocks, prefix_cache_*).
        """
        with self._lock:
            free = self._num_available_locked()
            used = self._num_blocks - free
            utilization = used / self._num_blocks if self._num_blocks > 0 else 0.0
            per_sequence = {}
            if include_per_sequence:
                per_sequence = {
                    sid: {
                        "num_blocks": len(ids),
                        "tokens_used": sum(self._blocks[bid].tokens_used for bid in ids),
                    }
                    for sid, ids in self._seq_to_blocks.items()
                }
            hit_rate = (
                self._cache_hit_tokens / self._cache_query_tokens
                if self._cache_query_tokens else 0.0
            )
            return {
                "num_blocks": self._num_blocks,
                "block_size": self._block_size,
                "free_blocks": free,
                "used_blocks": used,
                "active_sequences": len(self._seq_to_blocks),
                "total_allocated": self._total_allocated,
                "total_freed": self._total_freed,
                "total_evictions": self._total_evictions,
                "utilization": utilization,
                "per_sequence": per_sequence,
                "prefix_caching_enabled": self.enable_prefix_caching,
                "cached_blocks": len(self._evictable),
                "prefix_cache_query_tokens": self._cache_query_tokens,
                "prefix_cache_hit_tokens": self._cache_hit_tokens,
                "prefix_cache_hit_rate": hit_rate,
                "prefix_cache_evicted_blocks": self._cache_evicted_blocks,
            }
