"""
engine/cpu_swap_manager.py — Phase 9: CPU-side KV cache staging pool.

Maintains a CPU tensor pool that mirrors the layout of the device paged pool
(``[num_layers, num_cpu_blocks, block_size, num_kv_heads, head_dim]``).
Provides:

  swap_out()  — copy a sequence's blocks from the device pool to CPU, then
                release its device blocks.
  swap_in()   — allocate fresh device blocks, copy the data back, free the
                CPU slots.

Used by the ContinuousBatchingScheduler to preempt a running sequence under
memory pressure without throwing its work away.

Each swap is a single gather + a single device↔host copy covering every layer
and block of the sequence (not one small copy per block per layer).  On CUDA
the CPU pool is allocated in pinned memory so the copies run at full PCIe
bandwidth.

Thread-safe via threading.Lock.  No async anywhere.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from inference_engine.engine.kv_cache_config import KVCacheConfig

if TYPE_CHECKING:
    from inference_engine.engine.block_allocator import BlockAllocator
    from inference_engine.engine.paged_kv_cache import PagedKVCacheManager


# ── Exception ─────────────────────────────────────────────────────────────────


class CPUSwapError(Exception):
    """Raised when the CPU staging pool has insufficient free blocks."""

    def __init__(self, requested: int, available: int) -> None:
        super().__init__(
            f"Cannot swap out — need {requested} CPU blocks, "
            f"only {available} available"
        )
        self.requested = requested
        self.available = available


# ── Dataclass ─────────────────────────────────────────────────────────────────


@dataclass
class SwappedSequence:
    """Metadata for a sequence whose KV blocks have been moved to CPU memory.

    Fields
    ------
    seq_id
        UUID hex string of the swapped-out sequence.
    cpu_block_ids
        Indices into the CPU pools holding the copied KV data (same order as
        the original device block table).
    num_tokens
        Total number of filled token slots across all blocks at swap-out time.
    swapped_at
        ``time.perf_counter()`` timestamp of when swap-out occurred.
    original_num_blocks
        How many device blocks the sequence held before swap-out.  Required by
        swap_in() to re-allocate the same number of device blocks.
    """

    seq_id: str
    cpu_block_ids: list[int]
    num_tokens: int
    swapped_at: float
    original_num_blocks: int


# ── CPUSwapManager ────────────────────────────────────────────────────────────


class CPUSwapManager:
    """CPU-side KV cache staging pool for GPU ↔ CPU swapping.

    Parameters
    ----------
    kv_cache_config:
        Architectural metadata (num_layers, num_kv_heads, head_dim, dtype).
    block_size:
        Number of token slots per block — must match the device pool.
    num_cpu_blocks:
        Total number of CPU-side staging blocks to pre-allocate.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        block_size: int,
        num_cpu_blocks: int,
    ) -> None:
        self.kv_cache_config = kv_cache_config
        self.block_size = block_size
        self.num_cpu_blocks = num_cpu_blocks

        # Pinned host memory makes device↔host copies fast; only meaningful
        # (and only allowed) when a CUDA device is present.
        pin = kv_cache_config.device == "cuda" and torch.cuda.is_available()
        shape = [
            kv_cache_config.num_layers,
            num_cpu_blocks,
            block_size,
            kv_cache_config.num_kv_heads,
            kv_cache_config.head_dim,
        ]
        self.cpu_key_pool: torch.Tensor = torch.zeros(
            shape, dtype=kv_cache_config.dtype, device="cpu", pin_memory=pin
        )
        self.cpu_value_pool: torch.Tensor = torch.zeros(
            shape, dtype=kv_cache_config.dtype, device="cpu", pin_memory=pin
        )

        # Free CPU block index pool
        self._free_cpu_block_ids: list[int] = list(range(num_cpu_blocks))

        # seq_id → SwappedSequence for every currently-swapped sequence
        self._swapped: dict[str, SwappedSequence] = {}

        # Cumulative telemetry counters
        self._total_swap_outs: int = 0
        self._total_swap_ins: int = 0
        self._total_swap_out_tokens: int = 0
        self._total_swap_in_tokens: int = 0

        self._lock = threading.Lock()

    def num_free_blocks(self) -> int:
        with self._lock:
            return len(self._free_cpu_block_ids)

    # ── swap_out ──────────────────────────────────────────────────────────────

    def swap_out(
        self,
        seq_id: str,
        device_block_ids: list[int],
        paged_kv_cache: "PagedKVCacheManager",
        block_allocator: "BlockAllocator",
    ) -> SwappedSequence:
        """Copy *seq_id*'s KV blocks from the device pool to CPU, then free device blocks.

        Raises
        ------
        CPUSwapError
            If there are not enough free CPU blocks to hold all device blocks.
        """
        num_needed = len(device_block_ids)

        with self._lock:
            if seq_id in self._swapped:
                raise ValueError(f"seq_id={seq_id!r} is already swapped out")
            if num_needed > len(self._free_cpu_block_ids):
                raise CPUSwapError(
                    requested=num_needed,
                    available=len(self._free_cpu_block_ids),
                )
            cpu_block_ids = self._free_cpu_block_ids[:num_needed]
            del self._free_cpu_block_ids[:num_needed]

        try:
            if num_needed:
                dev_idx = torch.tensor(device_block_ids, dtype=torch.long,
                                       device=paged_kv_cache.key_pool.device)
                cpu_idx = torch.tensor(cpu_block_ids, dtype=torch.long)
                # One gather over all layers, one device→host copy.
                keys = paged_kv_cache.key_pool.index_select(1, dev_idx).cpu()
                values = paged_kv_cache.value_pool.index_select(1, dev_idx).cpu()
                self.cpu_key_pool[:, cpu_idx] = keys
                self.cpu_value_pool[:, cpu_idx] = values
        except Exception:
            with self._lock:
                self._free_cpu_block_ids.extend(cpu_block_ids)
            raise

        num_tokens = block_allocator.num_tokens_for_seq(seq_id)
        # Release device blocks.  Shared prefix blocks stay alive for their
        # other users; private hashed blocks become evictable cache entries.
        block_allocator.free(seq_id)

        swapped = SwappedSequence(
            seq_id=seq_id,
            cpu_block_ids=cpu_block_ids,
            num_tokens=num_tokens,
            swapped_at=time.perf_counter(),
            original_num_blocks=num_needed,
        )

        with self._lock:
            self._swapped[seq_id] = swapped
            self._total_swap_outs += 1
            self._total_swap_out_tokens += num_tokens

        return swapped

    # ── swap_in ───────────────────────────────────────────────────────────────

    def swap_in(
        self,
        seq_id: str,
        paged_kv_cache: "PagedKVCacheManager",
        block_allocator: "BlockAllocator",
    ) -> list[int]:
        """Restore *seq_id*'s KV blocks from CPU back to the device pool.

        Allocates *original_num_blocks* new device blocks (block IDs may
        differ from the original), copies data from CPU, then frees the CPU
        staging slots.

        Raises
        ------
        KeyError
            If *seq_id* is not currently swapped out.
        OutOfBlocksError
            If the device pool still has insufficient space.  The caller
            decides what to do — this manager's state is unchanged.
        """
        with self._lock:
            swapped = self._swapped.get(seq_id)
        if swapped is None:
            raise KeyError(f"seq_id={seq_id!r} is not currently swapped out")

        # Try to claim device blocks — let OutOfBlocksError propagate
        device_block_ids: list[int] = block_allocator.allocate(
            seq_id, swapped.original_num_blocks
        ) if swapped.original_num_blocks else []

        try:
            if device_block_ids:
                device = paged_kv_cache.key_pool.device
                cpu_idx = torch.tensor(swapped.cpu_block_ids, dtype=torch.long)
                dev_idx = torch.tensor(device_block_ids, dtype=torch.long, device=device)
                keys = self.cpu_key_pool.index_select(1, cpu_idx).to(device)
                values = self.cpu_value_pool.index_select(1, cpu_idx).to(device)
                paged_kv_cache.key_pool[:, dev_idx] = keys
                paged_kv_cache.value_pool[:, dev_idx] = values
        except Exception:
            block_allocator.free(seq_id)
            raise

        if device_block_ids:
            block_allocator.set_token_count(seq_id, swapped.num_tokens)

        with self._lock:
            self._free_cpu_block_ids.extend(swapped.cpu_block_ids)
            del self._swapped[seq_id]
            self._total_swap_ins += 1
            self._total_swap_in_tokens += swapped.num_tokens

        return device_block_ids

    def discard(self, seq_id: str) -> bool:
        """Drop a swapped sequence's CPU copy (e.g. the request was aborted)."""
        with self._lock:
            swapped = self._swapped.pop(seq_id, None)
            if swapped is None:
                return False
            self._free_cpu_block_ids.extend(swapped.cpu_block_ids)
            return True

    # ── Query ─────────────────────────────────────────────────────────────────

    def is_swapped(self, seq_id: str) -> bool:
        """Return True if *seq_id* currently has data staged in CPU memory."""
        with self._lock:
            return seq_id in self._swapped

    def get_swapped_sequence(self, seq_id: str) -> SwappedSequence | None:
        """Return swap metadata for *seq_id*, or None when it is resident."""
        with self._lock:
            return self._swapped.get(seq_id)

    def stats(self) -> dict:
        """Return a snapshot of CPUSwapManager state."""
        with self._lock:
            return {
                "num_cpu_blocks": self.num_cpu_blocks,
                "free_cpu_blocks": len(self._free_cpu_block_ids),
                "swapped_sequences": len(self._swapped),
                "total_swap_outs": self._total_swap_outs,
                "total_swap_ins": self._total_swap_ins,
                "total_swap_out_tokens": self._total_swap_out_tokens,
                "total_swap_in_tokens": self._total_swap_in_tokens,
                "swapped_seq_ids": list(self._swapped.keys()),
            }
