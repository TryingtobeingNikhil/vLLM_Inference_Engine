"""
engine/cpu_swap_manager.py — Phase 9: CPU-side KV cache staging pool.

Maintains a CPU tensor pool that mirrors the structure of the GPU/MPS paged
pool.  Provides:

  swap_out()  — copy blocks from device pool to CPU pool, free device blocks.
  swap_in()   — allocate new device blocks, copy from CPU pool back, free CPU
                slots.

Used by the ContinuousBatchingScheduler to avoid killing sequences under
memory pressure: instead of failing the incoming request with OOM, the
scheduler swaps out a large running sequence to CPU, frees device blocks,
and retries the allocation.

Thread-safe via threading.Lock.  No async anywhere.
No vLLM / TGI / TensorRT-LLM internals.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
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
        Indices into CPUSwapManager.cpu_key_pool / cpu_value_pool that hold
        the copied KV data (in the same order as the original device blocks).
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

        # ── CPU tensor pool — always pinned to CPU regardless of model device ─
        # Shape: [num_cpu_blocks, block_size, num_layers, num_kv_heads, head_dim]
        self.cpu_key_pool: torch.Tensor = torch.zeros(
            [
                num_cpu_blocks,
                block_size,
                kv_cache_config.num_layers,
                kv_cache_config.num_kv_heads,
                kv_cache_config.head_dim,
            ],
            dtype=kv_cache_config.dtype,
            device="cpu",
        )
        self.cpu_value_pool: torch.Tensor = torch.zeros_like(self.cpu_key_pool)

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

    # ── swap_out ──────────────────────────────────────────────────────────────

    def swap_out(
        self,
        seq_id: str,
        device_block_ids: list[int],
        paged_kv_cache: "PagedKVCacheManager",
        block_allocator: "BlockAllocator",
    ) -> SwappedSequence:
        """Copy *seq_id*'s KV blocks from the device pool to CPU, then free device blocks.

        Parameters
        ----------
        seq_id:
            The sequence whose KV cache will be moved to CPU.
        device_block_ids:
            List of device-side block IDs currently held by this sequence
            (in allocation order).  Typically obtained via
            ``block_allocator.get_blocks(seq_id)``.
        paged_kv_cache:
            The live device-side KV pool.
        block_allocator:
            The device-side block allocator.

        Returns
        -------
        SwappedSequence
            Metadata record for the swapped-out sequence.

        Raises
        ------
        CPUSwapError
            If there are not enough free CPU blocks to hold all device blocks.
        """
        num_needed = len(device_block_ids)

        with self._lock:
            if num_needed > len(self._free_cpu_block_ids):
                raise CPUSwapError(
                    requested=num_needed,
                    available=len(self._free_cpu_block_ids),
                )

            # Claim CPU block IDs while holding the lock
            cpu_block_ids: list[int] = []
            for _ in range(num_needed):
                cpu_block_ids.append(self._free_cpu_block_ids.pop(0))

        # Copy device → CPU (outside the lock — tensor ops can run freely)
        num_layers = self.kv_cache_config.num_layers
        for dev_bid, cpu_bid in zip(device_block_ids, cpu_block_ids):
            for layer_idx in range(num_layers):
                # Shape of each slice: [block_size, num_kv_heads, head_dim]
                self.cpu_key_pool[cpu_bid, :, layer_idx].copy_(
                    paged_kv_cache.key_pool[dev_bid, :, layer_idx]
                )
                self.cpu_value_pool[cpu_bid, :, layer_idx].copy_(
                    paged_kv_cache.value_pool[dev_bid, :, layer_idx]
                )

        # Count total filled tokens before zeroing/freeing
        num_tokens: int = sum(
            block_allocator._blocks[bid].tokens_used
            for bid in device_block_ids
        )

        # Zero device pool slots AFTER copying data (spec requirement)
        paged_kv_cache.clear_sequence(seq_id)
        # Release device blocks
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

        Parameters
        ----------
        seq_id:
            The sequence to restore.
        paged_kv_cache:
            The live device-side KV pool.
        block_allocator:
            The device-side block allocator.

        Returns
        -------
        list[int]
            Newly allocated device block IDs (in allocation order).

        Raises
        ------
        KeyError
            If *seq_id* is not currently swapped out.
        OutOfBlocksError
            If the device pool still has insufficient space.  The caller
            decides what to do — this manager's state is unchanged.
        """
        if seq_id not in self._swapped:
            raise KeyError(f"seq_id={seq_id!r} is not currently swapped out")

        swapped = self._swapped[seq_id]

        # Try to claim device blocks — let OutOfBlocksError propagate
        device_block_ids: list[int] = block_allocator.allocate(
            seq_id, swapped.original_num_blocks
        )

        # Copy CPU → device
        num_layers = self.kv_cache_config.num_layers
        for dev_bid, cpu_bid in zip(device_block_ids, swapped.cpu_block_ids):
            for layer_idx in range(num_layers):
                paged_kv_cache.key_pool[dev_bid, :, layer_idx].copy_(
                    self.cpu_key_pool[cpu_bid, :, layer_idx]
                )
                paged_kv_cache.value_pool[dev_bid, :, layer_idx].copy_(
                    self.cpu_value_pool[cpu_bid, :, layer_idx]
                )

        # Restore tokens_used on each device block
        # Full blocks get block_size tokens; the last block gets the remainder.
        remaining = swapped.num_tokens
        for i, dev_bid in enumerate(device_block_ids):
            if remaining >= self.block_size:
                block_allocator._blocks[dev_bid].tokens_used = self.block_size
                remaining -= self.block_size
            else:
                block_allocator._blocks[dev_bid].tokens_used = remaining
                remaining = 0
            block_allocator._blocks[dev_bid].is_dirty = True

        with self._lock:
            # Return CPU blocks to the free pool
            self._free_cpu_block_ids.extend(swapped.cpu_block_ids)
            del self._swapped[seq_id]
            self._total_swap_ins += 1
            self._total_swap_in_tokens += swapped.num_tokens

        return device_block_ids

    # ── Query ─────────────────────────────────────────────────────────────────

    def is_swapped(self, seq_id: str) -> bool:
        """Return True if *seq_id* currently has data staged in CPU memory."""
        return seq_id in self._swapped

    def stats(self) -> dict:
        """Return a snapshot of CPUSwapManager state.

        Returns
        -------
        dict with keys:
            num_cpu_blocks, free_cpu_blocks, swapped_sequences,
            total_swap_outs, total_swap_ins,
            total_swap_out_tokens, total_swap_in_tokens, swapped_seq_ids
        """
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
