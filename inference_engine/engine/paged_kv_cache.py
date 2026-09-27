"""
engine/paged_kv_cache.py — Phase 7: Paged KV Cache Manager.

Owns the actual torch tensor pool for all KV cache memory.  Uses
BlockAllocator (Phase 6) for block-level bookkeeping.

Layout
------
One key pool and one value pool, each shaped::

    [num_layers, num_blocks, block_size, num_kv_heads, head_dim]

``key_pool[layer]`` is a contiguous ``[num_blocks, block_size, H, D]`` tensor,
so the attention kernel for one layer touches one contiguous region.  A token
at logical position ``p`` of a sequence with block table ``bt`` lives at::

    physical block = bt[p // block_size]
    slot           = bt[p // block_size] * block_size + p % block_size

(the "slot" indexes the pool flattened to ``[num_blocks * block_size, H, D]``).

Hot path (used by the attention kernel every forward pass)
    write_slots()   — scatter a whole step's new K/V into the pool, per layer
    gather()        — assemble K/V for a batch of block tables, per layer

Slow path (tests, debugging, education)
    write_kv() / read_kv_sequence() / read_kv_block() — one token / one
    sequence at a time.

No custom CUDA kernels — standard torch tensor indexing throughout.
"""

from __future__ import annotations

import threading
from typing import Optional

import torch

from inference_engine.engine.block_allocator import BlockAllocator
from inference_engine.engine.kv_cache_config import KVCacheConfig, _DTYPE_BYTES


class PagedKVCacheManager:
    """Pre-allocated paged KV cache pool.

    Parameters
    ----------
    kv_cache_config:
        Architectural metadata (layers, heads, head_dim, dtype, device).
    block_allocator:
        Phase 6 block allocator — passed in, not owned here.
    config:
        Engine config; reads ``kv_block_size`` and ``device``.
    num_blocks:
        Pool size in blocks.  Defaults to the allocator's size.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        block_allocator: BlockAllocator,
        config,
        num_blocks: Optional[int] = None,
    ) -> None:
        self.kv_cache_config = kv_cache_config
        self.block_allocator = block_allocator
        self.config = config

        self.num_layers: int = kv_cache_config.num_layers
        self.num_kv_heads: int = kv_cache_config.num_kv_heads
        self.head_dim: int = kv_cache_config.head_dim
        self.block_size: int = config.kv_block_size
        self.num_blocks: int = num_blocks if num_blocks is not None else block_allocator.num_blocks
        self.device: str = config.device
        self.dtype: torch.dtype = kv_cache_config.dtype

        # ── Single large pre-allocated pool ───────────────────────────────────
        # Allocated ONCE here — never resized during inference.
        shape = [self.num_layers, self.num_blocks, self.block_size,
                 self.num_kv_heads, self.head_dim]
        self.key_pool: torch.Tensor = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.value_pool: torch.Tensor = torch.zeros_like(self.key_pool)

        # seq_id → next global token write cursor (slow-path bookkeeping)
        self._seq_write_cursors: dict[str, int] = {}
        self._lock = threading.Lock()

    # ── Hot path ──────────────────────────────────────────────────────────────

    def layer_caches(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(key, value)`` views for one layer: ``[N, block_size, H, D]``."""
        return self.key_pool[layer_idx], self.value_pool[layer_idx]

    def write_slots(
        self,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Scatter ``keys``/``values`` (``[T, H, D]``) into flat slots ``[T]``."""
        flat_k = self.key_pool[layer_idx].view(-1, self.num_kv_heads, self.head_dim)
        flat_v = self.value_pool[layer_idx].view(-1, self.num_kv_heads, self.head_dim)
        flat_k.index_put_((slot_mapping,), keys.to(self.dtype))
        flat_v.index_put_((slot_mapping,), values.to(self.dtype))

    def gather(
        self, layer_idx: int, block_tables: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K/V for ``block_tables`` (``[B, max_blocks]``).

        Returns tensors shaped ``[B, max_blocks * block_size, H, D]``; slots
        past each sequence's length contain garbage and must be masked.
        """
        b, nb = block_tables.shape
        k = self.key_pool[layer_idx][block_tables]  # [B, nb, bs, H, D]
        v = self.value_pool[layer_idx][block_tables]
        shape = (b, nb * self.block_size, self.num_kv_heads, self.head_dim)
        return k.view(shape), v.view(shape)

    # ── Slow path: single token / single sequence ─────────────────────────────

    def write_kv(
        self,
        seq_id: str,
        layer_idx: int,
        token_position: int,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> None:
        """Write a single token's KV pair (``[H, D]`` each) into the pool."""
        block_idx_in_seq = token_position // self.block_size
        slot_within_block = token_position % self.block_size

        block_ids = self.block_allocator.get_blocks(seq_id)
        physical_block_id = block_ids[block_idx_in_seq]

        self.key_pool[layer_idx, physical_block_id, slot_within_block] = key_tensor
        self.value_pool[layer_idx, physical_block_id, slot_within_block] = value_tensor

        with self._lock:
            cursor = self._seq_write_cursors.get(seq_id, 0)
            self._seq_write_cursors[seq_id] = max(cursor, token_position + 1)

    def read_kv_sequence(
        self,
        seq_id: str,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble all filled KV tensors for a sequence at one layer.

        Returns ``(keys, values)``, each ``[total_filled_tokens, H, D]``.
        Only filled slots (``tokens_used`` on each block) are returned.
        """
        block_ids = self.block_allocator.get_blocks(seq_id)

        key_slices: list[torch.Tensor] = []
        val_slices: list[torch.Tensor] = []
        for bid in block_ids:
            tokens_used = self.block_allocator.get_block(bid).tokens_used
            if tokens_used <= 0:
                continue
            key_slices.append(self.key_pool[layer_idx, bid, :tokens_used])
            val_slices.append(self.value_pool[layer_idx, bid, :tokens_used])

        if not key_slices:
            empty = torch.zeros(
                0, self.num_kv_heads, self.head_dim,
                dtype=self.dtype, device=self.device,
            )
            return empty, empty.clone()

        return torch.cat(key_slices, dim=0), torch.cat(val_slices, dim=0)

    def read_kv_block(
        self,
        block_id: int,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the raw ``[block_size, H, D]`` key/value tensors of a block.

        This is the low-level accessor — empty slots are NOT filtered out.
        """
        return self.key_pool[layer_idx, block_id], self.value_pool[layer_idx, block_id]

    # ── Maintenance ───────────────────────────────────────────────────────────

    def clear_sequence(self, seq_id: str) -> None:
        """Zero the pool slots of blocks *exclusively* owned by *seq_id*.

        Not needed for correctness (stale slots are always masked out) and not
        called on the hot path.  Shared prefix-cache blocks are left intact.
        Must be called BEFORE ``block_allocator.free(seq_id)``.
        """
        for bid in self.block_allocator.get_blocks(seq_id):
            blk = self.block_allocator.get_block(bid)
            if blk.ref_count <= 1 and blk.block_hash is None:
                self.key_pool[:, bid].zero_()
                self.value_pool[:, bid].zero_()

        with self._lock:
            self._seq_write_cursors.pop(seq_id, None)

    def copy_blocks(
        self,
        src_block_id: int,
        dst_block_id: int,
        layer_idx: Optional[int] = None,
    ) -> None:
        """Copy KV tensors from *src_block_id* to *dst_block_id* in-place.

        If *layer_idx* is None all layers are copied (copy-on-write building
        block, e.g. for beam search).
        """
        if layer_idx is None:
            self.key_pool[:, dst_block_id].copy_(self.key_pool[:, src_block_id])
            self.value_pool[:, dst_block_id].copy_(self.value_pool[:, src_block_id])
        else:
            self.key_pool[layer_idx, dst_block_id].copy_(self.key_pool[layer_idx, src_block_id])
            self.value_pool[layer_idx, dst_block_id].copy_(self.value_pool[layer_idx, src_block_id])

    # ── Stats ─────────────────────────────────────────────────────────────────

    def pool_size_mb(self) -> float:
        dtype_bytes = _DTYPE_BYTES.get(self.dtype, 2)
        total_elements = self.key_pool.nelement() + self.value_pool.nelement()
        return total_elements * dtype_bytes / (1024 * 1024)

    def stats(self) -> dict:
        """Return a snapshot of the pool state."""
        allocator_stats = self.block_allocator.stats(include_per_sequence=False)
        return {
            "pool_shape": list(self.key_pool.shape),
            "pool_size_mb": self.pool_size_mb(),
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "active_sequences": allocator_stats["active_sequences"],
            "token_capacity": self.num_blocks * self.block_size,
            "block_allocator": allocator_stats,
        }
