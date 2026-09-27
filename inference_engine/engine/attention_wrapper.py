"""
engine/attention_wrapper.py — Phase 8: Paged attention inside HuggingFace models.

The problem
-----------
A stock HuggingFace model computes attention against a ``past_key_values``
cache it owns: one contiguous tensor per layer, per sequence.  That is exactly
the memory layout paging is meant to replace, and it forces one forward pass
per sequence (or padded batches that waste compute on pad tokens).

The approach
------------
Modern transformers models call attention through a registry
(``ALL_ATTENTION_FUNCTIONS``) keyed by ``config._attn_implementation``.  We
register :func:`paged_attention_forward` under :data:`ATTN_IMPL_NAME`.  The
model still does everything else (embeddings, Q/K/V projections, RoPE,
MLPs, norms); when a layer reaches attention it hands us the query and the
*new* keys/values for this step, and we:

1. **Write** the new K/V into the paged pool at each token's slot
   (one scatter per layer for the whole batch).
2. **Attend** each query token to its sequence's full context, reading the
   context K/V from the paged pool through the sequence's block table.

The batch is *packed*: all scheduled tokens from all sequences form one
``[1, T]`` input with explicit ``position_ids`` — no padding tokens flow
through the MLPs.  Per-sequence structure (where each sequence's tokens are,
its block table, its context length) arrives through a *forward context*
(:class:`PagedAttentionMetadata`) set by the ModelRunner right before the
forward pass — the same trick vLLM uses to smuggle batch metadata past
model code it doesn't own.

Two attention paths
-------------------
* **Short queries** (decode = 1 token, speculative verification = k+1
  tokens, tiny prompts) — all such sequences are batched: gather every
  sequence's context from the pool into ``[B, C, H_kv, D]`` and compute
  masked attention with two batched matmuls.  GQA is handled by grouping
  query heads per KV head instead of materialising repeated KV.
* **Long queries** (prefill chunks) — one ``scaled_dot_product_attention``
  call per sequence.  A chunk with no cached prefix attends only to itself,
  so it uses SDPA's fused causal kernel directly on the new K/V.

This is a readable reference implementation in plain PyTorch; production
engines replace step 2 with a fused kernel (vLLM's PagedAttention,
FlashAttention/FlashInfer with block tables) that reads the pool in place
instead of gathering it.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

ATTN_IMPL_NAME = "pageserve_paged"

# Queries with at most this many tokens take the batched short-query path.
SHORT_QUERY_MAX_TOKENS = 16
# Cap on (sequences × context slots) gathered at once by the short-query path.
# Bounds the transient memory of a decode step independently of batch size:
# the batch is processed in row chunks that each gather at most this much.
SHORT_QUERY_MAX_GATHER_SLOTS = 1 << 17

_registered = False
_registry_lock = threading.Lock()
_context = threading.local()


# ── Metadata ──────────────────────────────────────────────────────────────────


@dataclass
class LongQuery:
    """One sequence handled by the per-sequence (prefill) path."""

    token_start: int                  # offset in the packed batch
    num_tokens: int                   # query length this step
    start_pos: int                    # position of the first query token
    block_table: Optional[torch.Tensor] = None  # [num_blocks] (only if start_pos > 0)


@dataclass
class PagedAttentionMetadata:
    """Everything the attention kernel needs about the packed batch."""

    # Flat pool slot for every token in the packed batch: [T]
    slot_mapping: torch.Tensor
    # Per-layer (key_cache, value_cache), each [num_blocks, block_size, H_kv, D]
    kv_caches: List[Tuple[torch.Tensor, torch.Tensor]]
    block_size: int

    # ── short-query group (batched) ──
    short_token_idx: Optional[torch.Tensor] = None     # [B, Q] indices into packed T
    short_query_pos: Optional[torch.Tensor] = None     # [B, Q] absolute positions
    short_block_tables: Optional[torch.Tensor] = None  # [B, max_blocks]
    # Scatter plan for the results, precomputed on the host so the kernel never
    # needs a data-dependent (GPU-syncing) boolean index:
    #   out[short_dest_idx] = short_out.view(B * Q, ...)[short_src_idx]
    short_dest_idx: Optional[torch.Tensor] = None      # [n_valid] packed positions
    short_src_idx: Optional[torch.Tensor] = None       # [n_valid] rows of B*Q

    # ── long-query group (per sequence) ──
    long_queries: List[LongQuery] = field(default_factory=list)

    # Masks depend only on (sliding_window) — build once per forward, reuse per layer.
    _mask_cache: dict = field(default_factory=dict)

    def short_kv_mask(self, sliding_window: Optional[int]) -> torch.Tensor:
        """[B, Q, C] bool: may query (b, q) attend to context slot c?"""
        key = ("short", sliding_window)
        mask = self._mask_cache.get(key)
        if mask is None:
            num_slots = self.short_block_tables.shape[1] * self.block_size
            kv_pos = torch.arange(num_slots, device=self.short_query_pos.device)
            q_pos = self.short_query_pos.unsqueeze(-1)            # [B, Q, 1]
            mask = kv_pos.view(1, 1, -1) <= q_pos                 # causal
            if sliding_window is not None:
                mask &= kv_pos.view(1, 1, -1) > q_pos - sliding_window
            self._mask_cache[key] = mask
        return mask


def get_forward_context() -> Optional[PagedAttentionMetadata]:
    return getattr(_context, "metadata", None)


@contextmanager
def set_forward_context(metadata: PagedAttentionMetadata) -> Iterator[None]:
    """Make *metadata* visible to :func:`paged_attention_forward` (per thread)."""
    previous = get_forward_context()
    _context.metadata = metadata
    try:
        yield
    finally:
        _context.metadata = previous


# ── Attention kernels (plain PyTorch) ─────────────────────────────────────────


def _short_query_attention(
    q: torch.Tensor,            # [T, H, D] packed queries
    meta: PagedAttentionMetadata,
    rows: slice,                # which short-query sequences to process
    k_cache: torch.Tensor,      # [N, bs, H_kv, D]
    v_cache: torch.Tensor,
    scale: float,
    sliding_window: Optional[int],
) -> torch.Tensor:
    """Batched attention for a slice of short-query sequences.  Returns [B, Q, H, D]."""
    idx = meta.short_token_idx[rows]               # [B, Q]
    bsz, qlen = idx.shape
    num_heads, head_dim = q.shape[1], q.shape[2]
    num_kv_heads = k_cache.shape[2]
    group = num_heads // num_kv_heads

    # Gather each sequence's context through its block table: [B, C, H_kv, D]
    tables = meta.short_block_tables[rows]
    num_slots = tables.shape[1] * meta.block_size
    keys = k_cache[tables].view(bsz, num_slots, num_kv_heads, head_dim)
    values = v_cache[tables].view(bsz, num_slots, num_kv_heads, head_dim)

    # Group query heads by the KV head they share (GQA) → [B, H_kv, Q*G, D]
    qs = (q[idx] * scale).view(bsz, qlen, num_kv_heads, group, head_dim)
    qs = qs.permute(0, 2, 1, 3, 4).reshape(bsz, num_kv_heads, qlen * group, head_dim)

    scores = torch.matmul(qs, keys.permute(0, 2, 3, 1))           # [B, H_kv, Q*G, C]
    if scores.dtype in (torch.float16, torch.bfloat16):
        scores = scores.float()   # softmax in fp32 for half-precision models
    mask = meta.short_kv_mask(sliding_window)[rows]                # [B, Q, C]
    mask = mask.unsqueeze(1).unsqueeze(3).expand(bsz, 1, qlen, group, num_slots)
    scores.masked_fill_(~mask.reshape(bsz, 1, qlen * group, num_slots), float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(values.dtype)

    out = torch.matmul(probs, values.permute(0, 2, 1, 3))          # [B, H_kv, Q*G, D]
    out = out.view(bsz, num_kv_heads, qlen, group, head_dim).permute(0, 2, 1, 3, 4)
    return out.reshape(bsz, qlen, num_heads, head_dim)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[H_kv, S, D] → [H_kv * n_rep, S, D] (GQA expansion for SDPA)."""
    if n_rep == 1:
        return x
    h, s, d = x.shape
    return x[:, None].expand(h, n_rep, s, d).reshape(h * n_rep, s, d)


def _long_query_attention(
    q: torch.Tensor,            # [T, H, D]
    k_new: torch.Tensor,        # [T, H_kv, D]
    v_new: torch.Tensor,
    lq: LongQuery,
    meta: PagedAttentionMetadata,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    sliding_window: Optional[int],
) -> torch.Tensor:
    """SDPA for one prefill chunk.  Returns [L, H, D]."""
    s, n = lq.token_start, lq.num_tokens
    num_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = q.shape[2]
    qi = q[s:s + n].transpose(0, 1)                                 # [H, L, D]

    if lq.start_pos == 0:
        # No cached prefix: the chunk only sees itself → fused causal kernel.
        ki = k_new[s:s + n].transpose(0, 1)
        vi = v_new[s:s + n].transpose(0, 1)
        ctx_len = n
    else:
        ctx_len = lq.start_pos + n
        ki = k_cache[lq.block_table].reshape(-1, num_kv_heads, head_dim)[:ctx_len].transpose(0, 1)
        vi = v_cache[lq.block_table].reshape(-1, num_kv_heads, head_dim)[:ctx_len].transpose(0, 1)

    group = num_heads // num_kv_heads
    ki = _repeat_kv(ki, group)
    vi = _repeat_kv(vi, group)

    attn_mask = None
    if lq.start_pos > 0 or sliding_window is not None:
        q_pos = torch.arange(lq.start_pos, lq.start_pos + n, device=q.device).unsqueeze(1)
        kv_pos = torch.arange(ctx_len, device=q.device).unsqueeze(0)
        attn_mask = kv_pos <= q_pos
        if sliding_window is not None:
            attn_mask &= kv_pos > q_pos - sliding_window

    out = F.scaled_dot_product_attention(
        qi.unsqueeze(0), ki.unsqueeze(0), vi.unsqueeze(0),
        attn_mask=attn_mask,
        is_causal=attn_mask is None,
        scale=scale,
    )
    return out[0].transpose(0, 1)                                   # [L, H, D]


def paged_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,        # [1, H, T, D]
    key: torch.Tensor,          # [1, H_kv, T, D]
    value: torch.Tensor,        # [1, H_kv, T, D]
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """Attention function registered with transformers' AttentionInterface.

    Returns ``(attn_output [1, T, H, D], None)`` like the built-in backends.
    Outside a PageServe forward context it falls back to HF's SDPA backend,
    so the same model object still works with ``model.generate()``.
    """
    meta = get_forward_context()
    if meta is None:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, **kwargs,
        )
    if softcap is not None:
        raise NotImplementedError("attention logit softcapping is not supported")
    if query.shape[0] != 1:
        raise ValueError("paged attention expects a packed batch of size 1")

    layer_idx = module.layer_idx
    k_cache, v_cache = meta.kv_caches[layer_idx]
    head_dim = query.shape[-1]
    scale = scaling if scaling is not None else head_dim ** -0.5

    q = query[0].transpose(0, 1)                                    # [T, H, D]
    k_new = key[0].transpose(0, 1)                                  # [T, H_kv, D]
    v_new = value[0].transpose(0, 1)

    # 1) Write this step's K/V into the paged pool (every layer, every token).
    #    index_put_ rather than index_copy_: identical on CUDA, but MPS
    #    implements index_copy_ by copying the whole destination tensor.
    flat_k = k_cache.view(-1, k_cache.shape[2], head_dim)
    flat_v = v_cache.view(-1, v_cache.shape[2], head_dim)
    flat_k.index_put_((meta.slot_mapping,), k_new.to(k_cache.dtype))
    flat_v.index_put_((meta.slot_mapping,), v_new.to(v_cache.dtype))

    # 2) Attend.
    out = torch.empty_like(q)
    if meta.short_token_idx is not None:
        num_rows, qlen = meta.short_token_idx.shape
        num_slots = meta.short_block_tables.shape[1] * meta.block_size
        rows_per_chunk = max(1, SHORT_QUERY_MAX_GATHER_SLOTS // num_slots)
        chunks = [
            _short_query_attention(q, meta, slice(r, r + rows_per_chunk),
                                   k_cache, v_cache, scale, sliding_window)
            for r in range(0, num_rows, rows_per_chunk)
        ]
        short_out = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
        short_out = short_out.reshape(num_rows * qlen, q.shape[1], head_dim)
        out.index_put_((meta.short_dest_idx,),
                       short_out.index_select(0, meta.short_src_idx).to(out.dtype))
    for lq in meta.long_queries:
        out[lq.token_start:lq.token_start + lq.num_tokens] = _long_query_attention(
            q, k_new, v_new, lq, meta, k_cache, v_cache, scale, sliding_window
        ).to(out.dtype)

    return out.unsqueeze(0), None


# ── Registration ──────────────────────────────────────────────────────────────


def register_paged_attention() -> str:
    """Register the paged backend with transformers (idempotent)."""
    global _registered
    with _registry_lock:
        if not _registered:
            from transformers import AttentionInterface

            AttentionInterface.register(ATTN_IMPL_NAME, paged_attention_forward)
            _registered = True
    return ATTN_IMPL_NAME


@contextmanager
def use_paged_attention(model: torch.nn.Module) -> Iterator[None]:
    """Temporarily route *model*'s attention through the paged backend."""
    config = model.config
    previous = config._attn_implementation
    config._attn_implementation = ATTN_IMPL_NAME
    try:
        yield
    finally:
        config._attn_implementation = previous
