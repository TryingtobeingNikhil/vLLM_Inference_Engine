"""
tests/test_attention_wrapper.py — Phase 8: paged attention kernel.

The kernel is checked against a naive reference: for every sequence,
concatenate its cached prefix K/V with this step's new K/V and run plain
causal softmax attention (with explicit GQA head repetition).
"""

from __future__ import annotations

import math

import pytest
import torch

from inference_engine.engine.attention_wrapper import (
    get_forward_context,
    paged_attention_forward,
    set_forward_context,
)
from inference_engine.engine.block_allocator import BlockAllocator
from inference_engine.engine.kv_cache_config import compute_kv_cache_config
from inference_engine.engine.model_runner import ModelRunner, SequenceInput
from inference_engine.engine.paged_kv_cache import PagedKVCacheManager
from inference_engine.tests.tiny_models import tiny_qwen2

BLOCK_SIZE = 4
NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 4, 2, 16


class _Cfg:
    kv_block_size = BLOCK_SIZE
    device = "cpu"


class _Layer(torch.nn.Module):
    layer_idx = 0


@pytest.fixture
def setup():
    model = tiny_qwen2()
    allocator = BlockAllocator(64, BLOCK_SIZE)
    pool = PagedKVCacheManager(compute_kv_cache_config(model, _Cfg), allocator, _Cfg)
    return ModelRunner(model, pool), allocator, pool


# (start_pos, num_new_tokens): decode, spec-verify, tiny prompt,
# long prefill from scratch, long chunk on top of a cached prefix.
CASES = [(13, 1), (7, 5), (0, 3), (0, 20), (9, 18)]


def _reference(q, k_ctx, v_ctx, start_pos, scale, sliding_window=None):
    """q: [n, H, D]; k_ctx/v_ctx: [start_pos + n, H_kv, D] → [n, H, D]."""
    group = NUM_HEADS // NUM_KV_HEADS
    k = k_ctx.repeat_interleave(group, dim=1)              # [C, H, D]
    v = v_ctx.repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,chd->hqc", q, k) * scale
    n, ctx = q.shape[0], k.shape[0]
    q_pos = torch.arange(start_pos, start_pos + n).unsqueeze(1)
    kv_pos = torch.arange(ctx).unsqueeze(0)
    allowed = kv_pos <= q_pos
    if sliding_window is not None:
        allowed &= kv_pos > q_pos - sliding_window
    scores = scores.masked_fill(~allowed, float("-inf"))
    return torch.einsum("hqc,chd->qhd", scores.softmax(-1), v)


@pytest.mark.parametrize("sliding_window", [None, 6])
def test_paged_attention_matches_reference(setup, sliding_window):
    runner, allocator, pool = setup
    torch.manual_seed(0)
    k_cache, v_cache = pool.layer_caches(0)

    inputs, prefixes = [], []
    for i, (start, n) in enumerate(CASES):
        allocator.allocate(f"s{i}", math.ceil((start + n) / BLOCK_SIZE))
        table = allocator.get_blocks(f"s{i}")
        # Pre-populate the cached prefix directly in the pool.
        k_pre = torch.randn(start, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float64)
        v_pre = torch.randn(start, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float64)
        for p in range(start):
            k_cache[table[p // BLOCK_SIZE], p % BLOCK_SIZE] = k_pre[p]
            v_cache[table[p // BLOCK_SIZE], p % BLOCK_SIZE] = v_pre[p]
        prefixes.append((k_pre, v_pre))
        inputs.append(SequenceInput(f"s{i}", [1] * n, start, table, num_logits=1))

    total = sum(n for _, n in CASES)
    query = torch.randn(1, NUM_HEADS, total, HEAD_DIM, dtype=torch.float64)
    key = torch.randn(1, NUM_KV_HEADS, total, HEAD_DIM, dtype=torch.float64)
    value = torch.randn(1, NUM_KV_HEADS, total, HEAD_DIM, dtype=torch.float64)
    scale = HEAD_DIM ** -0.5

    meta, *_ = runner.prepare(inputs)
    with set_forward_context(meta):
        out, weights = paged_attention_forward(
            _Layer(), query, key, value, None, scaling=scale, sliding_window=sliding_window
        )
    assert weights is None
    assert out.shape == (1, total, NUM_HEADS, HEAD_DIM)

    offset = 0
    for (start, n), (k_pre, v_pre) in zip(CASES, prefixes):
        q = query[0, :, offset:offset + n].transpose(0, 1)
        k_new = key[0, :, offset:offset + n].transpose(0, 1)
        v_new = value[0, :, offset:offset + n].transpose(0, 1)
        expected = _reference(
            q, torch.cat([k_pre, k_new]), torch.cat([v_pre, v_new]), start, scale, sliding_window
        )
        torch.testing.assert_close(out[0, offset:offset + n], expected, rtol=1e-9, atol=1e-9)
        offset += n


def test_new_kv_written_to_slots(setup):
    runner, allocator, pool = setup
    allocator.allocate("s", 2)
    inp = SequenceInput("s", [1, 2, 3], start_pos=3, block_table=allocator.get_blocks("s"))
    key = torch.randn(1, NUM_KV_HEADS, 3, HEAD_DIM, dtype=torch.float64)
    value = torch.randn(1, NUM_KV_HEADS, 3, HEAD_DIM, dtype=torch.float64)
    query = torch.randn(1, NUM_HEADS, 3, HEAD_DIM, dtype=torch.float64)
    meta, *_ = runner.prepare([inp])
    with set_forward_context(meta):
        paged_attention_forward(_Layer(), query, key, value, None)

    table = allocator.get_blocks("s")
    k_cache, _ = pool.layer_caches(0)
    for j, pos in enumerate(range(3, 6)):
        torch.testing.assert_close(k_cache[table[pos // 4], pos % 4], key[0, :, j])


def test_forward_context_is_scoped(setup):
    runner, allocator, _ = setup
    allocator.allocate("s", 1)
    meta, *_ = runner.prepare([SequenceInput("s", [1], 0, allocator.get_blocks("s"))])
    assert get_forward_context() is None
    with set_forward_context(meta):
        assert get_forward_context() is meta
    assert get_forward_context() is None


def test_falls_back_to_sdpa_without_context():
    """Outside an engine step the backend behaves like HF's SDPA, so the same
    model object still works with model.generate()."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    torch.manual_seed(0)
    module = _Layer()
    module.num_key_value_groups = NUM_HEADS // NUM_KV_HEADS
    module.is_causal = True
    q = torch.randn(1, NUM_HEADS, 5, HEAD_DIM)
    k = torch.randn(1, NUM_KV_HEADS, 5, HEAD_DIM)
    v = torch.randn(1, NUM_KV_HEADS, 5, HEAD_DIM)
    ours, _ = paged_attention_forward(module, q, k, v, None, scaling=0.25)
    ref, _ = sdpa_attention_forward(module, q, k, v, None, scaling=0.25)
    torch.testing.assert_close(ours, ref)
