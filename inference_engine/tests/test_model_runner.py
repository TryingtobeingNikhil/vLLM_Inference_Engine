"""
tests/test_model_runner.py — batched execution over the paged KV cache.

The engine must produce exactly HuggingFace's greedy tokens while packing
several sequences — at different stages (chunked prefill, decode) — into each
forward pass.  float64 tiny models make the comparison exact.
"""

from __future__ import annotations

import math

import pytest
import torch

from inference_engine.engine.block_allocator import BlockAllocator
from inference_engine.engine.kv_cache_config import compute_kv_cache_config
from inference_engine.engine.model_runner import ModelRunner, SequenceInput
from inference_engine.engine.paged_kv_cache import PagedKVCacheManager
from inference_engine.engine.sampler import sample
from inference_engine.engine.sequence import SamplingParams
from inference_engine.tests.tiny_models import hf_greedy, tiny_llama, tiny_qwen2

BLOCK_SIZE = 4


class _Cfg:
    kv_block_size = BLOCK_SIZE
    device = "cpu"


def _runner(model):
    allocator = BlockAllocator(256, BLOCK_SIZE)
    pool = PagedKVCacheManager(compute_kv_cache_config(model, _Cfg), allocator, _Cfg)
    return ModelRunner(model, pool), allocator


PROMPTS = [
    list(range(1, 60)),                     # long: first chunk takes the SDPA path
    [5, 6, 7],
    [9] * 7,
    [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6],
]


@pytest.mark.parametrize("build", [tiny_qwen2, tiny_llama], ids=["qwen2", "llama-gqa4"])
@pytest.mark.parametrize("chunk", [5, 17])
def test_batched_generation_matches_hf(build, chunk):
    model = build()
    runner, allocator = _runner(model)
    n_new = 10
    refs = [hf_greedy(model, p, n_new) for p in PROMPTS]

    state = [{"tokens": list(p), "computed": 0, "gen": []} for p in PROMPTS]
    for i, p in enumerate(PROMPTS):
        allocator.allocate(str(i), math.ceil((len(p) + n_new) / BLOCK_SIZE))

    while any(len(s["gen"]) < n_new for s in state):
        inputs, scheduled = [], {}
        for i, s in enumerate(state):
            if len(s["gen"]) >= n_new:
                continue
            start = s["computed"]
            n = min(len(s["tokens"]) - start, chunk)
            done = start + n == len(s["tokens"])
            inputs.append(SequenceInput(str(i), s["tokens"][start:start + n], start,
                                        allocator.get_blocks(str(i)), 1 if done else 0))
            scheduled[i] = n
        out = runner.execute(inputs)
        for i, n in scheduled.items():
            state[i]["computed"] += n
            if str(i) in out:
                token = out[str(i)][0]
                state[i]["gen"].append(token)
                state[i]["tokens"].append(token)

    for s, ref in zip(state, refs):
        assert s["gen"] == ref


def test_logits_only_for_requested_positions():
    model = tiny_qwen2()
    runner, allocator = _runner(model)
    allocator.allocate("a", 4)
    allocator.allocate("b", 4)
    out = runner.execute([
        SequenceInput("a", list(range(8)), 0, allocator.get_blocks("a"), num_logits=0),
        SequenceInput("b", list(range(6)), 0, allocator.get_blocks("b"), num_logits=3),
    ])
    assert "a" not in out
    assert len(out["b"]) == 3


def test_verification_logits_match_sequential_decoding():
    """Scoring k+1 tokens in one pass (speculative verification) must equal
    the greedy choices of k+1 separate decode steps."""
    model = tiny_qwen2()
    prompt = [4, 8, 15, 16, 23, 42]
    ref = hf_greedy(model, prompt, 5)

    runner, allocator = _runner(model)
    allocator.allocate("s", 4)
    table = allocator.get_blocks("s")
    runner.execute([SequenceInput("s", prompt, 0, table, num_logits=0)])
    # Feed the 5 reference tokens at once; each position predicts the next.
    out = runner.execute([SequenceInput("s", ref[:5], len(prompt), table, num_logits=5)])
    assert out["s"][:4] == ref[1:5]


def test_block_table_too_short_is_rejected():
    model = tiny_qwen2()
    runner, allocator = _runner(model)
    allocator.allocate("s", 1)
    with pytest.raises(ValueError, match="block table"):
        runner.execute([SequenceInput("s", list(range(6)), 0, allocator.get_blocks("s"))])


# ── Sampler ───────────────────────────────────────────────────────────────────


def test_sampler_greedy_and_temperature_rows():
    torch.manual_seed(0)
    logits = torch.randn(4, 50)
    params = [
        SamplingParams(),                               # greedy
        SamplingParams(temperature=1.0),
        SamplingParams(temperature=0.7, top_k=1),       # top-k=1 == greedy
        SamplingParams(temperature=1.0, top_p=1e-6),    # tiny nucleus == greedy
    ]
    tokens = sample(logits, params, torch.Generator().manual_seed(0))
    argmax = logits.argmax(-1).tolist()
    assert tokens[0] == argmax[0]
    assert tokens[2] == argmax[2]
    assert tokens[3] == argmax[3]
    assert 0 <= tokens[1] < 50


def test_sampler_is_reproducible_with_seeded_generator():
    logits = torch.randn(8, 100)
    params = [SamplingParams(temperature=1.0)] * 8
    a = sample(logits, params, torch.Generator().manual_seed(123))
    b = sample(logits, params, torch.Generator().manual_seed(123))
    assert a == b
