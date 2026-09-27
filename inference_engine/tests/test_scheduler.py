"""
tests/test_scheduler.py — Continuous batching scheduler, end to end.

Most tests use a tiny float64 model so that the engine's greedy output can
be compared token-for-token with HuggingFace ``generate()`` under every
combination of features that changes *how* tokens are computed — chunked
prefill, batching, prefix caching, swap / recompute preemption, n-gram and
draft-model speculative decoding — without changing *what* is computed.

The ``slow`` tests at the bottom repeat the check with the real
Qwen2-0.5B checkpoint (skipped when it cannot be loaded).

Each test drives the scheduler with ``asyncio.run`` from a normal test
function (no pytest-asyncio needed).
"""

from __future__ import annotations

import asyncio
import random

import pytest
import torch

from inference_engine.config import Config
from inference_engine.engine.scheduler import ContinuousBatchingScheduler
from inference_engine.engine.sequence import SamplingParams
from inference_engine.tests.tiny_models import FakeTokenizer, hf_greedy, tiny_qwen2

# ── Fixtures / helpers ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def model():
    return tiny_qwen2(seed=0)


@pytest.fixture(scope="module")
def workload(model):
    """Mixed prompts: half share a 40-token prefix, plus two repetitive ones."""
    rng = random.Random(1)
    shared = [rng.randrange(97) for _ in range(40)]
    prompts = []
    for i in range(20):
        base = shared if i % 2 == 0 else []
        prompts.append(base + [rng.randrange(97) for _ in range(rng.randint(1, 30))])
    prompts += [[1, 2, 3, 4, 5, 6, 7, 8] * 5] * 2
    lengths = [rng.randint(1, 25) for _ in prompts]
    refs = [hf_greedy(model, p, n) for p, n in zip(prompts, lengths)]
    return prompts, lengths, refs


def make_scheduler(model, draft=None, **overrides) -> ContinuousBatchingScheduler:
    settings = dict(
        device="cpu", kv_block_size=4, kv_num_blocks=512, max_batch_size=8,
        prefill_budget_tokens=24, prefill_chunk_size=10, max_queue_size=64,
    )
    settings.update(overrides)
    return ContinuousBatchingScheduler(
        model, FakeTokenizer(), Config(**settings), draft_model=draft, eos_token_ids=[]
    )


async def run_all(scheduler, prompts, lengths, **sampling):
    scheduler.start()
    try:
        requests = [
            await scheduler.add_request(
                "", prompt_token_ids=p, sampling=SamplingParams(max_new_tokens=n, **sampling)
            )
            for p, n in zip(prompts, lengths)
        ]
        return await asyncio.gather(*[future for _, future in requests])
    finally:
        await scheduler.stop()


def assert_clean(scheduler):
    """Every request finished and every resource was returned."""
    stats = scheduler.block_allocator.stats(include_per_sequence=False)
    assert stats["free_blocks"] == stats["num_blocks"], "leaked KV blocks"
    assert stats["active_sequences"] == 0
    assert not scheduler.running and not scheduler.swapped_out and not scheduler.preempted
    assert scheduler.kv_tracker.stats()["active_sequences"] == 0
    assert scheduler.cpu_swap_manager.stats()["swapped_sequences"] == 0
    assert not scheduler._futures


# ── Correctness under every feature combination ───────────────────────────────

CONFIGS = {
    "baseline": dict(enable_prefix_caching=False),
    "prefix_cache": dict(enable_prefix_caching=True),
    "tight_pool_swap": dict(kv_num_blocks=40, preemption_mode="swap", enable_prefix_caching=False),
    "tight_pool_recompute": dict(kv_num_blocks=40, preemption_mode="recompute"),
    "tight_pool_small_cpu": dict(kv_num_blocks=40, preemption_mode="swap", kv_num_cpu_blocks=12),
    "ngram_spec": dict(speculative_method="ngram", num_speculative_tokens=4),
    "ngram_spec_tight": dict(kv_num_blocks=40, speculative_method="ngram"),
    "one_seq_at_a_time": dict(max_batch_size=1),
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_outputs_match_hf_generate(model, workload, name):
    prompts, lengths, refs = workload
    scheduler = make_scheduler(model, **CONFIGS[name])
    results = asyncio.run(run_all(scheduler, prompts, lengths))
    for seq, ref in zip(results, refs):
        assert seq.generated_token_ids == ref
        assert seq.finish_reason == "length"
    assert_clean(scheduler)
    if name.startswith("tight"):
        assert scheduler.num_preemptions_swap + scheduler.num_preemptions_recompute > 0
    if name == "prefix_cache":
        assert scheduler.block_allocator.prefix_cache_hit_rate() > 0.2
    if name == "ngram_spec":
        assert scheduler.spec_decode_stats()["accepted_tokens"] > 0


@pytest.mark.parametrize("draft_kind", ["same_model", "different_model"])
def test_draft_model_speculation_matches_hf(model, workload, draft_kind):
    prompts, lengths, refs = workload
    draft = model if draft_kind == "same_model" else tiny_qwen2(seed=5, layers=1)
    scheduler = make_scheduler(
        model, draft=draft, speculative_method="draft", draft_model_name="tiny",
        num_speculative_tokens=3,
    )
    results = asyncio.run(run_all(scheduler, prompts, lengths))
    assert [s.generated_token_ids for s in results] == refs
    stats = scheduler.spec_decode_stats()
    if draft_kind == "same_model":
        assert stats["acceptance_rate"] == pytest.approx(1.0)
        assert stats["mean_tokens_per_step"] > 2.0
    assert_clean(scheduler)


# ── Scheduling limits ─────────────────────────────────────────────────────────


def test_batch_size_and_prefill_budget_respected(model, workload):
    prompts, lengths, _ = workload
    scheduler = make_scheduler(model, max_batch_size=3, prefill_budget_tokens=16)
    observed = []
    original = scheduler._plan_step

    async def spy():
        plan = await original()
        prefill = sum(i.num_tokens for i in plan.items if not i.is_decode)
        observed.append((len(scheduler.running), len(plan.items), prefill))
        return plan

    scheduler._plan_step = spy
    asyncio.run(run_all(scheduler, prompts, lengths))
    assert max(running for running, _, _ in observed) <= 3
    assert max(items for _, items, _ in observed) <= 3
    assert max(prefill for _, _, prefill in observed) <= 16


def test_decode_and_prefill_share_one_forward_pass(model):
    """While one request decodes, a newly arrived prompt is prefilled in the
    same step (the whole point of continuous batching)."""

    async def scenario():
        scheduler = make_scheduler(model)
        a, _ = await scheduler.add_request("", prompt_token_ids=[1, 2, 3],
                                           sampling=SamplingParams(max_new_tokens=20))
        for _ in range(3):
            await scheduler._schedule()
        assert a.state == "decoding"
        b, _ = await scheduler.add_request("", prompt_token_ids=list(range(30)),
                                           sampling=SamplingParams(max_new_tokens=5))
        plan = await scheduler._plan_step()
        kinds = {item.seq.seq_id: item.is_decode for item in plan.items}
        assert kinds == {a.seq_id: True, b.seq_id: False}
        await scheduler.stop()

    asyncio.run(scenario())


def test_prompt_longer_than_max_model_len_rejected(model):
    async def scenario():
        scheduler = make_scheduler(model, max_model_len=32)
        with pytest.raises(ValueError, match="max_model_len"):
            await scheduler.add_request("", prompt_token_ids=list(range(40)))
        await scheduler.stop()

    asyncio.run(scenario())


def test_generation_capped_at_max_model_len(model):
    scheduler = make_scheduler(model, max_model_len=24)
    (seq,) = asyncio.run(run_all(scheduler, [list(range(20))], [50]))
    assert seq.num_tokens() == 24
    assert seq.finish_reason == "length"


# ── Stop conditions ───────────────────────────────────────────────────────────


def test_eos_stops_unless_ignored(model):
    prompt = [3, 1, 4, 1, 5]
    ref = hf_greedy(model, prompt, 12)
    eos = ref[4]
    first_eos = ref.index(eos)

    async def generate(ignore_eos):
        scheduler = ContinuousBatchingScheduler(
            model, FakeTokenizer(),
            Config(device="cpu", kv_block_size=4, kv_num_blocks=64),
            eos_token_ids=[eos],
        )
        (seq,) = await run_all(scheduler, [prompt], [12], ignore_eos=ignore_eos)
        return seq

    stopped = asyncio.run(generate(False))
    assert stopped.finish_reason == "eos"
    assert stopped.generated_token_ids == ref[:first_eos + 1]
    full = asyncio.run(generate(True))
    assert full.generated_token_ids == ref


def test_sampling_requests_complete(model):
    scheduler = make_scheduler(model, speculative_method="ngram")
    results = asyncio.run(run_all(
        scheduler, [[1, 2, 3]] * 4, [10] * 4, temperature=0.8, top_p=0.9, top_k=20
    ))
    assert all(len(s.generated_token_ids) == 10 for s in results)
    # sampled requests are never speculated
    assert scheduler.spec_decode_stats()["draft_tokens"] == 0
    assert_clean(scheduler)


# ── Streaming, abort, timeouts ────────────────────────────────────────────────


def test_streaming_delivers_every_token_then_none(model):
    async def scenario():
        scheduler = make_scheduler(model, speculative_method="ngram")
        scheduler.start()
        seq, future = await scheduler.add_request(
            "", prompt_token_ids=[1, 2, 3, 4] * 4,
            sampling=SamplingParams(max_new_tokens=15), stream=True,
        )
        received = []
        while (chunk := await seq.stream.get()) is not None:
            received.extend(chunk)
        await future
        await scheduler.stop()
        return seq, received

    seq, received = asyncio.run(scenario())
    assert received == seq.generated_token_ids
    assert len(seq.token_times) == len(received)
    assert seq.ttft_ms > 0 and seq.e2e_latency_ms >= seq.ttft_ms


def test_abort_running_request_frees_blocks(model):
    async def scenario():
        scheduler = make_scheduler(model)
        scheduler.start()
        seq, future = await scheduler.add_request(
            "", prompt_token_ids=list(range(10)), sampling=SamplingParams(max_new_tokens=400)
        )
        while len(seq.generated_token_ids) < 3:
            await asyncio.sleep(0.005)
        await scheduler.abort(seq.seq_id)
        finished = await future
        await scheduler.stop()
        return scheduler, finished

    scheduler, seq = asyncio.run(scenario())
    assert seq.finish_reason == "abort"
    assert len(seq.generated_token_ids) < 400
    assert_clean(scheduler)


def test_abort_waiting_request_cancels_future(model):
    async def scenario():
        scheduler = make_scheduler(model)   # loop not started: request stays queued
        seq, future = await scheduler.add_request("", prompt_token_ids=[1, 2])
        await scheduler.abort(seq.seq_id)
        assert future.cancelled()
        assert seq.state == "cancelled"
        assert len(scheduler.request_queue) == 0
        await scheduler.stop()

    asyncio.run(scenario())


def test_queue_timeout_expires_request(model):
    async def scenario():
        scheduler = make_scheduler(model, request_timeout_ms=1.0)
        seq, future = await scheduler.add_request("", prompt_token_ids=[1, 2])
        await asyncio.sleep(0.01)
        await scheduler.request_queue.expire_timed_out()
        with pytest.raises(asyncio.TimeoutError):
            await future
        assert seq.state == "expired"
        await scheduler.stop()

    asyncio.run(scenario())


def test_metrics_report_structure(model, workload):
    prompts, lengths, _ = workload
    scheduler = make_scheduler(model, speculative_method="ngram")
    asyncio.run(run_all(scheduler, prompts[:6], lengths[:6]))
    report = scheduler.get_metrics()
    for key in ("system", "e2e_latency", "engine", "speculative_decoding",
                "engine_steps", "paged_kv_cache", "cpu_swap", "queue_stats"):
        assert key in report
    latency = report["e2e_latency"]
    assert {"ttft_ms", "itl_ms", "tpot_ms", "total_latency_ms", "queue_wait_ms"} <= set(latency)
    assert latency["ttft_ms"]["p99"] >= latency["ttft_ms"]["p50"] > 0
    assert report["engine_steps"]["total_steps"] > 0
    assert report["system"]["requests_finished_total"] == 6


# ── Real checkpoint (slow) ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def real_model():
    from inference_engine.models.loader import load_model_and_tokenizer

    try:
        loaded = load_model_and_tokenizer(Config(device="cpu", dtype="float32"))
    except Exception as exc:  # offline / not cached
        pytest.skip(f"real checkpoint unavailable: {exc}")
    return loaded


@pytest.mark.slow
def test_real_model_matches_hf_generate(real_model):
    model, tokenizer, _ = real_model
    prompts = ["The capital of France is", "def fibonacci(n):", "Water boils at"]
    n = 24
    refs = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt")
        out = model.generate(**ids, max_new_tokens=n, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        refs.append(out[0, ids.input_ids.shape[1]:].tolist())

    async def scenario():
        scheduler = ContinuousBatchingScheduler(
            model, tokenizer, Config(device="cpu", dtype="float32", kv_num_blocks=256)
        )
        scheduler.start()
        requests = [await scheduler.add_request(p, max_new_tokens=n) for p in prompts]
        results = await asyncio.gather(*[f for _, f in requests])
        await scheduler.stop()
        return results

    for seq, ref in zip(asyncio.run(scenario()), refs):
        # fp32 batched vs unbatched numerics can flip a near-tie late in a
        # generation; the first 16 tokens must agree exactly.
        assert seq.generated_token_ids[:16] == ref[:16]


def test_near_max_length_swapped_sequence_cannot_deadlock(model):
    """Regression: a swapped-out sequence holding ~the whole pool can't be
    swapped back in with room to grow.  It must fall back to recompute rather
    than wait forever (which also blocked every later request)."""
    prompts = [[1], [i % 97 for i in range(392)], [5, 6, 7]]
    lengths = [30, 6, 2]
    refs = [hf_greedy(model, p, n) for p, n in zip(prompts, lengths)]
    scheduler = make_scheduler(
        model, kv_num_blocks=100, prefill_budget_tokens=512, prefill_chunk_size=512,
        preemption_mode="swap", enable_prefix_caching=False,
    )

    async def scenario():
        return await asyncio.wait_for(run_all(scheduler, prompts, lengths), timeout=60)

    results = asyncio.run(scenario())
    assert [s.generated_token_ids for s in results] == refs
    assert scheduler.num_preemptions_swap >= 1
    assert_clean(scheduler)
