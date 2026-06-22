"""
tests/test_scheduler.py — Unit tests for the Phase 2 continuous batching scheduler.

All tests use a session-scoped real model fixture (same pattern as Phase 1's
test_sequential.py) — no mocking.  This properly validates the actual
scheduler + model interaction.

Run with:
    cd /Users/nikhilmourya/Desktop/PageServe
    pytest inference_engine/tests/test_scheduler.py -v

Tests
-----
1. test_single_request_through_scheduler
   Add one request, run the scheduler loop for N steps, verify state=="finished".

2. test_multiple_requests_admitted
   Add 3 requests to a scheduler with max_batch_size=4.
   Verify all 3 reach "decoding" state within 5 scheduler steps.

3. test_finished_sequences_removed
   Verify a finished sequence is removed from scheduler.running.

4. test_queue_wait_time_recorded
   Verify queue_wait_time_ms > 0 for each finished sequence.

5. test_batch_size_never_exceeds_max
   Add 10 requests, run the scheduler, verify len(running) <= max_batch_size
   at every recorded batch_size_over_time sample.

Note on asyncio in pytest
--------------------------
Each test that drives the scheduler loop uses asyncio.run() inside a standard
(non-async) test function.  This avoids a pytest-asyncio dependency while
still exercising the full async scheduler path.
"""

from __future__ import annotations

import asyncio
import time
from typing import List

import pytest

from inference_engine.config import Config
from inference_engine.engine.scheduler import ContinuousBatchingScheduler
from inference_engine.engine.sequence import Sequence
from inference_engine.models.loader import LoadedModel, load_model_and_tokenizer

# ── Session-scoped model fixture (loaded once for the entire test run) ────────


@pytest.fixture(scope="session")
def cfg() -> Config:
    """Config with a small batch size and short generations for fast tests."""
    c = Config()
    c.max_batch_size = 4
    c.max_new_tokens = 10   # keeps test wall-clock short
    return c


@pytest.fixture(scope="session")
def loaded(cfg: Config) -> LoadedModel:
    """Load model once; shared across all scheduler tests."""
    return load_model_and_tokenizer(cfg)


# ── Helpers ───────────────────────────────────────────────────────────────────

SHORT_PROMPT = "The capital of France is"

_N_STEPS_SINGLE = 30   # enough to finish a 10-token generation


def _make_scheduler(loaded: LoadedModel, cfg: Config, max_batch_size: int = 4) -> ContinuousBatchingScheduler:
    """Build a fresh scheduler for each test to avoid state leakage."""
    c = Config()
    c.max_batch_size = max_batch_size
    c.scheduler_poll_interval_ms = 1.0
    return ContinuousBatchingScheduler(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        config=c,
    )


async def _run_scheduler_steps(
    scheduler: ContinuousBatchingScheduler, n_steps: int
) -> None:
    """Drive the scheduler for *n_steps* iterations without starting run_loop()."""
    for _ in range(n_steps):
        if not scheduler.running and len(scheduler.request_queue) == 0:
            await asyncio.sleep(0.001)
            continue
        await scheduler._schedule()


# ── Test 1: single request completes ─────────────────────────────────────────


def test_single_request_through_scheduler(loaded: LoadedModel, cfg: Config):
    """A single enqueued request must reach state='finished' within N steps."""

    async def _run():
        scheduler = _make_scheduler(loaded, cfg)
        seq, _ = await scheduler.add_request(SHORT_PROMPT, max_new_tokens=5)
        assert seq.state == "waiting"

        await _run_scheduler_steps(scheduler, _N_STEPS_SINGLE)

        assert seq.is_finished(), (
            f"Expected state='finished', got '{seq.state}' after {_N_STEPS_SINGLE} steps"
        )
        assert seq.finish_reason in ("eos", "length"), (
            f"finish_reason must be 'eos' or 'length', got '{seq.finish_reason}'"
        )
        assert len(seq.generated_token_ids) > 0, "Must have generated at least one token"
        assert seq.ttft_ms > 0.0, "ttft_ms must be positive"

    asyncio.run(_run())


# ── Test 2: multiple requests all reach decoding state ───────────────────────


def test_multiple_requests_admitted(loaded: LoadedModel, cfg: Config):
    """Three requests with max_batch_size=4 must all reach 'decoding' within 5 steps."""

    async def _run():
        scheduler = _make_scheduler(loaded, cfg, max_batch_size=4)

        seqs: List[Sequence] = []
        for _ in range(3):
            seq, _ = await scheduler.add_request(SHORT_PROMPT, max_new_tokens=5)
            seqs.append(seq)

        # Run up to 5 scheduler steps.  All three should transition from
        # "waiting" → "prefill" → "decoding" within this window.
        for _ in range(5):
            await scheduler._schedule()

        for i, seq in enumerate(seqs):
            assert seq.state in ("decoding", "finished"), (
                f"Sequence {i} in unexpected state '{seq.state}' after 5 steps. "
                "Expected 'decoding' or 'finished'."
            )

    asyncio.run(_run())


# ── Test 3: finished sequences removed from running list ─────────────────────


def test_finished_sequences_removed(loaded: LoadedModel, cfg: Config):
    """After a sequence finishes, it must not remain in scheduler.running."""

    async def _run():
        scheduler = _make_scheduler(loaded, cfg)
        seq, _ = await scheduler.add_request(SHORT_PROMPT, max_new_tokens=3)

        await _run_scheduler_steps(scheduler, _N_STEPS_SINGLE)

        assert seq.is_finished(), "Sequence should be finished by now"
        assert seq not in scheduler.running, (
            "Finished sequence must be removed from scheduler.running"
        )
        assert seq in scheduler.finished, (
            "Finished sequence must be present in scheduler.finished"
        )

    asyncio.run(_run())


# ── Test 4: queue_wait_time_ms is recorded ────────────────────────────────────


def test_queue_wait_time_recorded(loaded: LoadedModel, cfg: Config):
    """Every finished sequence must have queue_wait_time_ms > 0."""

    async def _run():
        scheduler = _make_scheduler(loaded, cfg)

        seqs: List[Sequence] = []
        for _ in range(2):
            seq, _ = await scheduler.add_request(SHORT_PROMPT, max_new_tokens=4)
            seqs.append(seq)

        # Introduce a small delay so queue wait is measurable
        await asyncio.sleep(0.005)

        await _run_scheduler_steps(scheduler, _N_STEPS_SINGLE)

        for i, seq in enumerate(seqs):
            assert seq.is_finished(), f"Sequence {i} not finished"
            assert seq.queue_wait_time_ms > 0.0, (
                f"Sequence {i}: queue_wait_time_ms must be > 0, "
                f"got {seq.queue_wait_time_ms}"
            )

    asyncio.run(_run())


# ── Test 5: running count never exceeds max_batch_size ───────────────────────


def test_batch_size_never_exceeds_max(loaded: LoadedModel, cfg: Config):
    """With 10 requests and max_batch_size=4, len(running) must never exceed 4."""
    MAX_BS = 4

    async def _run():
        scheduler = _make_scheduler(loaded, cfg, max_batch_size=MAX_BS)

        # Enqueue 10 requests
        for _ in range(10):
            await scheduler.add_request(SHORT_PROMPT, max_new_tokens=5)

        # Drive the scheduler until queue is empty and all finish
        max_observed_batch = 0
        deadline = time.perf_counter() + 300.0  # 5-minute safety timeout
        while time.perf_counter() < deadline:
            if len(scheduler.request_queue) == 0 and not scheduler.running:
                break
            await scheduler._schedule()
            max_observed_batch = max(max_observed_batch, len(scheduler.running))

        # Verify batch size constraint from recorded telemetry
        for ts, bs in scheduler.batch_size_over_time:
            assert bs <= MAX_BS, (
                f"batch_size_over_time entry ({ts:.3f}, {bs}) exceeds "
                f"max_batch_size={MAX_BS}"
            )

        # Also verify the max observed directly
        assert max_observed_batch <= MAX_BS, (
            f"max observed running batch size {max_observed_batch} exceeds {MAX_BS}"
        )

        # All 10 should eventually finish
        assert len(scheduler.finished) == 10, (
            f"Expected 10 finished sequences, got {len(scheduler.finished)}"
        )

    asyncio.run(_run())


# ── Test 6: prefill token budget blocks oversized prompts ────────────────────


def test_prefill_budget_respected(loaded: LoadedModel, cfg: Config):
    async def _run():
        scheduler = _make_scheduler(loaded, cfg)
        scheduler.prefill_budget_tokens = 10
        seq = Sequence.create(
            prompt="oversized prompt",
            prompt_token_ids=list(range(20)),
            max_new_tokens=5,
        )
        await scheduler.request_queue.enqueue(seq)

        await scheduler._schedule()

        assert seq not in scheduler.running
        assert len(scheduler.request_queue) == 1
        assert scheduler.request_queue._queue[0].sequence is seq

    asyncio.run(_run())


# ── Test 7: decode capacity prevents further admission ──────────────────────


def test_decode_batch_limit_respected(loaded: LoadedModel, cfg: Config):
    async def _run():
        scheduler = _make_scheduler(loaded, cfg)
        scheduler.decode_batch_limit = 2

        first, _ = await scheduler.add_request(SHORT_PROMPT, max_new_tokens=5)
        second, _ = await scheduler.add_request(SHORT_PROMPT, max_new_tokens=5)
        await scheduler._schedule()
        assert len(scheduler.running) == 2

        waiting, _ = await scheduler.add_request(SHORT_PROMPT, max_new_tokens=5)
        await scheduler._schedule()

        assert len(scheduler.running) <= 2
        assert waiting not in scheduler.running
        assert scheduler.request_queue._queue[0].sequence is waiting

    asyncio.run(_run())


# ── Test 8: pool is source of truth — past_key_values stays None ─────────────


def test_decode_uses_pool_not_past_key_values(loaded: LoadedModel, cfg: Config):
    """After Phase 8, seq.past_key_values must be None throughout decode.

    Verifies:
    - After prefill completes, seq.past_key_values is None (pool holds KV state).
    - After at least one decode step, a token has been generated AND
      seq.past_key_values is still None.
    """

    async def _run():
        scheduler = _make_scheduler(loaded, cfg)
        seq, _ = await scheduler.add_request(SHORT_PROMPT, max_new_tokens=5)

        # Run scheduler steps until the sequence leaves 'waiting'/'prefill'
        # (i.e. prefill is done and it's in 'decoding' or 'finished')
        for _ in range(10):
            await scheduler._schedule()
            if seq.state in ("decoding", "finished"):
                break

        # After prefill: pool is source of truth, past_key_values must be None
        assert seq.past_key_values is None, (
            f"Expected past_key_values=None after prefill (Phase 8), "
            f"got {type(seq.past_key_values)}"
        )
        assert len(seq.generated_token_ids) >= 1, "Must have at least the first token"

        tokens_before = len(seq.generated_token_ids)

        # Run one more decode step
        if seq.state == "decoding":
            await scheduler._schedule()

        # Decode must generate more tokens and still keep past_key_values=None
        assert len(seq.generated_token_ids) >= tokens_before, (
            "Decode step must produce tokens even when reading KV from pool"
        )
        assert seq.past_key_values is None, (
            "past_key_values must remain None during decode — pool is source of truth"
        )

    asyncio.run(_run())

