"""
tests/test_request_queue.py — Unit tests for Phase 3 RequestQueue.

All tests are pure asyncio (pytest-asyncio, mode=strict).  No model is
loaded — Sequence objects are constructed with a minimal helper that
bypasses Sequence.create() so that no tokenizer is needed.

Run with:
    cd /Users/nikhilmourya/Desktop/PageServe
    pytest inference_engine/tests/test_request_queue.py -v

Tests
-----
1. test_enqueue_and_dequeue_fifo           FIFO ordering preserved across 3 items
2. test_queue_full_raises                  QueueFullError on overflow
3. test_timeout_expires_waiting_request    Expiry sets state, resolves future
4. test_cancel_removes_from_queue          cancel() removes item, sets state
5. test_cancel_nonexistent_returns_false   cancel() returns False on miss
6. test_stats_accuracy                     stats() fields match expected values
7. test_dequeue_returns_none_on_empty_queue  Empty queue dequeue → None
"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest

from inference_engine.engine.request_queue import QueueFullError, RequestQueue
from inference_engine.engine.sequence import Sequence


# ── Helper ────────────────────────────────────────────────────────────────────


def make_sequence(seq_id: str | None = None) -> Sequence:
    """Construct a minimal waiting Sequence without a tokenizer."""
    s = Sequence.create(prompt="test", prompt_token_ids=[1, 2, 3], max_new_tokens=10)
    if seq_id is not None:
        s.seq_id = seq_id
    return s


# ── Test 1: FIFO ordering ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_enqueue_and_dequeue_fifo():
    """Three sequences enqueued must be dequeued in insertion (FIFO) order."""
    queue = RequestQueue(maxsize=10)

    ids = [uuid.uuid4().hex for _ in range(3)]
    for sid in ids:
        await queue.enqueue(make_sequence(sid))

    dequeued_ids = []
    for _ in range(3):
        item = await queue.dequeue()
        assert item is not None, "Expected a QueuedRequest, got None"
        dequeued_ids.append(item.sequence.seq_id)

    assert dequeued_ids == ids, (
        f"FIFO order violated: expected {ids}, got {dequeued_ids}"
    )


# ── Test 2: overflow raises QueueFullError ────────────────────────────────────


@pytest.mark.asyncio
async def test_queue_full_raises():
    """Enqueuing beyond maxsize must raise QueueFullError immediately."""
    queue = RequestQueue(maxsize=2)

    await queue.enqueue(make_sequence())
    await queue.enqueue(make_sequence())

    with pytest.raises(QueueFullError) as exc_info:
        await queue.enqueue(make_sequence())

    assert exc_info.value.maxsize == 2
    assert "2" in str(exc_info.value)


@pytest.mark.asyncio
async def test_enqueue_reclaims_expired_capacity():
    """A stale full queue must expire old entries before rejecting new work."""
    queue = RequestQueue(maxsize=1, request_timeout_ms=1.0)
    stale = make_sequence("stale")
    stale_future = await queue.enqueue(stale)
    await asyncio.sleep(0.01)

    fresh = make_sequence("fresh")
    await queue.enqueue(fresh)

    assert stale.state == "expired"
    assert stale_future.done()
    with pytest.raises(asyncio.TimeoutError):
        stale_future.result()
    assert len(queue) == 1


# ── Test 3: timeout expiry ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_timeout_expires_waiting_request():
    """A request that waits beyond request_timeout_ms must be expired."""
    queue = RequestQueue(maxsize=10, request_timeout_ms=50.0)  # 50 ms

    seq = make_sequence()
    future = await queue.enqueue(seq)

    # Wait longer than the timeout
    await asyncio.sleep(0.1)  # 100 ms > 50 ms

    expired_count = await queue.expire_timed_out()

    assert expired_count == 1, f"Expected 1 expired, got {expired_count}"
    assert seq.state == "expired", (
        f"Expected state='expired', got '{seq.state}'"
    )
    assert future.done(), "Future must be resolved after expiry"
    with pytest.raises(asyncio.TimeoutError):
        future.result()


# ── Test 4: explicit cancellation ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_removes_from_queue():
    """cancel() on an existing seq_id removes it and updates state."""
    queue = RequestQueue(maxsize=10)

    seq_a = make_sequence("aaaa")
    seq_b = make_sequence("bbbb")
    future_a = await queue.enqueue(seq_a)
    await queue.enqueue(seq_b)

    assert len(queue) == 2

    result = await queue.cancel("aaaa")

    assert result is True, "cancel() must return True when the item is found"
    assert len(queue) == 1, f"Queue depth should be 1, got {len(queue)}"
    assert seq_a.state == "cancelled", (
        f"Expected state='cancelled', got '{seq_a.state}'"
    )
    assert future_a.done(), "Future must be resolved after cancellation"
    assert future_a.cancelled(), "Cancellation must cancel the future"

    # seq_b must still be in the queue
    remaining = await queue.dequeue()
    assert remaining is not None
    assert remaining.sequence.seq_id == "bbbb"


# ── Test 5: cancel of non-existent seq_id ────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_nonexistent_returns_false():
    """cancel() must return False when the seq_id is not in the queue."""
    queue = RequestQueue(maxsize=10)

    result = await queue.cancel("does-not-exist")
    assert result is False, f"Expected False, got {result}"


# ── Test 6: stats accuracy ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stats_accuracy():
    """stats() must reflect accurate counts after enqueue and dequeue."""
    queue = RequestQueue(maxsize=10)

    await queue.enqueue(make_sequence())
    await queue.enqueue(make_sequence())
    await queue.enqueue(make_sequence())

    # Dequeue one item
    item = await queue.dequeue()
    assert item is not None

    s = queue.stats()

    assert s["queue_depth"] == 2, f"queue_depth: expected 2, got {s['queue_depth']}"
    assert s["total_enqueued"] == 3, (
        f"total_enqueued: expected 3, got {s['total_enqueued']}"
    )
    assert s["maxsize"] == 10
    assert s["oldest_wait_ms"] > 0.0, (
        "oldest_wait_ms must be > 0 for non-empty queue"
    )

    # NOTE: _total_admitted is incremented by the *scheduler* when the sequence
    # enters prefill, not by dequeue() itself.  In this isolated test we are not
    # running the scheduler, so total_admitted remains 0 after dequeue.
    # To simulate what the scheduler does, increment it manually here.
    assert s["total_admitted"] == 0, (
        "total_admitted must be 0 here — scheduler increments it, not dequeue(). "
        f"Got {s['total_admitted']}"
    )

    # Simulate scheduler incrementing admitted after prefill
    queue.mark_admitted()
    s2 = queue.stats()
    assert s2["total_admitted"] == 1, (
        f"After manual increment, total_admitted should be 1, got {s2['total_admitted']}"
    )


# ── Test 7: dequeue on empty queue returns None ───────────────────────────────


@pytest.mark.asyncio
async def test_dequeue_returns_none_on_empty_queue():
    """dequeue() on an empty RequestQueue must return None without raising."""
    queue = RequestQueue(maxsize=10)

    result = await queue.dequeue()
    assert result is None, f"Expected None from empty queue, got {result}"


@pytest.mark.asyncio
async def test_requeue_front_restores_head():
    """A request the scheduler could not admit goes back to the front."""
    queue = RequestQueue(maxsize=10)
    first = make_sequence("first")
    second = make_sequence("second")
    await queue.enqueue(first)
    await queue.enqueue(second)

    item = await queue.dequeue()
    assert item.sequence is first
    await queue.requeue_front(item)

    assert len(queue) == 2
    again = await queue.dequeue()
    assert again.sequence is first
