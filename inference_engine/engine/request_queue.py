"""
engine/request_queue.py — Phase 3 request queue with timeout, cancellation,
and backpressure.

Replaces the raw asyncio.Queue used in Phase 2 with a structured layer that
adds:
  - Hard capacity cap (maxsize) with QueueFullError on overflow
  - Per-request timeout: requests that wait longer than request_timeout_ms
    before entering prefill are expired and their futures resolved with
    asyncio.TimeoutError
  - Explicit cancellation by seq_id
  - Structured stats (queue_depth, total_enqueued, total_expired, total_admitted)

Internal storage
----------------
Uses a plain list (not asyncio.Queue) to allow O(n) expiry scanning and
O(1) front-of-queue dequeue.  All mutations are protected by asyncio.Lock,
which is non-blocking in the asyncio sense (it does not release the GIL to
a thread pool) and appropriate here because every operation is called from
the scheduler's single event-loop task.

Priority field
--------------
QueuedRequest.priority is wired and stored but ordering is NOT changed in
Phase 3 — insertion order (FIFO within all priorities) is preserved.
Priority-aware scheduling is reserved for a future phase.

Future wiring
-------------
Each QueuedRequest carries an asyncio.Future.  It is resolved with an
exception on timeout or cancellation.  The scheduler will resolve it with
the finished Sequence in a future phase; in Phase 3 the /generate endpoint
still uses the polling loop (seq.state == "finished") and the future is
created but not awaited by the server layer.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional

from inference_engine.engine.sequence import Sequence


# ── Exception ─────────────────────────────────────────────────────────────────


class QueueFullError(Exception):
    """Raised by RequestQueue.enqueue() when the queue has reached maxsize."""

    def __init__(self, maxsize: int) -> None:
        super().__init__(f"Request queue is full ({maxsize} requests waiting)")
        self.maxsize = maxsize


# ── QueuedRequest dataclass ───────────────────────────────────────────────────


@dataclass
class QueuedRequest:
    """Wraps a Sequence with queue-management metadata.

    Fields
    ------
    sequence
        The Sequence object representing this generation request.
    enqueue_time
        ``time.perf_counter()`` when the request entered the queue.
        Used by expire_timed_out() to compute elapsed wait time.
    future
        An ``asyncio.Future`` created at enqueue time.  Resolved with:
        - ``asyncio.TimeoutError`` on expiry
        - ``asyncio.CancelledError`` on explicit cancellation
        - (future phase) the finished Sequence on successful completion
    priority
        Lower integer = higher priority.  Reserved for future use; ordering
        is NOT changed in Phase 3 — FIFO is preserved across all priorities.
    """

    sequence: Sequence
    enqueue_time: float
    future: asyncio.Future
    priority: int = 0


# ── RequestQueue ──────────────────────────────────────────────────────────────


class RequestQueue:
    """Structured request queue with timeout enforcement and cancellation.

    Parameters
    ----------
    maxsize
        Hard cap on the number of waiting requests.  Attempting to enqueue
        beyond this limit raises QueueFullError immediately.
    request_timeout_ms
        Maximum time (ms) a request may wait before entering prefill.
        Checked lazily on every dequeue() call via expire_timed_out().
    """

    def __init__(
        self,
        maxsize: int,
        request_timeout_ms: float = 30_000.0,
    ) -> None:
        self.maxsize = maxsize
        self.request_timeout_ms = request_timeout_ms

        # Internal storage — list preserves insertion order for FIFO
        self._queue: List[QueuedRequest] = []
        self._lock: asyncio.Lock = asyncio.Lock()

        # Counters
        self._total_enqueued: int = 0
        self._total_expired: int = 0
        self._total_admitted: int = 0   # incremented by scheduler on prefill entry

    # ── Enqueue ───────────────────────────────────────────────────────────────

    async def enqueue(
        self,
        sequence: Sequence,
        priority: int = 0,
    ) -> asyncio.Future:
        """Add *sequence* to the waiting queue and return its asyncio.Future.

        Parameters
        ----------
        sequence
            A Sequence in state ``"waiting"``.
        priority
            Lower = higher priority.  Stored but not used for ordering in
            Phase 3 (FIFO preserved).

        Returns
        -------
        asyncio.Future
            Resolved with TimeoutError on expiry or CancelledError on cancel.
            In a future phase it will be resolved with the finished Sequence.

        Raises
        ------
        QueueFullError
            If ``len(_queue) >= maxsize`` at the time of enqueue.
        """
        async with self._lock:
            if len(self._queue) >= self.maxsize:
                raise QueueFullError(self.maxsize)

            loop = asyncio.get_event_loop()
            fut: asyncio.Future = loop.create_future()

            item = QueuedRequest(
                sequence=sequence,
                enqueue_time=time.perf_counter(),
                future=fut,
                priority=priority,
            )
            self._queue.append(item)
            self._total_enqueued += 1

        return fut

    # ── Dequeue ───────────────────────────────────────────────────────────────

    async def dequeue(self) -> Optional[QueuedRequest]:
        """Admit the next waiting request into the prefill stage.

        Calls expire_timed_out() first to remove stale entries, then returns
        the first remaining QueuedRequest (FIFO) and removes it from the queue.

        Returns None if the queue is empty after expiry scanning.

        Note: _total_admitted is NOT incremented here — the scheduler
        increments it after the sequence successfully completes prefill,
        separating "dequeued for prefill" from "prefill started".
        """
        # Expire stale entries before considering admission
        await self.expire_timed_out()

        async with self._lock:
            if not self._queue:
                return None
            item = self._queue.pop(0)
            return item

    # ── Expiry ────────────────────────────────────────────────────────────────

    async def expire_timed_out(self) -> int:
        """Scan the queue and expire all requests that have exceeded the timeout.

        For each expired QueuedRequest:
        - Sets ``sequence.state = "expired"``
        - Resolves the future with asyncio.TimeoutError
        - Increments _total_expired
        - Removes the item from _queue

        Returns
        -------
        int
            Number of requests expired in this call.
        """
        now = time.perf_counter()
        expired_count = 0
        still_waiting: List[QueuedRequest] = []

        async with self._lock:
            for item in self._queue:
                elapsed_ms = (now - item.enqueue_time) * 1000.0
                if elapsed_ms > self.request_timeout_ms:
                    item.sequence.state = "expired"
                    if not item.future.done():
                        item.future.set_exception(
                            asyncio.TimeoutError(
                                f"Request {item.sequence.seq_id} timed out "
                                f"after {elapsed_ms:.1f} ms"
                            )
                        )
                    self._total_expired += 1
                    expired_count += 1
                else:
                    still_waiting.append(item)

            self._queue = still_waiting

        return expired_count

    # ── Cancellation ─────────────────────────────────────────────────────────

    async def cancel(self, seq_id: str) -> bool:
        """Cancel a waiting request by seq_id.

        If the request is found in the queue and not already in a terminal
        state, sets its state to ``"cancelled"``, resolves the future with
        asyncio.CancelledError, removes it from the queue, and returns True.

        Returns False if no matching request is found or it has already been
        removed (e.g. already admitted or expired).
        """
        async with self._lock:
            for i, item in enumerate(self._queue):
                if item.sequence.seq_id == seq_id:
                    # Only cancel if still in a cancellable state
                    if item.sequence.state not in ("expired", "cancelled",
                                                   "decoding", "finished"):
                        item.sequence.state = "cancelled"
                        if not item.future.done():
                            item.future.set_exception(
                                asyncio.CancelledError(
                                    f"Request {seq_id} cancelled"
                                )
                            )
                    del self._queue[i]
                    return True

        return False

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a snapshot of queue statistics.

        Returns
        -------
        dict with keys:
            queue_depth      Current number of waiting requests.
            total_enqueued   Cumulative requests ever added to the queue.
            total_expired    Cumulative requests removed by timeout.
            total_admitted   Cumulative requests that entered prefill
                             (incremented by the scheduler, not by dequeue).
            maxsize          The hard capacity cap.
            oldest_wait_ms   Age (ms) of the front-of-queue item;
                             0.0 if the queue is empty.
        """
        now = time.perf_counter()
        oldest_wait_ms = 0.0
        if self._queue:
            oldest_wait_ms = (now - self._queue[0].enqueue_time) * 1000.0

        return {
            "queue_depth": len(self._queue),
            "total_enqueued": self._total_enqueued,
            "total_expired": self._total_expired,
            "total_admitted": self._total_admitted,
            "maxsize": self.maxsize,
            "oldest_wait_ms": oldest_wait_ms,
        }

    # ── Sizing ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return the current number of waiting requests."""
        return len(self._queue)
