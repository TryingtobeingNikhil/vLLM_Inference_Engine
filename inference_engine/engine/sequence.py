"""
engine/sequence.py — Sequence state dataclass for the Phase 2 continuous
batching scheduler.

Each request submitted to the scheduler is wrapped in a Sequence object that
carries all mutable state through its lifetime:


    waiting   →  prefill   →  decoding   →  finished
       ↑ created    ↑ admitted    ↑ first token    ↑ EOS or max_new_tokens

    waiting   →  expired    (timed out in queue before prefill)
    waiting   →  cancelled  (explicit cancel before prefill)
    decoding  →  swapped    (KV blocks evicted to CPU under memory pressure;
                             re-enters decoding after swap-in)

    Full valid states in lifecycle order:
        waiting | prefill | decoding | finished | expired | cancelled | swapped

The `past_key_values` field holds the per-sequence HuggingFace KV cache.
Storing KV caches per-sequence is the Phase 2 approach; unified KV cache
management (with eviction/swapping) is Phase 9.

Memory note: for Qwen2-0.5B (fp16) each sequence's KV cache grows by
approximately 3 MB per generated token.  At max_batch_size=4 and
max_new_tokens=50 this adds ~600 MB on top of the ~950 MB model weight
footprint.  Keep max_batch_size ≤ 4 and max_new_tokens ≤ 50 unless you have
confirmed available headroom.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class Sequence:
    """Full mutable state for one scheduled generation request.

    Fields
    ------
    seq_id
        UUID4 hex string — unique identifier for this sequence.
    prompt
        Raw text of the user prompt.
    prompt_token_ids
        Token-id list produced by the tokenizer for `prompt`.
    generated_token_ids
        Token ids produced so far, including the first token from prefill.
        Grows by one per scheduler decode step.
    max_new_tokens
        Upper bound on tokens to generate.
    state
        Lifecycle state.  Mutated exclusively by the scheduler or
        RequestQueue:

        Normal path:  ``"waiting"`` → ``"prefill"`` → ``"decoding"`` → ``"finished"``
        Timeout path: ``"waiting"`` → ``"expired"``
        Cancel path:  ``"waiting"`` → ``"cancelled"``
        Swap path:    ``"decoding"`` → ``"swapped"`` → ``"decoding"``

        All valid states: ``waiting | prefill | decoding | finished | expired | cancelled | swapped``
    past_key_values
        HuggingFace KV cache returned by the last model forward pass.
        ``None`` until prefill completes.
    ttft_ms
        Time-to-first-token in milliseconds.  Set at the end of prefill.
    arrival_time
        ``time.perf_counter()`` timestamp when the Sequence was created.
        Used to compute ``queue_wait_time_ms``.
    first_token_time
        ``time.perf_counter()`` timestamp when the first generated token was
        appended (i.e. at the end of prefill).  ``0.0`` until then.
    per_token_latencies_ms
        Wall-clock latency (ms) for each individual decode step.
        Does **not** include the prefill step; that is captured in ``ttft_ms``.
    finish_reason
        ``"length"``  — stopped because ``len(generated_token_ids) >= max_new_tokens``
        ``"eos"``     — stopped because EOS token was produced
        ``""``        — not yet finished
    queue_wait_time_ms
        Time (ms) from ``arrival_time`` to the moment prefill began.
        Set by the scheduler when the sequence is admitted from the waiting
        queue.  ``0.0`` until then.
    """

    seq_id: str
    prompt: str
    prompt_token_ids: List[int]
    generated_token_ids: List[int]
    max_new_tokens: int
    state: str  # waiting | prefill | decoding | finished | expired | cancelled | swapped
    past_key_values: Any              # HuggingFace KV cache; None until prefill done
    ttft_ms: float
    arrival_time: float
    first_token_time: float
    per_token_latencies_ms: List[float]
    finish_reason: str                # "length" | "eos" | ""
    queue_wait_time_ms: float         # arrival → prefill start; set by scheduler
    kv_token_count: int = 0
    kv_memory_mb: float = 0.0

    # ── Convenience helpers ───────────────────────────────────────────────────

    def is_finished(self) -> bool:
        """Return True when the sequence has reached terminal state."""
        return self.state == "finished"

    def total_tokens(self) -> int:
        """Total token count: prompt tokens + generated tokens so far."""
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    def update_kv_stats(self, token_count: int, memory_mb: float) -> None:
        """Update the informational snapshot of this sequence's KV cache."""
        self.kv_token_count = token_count
        self.kv_memory_mb = memory_mb

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        prompt: str,
        prompt_token_ids: List[int],
        max_new_tokens: int,
    ) -> "Sequence":
        """Create a new Sequence in the ``'waiting'`` state.

        Assigns a fresh UUID4, records the current time as ``arrival_time``,
        and initialises all mutable fields to empty / zero.
        """
        return cls(
            seq_id=uuid.uuid4().hex,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=[],
            max_new_tokens=max_new_tokens,
            state="waiting",
            past_key_values=None,
            ttft_ms=0.0,
            arrival_time=time.perf_counter(),
            first_token_time=0.0,
            per_token_latencies_ms=[],
            finish_reason="",
            queue_wait_time_ms=0.0,
        )
