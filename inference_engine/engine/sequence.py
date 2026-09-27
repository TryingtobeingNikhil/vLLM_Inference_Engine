"""
engine/sequence.py — Per-request state for the continuous batching scheduler.

Each request submitted to the scheduler is wrapped in a Sequence object that
carries all mutable state through its lifetime:

    waiting ──► prefill ──► decoding ──► finished
       │           │            │
       │           └────────────┴──► swapped    (KV blocks copied to CPU;
       │                                          resumes in "decoding")
       │           └────────────┴──► preempted  (KV blocks dropped; resumes
       │                                          by re-prefilling)
       ├──► expired    (timed out in the queue)
       └──► cancelled  (client went away before admission)

The one number that drives scheduling
-------------------------------------
``num_computed_tokens`` counts how many of the sequence's tokens
(prompt + generated) already have their K/V in the paged cache.  Every step a
sequence processes some of its *uncomputed* tokens:

* prefill          — many uncomputed prompt tokens (processed in chunks)
* decode           — exactly one uncomputed token: the last sampled one
* after preemption — every token is uncomputed again (recompute)

When a step processes the final uncomputed token, the logits at that position
predict the next token.  Prefill, chunked prefill, decode and recompute are
therefore the *same* operation with different token counts, which is what lets
one batched forward pass mix them freely.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SamplingParams:
    """Per-request generation controls.

    ``temperature == 0`` means greedy decoding (the default, and the only
    mode speculative decoding applies to).  ``ignore_eos`` keeps generating
    until ``max_new_tokens`` — benchmarks use it to fix output lengths.
    """

    max_new_tokens: int = 50
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    ignore_eos: bool = False
    stop_token_ids: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0


@dataclass
class Sequence:
    """Full mutable state for one scheduled generation request.

    Timing fields are ``time.perf_counter()`` timestamps; derived latencies
    are exposed as properties so they are always consistent with each other:

    * ``ttft_ms``            arrival → first generated token (includes queueing)
    * ``queue_wait_time_ms`` arrival → first scheduled
    * ``itl_ms``             gaps between consecutive token emissions
    * ``per_token_latencies_ms`` alias of ``itl_ms`` (kept for API stability)
    """

    seq_id: str
    prompt: str
    prompt_token_ids: List[int]
    sampling: SamplingParams
    generated_token_ids: List[int] = field(default_factory=list)
    state: str = "waiting"   # waiting | prefill | decoding | swapped | preempted | finished | expired | cancelled
    finish_reason: str = ""  # "length" | "eos" | "stop" | "abort" | "oom" | "error"
    error_message: str = ""

    # ── Cache bookkeeping ─────────────────────────────────────────────────────
    num_computed_tokens: int = 0
    num_cached_tokens: int = 0        # prompt tokens served from the prefix cache
    num_preemptions: int = 0
    # Speculative decoding
    draft_num_computed_tokens: int = 0  # draft-model KV progress
    num_draft_tokens: int = 0           # proposed
    num_accepted_tokens: int = 0        # accepted by the target model

    # ── Timing ────────────────────────────────────────────────────────────────
    arrival_time: float = field(default_factory=time.perf_counter)
    first_scheduled_time: float = 0.0
    first_token_time: float = 0.0
    finish_time: float = 0.0
    token_times: List[float] = field(default_factory=list)

    # KV tracker snapshot (informational)
    kv_token_count: int = 0
    kv_memory_mb: float = 0.0

    # Streaming: the scheduler pushes lists of new token ids, then None.
    stream: Optional[asyncio.Queue] = None
    abort_requested: bool = False

    # ── Convenience helpers ───────────────────────────────────────────────────

    @property
    def max_new_tokens(self) -> int:
        return self.sampling.max_new_tokens

    def num_tokens(self) -> int:
        """Total token count: prompt tokens + generated tokens so far."""
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    total_tokens = num_tokens  # backwards-compatible name

    def all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.generated_token_ids

    def num_uncomputed_tokens(self) -> int:
        return self.num_tokens() - self.num_computed_tokens

    def is_prefill_done(self) -> bool:
        """True once only the last sampled token (if any) lacks KV."""
        return self.num_uncomputed_tokens() <= 1 and bool(self.generated_token_ids)

    def is_finished(self) -> bool:
        """Return True when the sequence has reached a terminal state."""
        return self.state in ("finished", "expired", "cancelled")

    def update_kv_stats(self, token_count: int, memory_mb: float) -> None:
        """Update the informational snapshot of this sequence's KV cache."""
        self.kv_token_count = token_count
        self.kv_memory_mb = memory_mb

    # ── Derived latency metrics ───────────────────────────────────────────────

    @property
    def ttft_ms(self) -> float:
        if not self.first_token_time:
            return 0.0
        return (self.first_token_time - self.arrival_time) * 1000.0

    @property
    def queue_wait_time_ms(self) -> float:
        if not self.first_scheduled_time:
            return 0.0
        return (self.first_scheduled_time - self.arrival_time) * 1000.0

    @property
    def itl_ms(self) -> List[float]:
        """Gaps between token *emissions*.  A speculative step emits several
        tokens at once; those count as one emission (as a streaming client
        sees them), so ITL never contains artificial zero gaps."""
        times = self.token_times
        return [(b - a) * 1000.0 for a, b in zip(times, times[1:]) if b > a]

    @property
    def per_token_latencies_ms(self) -> List[float]:
        return self.itl_ms

    @property
    def e2e_latency_ms(self) -> float:
        end = self.finish_time or time.perf_counter()
        return (end - self.arrival_time) * 1000.0

    @property
    def tpot_ms(self) -> float:
        """Mean time per output token after the first one."""
        n = len(self.generated_token_ids)
        if n < 2 or not self.first_token_time or not self.token_times:
            return 0.0
        return (self.token_times[-1] - self.first_token_time) * 1000.0 / (n - 1)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        prompt: str,
        prompt_token_ids: List[int],
        max_new_tokens: Optional[int] = None,
        sampling: Optional[SamplingParams] = None,
    ) -> "Sequence":
        """Create a new Sequence in the ``'waiting'`` state."""
        if sampling is None:
            sampling = SamplingParams(max_new_tokens=max_new_tokens or 50)
        elif max_new_tokens is not None:
            sampling.max_new_tokens = max_new_tokens
        return cls(
            seq_id=uuid.uuid4().hex,
            prompt=prompt,
            prompt_token_ids=list(prompt_token_ids),
            sampling=sampling,
        )
