"""
engine/scheduler.py — Continuous Batching Scheduler.

Architecture
------------
The scheduler owns one background asyncio task (``run_loop``).  Each
iteration is one *step*:

    add_request() ──► RequestQueue (FIFO, timeouts, backpressure)
                          │
                          ▼  _schedule()
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 1. plan     (event loop, CPU only)                                  │
    │    a. every running sequence in decode gets 1 token (+k speculative │
    │       slots); grow its block table, preempting the NEWEST running    │
    │       sequence (swap to CPU or drop for recompute) if the pool is    │
    │       exhausted                                                      │
    │    b. sequences mid-prefill get their next chunk (≤ chunk size),     │
    │       bounded by the per-step prefill token budget                   │
    │    c. swapped-out sequences are swapped back in when blocks allow    │
    │    d. new requests are admitted FCFS while there are free batch      │
    │       slots, budget and enough KV blocks for their whole prompt —    │
    │       prefix-cache hits are shared instead of recomputed             │
    │ 2. execute  (worker thread) — ONE packed forward pass for the whole  │
    │    plan: prefill chunks and decode tokens together                   │
    │ 3. apply    (event loop) — append sampled tokens, verify speculative │
    │    drafts, stream tokens to clients, publish full blocks to the      │
    │    prefix cache, finish sequences and free their blocks              │
    └─────────────────────────────────────────────────────────────────────┘

Why plan and apply run on the event loop: every piece of scheduler state is
mutated from exactly one thread, so there are no locks to reason about in
the scheduling logic.  Only the GPU work runs in the executor thread, and
the event loop stays free to accept HTTP requests while it runs.

Scheduling policy (FCFS with LIFO preemption, as in vLLM)
---------------------------------------------------------
* ``running`` is kept in arrival order.  When memory runs out, the most
  recently arrived running sequence is preempted first, so older requests
  keep making progress and cannot be starved by a stream of new arrivals.
* Nothing new is admitted in a step that preempted, or while swapped-out
  sequences are waiting to come back.
* Admission requires blocks for the *whole* prompt up front (plus a small
  watermark), so admitted prefills never stall half-way for lack of memory.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence as Seq, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from inference_engine.config import Config
from inference_engine.engine.block_allocator import BlockAllocator, OutOfBlocksError
from inference_engine.engine.cpu_swap_manager import CPUSwapError, CPUSwapManager
from inference_engine.engine.kv_cache_config import (
    compute_kv_cache_config,
    format_kv_cache_report,
)
from inference_engine.engine.kv_cache_tracker import KVCacheTracker
from inference_engine.engine.metrics_aggregator import MetricsAggregator
from inference_engine.engine.model_runner import (
    ModelRunner,
    SequenceInput,
    determine_num_blocks,
    resolve_max_model_len,
)
from inference_engine.engine.paged_kv_cache import PagedKVCacheManager
from inference_engine.engine.request_queue import RequestQueue
from inference_engine.engine.sequence import SamplingParams, Sequence
from inference_engine.engine.sequential import GenerationResult, get_memory_stats
from inference_engine.engine.spec_decode import DraftModelProposer, DraftRequest, NgramProposer
from inference_engine.engine.stage_tracker import StageTracker
from inference_engine.metrics.collector import MetricsCollector
from inference_engine.models.loader import get_eos_token_ids

logger = logging.getLogger(__name__)


@dataclass
class ScheduledItem:
    """One sequence's share of a step."""

    seq: Sequence
    num_tokens: int                     # uncomputed tokens processed this step
    num_lookahead: int = 0              # speculative slots reserved
    draft_tokens: List[int] = field(default_factory=list)
    is_decode: bool = False


@dataclass
class StepPlan:
    items: List[ScheduledItem] = field(default_factory=list)
    num_preempted: int = 0

    def remove(self, seq: Sequence) -> None:
        self.items = [item for item in self.items if item.seq is not seq]


class ContinuousBatchingScheduler:
    """Iteration-level continuous batching scheduler over a paged KV cache.

    Parameters
    ----------
    model
        A loaded HuggingFace causal LM in eval mode.
    tokenizer
        The corresponding tokenizer.
    config
        Engine Config.
    metrics_collector
        Optional shared per-request result store.
    draft_model
        Small model for ``speculative_method="draft"``.
    eos_token_ids
        Override the stop tokens (defaults to generation_config + tokenizer).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: Config,
        metrics_collector: Optional[MetricsCollector] = None,
        draft_model: Optional[PreTrainedModel] = None,
        eos_token_ids: Optional[List[int]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._device: torch.device = next(model.parameters()).device
        self.eos_token_ids = set(
            eos_token_ids if eos_token_ids is not None else get_eos_token_ids(model, tokenizer)
        )

        # ── KV cache sizing ───────────────────────────────────────────────────
        self.kv_cache_config = compute_kv_cache_config(model, config)
        self.max_model_len = resolve_max_model_len(model, config)
        models = [model]
        kv_configs = [self.kv_cache_config]
        self.draft_kv_cache_config = None
        if config.speculative_method == "draft":
            if draft_model is None:
                raise ValueError("speculative_method='draft' requires a draft_model")
            self.draft_kv_cache_config = compute_kv_cache_config(draft_model, config)
            models.append(draft_model)
            kv_configs.append(self.draft_kv_cache_config)
        num_blocks = determine_num_blocks(models, kv_configs, config, self.max_model_len)
        block_size = config.kv_block_size
        if num_blocks * block_size < self.max_model_len:
            logger.warning(
                "KV pool holds %d tokens < max_model_len=%d; capping max_model_len",
                num_blocks * block_size, self.max_model_len,
            )
            self.max_model_len = num_blocks * block_size

        # Phase 6: block allocator (+ prefix cache)
        self.block_allocator = BlockAllocator(
            num_blocks=num_blocks,
            block_size=block_size,
            enable_prefix_caching=config.enable_prefix_caching,
        )
        # Phase 7: the device tensor pool
        self.paged_kv_cache = PagedKVCacheManager(
            kv_cache_config=self.kv_cache_config,
            block_allocator=self.block_allocator,
            config=config,
        )
        # Phase 8: the batched model runner
        self.runner = ModelRunner(model, self.paged_kv_cache, seed=config.seed)
        # Phase 9: CPU staging pool for GPU ⇔ CPU swapping
        num_cpu_blocks = config.kv_num_cpu_blocks or max(
            1, int(config.swap_space_gb * 2**30 // self.kv_cache_config.bytes_per_block(block_size))
        )
        self.cpu_swap_manager = CPUSwapManager(
            kv_cache_config=self.kv_cache_config,
            block_size=block_size,
            num_cpu_blocks=num_cpu_blocks,
        )
        # Phase 5: logical KV accounting
        self.kv_tracker = KVCacheTracker(
            self.kv_cache_config, max_memory_mb=self.paged_kv_cache.pool_size_mb()
        )

        # Speculative decoding
        self.ngram_proposer: Optional[NgramProposer] = None
        self.draft_proposer: Optional[DraftModelProposer] = None
        if config.speculative_method == "ngram":
            self.ngram_proposer = NgramProposer(config.ngram_max, config.ngram_min)
        elif config.speculative_method == "draft":
            self.draft_kv_cache = PagedKVCacheManager(
                kv_cache_config=self.draft_kv_cache_config,
                block_allocator=self.block_allocator,
                config=config,
            )
            self.draft_proposer = DraftModelProposer(
                ModelRunner(draft_model, self.draft_kv_cache, seed=config.seed)
            )

        # ── Scheduling state ──────────────────────────────────────────────────
        self.max_batch_size: int = config.max_batch_size
        self.prefill_budget_tokens: int = config.prefill_budget_tokens
        self.decode_batch_limit: int = config.decode_batch_limit
        self.prefill_chunk_size: int = config.prefill_chunk_size
        self.watermark_blocks: int = int(0.01 * num_blocks)

        self.request_queue: RequestQueue = RequestQueue(
            maxsize=config.max_queue_size,
            request_timeout_ms=config.request_timeout_ms,
        )
        self.running: List[Sequence] = []        # arrival order (FCFS)
        self.swapped_out: List[Sequence] = []    # KV on CPU, waiting to resume
        self.preempted: List[Sequence] = []      # KV dropped, waiting to re-prefill
        history_size = max(1, config.metrics_history_size)
        self.finished: deque[Sequence] = deque(maxlen=history_size)
        self.total_finished: int = 0
        self._futures: Dict[str, asyncio.Future] = {}
        self._pending_aborts: set[str] = set()

        # Counters
        self.num_preemptions_swap = 0
        self.num_preemptions_recompute = 0
        self.num_spec_draft_tokens = 0
        self.num_spec_accepted_tokens = 0
        self.num_spec_steps = 0

        # Metrics
        self.stage_tracker = StageTracker(history_size=500)
        if metrics_collector is None:
            metrics_collector = MetricsCollector(history_size=history_size)
        self._metrics_collector = metrics_collector
        self.metrics_aggregator = MetricsAggregator(
            metrics_collector=self._metrics_collector,
            stage_tracker=self.stage_tracker,
            kv_tracker=self.kv_tracker,
            block_allocator=self.block_allocator,
            paged_kv_cache=self.paged_kv_cache,
            cpu_swap_manager=self.cpu_swap_manager,
            request_queue=self.request_queue,
            history_window_seconds=60.0,
        )
        scheduler_history_size = history_size * 10
        self.batch_size_over_time: deque[Tuple[float, int]] = deque(maxlen=scheduler_history_size)
        self.scheduler_step_latency_ms: deque[float] = deque(maxlen=scheduler_history_size)

        # Control
        self._stop_event: asyncio.Event = asyncio.Event()
        self._loop_task: Optional[asyncio.Task] = None
        # One warm thread for the model (forward passes are serialised) and a
        # separate one for tokenization so new requests never queue behind a
        # forward pass just to be tokenized.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model_runner")
        self._tokenizer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tokenizer")

        logger.info(
            "%s\n  KV pool: %d blocks × %d tokens = %d tokens (%.1f MB)"
            "\n  max_model_len=%d max_batch_size=%d prefill_budget=%d chunk=%d"
            " prefix_caching=%s speculative=%s",
            format_kv_cache_report(self.kv_cache_config),
            num_blocks, block_size, num_blocks * block_size,
            self.paged_kv_cache.pool_size_mb(), self.max_model_len,
            self.max_batch_size, self.prefill_budget_tokens, self.prefill_chunk_size,
            config.enable_prefix_caching, config.speculative_method or "off",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def add_request(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        sampling: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[int]] = None,
        stream: bool = False,
    ) -> Tuple[Sequence, asyncio.Future]:
        """Tokenize *prompt*, create a waiting Sequence and enqueue it.

        Returns ``(sequence, future)``.  The future resolves with the finished
        Sequence, or raises TimeoutError (expired in the queue) /
        CancelledError (aborted before admission).  With ``stream=True``,
        ``sequence.stream`` receives lists of new token ids, then ``None``.

        Raises QueueFullError when the queue is at capacity and ValueError for
        requests that can never be served (empty or too-long prompts).
        """
        if sampling is None:
            sampling = SamplingParams(max_new_tokens=max_new_tokens or self.config.max_new_tokens)
        elif max_new_tokens is not None:
            sampling.max_new_tokens = max_new_tokens
        if sampling.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")

        if prompt_token_ids is None:
            loop = asyncio.get_running_loop()
            prompt_token_ids = await loop.run_in_executor(
                self._tokenizer_executor,
                lambda: self.tokenizer(prompt, add_special_tokens=True)["input_ids"],
            )
        if not prompt_token_ids:
            raise ValueError("The prompt produced no input tokens")
        if len(prompt_token_ids) >= self.max_model_len:
            raise ValueError(
                f"Prompt has {len(prompt_token_ids)} tokens; the engine's "
                f"max_model_len is {self.max_model_len}"
            )

        seq = Sequence.create(prompt=prompt, prompt_token_ids=prompt_token_ids, sampling=sampling)
        if stream:
            seq.stream = asyncio.Queue()
        # QueueFullError propagates to caller — do NOT catch here.
        future = await self.request_queue.enqueue(seq)
        self._futures[seq.seq_id] = future

        def _on_done(done_future: asyncio.Future) -> None:
            if self._futures.get(seq.seq_id) is done_future:
                self._futures.pop(seq.seq_id, None)
            # Expiry / cancellation happen in the queue: unblock stream readers.
            if seq.stream is not None:
                seq.stream.put_nowait(None)

        future.add_done_callback(_on_done)
        return seq, future

    async def abort(self, seq_id: str) -> None:
        """Stop a request (e.g. its client disconnected) and free its memory."""
        if await self.request_queue.cancel(seq_id):
            return
        self._pending_aborts.add(seq_id)

    # ── Step 1: plan ──────────────────────────────────────────────────────────

    def _running_limit(self) -> int:
        return min(self.max_batch_size, self.decode_batch_limit)

    def _lookahead_for(self, seq: Sequence) -> int:
        """Speculative slots for a decode-phase sequence this step."""
        if not self.config.speculative_method or not seq.sampling.is_greedy:
            return 0
        remaining = seq.max_new_tokens - len(seq.generated_token_ids)
        room = self.max_model_len - seq.num_tokens()
        return max(0, min(self.config.num_speculative_tokens, remaining - 1, room))

    def _insert_running(self, seq: Sequence) -> None:
        """Insert into ``running`` keeping arrival (FCFS) order."""
        idx = len(self.running)
        while idx > 0 and self.running[idx - 1].arrival_time > seq.arrival_time:
            idx -= 1
        self.running.insert(idx, seq)

    def _preempt(self, seq: Sequence, plan: StepPlan) -> None:
        """Evict *seq*'s KV blocks: swap them to CPU, or drop them for recompute."""
        plan.remove(seq)
        plan.num_preempted += 1
        self.running.remove(seq)
        seq.num_preemptions += 1
        seq.draft_num_computed_tokens = 0

        # Swapping only pays off for sequences whose blocks are fully computed.
        if self.config.preemption_mode == "swap" and seq.num_uncomputed_tokens() == 1:
            try:
                self.cpu_swap_manager.swap_out(
                    seq.seq_id,
                    self.block_allocator.get_blocks(seq.seq_id),
                    self.paged_kv_cache,
                    self.block_allocator,
                )
                seq.state = "swapped"
                self.swapped_out.append(seq)
                self.num_preemptions_swap += 1
                logger.debug("Swapped out seq_id=%s", seq.seq_id)
                return
            except CPUSwapError:
                pass  # CPU pool full → fall back to recompute
            except Exception:
                # e.g. CUDA OOM on the copy's temporaries: never orphan the
                # sequence — recompute is always possible.
                logger.exception("swap-out failed for seq_id=%s; recomputing", seq.seq_id)
                self.cpu_swap_manager.discard(seq.seq_id)

        self._demote_to_recompute(seq)

    def _demote_to_recompute(self, seq: Sequence) -> None:
        """Drop *seq*'s device KV; it will re-prefill (usually from cache)."""
        self.block_allocator.free(seq.seq_id)
        self.kv_tracker.unregister_sequence(seq.seq_id)
        seq.num_computed_tokens = 0
        seq.draft_num_computed_tokens = 0
        seq.state = "preempted"
        idx = len(self.preempted)
        while idx > 0 and self.preempted[idx - 1].arrival_time > seq.arrival_time:
            idx -= 1
        self.preempted.insert(idx, seq)
        self.num_preemptions_recompute += 1
        logger.debug("Preempted (recompute) seq_id=%s", seq.seq_id)

    def _reserve(self, seq: Sequence, total_tokens: int, plan: StepPlan) -> bool:
        """Make *seq*'s block table cover *total_tokens*, preempting newer
        running sequences if needed.  Returns False if *seq* itself had to be
        preempted."""
        while True:
            try:
                self.block_allocator.ensure_capacity(seq.seq_id, total_tokens)
                return True
            except OutOfBlocksError:
                victims = [s for s in self.running if s is not seq]
                if not victims or self.running[-1] is seq:
                    self._preempt(seq, plan)
                    return False
                self._preempt(self.running[-1], plan)

    def _admit(self, seq: Sequence, now: float) -> bool:
        """Allocate blocks for all of *seq*'s tokens (sharing cached prefixes)."""
        # The watermark reserves headroom for running sequences to grow; with
        # nothing running it would only block the one request that fits.
        watermark = self.watermark_blocks if self.running else 0
        try:
            cached = self.block_allocator.allocate_for_tokens(
                seq.seq_id, seq.all_token_ids(), watermark,
                # re-admissions after preemption would inflate the hit rate
                count_stats=not seq.first_scheduled_time,
            )
        except OutOfBlocksError:
            return False
        if not seq.first_scheduled_time:
            seq.first_scheduled_time = now
            seq.num_cached_tokens = cached
        seq.num_computed_tokens = cached
        seq.state = "prefill"
        self.kv_tracker.register_sequence(seq.seq_id, cached)
        self._insert_running(seq)
        return True

    async def _plan_step(self) -> StepPlan:
        plan = StepPlan()
        budget = self.prefill_budget_tokens
        now = time.perf_counter()

        # a) Decode-phase sequences: one token each (+ speculative slots).
        for seq in list(self.running):
            if seq.state not in ("prefill", "decoding") or seq.num_uncomputed_tokens() != 1:
                continue
            if seq not in self.running:          # preempted as a victim above
                continue
            # A sequence finishing its prompt has nothing to speculate from yet.
            k = self._lookahead_for(seq) if seq.generated_token_ids else 0
            target = seq.num_computed_tokens + 1
            if k:
                try:  # speculation is optional: never preempt just for it
                    self.block_allocator.ensure_capacity(seq.seq_id, target + k)
                except OutOfBlocksError:
                    k = 0
            if not self._reserve(seq, target, plan):
                continue
            plan.items.append(ScheduledItem(seq, 1, num_lookahead=k, is_decode=True))

        # b) Sequences mid-prefill continue with their next chunk (FCFS).
        for seq in list(self.running):
            if budget <= 0:
                break
            if seq not in self.running or seq.num_uncomputed_tokens() <= 1:
                continue
            n = min(seq.num_uncomputed_tokens(), self.prefill_chunk_size, budget)
            if not self._reserve(seq, seq.num_computed_tokens + n, plan):
                continue
            plan.items.append(ScheduledItem(seq, n))
            budget -= n

        # c) Swap sequences back in (oldest first) once memory allows.
        if not plan.num_preempted:
            for seq in list(self.swapped_out):
                if len(self.running) >= self._running_limit():
                    break
                record = self.cpu_swap_manager.get_swapped_sequence(seq.seq_id)
                # Room for its blocks plus the next token (and headroom for
                # others only if anyone else is running).
                needed = max(record.original_num_blocks,
                             self.block_allocator.blocks_needed(seq.num_computed_tokens + 1))
                if self.running:
                    needed += self.watermark_blocks
                free = self.block_allocator.num_free_blocks()
                if free < needed:
                    if self.running:
                        break
                    # Even an empty pool can't take it back with room to grow
                    # (a near-max-length sequence): recompute instead of
                    # waiting forever and blocking every later request.
                    self.cpu_swap_manager.discard(seq.seq_id)
                    self.swapped_out.remove(seq)
                    self.num_preemptions_recompute += 1
                    self._demote_to_recompute(seq)
                    continue
                self.swapped_out.remove(seq)
                try:
                    self.cpu_swap_manager.swap_in(seq.seq_id, self.paged_kv_cache, self.block_allocator)
                except Exception:
                    logger.exception("swap-in failed for seq_id=%s; recomputing", seq.seq_id)
                    self.cpu_swap_manager.discard(seq.seq_id)
                    self._demote_to_recompute(seq)
                    continue
                seq.state = "decoding"
                self._insert_running(seq)
                if self._reserve(seq, seq.num_computed_tokens + 1, plan):
                    plan.items.append(ScheduledItem(seq, 1, is_decode=True))

        # d) Admit: recompute-preempted sequences first, then new requests.
        if not plan.num_preempted and not self.swapped_out:
            while len(self.running) < self._running_limit() and budget > 0:
                if self.preempted:
                    seq = self.preempted[0]
                    if not self._admit(seq, now):
                        break
                    self.preempted.pop(0)
                else:
                    queued = await self.request_queue.dequeue()
                    if queued is None:
                        break
                    seq = queued.sequence
                    if not self._admit(seq, now):
                        await self.request_queue.requeue_front(queued)
                        break
                    self.request_queue.mark_admitted()
                n = min(seq.num_uncomputed_tokens(), self.prefill_chunk_size, budget)
                plan.items.append(ScheduledItem(seq, n))
                budget -= n

        # Speculation: n-gram drafts are cheap CPU work, propose them now.
        if self.ngram_proposer is not None:
            for item in plan.items:
                if item.num_lookahead:
                    item.draft_tokens = self.ngram_proposer.propose(
                        item.seq.all_token_ids(), item.num_lookahead
                    )
        return plan

    # ── Step 2: execute (worker thread) ───────────────────────────────────────

    def _execute(self, plan: StepPlan) -> Dict[str, List[int]]:
        if self.draft_proposer is not None:
            spec_items = [item for item in plan.items if item.num_lookahead]
            if spec_items:
                drafts = self.draft_proposer.propose([
                    DraftRequest(
                        seq_id=item.seq.seq_id,
                        token_ids=item.seq.all_token_ids(),
                        draft_computed=item.seq.draft_num_computed_tokens,
                        block_table=self.block_allocator.get_blocks(item.seq.seq_id),
                        k=item.num_lookahead,
                    )
                    for item in spec_items
                ])
                for item in spec_items:
                    item.draft_tokens = drafts[item.seq.seq_id]

        inputs: List[SequenceInput] = []
        for item in plan.items:
            seq = item.seq
            start = seq.num_computed_tokens
            if item.draft_tokens:
                tokens = [seq.generated_token_ids[-1]] + item.draft_tokens
                num_logits = len(tokens)
            else:
                tokens = seq.all_token_ids()[start:start + item.num_tokens]
                num_logits = 1 if start + item.num_tokens == seq.num_tokens() else 0
            inputs.append(SequenceInput(
                seq_id=seq.seq_id,
                token_ids=tokens,
                start_pos=start,
                block_table=self.block_allocator.get_blocks(seq.seq_id),
                num_logits=num_logits,
                sampling=seq.sampling,
            ))
        return self.runner.execute(inputs)

    # ── Step 3: apply ─────────────────────────────────────────────────────────

    def _stop_reason(self, seq: Sequence, token_id: int) -> str:
        sampling = seq.sampling
        if not sampling.ignore_eos and token_id in self.eos_token_ids:
            return "eos"
        if sampling.stop_token_ids and token_id in sampling.stop_token_ids:
            return "stop"
        if len(seq.generated_token_ids) >= sampling.max_new_tokens:
            return "length"
        if seq.num_tokens() >= self.max_model_len:
            return "length"
        return ""

    def _apply(self, plan: StepPlan, outputs: Dict[str, List[int]]) -> None:
        now = time.perf_counter()
        for item in plan.items:
            seq = item.seq
            if seq.state not in ("prefill", "decoding"):
                continue
            sampled = outputs.get(seq.seq_id)
            before = seq.num_computed_tokens

            if item.draft_tokens:
                draft = item.draft_tokens
                accepted = 0
                while accepted < len(draft) and draft[accepted] == sampled[accepted]:
                    accepted += 1
                new_tokens = draft[:accepted] + [sampled[accepted]]
                seq.num_computed_tokens = before + 1 + accepted
                seq.num_draft_tokens += len(draft)
                seq.num_accepted_tokens += accepted
                self.num_spec_draft_tokens += len(draft)
                self.num_spec_accepted_tokens += accepted
                self.num_spec_steps += 1
                if self.draft_proposer is not None:
                    seq.draft_num_computed_tokens = (before + 1) + min(accepted, len(draft) - 1)
            else:
                seq.num_computed_tokens = before + item.num_tokens
                new_tokens = sampled[:1] if sampled else []

            emitted: List[int] = []
            reason = ""
            for token_id in new_tokens:
                seq.generated_token_ids.append(token_id)
                seq.token_times.append(now)
                if not seq.first_token_time:
                    seq.first_token_time = now
                emitted.append(token_id)
                reason = self._stop_reason(seq, token_id)
                if reason:
                    break

            if seq.generated_token_ids:
                seq.state = "decoding"
            computed = min(seq.num_computed_tokens, seq.num_tokens())
            self.block_allocator.set_token_count(seq.seq_id, computed)
            self.block_allocator.register_computed_blocks(seq.seq_id, seq.all_token_ids(), computed)
            self.kv_tracker.update_sequence(seq.seq_id, computed)
            seq.update_kv_stats(computed, self.kv_tracker.sequence_memory_mb(seq.seq_id))

            if emitted:
                self.metrics_aggregator.record_token_generated(len(emitted))
                if seq.stream is not None:
                    seq.stream.put_nowait(emitted)
            if reason:
                self._finish(seq, reason)

    def _finish(self, seq: Sequence, reason: str, error: str = "") -> None:
        """Move *seq* to a terminal state, release its memory, notify waiters."""
        for pool in (self.running, self.swapped_out, self.preempted):
            if seq in pool:
                pool.remove(seq)
        self.cpu_swap_manager.discard(seq.seq_id)
        self.block_allocator.free(seq.seq_id)
        self.kv_tracker.unregister_sequence(seq.seq_id)

        seq.state = "finished"
        seq.finish_reason = reason
        seq.error_message = error
        seq.finish_time = time.perf_counter()
        self.finished.append(seq)
        self.total_finished += 1
        self._record_result(seq)
        self.metrics_aggregator.record_request_finished(reason)

        future = self._futures.pop(seq.seq_id, None)
        if future is not None and not future.done():
            future.set_result(seq)
        elif seq.stream is not None:
            seq.stream.put_nowait(None)

    def _fail_sequence(self, seq: Sequence, exc: Exception) -> None:
        """Move a sequence to a terminal error state without killing the loop."""
        logger.error("Inference failed for seq_id=%s: %s", seq.seq_id, exc)
        self._finish(seq, "error", str(exc))

    def _record_result(self, seq: Sequence) -> None:
        try:
            allocated_mb, reserved_mb = get_memory_stats(self._device.type)
            total_latency_ms = seq.e2e_latency_ms
            n_gen = len(seq.generated_token_ids)
            self._metrics_collector.append(
                GenerationResult(
                    prompt=seq.prompt,
                    generated_text="",   # decoded lazily by the server; keeps the loop fast
                    prompt_tokens=len(seq.prompt_token_ids),
                    generated_tokens=n_gen,
                    ttft_ms=seq.ttft_ms,
                    total_latency_ms=total_latency_ms,
                    tokens_per_second=(n_gen / total_latency_ms * 1000.0) if total_latency_ms > 0 else 0.0,
                    per_token_latencies_ms=seq.itl_ms,
                    gpu_memory_allocated_mb=allocated_mb,
                    gpu_memory_reserved_mb=reserved_mb,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    finish_reason=seq.finish_reason,
                    queue_wait_time_ms=seq.queue_wait_time_ms,
                    tpot_ms=seq.tpot_ms,
                    cached_prompt_tokens=seq.num_cached_tokens,
                    num_preemptions=seq.num_preemptions,
                    spec_draft_tokens=seq.num_draft_tokens,
                    spec_accepted_tokens=seq.num_accepted_tokens,
                )
            )
        except Exception:
            logger.exception("Failed to record metrics for seq_id=%s", seq.seq_id)

    def _process_aborts(self) -> None:
        if not self._pending_aborts:
            return
        for pool in (self.running, self.swapped_out, self.preempted):
            for seq in list(pool):
                if seq.seq_id in self._pending_aborts or seq.abort_requested:
                    self._finish(seq, "abort")
        self._pending_aborts.clear()

    # ── Scheduler step ────────────────────────────────────────────────────────

    async def _schedule(self) -> None:
        """Run one step: plan → one batched forward pass → apply."""
        step_start = time.perf_counter()
        await self.request_queue.expire_timed_out()
        self._process_aborts()

        plan = await self._plan_step()
        if not plan.items:
            await asyncio.sleep(self.config.scheduler_poll_interval_ms / 1000.0)
            return

        forward_start = time.perf_counter()
        loop = asyncio.get_running_loop()
        try:
            outputs = await loop.run_in_executor(self._executor, self._execute, plan)
        except Exception as exc:  # keep serving other requests
            logger.exception("Model execution failed; failing %d sequences", len(plan.items))
            for item in plan.items:
                if not item.seq.is_finished():
                    self._fail_sequence(item.seq, exc)
            return
        forward_ms = (time.perf_counter() - forward_start) * 1000.0
        self._apply(plan, outputs)

        # ── Telemetry ─────────────────────────────────────────────────────────
        decode_items = [item for item in plan.items if item.is_decode]
        prefill_items = [item for item in plan.items if not item.is_decode]
        prefill_tokens = sum(item.num_tokens for item in prefill_items)
        num_tokens = prefill_tokens + sum(1 + len(item.draft_tokens) for item in decode_items)
        self.stage_tracker.record_prefill(
            sequences_prefilled=sum(1 for item in prefill_items if item.seq.generated_token_ids),
            tokens_prefilled=prefill_tokens,
            latency_ms=forward_ms,
            budget_tokens=self.prefill_budget_tokens,
        )
        self.stage_tracker.record_decode(
            sequences_decoded=len(decode_items),
            latency_ms=forward_ms,
            batch_limit=self.decode_batch_limit,
        )
        step_ms = (time.perf_counter() - step_start) * 1000.0
        num_blocks = self.block_allocator.num_blocks
        self.metrics_aggregator.record_step(
            num_seqs=len(plan.items),
            num_tokens=num_tokens,
            num_prefill_tokens=prefill_tokens,
            num_decode_seqs=len(decode_items),
            num_preempted=plan.num_preempted,
            kv_utilization=self.block_allocator.num_used_blocks() / num_blocks,
            forward_ms=forward_ms,
            step_ms=step_ms,
        )
        self.scheduler_step_latency_ms.append(step_ms)
        self.batch_size_over_time.append((time.perf_counter(), len(self.running)))

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def has_work(self) -> bool:
        return bool(
            self.running or self.swapped_out or self.preempted or len(self.request_queue)
        )

    async def run_loop(self) -> None:
        """Main scheduler loop — runs as a background asyncio Task."""
        idle_sleep_s = self.config.scheduler_poll_interval_ms / 1000.0
        logger.info("Scheduler run_loop started (idle_sleep=%.3f s)", idle_sleep_s)
        while not self._stop_event.is_set():
            if not self.has_work():
                self._process_aborts()
                await asyncio.sleep(idle_sleep_s)
                continue
            try:
                await self._schedule()
            except Exception:
                logger.exception("Scheduler step failed")
                await asyncio.sleep(idle_sleep_s)
        logger.info("Scheduler run_loop stopped.")

    def start(self) -> None:
        """Schedule run_loop() as a background asyncio Task (needs a running loop)."""
        if self._stop_event.is_set():
            raise RuntimeError("A stopped scheduler cannot be restarted")
        if self._loop_task is not None and not self._loop_task.done():
            raise RuntimeError("Scheduler is already running")
        self._loop_task = asyncio.create_task(self.run_loop(), name="scheduler_loop")

    async def stop(self) -> None:
        """Gracefully stop the scheduler loop and await its completion."""
        self._stop_event.set()
        if self._loop_task is not None:
            try:
                await asyncio.wait_for(self._loop_task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Scheduler loop did not stop within 10 s; cancelling.")
                self._loop_task.cancel()
                try:
                    await self._loop_task
                except asyncio.CancelledError:
                    pass
        for future in list(self._futures.values()):
            if not future.done():
                future.cancel()
        self._futures.clear()
        self._executor.shutdown(wait=False)
        self._tokenizer_executor.shutdown(wait=False)
        logger.info("Scheduler stopped. Finished %d sequences.", self.total_finished)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def spec_decode_stats(self) -> dict:
        drafted = self.num_spec_draft_tokens
        steps = self.num_spec_steps
        return {
            "method": self.config.speculative_method or "off",
            "num_speculative_tokens": self.config.num_speculative_tokens,
            "draft_tokens": drafted,
            "accepted_tokens": self.num_spec_accepted_tokens,
            "acceptance_rate": self.num_spec_accepted_tokens / drafted if drafted else 0.0,
            # tokens emitted per verification step (1.0 = no gain)
            "mean_tokens_per_step": (
                (self.num_spec_accepted_tokens + steps) / steps if steps else 0.0
            ),
        }

    def get_metrics(self) -> dict:
        """Return a single unified metrics dict via MetricsAggregator."""
        report = self.metrics_aggregator.full_report(
            requests_in_flight=len(self.running) + len(self.swapped_out) + len(self.preempted),
            requests_waiting=len(self.request_queue),
        )
        report["summary"] = self._metrics_collector.compute_summary()
        report["engine"] = {
            "model": getattr(self.model.config, "_name_or_path", ""),
            "device": str(self._device),
            "dtype": str(self.kv_cache_config.dtype),
            "max_model_len": self.max_model_len,
            "max_batch_size": self.max_batch_size,
            "prefill_budget_tokens": self.prefill_budget_tokens,
            "prefill_chunk_size": self.prefill_chunk_size,
            "num_running": len(self.running),
            "num_swapped": len(self.swapped_out),
            "num_preempted_waiting": len(self.preempted),
            "preemptions_swap": self.num_preemptions_swap,
            "preemptions_recompute": self.num_preemptions_recompute,
            "kv_blocks_total": self.block_allocator.num_blocks,
            "kv_blocks_used": self.block_allocator.num_used_blocks(),
            "prefix_cache_hit_rate": self.block_allocator.prefix_cache_hit_rate(),
        }
        report["speculative_decoding"] = self.spec_decode_stats()
        if self._device.type == "cuda":
            report["gpu_memory"] = {
                "allocated_mb": torch.cuda.memory_allocated() / 2**20,
                "reserved_mb": torch.cuda.memory_reserved() / 2**20,
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            }
        report["scheduler"] = {
            "batch_size_over_time": list(self.batch_size_over_time)[-500:],
            "scheduler_step_latency_ms": list(self.scheduler_step_latency_ms)[-500:],
        }
        report["sequences"] = [
            {
                "seq_id": seq.seq_id,
                "state": seq.state,
                "finish_reason": seq.finish_reason,
                "prompt_tokens": len(seq.prompt_token_ids),
                "generated_tokens": len(seq.generated_token_ids),
                "cached_prompt_tokens": seq.num_cached_tokens,
                "ttft_ms": seq.ttft_ms,
                "tpot_ms": seq.tpot_ms,
                "queue_wait_time_ms": seq.queue_wait_time_ms,
                "num_preemptions": seq.num_preemptions,
            }
            for seq in list(self.finished)[-100:]
        ]
        return report
