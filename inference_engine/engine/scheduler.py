"""
engine/scheduler.py — Continuous Batching Scheduler (Phase 2).

Architecture
------------
The scheduler owns a single background asyncio task (`run_loop`) that runs
continuously for the lifetime of the server. Blocking model operations are
dispatched to worker threads so they do not stall the event loop.

                       ┌─────────────────────────────────────────┐
  add_request()  ──►  │  RequestQueue  (Phase 3)               │
                       └────────────────────┬────────────────────┘
                                            │  _schedule()
                                            │   1. evict finished
                                            │   2. admit new (up to headroom)
                                            │   3. prefill each new sequence
                                            │   4. one decode step per running seq
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │  running  (list[Sequence])              │
                       └────────────────────┬────────────────────┘
                                            │  state == "finished"
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │  finished  (list[Sequence])             │
                       └─────────────────────────────────────────┘

Decode step design (Gap 1 resolution)
--------------------------------------
`_decode_step` does NOT call Phase 1's `decode()` function, which would run
the full remaining-token loop and block all other sequences for their entire
generation.  Instead, it performs a raw single-token model forward pass — the
same pattern as the inner loop of `decode()` in sequential.py — one step at a
time.  After each step the scheduler can preempt and serve other sequences.

Threading
---------
Tokenization retains the scheduler's shared executor. Phase 4 prefill and
decode helpers are blocking functions dispatched with ``asyncio.to_thread``.
The scheduler awaits each call, so model execution remains serial until true
tensor-level batching is introduced in Phase 8.

Memory (Gap 3 acknowledgement)
--------------------------------
Each Sequence holds a separate `past_key_values` object.  For Qwen2-0.5B
(fp16) this grows by ~3 MB per generated token.  At max_batch_size=4 and
max_new_tokens=50 this adds ~600 MB on top of the ~950 MB model weight
footprint.  KV cache eviction and swapping are Phase 9 features.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from inference_engine.config import Config
from inference_engine.engine.attention_wrapper import build_past_key_values, extract_new_token_kv
from inference_engine.engine.block_allocator import BlockAllocator, OutOfBlocksError
from inference_engine.engine.kv_cache_config import (
    compute_kv_cache_config,
    format_kv_cache_report,
)
from inference_engine.engine.kv_cache_tracker import KVCacheTracker
from inference_engine.engine.paged_kv_cache import PagedKVCacheManager
from inference_engine.engine.prefill_utils import run_prefill_single
from inference_engine.engine.request_queue import RequestQueue
from inference_engine.engine.sequence import Sequence
from inference_engine.engine.stage_tracker import StageTracker
from inference_engine.engine.sequential import get_memory_stats

logger = logging.getLogger(__name__)


# ── Compatibility helper ───────────────────────────────────────────────────────

def _extract_kv_layer(
    past_key_values,
    layer_idx: int,
) -> tuple:
    """Extract (key, value) tensors for one layer from past_key_values.

    Handles three formats returned by different transformers versions:

    1. Legacy tuple-of-tuples (transformers < ~4.36):
       ``past_key_values[layer_idx]`` → ``(key_tensor, value_tensor)``
       shape: [batch, num_kv_heads, seq_len, head_dim]

    2. DynamicCache with key_cache / value_cache lists (transformers ~4.36-4.40):
       ``past_key_values.key_cache[layer_idx]`` → key tensor
       ``past_key_values.value_cache[layer_idx]`` → value tensor

    3. DynamicCache with layers list (transformers >= ~4.41):
       ``past_key_values.layers[layer_idx].keys`` → key tensor
       ``past_key_values.layers[layer_idx].values`` → value tensor
       shape: [batch, num_kv_heads, seq_len, head_dim]

    Returns
    -------
    (key_tensor, value_tensor)
        Both shaped [batch, num_kv_heads, seq_len, head_dim].
    """
    # Format 3: DynamicCache with .layers list holding DynamicLayer objects
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_idx]
        return layer.keys, layer.values

    # Format 2: DynamicCache with .key_cache / .value_cache lists
    if hasattr(past_key_values, "key_cache"):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]

    # Format 1: Legacy tuple-of-tuples
    layer = past_key_values[layer_idx]
    return layer[0], layer[1]


class ContinuousBatchingScheduler:
    """Iteration-level continuous batching scheduler.

    Parameters
    ----------
    model
        A loaded HuggingFace causal LM in eval mode.
    tokenizer
        The corresponding tokenizer (pad_token already set).
    config
        Engine Config; reads ``max_batch_size`` and ``scheduler_poll_interval_ms``.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: Config,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.kv_cache_config = compute_kv_cache_config(model, config)
        allocated_mb, _ = get_memory_stats(config.device)
        available_mb = getattr(config, "kv_cache_max_memory_mb", 1024.0)
        self.kv_tracker = KVCacheTracker(
            self.kv_cache_config, max_memory_mb=available_mb
        )
        logger.info(
            "%s\n  Current allocated memory: %.2f MB",
            format_kv_cache_report(self.kv_cache_config),
            allocated_mb,
        )

        # Phase 6: Block allocator
        self.block_allocator = BlockAllocator(
            num_blocks=config.kv_num_blocks,
            block_size=config.kv_block_size,
        )

        # Phase 7: Paged KV cache tensor pool
        self.paged_kv_cache = PagedKVCacheManager(
            kv_cache_config=self.kv_cache_config,
            block_allocator=self.block_allocator,
            config=config,
        )
        kv_stats = self.paged_kv_cache.stats()
        print(
            f"[PagedKVCache] Pool size: {kv_stats['pool_size_mb']:.1f} MB "
            f"({kv_stats['num_blocks']} blocks \u00d7 {kv_stats['block_size']} tokens)"
        )

        self.max_batch_size: int = config.max_batch_size
        self.prefill_budget_tokens: int = config.prefill_budget_tokens
        self.decode_batch_limit: int = config.decode_batch_limit
        self.stage_tracker = StageTracker(history_size=500)
        self._futures: dict[str, asyncio.Future] = {}

        # Phase 3 request queue — replaces raw asyncio.Queue.
        # maxsize = 8× batch size gives reasonable queue depth before 503.
        # request_timeout_ms comes from config (default 30 s).
        self.request_queue: RequestQueue = RequestQueue(
            maxsize=config.max_batch_size * 8,
            request_timeout_ms=getattr(config, "request_timeout_ms", 30_000.0),
        )
        self.running: List[Sequence] = []      # currently active sequences
        self.finished: List[Sequence] = []     # completed sequences

        # Control
        self._stop_event: asyncio.Event = asyncio.Event()
        self._loop_task: Optional[asyncio.Task] = None

        # Single shared executor — one warm thread, no per-token spawning.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="scheduler_inference"
        )

        # Metrics
        # list of (timestamp: float, batch_size: int)
        self.batch_size_over_time: List[Tuple[float, int]] = []
        # wall-clock duration of each _schedule() call in ms
        self.scheduler_step_latency_ms: List[float] = []

        # Device resolved once from model parameters
        self._device: torch.device = next(model.parameters()).device

        logger.info(
            "ContinuousBatchingScheduler init: max_batch_size=%d device=%s",
            self.max_batch_size,
            self._device,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def add_request(
        self, prompt: str, max_new_tokens: int
    ) -> Tuple[Sequence, asyncio.Future]:
        """Tokenize *prompt*, create a Sequence in state 'waiting', enqueue it.

        Returns a ``(Sequence, asyncio.Future)`` tuple:
        - Sequence: the caller's handle for polling ``seq.state == 'finished'``
          (used by the Phase 3 server polling loop).
        - asyncio.Future: wired to the request lifecycle; resolved with
          TimeoutError on expiry or CancelledError on cancellation.  Will be
          resolved with the finished Sequence in a later phase.

        Raises QueueFullError if the RequestQueue is at capacity — propagated
        to the caller; the server layer converts it to HTTP 503.

        Tokenization runs in the shared executor to avoid blocking the event
        loop on long prompts.
        """
        loop = asyncio.get_event_loop()
        prompt_token_ids: List[int] = await loop.run_in_executor(
            self._executor,
            lambda: self.tokenizer(prompt, add_special_tokens=True)["input_ids"],
        )

        seq = Sequence.create(
            prompt=prompt,
            prompt_token_ids=list(prompt_token_ids),
            max_new_tokens=max_new_tokens,
        )
        # QueueFullError propagates to caller — do NOT catch here.
        future = await self.request_queue.enqueue(seq)
        self._futures[seq.seq_id] = future
        logger.debug("Enqueued seq_id=%s prompt_len=%d", seq.seq_id, len(prompt_token_ids))
        return seq, future

    # ── Internal: prefill ─────────────────────────────────────────────────────

    def _prefill_sequence(self, seq: Sequence) -> None:
        """Run the prefill pass for *seq* and transition it to 'decoding'.

        Sets:
        - seq.past_key_values
        - seq.generated_token_ids (first token appended)
        - seq.ttft_ms
        - seq.first_token_time
        - seq.queue_wait_time_ms
        - seq.state = "decoding"
        """
        # Record queue wait before blocking
        prefill_start = time.perf_counter()
        seq.queue_wait_time_ms = (prefill_start - seq.arrival_time) * 1000.0
        seq.state = "prefill"

        past_key_values, first_token_id, ttft_ms = run_prefill_single(
            self.model,
            seq.prompt_token_ids,
            self._device,
        )

        seq.past_key_values = past_key_values
        seq.ttft_ms = ttft_ms
        seq.first_token_time = time.perf_counter()
        seq.generated_token_ids.append(first_token_id)
        seq.state = "decoding"
        prompt_token_count = len(seq.prompt_token_ids)
        self.kv_tracker.register_sequence(seq.seq_id, prompt_token_count)
        seq.update_kv_stats(
            token_count=prompt_token_count,
            memory_mb=self.kv_tracker.sequence_memory_mb(seq.seq_id),
        )

        # Phase 6: allocate KV cache blocks for the prompt
        import math
        blocks_needed = math.ceil(prompt_token_count / self.config.kv_block_size)
        try:
            block_ids = self.block_allocator.allocate(seq.seq_id, blocks_needed)
            # Mark tokens_used in the last block
            tokens_in_last_block = prompt_token_count % self.config.kv_block_size
            if tokens_in_last_block > 0:
                self.block_allocator._blocks[block_ids[-1]].tokens_used = tokens_in_last_block
                self.block_allocator._blocks[block_ids[-1]].is_dirty = True
            else:
                # Prompt exactly filled all blocks
                self.block_allocator._blocks[block_ids[-1]].tokens_used = self.config.kv_block_size
                self.block_allocator._blocks[block_ids[-1]].is_dirty = True
        except OutOfBlocksError:
            seq.state = "finished"
            seq.finish_reason = "oom"
            return

        # Phase 7: write prompt KV tensors into paged pool
        # HuggingFace past_key_values may be a tuple-of-tuples (legacy) or
        # DynamicCache (transformers >= 4.38). Use the compat helper.
        past_kv = seq.past_key_values
        seq_len = len(seq.prompt_token_ids)
        for layer_idx in range(self.kv_cache_config.num_layers):
            layer_key, layer_val = _extract_kv_layer(past_kv, layer_idx)
            # shape: [1, num_kv_heads, seq_len, head_dim]
            for token_position in range(seq_len):
                key_slice = layer_key[0, :, token_position, :]
                val_slice = layer_val[0, :, token_position, :]
                self.paged_kv_cache.write_kv(
                    seq.seq_id, layer_idx, token_position, key_slice, val_slice
                )
        # Phase 8: release HuggingFace KV tensor — pool is now source of truth
        seq.past_key_values = None

        logger.debug(
            "Prefill done: seq_id=%s ttft=%.1f ms first_token=%d",
            seq.seq_id,
            ttft_ms,
            first_token_id,
        )

    # ── Internal: single decode step ──────────────────────────────────────────

    def _decode_one_step_blocking(self, seq: Sequence) -> None:
        """Run ONE raw model forward pass for *seq*.

        This is the inner loop of sequential.py's decode() extracted to run
        exactly once.  Bypasses decode() entirely — that function would block
        for the full remaining-token loop.

        Updates seq.past_key_values, appends the new token, records latency,
        and sets finish state if EOS or max_new_tokens is reached.

        This is a BLOCKING function intended to run inside the shared executor.
        """
        self._decode_step_single(seq)

    def _decode_step_single(self, seq: Sequence) -> None:
        """Run one blocking decode step — reads KV state from the paged pool.

        Phase 8: past_key_values is reconstructed from the paged pool before
        each forward pass.  seq.past_key_values remains None throughout decode.
        """
        t_step = time.perf_counter()

        # Reconstruct past_key_values from paged pool (Phase 8 pool-read path)
        past_kv = build_past_key_values(
            seq_id=seq.seq_id,
            paged_kv_cache=self.paged_kv_cache,
            num_layers=self.kv_cache_config.num_layers,
            device=str(self._device),
            use_dynamic_cache=True,
        )

        # Determine next input token
        last_token_id = (
            seq.generated_token_ids[-1]
            if seq.generated_token_ids
            else seq.prompt_token_ids[-1]
        )
        input_ids = torch.tensor(
            [[last_token_id]], dtype=torch.long, device=self._device
        )

        # Forward pass — one token, with reconstructed KV from paged pool
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_kv,
                use_cache=True,
            )

        # Greedy sample
        next_token_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
        seq.generated_token_ids.append(next_token_id)

        # Write new token's KV into the paged pool
        new_past_kv = outputs.past_key_values
        token_position = len(seq.prompt_token_ids) + len(seq.generated_token_ids) - 1
        for layer_idx in range(self.kv_cache_config.num_layers):
            key_slice, value_slice = extract_new_token_kv(
                new_past_kv, layer_idx, token_position
            )
            self.paged_kv_cache.write_kv(
                seq.seq_id, layer_idx, token_position, key_slice, value_slice
            )
        # Do NOT store new_past_kv on seq — pool is source of truth
        # seq.past_key_values remains None after Phase 8

        # Update block allocator token tracking
        try:
            self.block_allocator.write_token(seq.seq_id, count=1)
        except OutOfBlocksError:
            seq.state = "finished"
            seq.finish_reason = "oom"
            return

        # Update KV tracker
        total_tokens = len(seq.prompt_token_ids) + len(seq.generated_token_ids)
        self.kv_tracker.update_sequence(seq.seq_id, total_tokens)
        seq.update_kv_stats(
            token_count=total_tokens,
            memory_mb=self.kv_tracker.sequence_memory_mb(seq.seq_id),
        )

        # Record per-token latency
        step_ms = (time.perf_counter() - t_step) * 1000.0
        seq.per_token_latencies_ms.append(step_ms)

        # Check termination
        eos_id = self.tokenizer.eos_token_id
        eos_hit = eos_id is not None and next_token_id == eos_id
        length_hit = len(seq.generated_token_ids) >= seq.max_new_tokens

        if eos_hit:
            seq.finish_reason = "eos"
            seq.state = "finished"
            logger.debug("seq_id=%s finished (eos)", seq.seq_id)
        elif length_hit:
            seq.finish_reason = "length"
            seq.state = "finished"
            logger.debug("seq_id=%s finished (length)", seq.seq_id)

        if seq.is_finished():
            self.kv_tracker.unregister_sequence(seq.seq_id)

    async def _decode_step(self) -> None:
        """Run one decode step for every sequence in self.running.

        Sequences are decoded serially within the shared executor (one warm
        thread).  This is the Phase 2 constraint: true tensor-level batching
        across sequences is deferred to Phase 8.

        The iteration-level scheduling value is still demonstrated: at every
        call to _schedule(), all running sequences advance by exactly one token
        before any one sequence monopolises future steps.
        """
        if not self.running:
            return

        loop = asyncio.get_event_loop()
        for seq in list(self.running):   # snapshot — finished ones removed later
            if seq.state != "decoding":
                continue
            await loop.run_in_executor(
                self._executor,
                self._decode_step_single,
                seq,
            )

    def _resolve_sequence_future(self, seq: Sequence) -> None:
        """Resolve the lifecycle future associated with a finished sequence."""
        future = self._futures.get(seq.seq_id)
        if future is not None and not future.done():
            future.set_result(seq)
        # Phase 7: zero paged pool slots before releasing blocks
        self.paged_kv_cache.clear_sequence(seq.seq_id)
        # Phase 6: release block allocator memory for this sequence
        self.block_allocator.free(seq.seq_id)

    # ── Internal: scheduler step ──────────────────────────────────────────────

    async def _schedule(self) -> None:
        """Run the prefill stage, then one decode pass over active sequences."""
        step_start = time.perf_counter()

        # --- Stage 1: Prefill ---
        prefill_start = time.perf_counter()
        tokens_this_iteration = 0
        sequences_prefilled = 0
        running_limit = min(self.max_batch_size, self.decode_batch_limit)

        while len(self.running) < running_limit:
            queued = await self.request_queue.dequeue()
            if queued is None:
                break
            seq = queued.sequence
            prompt_len = len(seq.prompt_token_ids)

            if tokens_this_iteration + prompt_len > self.prefill_budget_tokens:
                async with self.request_queue._lock:
                    self.request_queue._queue.insert(0, queued)
                break

            tokens_this_iteration += prompt_len
            await asyncio.to_thread(self._prefill_sequence, seq)
            self.running.append(seq)
            self.request_queue._total_admitted += 1
            sequences_prefilled += 1

        prefill_latency_ms = (time.perf_counter() - prefill_start) * 1000.0
        self.stage_tracker.record_prefill(
            sequences_prefilled=sequences_prefilled,
            tokens_prefilled=tokens_this_iteration,
            latency_ms=prefill_latency_ms,
            budget_tokens=self.prefill_budget_tokens,
        )

        # --- Stage 2: Decode ---
        decode_start = time.perf_counter()
        sequences_decoded = 0
        still_running: List[Sequence] = []
        for seq in self.running:
            if seq.state == "decoding":
                await asyncio.to_thread(self._decode_step_single, seq)
                sequences_decoded += 1
            if not seq.is_finished():
                still_running.append(seq)
            else:
                self.finished.append(seq)
                self._resolve_sequence_future(seq)
        self.running = still_running

        decode_latency_ms = (time.perf_counter() - decode_start) * 1000.0
        self.stage_tracker.record_decode(
            sequences_decoded=sequences_decoded,
            latency_ms=decode_latency_ms,
            batch_limit=self.decode_batch_limit,
        )

        step_latency_ms = (time.perf_counter() - step_start) * 1000.0
        self.scheduler_step_latency_ms.append(step_latency_ms)
        self.batch_size_over_time.append((time.perf_counter(), len(self.running)))

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def run_loop(self) -> None:
        """Main scheduler loop — runs as a background asyncio Task.

        Idle sleep interval is ``scheduler_poll_interval_ms / 1000`` seconds
        (default 0.001 s = 1 ms) when both the waiting queue and running list
        are empty.  This is separate from the 5 ms polling in app_v2.py's
        /generate handler.
        """
        idle_sleep_s = self.config.scheduler_poll_interval_ms / 1000.0
        logger.info("Scheduler run_loop started (idle_sleep=%.3f s)", idle_sleep_s)

        while not self._stop_event.is_set():
            if not self.running and len(self.request_queue) == 0:
                await asyncio.sleep(idle_sleep_s)
                continue
            await self._schedule()

        logger.info("Scheduler run_loop stopped.")

    def start(self) -> None:
        """Schedule run_loop() as a background asyncio Task.

        Must be called from within a running event loop (e.g. inside a FastAPI
        lifespan context).
        """
        self._loop_task = asyncio.create_task(self.run_loop(), name="scheduler_loop")

    async def stop(self) -> None:
        """Gracefully stop the scheduler loop and await its completion."""
        self._stop_event.set()
        if self._loop_task is not None:
            try:
                await asyncio.wait_for(self._loop_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Scheduler loop did not stop within 5 s; cancelling.")
                self._loop_task.cancel()
        self._executor.shutdown(wait=False)
        logger.info("Scheduler stopped. Finished %d sequences.", len(self.finished))
