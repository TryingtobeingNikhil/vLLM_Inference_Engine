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
from inference_engine.engine.cpu_swap_manager import CPUSwapManager, CPUSwapError
from inference_engine.engine.kv_cache_config import (
    compute_kv_cache_config,
    format_kv_cache_report,
)
from inference_engine.engine.kv_cache_tracker import KVCacheTracker
from inference_engine.engine.metrics_aggregator import MetricsAggregator
from inference_engine.engine.paged_kv_cache import PagedKVCacheManager
from inference_engine.engine.prefill_utils import run_prefill_single
from inference_engine.engine.request_queue import RequestQueue
from inference_engine.engine.sequence import Sequence
from inference_engine.engine.stage_tracker import StageTracker
from inference_engine.engine.sequential import get_memory_stats
from inference_engine.metrics.collector import MetricsCollector

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
        metrics_collector: Optional[MetricsCollector] = None,
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

        # Phase 9: CPU staging pool for GPU ⇔ CPU swapping
        self.cpu_swap_manager = CPUSwapManager(
            kv_cache_config=self.kv_cache_config,
            block_size=config.kv_block_size,
            num_cpu_blocks=config.kv_num_cpu_blocks,
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
        self.swapped_out: List[Sequence] = []  # Phase 9: sequences swapped to CPU

        # Phase 10: Unified metrics aggregator
        # If no MetricsCollector is provided by the server layer, create a
        # local one so the aggregator always has something to read from.
        if metrics_collector is None:
            metrics_collector = MetricsCollector(
                history_size=getattr(config, "metrics_history_size", 100)
            )
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
            # Phase 9: instead of killing the sequence, try to swap out the
            # largest running sequence to free device blocks, then retry.
            swapped_ok = self._try_swap_out_victim()
            if not swapped_ok:
                seq.state = "finished"
                seq.finish_reason = "oom"
                return
            # Retry allocation once after swap-out freed some blocks
            try:
                block_ids = self.block_allocator.allocate(seq.seq_id, blocks_needed)
                # Mark tokens_used in the last block
                tokens_in_last_block = prompt_token_count % self.config.kv_block_size
                if tokens_in_last_block > 0:
                    self.block_allocator._blocks[block_ids[-1]].tokens_used = tokens_in_last_block
                    self.block_allocator._blocks[block_ids[-1]].is_dirty = True
                else:
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

    # ── Internal: chunked prefill ──────────────────────────────────────────────

    def _prefill_chunk_blocking(self, seq: Sequence) -> None:
        """Run ONE chunk of the prefill pass for *seq*.

        This is the core of the chunked-prefill feature.  Instead of processing
        the full prompt in a single blocking forward pass (which delays decode
        for all other sequences), we advance by at most ``seq.prefill_chunk_size``
        tokens per scheduler step, then yield back to the event loop so other
        sequences can decode.

        Chunk logic
        -----------
        - First chunk  (seq.prefill_offset == 0): cold forward pass, no prior KV.
        - Middle chunks: reconstruct accumulated KV from the paged pool, then
          run a forward pass over the next slice of prompt tokens.
        - Final chunk  (offset+chunk reaches end of prompt): same as middle, but
          additionally samples the first output token and transitions to decoding.

        After each chunk:
        - New KV tensors are written into the paged pool.
        - Block-allocator token accounting is updated.
        - ``seq.prefill_offset`` is advanced.

        This is a BLOCKING function intended to run inside the shared executor.
        """
        import math

        t_chunk_start = time.perf_counter()

        prompt_ids = seq.prompt_token_ids
        prompt_len = len(prompt_ids)
        chunk_start = seq.prefill_offset
        chunk_end = min(chunk_start + seq.prefill_chunk_size, prompt_len)
        chunk_ids = prompt_ids[chunk_start:chunk_end]
        chunk_len = len(chunk_ids)
        is_first_chunk = chunk_start == 0
        is_last_chunk = chunk_end >= prompt_len

        # ── Record queue wait and start time on the first chunk ───────────────
        if is_first_chunk:
            seq.queue_wait_time_ms = (t_chunk_start - seq.arrival_time) * 1000.0
            seq.prefill_start_time = t_chunk_start
            seq.state = "chunked_prefilling"

            # Allocate blocks for the full prompt upfront (same as full prefill)
            blocks_needed = math.ceil(prompt_len / self.config.kv_block_size)
            try:
                block_ids = self.block_allocator.allocate(seq.seq_id, blocks_needed)
                tokens_in_last_block = prompt_len % self.config.kv_block_size
                last_fill = tokens_in_last_block if tokens_in_last_block > 0 else self.config.kv_block_size
                self.block_allocator._blocks[block_ids[-1]].tokens_used = last_fill
                self.block_allocator._blocks[block_ids[-1]].is_dirty = True
            except OutOfBlocksError:
                swapped_ok = self._try_swap_out_victim()
                if not swapped_ok:
                    seq.state = "finished"
                    seq.finish_reason = "oom"
                    return
                try:
                    block_ids = self.block_allocator.allocate(seq.seq_id, blocks_needed)
                    tokens_in_last_block = prompt_len % self.config.kv_block_size
                    last_fill = tokens_in_last_block if tokens_in_last_block > 0 else self.config.kv_block_size
                    self.block_allocator._blocks[block_ids[-1]].tokens_used = last_fill
                    self.block_allocator._blocks[block_ids[-1]].is_dirty = True
                except OutOfBlocksError:
                    seq.state = "finished"
                    seq.finish_reason = "oom"
                    return

            # Register with KV tracker using prompt length
            self.kv_tracker.register_sequence(seq.seq_id, prompt_len)

        # ── Build past_key_values from paged pool (empty on first chunk) ──────
        if is_first_chunk:
            past_kv = None
        else:
            past_kv = build_past_key_values(
                seq_id=seq.seq_id,
                paged_kv_cache=self.paged_kv_cache,
                num_layers=self.kv_cache_config.num_layers,
                device=str(self._device),
                use_dynamic_cache=True,
            )

        # ── Forward pass over this chunk ──────────────────────────────────────
        input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=self._device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_kv,
                use_cache=True,
            )

        new_past_kv = outputs.past_key_values

        # ── Write new chunk KV into paged pool ────────────────────────────────
        # token_position is the global index within the full prompt.
        for token_pos_in_chunk in range(chunk_len):
            global_pos = chunk_start + token_pos_in_chunk
            for layer_idx in range(self.kv_cache_config.num_layers):
                layer_key, layer_val = _extract_kv_layer(new_past_kv, layer_idx)
                # layer_key shape: [1, num_kv_heads, total_tokens_so_far, head_dim]
                # We need the column corresponding to this specific prompt token.
                # The output KV spans [prior_kv_len .. prior_kv_len + chunk_len].
                # token_pos_in_chunk maps to that column.
                key_slice = layer_key[0, :, token_pos_in_chunk, :]
                val_slice = layer_val[0, :, token_pos_in_chunk, :]
                self.paged_kv_cache.write_kv(
                    seq.seq_id, layer_idx, global_pos, key_slice, val_slice
                )

        # ── Advance offset ────────────────────────────────────────────────────
        seq.prefill_offset = chunk_end

        # ── If this was the last chunk, sample first token & transition ───────
        if is_last_chunk:
            logits = outputs.logits       # [1, chunk_len, vocab_size]
            first_token_id = int(logits[:, -1, :].argmax(dim=-1).item())
            seq.generated_token_ids.append(first_token_id)

            seq.ttft_ms = (time.perf_counter() - seq.prefill_start_time) * 1000.0
            seq.first_token_time = time.perf_counter()
            seq.state = "decoding"
            seq.update_kv_stats(
                token_count=prompt_len,
                memory_mb=self.kv_tracker.sequence_memory_mb(seq.seq_id),
            )
            logger.debug(
                "Chunked prefill done: seq_id=%s chunks=%d ttft=%.1f ms first_token=%d",
                seq.seq_id,
                math.ceil(prompt_len / seq.prefill_chunk_size),
                seq.ttft_ms,
                first_token_id,
            )
        else:
            logger.debug(
                "Chunk %d-%d / %d done for seq_id=%s (%.1f ms)",
                chunk_start,
                chunk_end - 1,
                prompt_len - 1,
                seq.seq_id,
                (time.perf_counter() - t_chunk_start) * 1000.0,
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

        # Phase 10: record token throughput for aggregated metrics
        self.metrics_aggregator.record_token_generated(1)

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
        # Phase 10: record completion for throughput and SLO tracking
        self.metrics_aggregator.record_request_finished(seq.finish_reason)
        # Phase 7: zero paged pool slots before releasing blocks
        self.paged_kv_cache.clear_sequence(seq.seq_id)
        # Phase 6: release block allocator memory for this sequence
        self.block_allocator.free(seq.seq_id)

    # ── Internal: Phase 9 swap helpers ────────────────────────────────────────

    def _try_swap_out_victim(self) -> bool:
        """Find the largest running sequence and swap its KV blocks to CPU.

        Selection policy: largest by block count (most device memory freed per
        swap) rather than LRU.  Freeing one large sequence is more efficient
        than several small ones.

        Returns
        -------
        bool
            True if a victim was successfully swapped out; False if no
            candidates exist or the CPU pool is full.
        """
        if not self.running:
            return False

        # Pick the sequence holding the most device blocks
        victim = max(
            self.running,
            key=lambda s: self.block_allocator.num_blocks_for_seq(s.seq_id),
        )

        device_block_ids = self.block_allocator.get_blocks(victim.seq_id)
        if not device_block_ids:
            # Nothing to swap — sequence holds no blocks yet
            return False

        try:
            self.cpu_swap_manager.swap_out(
                victim.seq_id,
                device_block_ids,
                self.paged_kv_cache,
                self.block_allocator,
            )
        except CPUSwapError:
            return False

        # Transition victim to 'swapped' state and move it out of running
        victim.state = "swapped"
        self.running.remove(victim)
        self.swapped_out.append(victim)
        logger.debug(
            "Swapped out seq_id=%s (%d blocks) to CPU",
            victim.seq_id,
            len(device_block_ids),
        )
        return True

    # ── Internal: scheduler step ──────────────────────────────────────────────

    async def _schedule(self) -> None:
        """Run prefill admission, chunked-prefill advancement, then one decode pass.

        Scheduling order per step
        -------------------------
        0. Swap-in: restore any sequences whose KV blocks were evicted to CPU.
        1. Admit: pull new sequences from the waiting queue into self.running,
           stamping their prefill_chunk_size and setting state to
           "chunked_prefilling".  No forward pass happens here.
        2. Chunked prefill: for each "chunked_prefilling" sequence, run exactly
           one chunk (≤ prefill_chunk_size tokens) via _prefill_chunk_blocking.
           This is bounded by the prefill_budget_tokens cap so a very long
           prompt cannot monopolise an entire step.  When the last chunk
           finishes the sequence transitions to "decoding" automatically.
        3. Decode: advance every "decoding" sequence by one token.
        4. Eviction: move finished sequences out of self.running.

        Why this order matters
        ----------------------
        By separating admission (Step 1) from the forward pass (Step 2), a
        newly-admitted sequence sits in self.running immediately but only
        processes chunk_size tokens before decode runs.  This bounds the
        maximum delay experienced by already-decoding sequences to the cost of
        one chunk forward pass rather than the full prompt forward pass.
        """
        step_start = time.perf_counter()

        # --- Phase 9: Swap-in check ---
        # Before admitting new sequences, try to restore any swapped-out
        # sequences if the device pool now has enough room.
        for victim in list(self.swapped_out):
            swap_record = self.cpu_swap_manager._swapped.get(victim.seq_id)
            if swap_record is None:
                # Already cleaned up — remove from list
                self.swapped_out.remove(victim)
                continue
            if self.block_allocator.num_free_blocks() >= swap_record.original_num_blocks:
                try:
                    self.cpu_swap_manager.swap_in(
                        victim.seq_id,
                        self.paged_kv_cache,
                        self.block_allocator,
                    )
                    victim.state = "decoding"
                    self.swapped_out.remove(victim)
                    self.running.append(victim)
                    logger.debug(
                        "Swapped in seq_id=%s back to decoding", victim.seq_id
                    )
                except OutOfBlocksError:
                    continue  # still not enough room, try next iteration

        # --- Stage 1: Admit new sequences (no prefill forward pass here) ---
        # Sequences are added to self.running with state "chunked_prefilling".
        # The actual forward pass happens in Stage 2 below.
        admit_start = time.perf_counter()
        sequences_admitted = 0
        running_limit = min(self.max_batch_size, self.decode_batch_limit)

        while len(self.running) < running_limit:
            queued = await self.request_queue.dequeue()
            if queued is None:
                break
            seq = queued.sequence
            # Stamp the chunk size from config (frozen for this sequence's lifetime)
            seq.prefill_chunk_size = self.config.prefill_chunk_size
            seq.state = "chunked_prefilling"
            self.running.append(seq)
            self.request_queue._total_admitted += 1
            sequences_admitted += 1

        # --- Stage 2: Advance chunked prefill for in-progress sequences ---
        prefill_start = time.perf_counter()
        tokens_this_iteration = 0
        sequences_prefilled = 0   # counts sequences that finished their last chunk

        for seq in list(self.running):
            if seq.state != "chunked_prefilling":
                continue
            # Respect the per-step token budget to avoid starving decode.
            chunk_tokens = min(seq.prefill_chunk_size, len(seq.prompt_token_ids) - seq.prefill_offset)
            if tokens_this_iteration + chunk_tokens > self.prefill_budget_tokens:
                # Budget exhausted — this sequence will get its chunk next step.
                break
            tokens_this_iteration += chunk_tokens
            await asyncio.to_thread(self._prefill_chunk_blocking, seq)
            if seq.state == "decoding":
                sequences_prefilled += 1  # last chunk completed
            elif seq.state == "finished":
                # OOM during chunk allocation — will be evicted in Stage 3
                pass

        prefill_latency_ms = (time.perf_counter() - prefill_start) * 1000.0
        self.stage_tracker.record_prefill(
            sequences_prefilled=sequences_prefilled,
            tokens_prefilled=tokens_this_iteration,
            latency_ms=prefill_latency_ms,
            budget_tokens=self.prefill_budget_tokens,
        )

        # --- Stage 3: Decode + eviction ---
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

    # ── Public metrics API (Phase 10) ─────────────────────────────────────────

    def get_metrics(self) -> dict:
        """Return a single unified metrics dict via MetricsAggregator.

        This is the ONLY method the server layer should call for metrics.
        All tracker stats, derived throughput, SLO compliance, and latency
        percentiles are assembled inside MetricsAggregator.full_report().
        """
        return self.metrics_aggregator.full_report(
            requests_in_flight=len(self.running),
            requests_waiting=len(self.request_queue),
        )
