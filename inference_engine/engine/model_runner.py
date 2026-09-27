"""
engine/model_runner.py — Execute one scheduler step as ONE batched forward pass.

The scheduler decides *what* to run (which sequences, how many tokens each);
the ModelRunner turns that into tensors, runs the model and samples.

Packing
-------
Given per-sequence inputs, e.g. one prefill chunk and two decodes::

    seq A (prefill chunk): tokens a0..a4 at positions 0..4
    seq B (decode):        token  b9     at position 9
    seq C (decode):        token  c3     at position 3

the runner builds a single packed batch::

    input_ids    = [a0 a1 a2 a3 a4 b9 c3]        shape [1, T=7]
    position_ids = [ 0  1  2  3  4  9  3]
    slot_mapping = [pool slot for each token, via each sequence's block table]

Every linear layer, MLP and norm then processes all 7 tokens in one matmul —
this is where continuous batching gets its GPU efficiency.  Only attention
needs per-sequence structure, which it reads from the forward context (see
attention_wrapper.py).  Logits are computed only for positions that are
actually sampled (the last token of a finished prefill, every decode token,
every speculative-verification token), not for the whole batch.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence as Seq

import numpy as np
import torch

from inference_engine.engine.attention_wrapper import (
    SHORT_QUERY_MAX_GATHER_SLOTS,
    SHORT_QUERY_MAX_TOKENS,
    LongQuery,
    PagedAttentionMetadata,
    register_paged_attention,
    set_forward_context,
    use_paged_attention,
)
from inference_engine.engine.kv_cache_config import KVCacheConfig
from inference_engine.engine.paged_kv_cache import PagedKVCacheManager
from inference_engine.engine.sampler import sample
from inference_engine.engine.sequence import SamplingParams

logger = logging.getLogger(__name__)

_GREEDY = SamplingParams(max_new_tokens=1)


@dataclass
class SequenceInput:
    """What one sequence contributes to a forward pass."""

    seq_id: str
    token_ids: List[int]          # tokens to process this step
    start_pos: int                # absolute position of token_ids[0]
    block_table: List[int]        # must cover positions < start_pos + len(token_ids)
    num_logits: int = 1           # trailing positions to sample (0 = none)
    sampling: SamplingParams = field(default_factory=lambda: _GREEDY)


class ModelRunner:
    """Owns the model + its paged KV pool; runs packed batched steps."""

    def __init__(
        self,
        model: torch.nn.Module,
        paged_kv_cache: PagedKVCacheManager,
        seed: int = 0,
    ) -> None:
        register_paged_attention()
        self.model = model
        self.paged_kv_cache = paged_kv_cache
        self.block_size = paged_kv_cache.block_size
        self.device = next(model.parameters()).device
        self.decoder = model.get_decoder() if hasattr(model, "get_decoder") else model.model
        self.lm_head = model.get_output_embeddings()
        self.final_logit_softcapping = getattr(model.config, "final_logit_softcapping", None)
        self.kv_caches = [
            paged_kv_cache.layer_caches(layer) for layer in range(paged_kv_cache.num_layers)
        ]
        # A ready-made 4D mask makes transformers skip building its own
        # O(T²) causal mask; our attention kernel ignores it anyway.
        self._dummy_mask = torch.zeros(1, 1, 1, 1, dtype=torch.bool, device=self.device)
        try:
            self.generator: Optional[torch.Generator] = torch.Generator(device=self.device)
            self.generator.manual_seed(seed)
        except RuntimeError:  # some backends lack device generators
            self.generator = None
            torch.manual_seed(seed)

    # ── Input preparation ─────────────────────────────────────────────────────

    def _to_device(self, data, dtype=torch.long) -> torch.Tensor:
        return torch.as_tensor(data, dtype=dtype).to(self.device)

    def prepare(self, inputs: Seq[SequenceInput]):
        """Build packed tensors + attention metadata for *inputs*."""
        bs = self.block_size
        input_ids: List[int] = []
        positions: List[np.ndarray] = []
        slots: List[np.ndarray] = []
        logits_idx: List[int] = []
        row_params: List[SamplingParams] = []
        short: List[tuple] = []   # (token_start, n, start_pos, num_blocks, block_table)
        long_queries: List[LongQuery] = []

        offset = 0
        for inp in inputs:
            n = len(inp.token_ids)
            if n == 0:
                raise ValueError(f"empty input for seq {inp.seq_id}")
            end_pos = inp.start_pos + n
            num_blocks = math.ceil(end_pos / bs)
            if num_blocks > len(inp.block_table):
                raise ValueError(
                    f"block table of seq {inp.seq_id} covers {len(inp.block_table) * bs} "
                    f"tokens, need {end_pos}"
                )
            pos = np.arange(inp.start_pos, end_pos, dtype=np.int64)
            table = np.asarray(inp.block_table[:num_blocks], dtype=np.int64)
            input_ids.extend(inp.token_ids)
            positions.append(pos)
            slots.append(table[pos // bs] * bs + pos % bs)
            for j in range(n - inp.num_logits, n):
                logits_idx.append(offset + j)
                row_params.append(inp.sampling)

            if n <= SHORT_QUERY_MAX_TOKENS:
                short.append((offset, n, inp.start_pos, num_blocks, table))
            else:
                long_queries.append(LongQuery(
                    token_start=offset,
                    num_tokens=n,
                    start_pos=inp.start_pos,
                    block_table=self._to_device(table) if inp.start_pos > 0 else None,
                ))
            offset += n

        meta = PagedAttentionMetadata(
            slot_mapping=self._to_device(np.concatenate(slots)),
            kv_caches=self.kv_caches,
            block_size=bs,
            long_queries=long_queries,
        )
        if short:
            num_rows = len(short)
            qlen = max(s[1] for s in short)
            max_blocks = max(s[3] for s in short)
            token_idx = np.zeros((num_rows, qlen), dtype=np.int64)
            query_pos = np.zeros((num_rows, qlen), dtype=np.int64)
            tables = np.zeros((num_rows, max_blocks), dtype=np.int64)
            dest, src = [], []
            for r, (start, n, start_pos, nb, table) in enumerate(short):
                # Padding repeats the row's last real query (valid, never scattered).
                token_idx[r, :n] = np.arange(start, start + n)
                token_idx[r, n:] = start + n - 1
                query_pos[r, :n] = np.arange(start_pos, start_pos + n)
                query_pos[r, n:] = start_pos + n - 1
                tables[r, :nb] = table
                dest.extend(range(start, start + n))
                src.extend(range(r * qlen, r * qlen + n))
            meta.short_token_idx = self._to_device(token_idx)
            meta.short_query_pos = self._to_device(query_pos)
            meta.short_block_tables = self._to_device(tables)
            meta.short_dest_idx = self._to_device(dest)
            meta.short_src_idx = self._to_device(src)

        ids = self._to_device(input_ids).unsqueeze(0)
        pos = self._to_device(np.concatenate(positions)).unsqueeze(0)
        return meta, ids, pos, self._to_device(logits_idx), row_params

    # ── Execution ─────────────────────────────────────────────────────────────

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden)
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        if self.final_logit_softcapping:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits

    @torch.inference_mode()
    def forward_logits(self, inputs: Seq[SequenceInput]) -> torch.Tensor:
        """Run the packed forward pass; return logits for the sampled rows."""
        meta, input_ids, positions, logits_idx, _ = self.prepare(inputs)
        with set_forward_context(meta), use_paged_attention(self.model):
            out = self.decoder(
                input_ids=input_ids,
                position_ids=positions,
                attention_mask=self._dummy_mask,
                use_cache=False,
            )
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return self.compute_logits(hidden[0].index_select(0, logits_idx))

    def execute(self, inputs: Seq[SequenceInput]) -> Dict[str, List[int]]:
        """Run one step.  Returns ``{seq_id: sampled token ids}`` for every
        sequence with ``num_logits > 0`` (one id per logit position)."""
        if not inputs:
            return {}
        row_params = [inp.sampling for inp in inputs for _ in range(inp.num_logits)]
        logits = self.forward_logits(inputs)
        if logits.shape[0] == 0:
            return {}
        with torch.inference_mode():
            tokens = sample(logits, row_params, self.generator)
        results: Dict[str, List[int]] = {}
        cursor = 0
        for inp in inputs:
            if inp.num_logits:
                results[inp.seq_id] = tokens[cursor:cursor + inp.num_logits]
                cursor += inp.num_logits
        return results


# ── KV pool sizing ────────────────────────────────────────────────────────────


def resolve_max_model_len(model: torch.nn.Module, config) -> int:
    model_max = getattr(model.config, "max_position_embeddings", None) or 4096
    if config.max_model_len > 0:
        if config.max_model_len > model_max:
            logger.warning(
                "max_model_len=%d exceeds the model's max_position_embeddings=%d",
                config.max_model_len, model_max,
            )
        return config.max_model_len
    return min(model_max, 4096)


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def determine_num_blocks(
    models: Seq[torch.nn.Module],
    kv_configs: Seq[KVCacheConfig],
    config,
    max_model_len: int,
) -> int:
    """How many KV blocks fit in memory (for all models sharing block tables).

    * ``config.kv_num_blocks > 0`` — use it.
    * CPU / MPS — ``kv_cache_max_memory_mb`` worth of blocks.
    * CUDA — profile like vLLM: run a worst-case prefill step with a
      throw-away pool, measure peak activation memory, then give the KV pool
      everything that remains under ``gpu_memory_utilization`` minus the
      transient memory of the largest decode-attention gather.
    """
    bs = config.kv_block_size
    bytes_per_block = sum(c.bytes_per_block(bs) for c in kv_configs)
    min_blocks = math.ceil(max_model_len / bs) + 1

    if config.kv_num_blocks > 0:
        return config.kv_num_blocks
    if config.device != "cuda":
        budget = config.kv_cache_max_memory_mb * 1024 * 1024
        return max(min_blocks, int(budget // bytes_per_block))

    from inference_engine.engine.block_allocator import BlockAllocator

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # Worst-case step: a full prefill budget plus every running sequence
    # decoding — with k speculative tokens each when speculation is on (those
    # all need logits, which dominate at vocab sizes of ~150k).
    lookahead = config.num_speculative_tokens if config.speculative_method else 0
    decode_rows = config.max_batch_size * (1 + lookahead)
    profile_tokens = config.prefill_budget_tokens + decode_rows
    profile_blocks = math.ceil(profile_tokens / bs) + 1
    temp_pool_bytes = 0
    for model, kv_cfg in zip(models, kv_configs):
        allocator = BlockAllocator(profile_blocks, bs)
        pool = PagedKVCacheManager(kv_cfg, allocator, config, num_blocks=profile_blocks)
        temp_pool_bytes += int(pool.pool_size_mb() * 1024 * 1024)
        runner = ModelRunner(model, pool)
        vocab = model.config.vocab_size
        dummy = SequenceInput(
            seq_id="__profile__",
            token_ids=[i % vocab for i in range(profile_tokens)],
            start_pos=0,
            block_table=list(range(profile_blocks)),
            num_logits=min(profile_tokens, decode_rows),
        )
        runner.forward_logits([dummy])
        del runner, pool
    _sync("cuda")
    peak = torch.cuda.max_memory_allocated() - temp_pool_bytes
    torch.cuda.empty_cache()

    free, total = torch.cuda.mem_get_info()
    non_torch = max(0, (total - free) - torch.cuda.memory_reserved())
    # Decode attention gathers up to SHORT_QUERY_MAX_GATHER_SLOTS context slots
    # per layer at a time (K and V), plus fp32 scores and permuted copies.
    per_slot = max(c.bytes_per_token // c.num_layers for c in kv_configs)
    decode_transient = SHORT_QUERY_MAX_GATHER_SLOTS * per_slot * 3
    budget = total * config.gpu_memory_utilization - peak - non_torch - decode_transient
    num_blocks = int(budget // bytes_per_block)
    logger.info(
        "KV profile: total=%.0f MB util=%.2f peak_activations+weights=%.0f MB "
        "non_torch=%.0f MB → %d blocks (%.0f MB)",
        total / 2**20, config.gpu_memory_utilization, peak / 2**20, non_torch / 2**20,
        num_blocks, num_blocks * bytes_per_block / 2**20,
    )
    if num_blocks < min_blocks:
        raise RuntimeError(
            f"Not enough GPU memory for the KV cache: {num_blocks} blocks fit but at "
            f"least {min_blocks} are needed for max_model_len={max_model_len}. Use a "
            f"smaller model, lower MAX_MODEL_LEN / PREFILL_BUDGET_TOKENS, or raise "
            f"GPU_MEMORY_UTILIZATION."
        )
    return num_blocks
