"""
engine/sequential.py — Core inference logic for sequential serving.

Three clearly separated concerns
---------------------------------
1. prefill()  — full-prompt forward pass; produces KV cache + first token.
2. decode()   — autoregressive loop; consumes KV cache token-by-token.
3. generate() — orchestrates prefill → decode and builds GenerationResult.

Design constraints (intentional — this is the naive baseline)
-------------------------------------------------------------
* No batching.
* No custom KV cache management (uses HuggingFace past_key_values as-is).
* No attention optimisation (Flash Attention, etc.).
* Greedy sampling only.

Memory instrumentation
----------------------
get_memory_stats() is device-aware:
  cuda  → torch.cuda.memory_allocated / reserved
  mps   → torch.mps.current_allocated_memory + psutil RSS for "reserved"
  cpu   → psutil virtual_memory used for both fields
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import psutil
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# ── GenerationResult ──────────────────────────────────────────────────────────


@dataclass
class GenerationResult:
    """Fully-populated result of one generate() call."""

    prompt: str
    generated_text: str
    prompt_tokens: int
    generated_tokens: int

    # Latency in milliseconds
    ttft_ms: float          # Time-To-First-Token
    total_latency_ms: float # prefill + all decode steps
    tokens_per_second: float

    # Per-token decode latencies (ms); length == generated_tokens - 1
    # (first token comes from prefill, so no separate decode step for it)
    per_token_latencies_ms: List[float]

    # Memory at the END of inference (post-generate snapshot)
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float

    # ISO-8601 UTC timestamp when generate() was called
    timestamp: str


# ── Memory helpers ────────────────────────────────────────────────────────────


def _bytes_to_mb(n: int) -> float:
    return n / (1024 ** 2)


def get_memory_stats(device: str) -> Tuple[float, float]:
    """
    Return (allocated_mb, reserved_mb) for the given device.

    The semantics differ by device:

    CUDA
        allocated = torch.cuda.memory_allocated()  (tensors currently in use)
        reserved  = torch.cuda.memory_reserved()   (total pool held by PyTorch)

    MPS (Apple Silicon)
        allocated = torch.mps.current_allocated_memory()
                    (driver-level allocation for MPS tensors)
        reserved  = process RSS from psutil
                    (MPS has no concept of a reserved pool)
        Falls back to psutil only if the MPS API is unavailable (PyTorch < 2.1).

    CPU
        Both fields are set to psutil virtual_memory().used because there is
        no meaningful distinction between "allocated by tensors" and "reserved"
        in a CPU-only setting.
    """
    if device == "cuda":
        allocated = _bytes_to_mb(torch.cuda.memory_allocated())
        reserved = _bytes_to_mb(torch.cuda.memory_reserved())
        return allocated, reserved

    if device == "mps":
        # torch.mps.current_allocated_memory() — available since PyTorch 2.1
        try:
            allocated = _bytes_to_mb(torch.mps.current_allocated_memory())
        except AttributeError:
            # PyTorch < 2.1 — fall back to process RSS
            allocated = _bytes_to_mb(
                psutil.Process().memory_info().rss
            )
        # MPS has no "reserved pool" concept; use process RSS as a proxy.
        reserved = _bytes_to_mb(psutil.Process().memory_info().rss)
        return allocated, reserved

    # CPU fallback
    vm = psutil.virtual_memory()
    used_mb = _bytes_to_mb(vm.used)
    return used_mb, used_mb


# ── Prefill ───────────────────────────────────────────────────────────────────


def prefill(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> Tuple[object, int, float]:
    """
    Tokenize *prompt*, run one full forward pass, return KV cache + first token.

    Returns
    -------
    past_key_values
        HuggingFace KV cache (tuple of layer tensors).
    first_token_id : int
        Greedily sampled token from the last position's logits.
    ttft_ms : float
        Time in milliseconds from start of forward pass to logits available.
        This is the canonical TTFT measurement for the sequential baseline.
    """
    device = next(model.parameters()).device

    # Tokenise — stay on CPU, then move to model device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # ── TTFT measurement starts here ──────────────────────────────────────────
    t0 = time.perf_counter()

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,         # ask HF to return past_key_values
            return_dict=True,
        )

    # Logits available → TTFT
    ttft_ms = (time.perf_counter() - t0) * 1000.0

    logits = outputs.logits          # shape: (1, seq_len, vocab_size)
    past_key_values = outputs.past_key_values

    # Greedy sample from last position
    first_token_id: int = int(logits[:, -1, :].argmax(dim=-1).item())

    logger.debug(
        "prefill: %d prompt tokens, first_token_id=%d, ttft=%.1f ms",
        input_ids.shape[1],
        first_token_id,
        ttft_ms,
    )

    return past_key_values, first_token_id, ttft_ms


# ── Decode ────────────────────────────────────────────────────────────────────


def decode(
    model: PreTrainedModel,
    past_key_values: object,
    first_token_id: int,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> Tuple[List[int], List[float]]:
    """
    Autoregressive decode loop starting from *first_token_id*.

    Each step feeds only the last token and the accumulated KV cache —
    no recomputation of the prompt.

    Returns
    -------
    generated_ids : list[int]
        Token ids produced, INCLUDING first_token_id.
    per_token_latencies_ms : list[float]
        Wall-clock latency (ms) for each decode step AFTER the first token.
        Length is len(generated_ids) - 1.
    """
    device = next(model.parameters()).device

    generated_ids: List[int] = [first_token_id]
    per_token_latencies_ms: List[float] = []

    current_token = torch.tensor([[first_token_id]], dtype=torch.long, device=device)

    for step in range(max_new_tokens - 1):  # -1 because first token already counted
        t_step = time.perf_counter()

        with torch.inference_mode():
            outputs = model(
                input_ids=current_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        step_ms = (time.perf_counter() - t_step) * 1000.0
        per_token_latencies_ms.append(step_ms)

        logits = outputs.logits            # shape: (1, 1, vocab_size)
        past_key_values = outputs.past_key_values

        next_token_id = int(logits[:, -1, :].argmax(dim=-1).item())
        generated_ids.append(next_token_id)
        current_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

        # EOS check
        if eos_token_id is not None and next_token_id == eos_token_id:
            logger.debug("decode: EOS hit at step %d", step + 1)
            break

    logger.debug(
        "decode: generated %d tokens, avg step %.1f ms",
        len(generated_ids),
        sum(per_token_latencies_ms) / max(len(per_token_latencies_ms), 1),
    )

    return generated_ids, per_token_latencies_ms


# ── Generate (orchestrator) ───────────────────────────────────────────────────


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> GenerationResult:
    """
    Full prefill → decode pipeline.  Returns a fully-populated GenerationResult.

    Memory snapshots are taken AFTER inference completes (post-generate state),
    which is the most informative point for a sequential baseline: it shows the
    peak footprint a single request leaves behind.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    # Count prompt tokens (don't move to device yet; prefill() handles that)
    prompt_token_count = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])

    # ── Wall-clock start ──────────────────────────────────────────────────────
    t_total_start = time.perf_counter()

    past_key_values, first_token_id, ttft_ms = prefill(model, tokenizer, prompt)

    generated_ids, per_token_latencies_ms = decode(
        model=model,
        past_key_values=past_key_values,
        first_token_id=first_token_id,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )

    total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0

    # ── Memory snapshot ───────────────────────────────────────────────────────
    allocated_mb, reserved_mb = get_memory_stats(device)

    # ── Decode text ───────────────────────────────────────────────────────────
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    generated_token_count = len(generated_ids)
    tps = (generated_token_count / total_latency_ms * 1000.0) if total_latency_ms > 0 else 0.0

    return GenerationResult(
        prompt=prompt,
        generated_text=generated_text,
        prompt_tokens=prompt_token_count,
        generated_tokens=generated_token_count,
        ttft_ms=ttft_ms,
        total_latency_ms=total_latency_ms,
        tokens_per_second=tps,
        per_token_latencies_ms=per_token_latencies_ms,
        gpu_memory_allocated_mb=allocated_mb,
        gpu_memory_reserved_mb=reserved_mb,
        timestamp=timestamp,
    )
