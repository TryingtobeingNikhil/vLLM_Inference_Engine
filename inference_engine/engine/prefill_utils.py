"""
engine/prefill_utils.py — Thin blocking helper for single-sequence prefill.

Separated from sequential.py to give the scheduler a clean entry point that:
- Accepts already-tokenized `prompt_token_ids` (tokenization happens
  before this call, in the event loop via a separate executor dispatch).
- Returns (past_key_values, first_token_id, ttft_ms) — same contract as
  the Phase 1 `prefill()` function.
- Is fully safe to call from a ThreadPoolExecutor worker thread.

Why not just call sequential.prefill() directly?
-------------------------------------------------
sequential.prefill() tokenizes internally, which doubles the executor round
trip (tokenize + forward pass in one blocking call, mixing two concerns).
The scheduler already tokenizes in add_request() to know prompt_token_ids
before Sequence creation.  Splitting tokenize vs. forward-pass into two
separate executor calls would require awaiting twice; this helper collapses
the forward-pass portion cleanly.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import torch
from transformers import PreTrainedModel


def run_prefill_single(
    model: PreTrainedModel,
    prompt_token_ids: List[int],
    device: torch.device,
) -> Tuple[object, int, float]:
    """Run one full forward pass over *prompt_token_ids* and return the KV cache.

    Parameters
    ----------
    model
        HuggingFace causal LM in eval mode.
    prompt_token_ids
        Already-tokenized integer token ids for the prompt.
    device
        The torch device the model lives on.

    Returns
    -------
    past_key_values
        HuggingFace KV cache after the prefill pass.
    first_token_id : int
        Greedily sampled token from the last position's logits.
    ttft_ms : float
        Wall-clock milliseconds from start of forward pass to logits available.
    """
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)

    t0 = time.perf_counter()

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )

    ttft_ms = (time.perf_counter() - t0) * 1000.0

    logits = outputs.logits               # (1, seq_len, vocab_size)
    past_key_values = outputs.past_key_values

    first_token_id: int = int(logits[:, -1, :].argmax(dim=-1).item())

    return past_key_values, first_token_id, ttft_ms
