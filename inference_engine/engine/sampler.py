"""
engine/sampler.py — Turn next-token logits into token ids.

One call handles a whole batch of logit rows with mixed per-request settings:
greedy rows (temperature 0) take the argmax; sampled rows apply temperature,
top-k and top-p (nucleus) filtering and draw from the resulting distribution.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from inference_engine.engine.sequence import SamplingParams


def _apply_top_k_top_p(
    logits: torch.Tensor, top_ks: torch.Tensor, top_ps: torch.Tensor
) -> torch.Tensor:
    """Mask logits outside each row's top-k / top-p set with -inf."""
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    vocab = logits.shape[-1]

    # top-k: keep ranks < k (k <= 0 means disabled)
    ranks = torch.arange(vocab, device=logits.device).unsqueeze(0)
    k = torch.where(top_ks > 0, top_ks, torch.full_like(top_ks, vocab)).unsqueeze(1)
    remove = ranks >= k

    # top-p: drop tokens once the cumulative probability *before* them
    # exceeds p (the top-1 token always survives).
    probs = torch.softmax(sorted_logits.masked_fill(remove, float("-inf")), dim=-1)
    cumulative = probs.cumsum(dim=-1) - probs
    remove |= cumulative > top_ps.unsqueeze(1)

    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.empty_like(logits).scatter_(1, sorted_idx, sorted_logits)


def sample(
    logits: torch.Tensor,
    params: Sequence[SamplingParams],
    generator: Optional[torch.Generator] = None,
) -> List[int]:
    """Sample one token per row of ``logits`` (``[M, V]``, float32)."""
    greedy = logits.argmax(dim=-1)
    if all(p.is_greedy for p in params):
        return greedy.tolist()

    device = logits.device
    temps = torch.tensor([p.temperature if p.temperature > 0 else 1.0 for p in params],
                         device=device, dtype=torch.float32)
    top_ks = torch.tensor([p.top_k for p in params], device=device, dtype=torch.long)
    top_ps = torch.tensor([p.top_p for p in params], device=device, dtype=torch.float32)
    is_greedy = torch.tensor([p.is_greedy for p in params], device=device)

    scaled = logits / temps.unsqueeze(1)
    if bool((top_ks > 0).any()) or bool((top_ps < 1.0).any()):
        scaled = _apply_top_k_top_p(scaled, top_ks, top_ps)
    probs = torch.softmax(scaled, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
    return torch.where(is_greedy, greedy, sampled).tolist()
