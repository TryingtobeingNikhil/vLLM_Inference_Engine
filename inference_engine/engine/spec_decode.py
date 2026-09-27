"""
engine/spec_decode.py — Speculative decoding proposers.

Decode is memory-bandwidth bound: a forward pass over 1 token per sequence
costs about the same as one over 5 tokens per sequence, because the time goes
into streaming the weights, not into arithmetic.  Speculative decoding
exploits that:

1. A cheap *proposer* guesses the next k tokens: d1..dk.
2. The target model runs ONE forward pass over [last_token, d1..dk] and
   produces its own greedy choice t0..tk at every position.
3. Drafts are accepted while they agree (d1 == t0, d2 == t1, …); the first
   disagreement is replaced by the target's token, and if everything matched
   the target's extra prediction tk comes for free ("bonus" token).

Every emitted token is the target model's own greedy choice, so the output is
identical to normal greedy decoding — only faster when drafts are good.
(Speculation is applied to greedy requests only; sampled requests would need
rejection sampling to preserve their distribution.)

Proposers
---------
* NgramProposer — "prompt lookup decoding": find the most recent earlier
  occurrence of the sequence's last n tokens and propose what followed it.
  Free to compute, great when output copies from the input (summaries,
  code edits, RAG answers, structured extraction).
* DraftModelProposer — a small model from the same family (e.g.
  Qwen2.5-0.5B for Qwen2.5-7B) runs k cheap decode steps.  It has its own
  paged KV pool but shares the target's block tables, so the scheduler's
  block allocation covers both models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence as Seq

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from inference_engine.engine.model_runner import ModelRunner, SequenceInput


class NgramProposer:
    """Propose tokens by matching the sequence's suffix against its own history."""

    def __init__(self, ngram_max: int = 4, ngram_min: int = 2) -> None:
        self.ngram_max = ngram_max
        self.ngram_min = ngram_min

    def propose(self, token_ids: Seq[int], k: int) -> List[int]:
        if k <= 0:
            return []
        tokens = np.asarray(token_ids, dtype=np.int64)
        length = len(tokens)
        for n in range(self.ngram_max, self.ngram_min - 1, -1):
            if length < n + 1:
                continue
            pattern = tokens[-n:]
            # Candidate windows start at 0 .. length-n-1 (the suffix itself is
            # excluded because it would "match" with nothing after it).
            windows = sliding_window_view(tokens[:-1], n)
            hits = np.flatnonzero((windows == pattern).all(axis=1))
            if hits.size:
                start = int(hits[-1]) + n           # most recent occurrence
                return tokens[start:start + k].tolist()
        return []


@dataclass
class DraftRequest:
    seq_id: str
    token_ids: List[int]      # all tokens of the sequence (prompt + generated)
    draft_computed: int       # tokens whose KV the draft model already has
    block_table: List[int]
    k: int                    # tokens to propose


class DraftModelProposer:
    """Run a small draft model for k greedy steps (batched across sequences)."""

    def __init__(self, runner: ModelRunner) -> None:
        self.runner = runner

    def propose(self, requests: Seq[DraftRequest]) -> Dict[str, List[int]]:
        drafts: Dict[str, List[int]] = {r.seq_id: [] for r in requests}
        active = [r for r in requests if r.k > 0]
        if not active:
            return drafts

        # Pass 1: catch the draft up on every token it hasn't seen (usually
        # just the last one or two; the whole prompt for a new sequence) and
        # predict d1 from the final position.
        first = self.runner.execute([
            SequenceInput(
                seq_id=r.seq_id,
                token_ids=r.token_ids[r.draft_computed:],
                start_pos=r.draft_computed,
                block_table=r.block_table,
                num_logits=1,
            )
            for r in active
        ])
        for r in active:
            drafts[r.seq_id].append(first[r.seq_id][0])

        # Passes 2..k: one token per sequence per pass.
        for step in range(1, max(r.k for r in active)):
            batch = [r for r in active if r.k > step]
            if not batch:
                break
            out = self.runner.execute([
                SequenceInput(
                    seq_id=r.seq_id,
                    token_ids=[drafts[r.seq_id][-1]],
                    start_pos=len(r.token_ids) + step - 1,
                    block_table=r.block_table,
                    num_logits=1,
                )
                for r in batch
            ])
            for r in batch:
                drafts[r.seq_id].append(out[r.seq_id][0])
        return drafts
