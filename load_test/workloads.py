"""
load_test/workloads.py — Reproducible benchmark workloads.

Every workload is a deterministic function of its arguments and ``seed``, so
two runs (or two GPUs) see exactly the same prompts and output lengths.

Workloads
---------
random         Independent prompts with input/output lengths drawn uniformly
               from ranges.  The standard throughput/latency workload.
shared_prefix  One long shared "system prompt" + a short unique question per
               request — the case prefix caching is built for.
repetitive     "Copy this passage" requests whose outputs repeat the input —
               where n-gram speculative decoding shines.
chat           The original 10-prompt pool with varied output lengths.

Prompt text is built from a fixed word list.  Pass a HuggingFace tokenizer
to get prompts with (almost) exactly the requested number of tokens;
without one, lengths are approximate (~1.3 tokens per word).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from load_test.profiles import default_prompt_pool

_WORDS = (
    "time person year way day thing man world life hand part child eye woman place "
    "work week case point government company number group problem fact system program "
    "question night home water room mother area money story month lot right study book "
    "job word business issue side kind head house service friend father power hour game "
    "line end member law car city community name president team minute idea kid body "
    "information back parent face others level office door health art war history party "
    "result change morning reason research girl guy moment air teacher force education "
    "memory network kernel tensor cache block attention batch token latency model engine "
    "scheduler request memory bandwidth compute graph layer weight vector matrix signal"
).split()


@dataclass
class WorkloadItem:
    prompt: str
    max_new_tokens: int
    tag: str = ""


def _text(rng: random.Random, num_words: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(num_words))


def _fit_tokens(text: str, num_tokens: int, tokenizer) -> str:
    """Trim/extend *text* to about *num_tokens* tokens."""
    if tokenizer is None:
        return text
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    while len(ids) < num_tokens:
        ids = ids + ids
    return tokenizer.decode(ids[:num_tokens])


def random_workload(
    num_requests: int,
    input_len: Tuple[int, int] = (128, 512),
    output_len: Tuple[int, int] = (64, 256),
    seed: int = 0,
    tokenizer=None,
) -> List[WorkloadItem]:
    rng = random.Random(seed)
    items = []
    for i in range(num_requests):
        n_in = rng.randint(*input_len)
        n_out = rng.randint(*output_len)
        text = _text(rng, max(1, int(n_in / 1.3)))
        items.append(WorkloadItem(_fit_tokens(text, n_in, tokenizer), n_out, "random"))
    return items


def shared_prefix_workload(
    num_requests: int,
    prefix_len: int = 1024,
    suffix_len: Tuple[int, int] = (16, 64),
    output_len: Tuple[int, int] = (64, 128),
    num_prefixes: int = 1,
    seed: int = 0,
    tokenizer=None,
) -> List[WorkloadItem]:
    rng = random.Random(seed)
    prefixes = [
        "You are a helpful assistant. Reference document:\n"
        + _fit_tokens(_text(rng, int(prefix_len / 1.3)), prefix_len, tokenizer)
        + "\n\n"
        for _ in range(num_prefixes)
    ]
    items = []
    for i in range(num_requests):
        n_suffix = rng.randint(*suffix_len)
        question = f"Question {i}: " + _fit_tokens(_text(rng, int(n_suffix / 1.3)), n_suffix, tokenizer)
        items.append(WorkloadItem(prefixes[i % num_prefixes] + question + "\nAnswer:",
                                  rng.randint(*output_len), "shared_prefix"))
    return items


def repetitive_workload(
    num_requests: int,
    passage_len: Tuple[int, int] = (128, 256),
    seed: int = 0,
    tokenizer=None,
) -> List[WorkloadItem]:
    rng = random.Random(seed)
    items = []
    for _ in range(num_requests):
        n = rng.randint(*passage_len)
        passage = _fit_tokens(_text(rng, int(n / 1.3)), n, tokenizer)
        prompt = (
            "Repeat the following passage exactly, word for word.\n\n"
            f"Passage:\n{passage}\n\nRepeated passage:\n{passage[:40]}"
        )
        items.append(WorkloadItem(prompt, n, "repetitive"))
    return items


def chat_workload(num_requests: int, seed: int = 0, tokenizer=None,
                  output_len: Tuple[int, int] = (32, 256)) -> List[WorkloadItem]:
    rng = random.Random(seed)
    pool = default_prompt_pool()
    return [WorkloadItem(pool[i % len(pool)], rng.randint(*output_len), "chat")
            for i in range(num_requests)]


WORKLOADS: Dict[str, Callable[..., List[WorkloadItem]]] = {
    "random": random_workload,
    "shared_prefix": shared_prefix_workload,
    "repetitive": repetitive_workload,
    "chat": chat_workload,
}


def make_workload(name: str, num_requests: int, seed: int = 0,
                  tokenizer=None, **kwargs) -> List[WorkloadItem]:
    if name not in WORKLOADS:
        raise ValueError(f"unknown workload {name!r}; choose from {sorted(WORKLOADS)}")
    return WORKLOADS[name](num_requests, seed=seed, tokenizer=tokenizer, **kwargs)


def load_tokenizer(model_name: Optional[str]):
    """Best-effort tokenizer for exact prompt lengths (None if unavailable)."""
    if not model_name:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name)
    except Exception:
        return None
