#!/usr/bin/env python
"""
scripts/gpu_smoke_test.py — Verify the engine on this machine's accelerator.

Run this first on a new GPU (the Colab notebook does).  Two checks:

1. Exactness (hard fail): a tiny float64 model runs through the full
   scheduler on the GPU under every feature combination — batching, chunked
   prefill, prefix caching, swap and recompute preemption, n-gram and
   draft-model speculation — and must reproduce HuggingFace generate()
   token-for-token.  This exercises every CUDA code path (paged writes,
   gathers, masks, pinned-memory swapping) with numerics that leave no room
   for "close enough".
2. Real model (sanity): the benchmark model's greedy output from the engine
   is compared with HF generate().  Half-precision batched and unbatched
   kernels round differently, so late divergence is normal; the check
   requires most prompts to agree on their first 16 tokens.

    python scripts/gpu_smoke_test.py [--model auto] [--skip-real]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from benchmarks.common import pick_model
from inference_engine.config import Config
from inference_engine.engine.scheduler import ContinuousBatchingScheduler
from inference_engine.engine.sequence import SamplingParams
from inference_engine.models.loader import load_model_and_tokenizer
from inference_engine.tests.tiny_models import FakeTokenizer, hf_greedy, tiny_qwen2

CONFIGS = {
    "baseline": dict(enable_prefix_caching=False),
    "prefix_cache": dict(enable_prefix_caching=True),
    "swap_preemption": dict(kv_num_blocks=40, preemption_mode="swap", enable_prefix_caching=False),
    "recompute_preemption": dict(kv_num_blocks=40, preemption_mode="recompute"),
    "ngram_spec": dict(speculative_method="ngram"),
    "draft_spec": dict(speculative_method="draft", draft_model_name="tiny", num_speculative_tokens=3),
}


async def _run(scheduler, prompts, lengths):
    scheduler.start()
    reqs = [await scheduler.add_request("", prompt_token_ids=p, sampling=SamplingParams(max_new_tokens=n))
            for p, n in zip(prompts, lengths)]
    out = await asyncio.gather(*[f for _, f in reqs])
    await scheduler.stop()
    return out


def exactness_check(device: str) -> bool:
    # float64 makes "exact" meaningful; MPS has no float64 kernels.
    dtype = torch.float32 if device == "mps" else torch.float64
    print(f"\n[1/2] Exactness on {device} (tiny {str(dtype)[6:]} model vs HF generate)")
    model = tiny_qwen2(seed=0, dtype=dtype).to(device)
    rng = random.Random(1)
    shared = [rng.randrange(97) for _ in range(40)]
    prompts = [(shared if i % 2 == 0 else []) + [rng.randrange(97) for _ in range(rng.randint(1, 30))]
               for i in range(20)] + [[1, 2, 3, 4, 5, 6, 7, 8] * 5] * 2
    lengths = [rng.randint(1, 25) for _ in prompts]
    refs = [hf_greedy(model, p, n) for p, n in zip(prompts, lengths)]
    ok_all = True
    for name, overrides in CONFIGS.items():
        settings = dict(device=device, kv_block_size=4, kv_num_blocks=512, max_batch_size=8,
                        prefill_budget_tokens=24, prefill_chunk_size=10, max_queue_size=64)
        cfg = Config(**{**settings, **overrides})
        draft = model if name == "draft_spec" else None
        scheduler = ContinuousBatchingScheduler(model, FakeTokenizer(), cfg, draft_model=draft,
                                                eos_token_ids=[])
        results = asyncio.run(_run(scheduler, prompts, lengths))
        matches = sum(r.generated_token_ids == ref for r, ref in zip(results, refs))
        free = scheduler.block_allocator.num_free_blocks() == scheduler.block_allocator.num_blocks
        ok = matches == len(refs) and free
        ok_all &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:22s} {matches}/{len(refs)} exact"
              f"{'' if free else '  (leaked KV blocks!)'}")
    return ok_all


def real_model_check(model_name: str, device: str) -> bool:
    print(f"\n[2/2] Real model {model_name} on {device}")
    model, tokenizer, _ = load_model_and_tokenizer(Config(model_name=model_name, device=device))
    prompts = [
        "The capital of France is", "Explain how a transformer neural network works:",
        "def fibonacci(n):", "Water boils at", "List three prime numbers:",
        "The theory of relativity states that", "Once upon a time,", "SQL versus NoSQL:",
    ]
    n = 64
    refs = []
    t0 = time.perf_counter()
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").to(device)
        out = model.generate(**ids, max_new_tokens=n, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        refs.append(out[0, ids.input_ids.shape[1]:].tolist())
    hf_s = time.perf_counter() - t0

    scheduler = ContinuousBatchingScheduler(model, tokenizer, Config(model_name=model_name, device=device))

    async def go():
        scheduler.start()
        _, f = await scheduler.add_request("warm up", sampling=SamplingParams(max_new_tokens=4))
        await f
        t = time.perf_counter()
        reqs = [await scheduler.add_request(p, sampling=SamplingParams(max_new_tokens=n))
                for p in prompts]
        out = await asyncio.gather(*[f for _, f in reqs])
        el = time.perf_counter() - t
        await scheduler.stop()
        return out, el

    results, engine_s = asyncio.run(go())
    # Both sides decode greedily and stop at EOS, so they are comparable token
    # for token; "agree" = identical for the first 16 tokens, or identical
    # outright (a short answer that ended at the same EOS).
    agree = 0
    for seq, ref in zip(results, refs):
        got = seq.generated_token_ids
        div = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), min(len(got), len(ref)))
        agree += div >= 16 or got == ref
        text = tokenizer.decode(seq.generated_token_ids[:20], skip_special_tokens=True).replace("\n", " ")
        print(f"  first divergence at {div:>2}/{min(len(got), len(ref))} | {text[:70]}")
    ok = agree >= 0.75 * len(prompts)
    hf_tokens = sum(len(r) for r in refs)
    engine_tokens = sum(len(s.generated_token_ids) for s in results)
    print(f"  HF sequential: {hf_tokens / hf_s:7.1f} tok/s | engine (batched): "
          f"{engine_tokens / engine_s:7.1f} tok/s | {agree}/{len(prompts)} agree on 16+ tokens"
          f" → {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="auto")
    p.add_argument("--skip-real", action="store_true")
    args = p.parse_args()
    device = Config().device
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    ok = exactness_check(device)
    if not args.skip_real:
        ok &= real_model_check(pick_model(args.model), device)
    print("\nSMOKE TEST", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
