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
2. Real model (accuracy): the engine and HF's own half-precision forward
   pass score the same token sequences (teacher forcing); each is compared
   with an fp32 copy of the model.  The engine must be about as accurate as
   HF.  Throughput of both is printed for information.

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
from inference_engine.models.loader import load_model, load_model_and_tokenizer
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


PROMPTS = [
    "The capital of France is", "Explain how a transformer neural network works:",
    "def fibonacci(n):", "Water boils at", "List three prime numbers:",
    "The theory of relativity states that", "Once upon a time,", "SQL versus NoSQL:",
]


def _teacher_forced_logits(model, sequences, device) -> list:
    """HF forward over prompt + continuation; logits predicting each continuation token."""
    out = []
    for plen, full in sequences:
        with torch.inference_mode():
            logits = model(torch.tensor([full], device=device)).logits[0, plen - 1:-1]
        out.append(logits.float().cpu())
    return out


def _engine_teacher_forced_logits(model, sequences) -> list:
    """Same continuation through the engine exactly as it serves traffic: every
    prompt prefilled in one packed step, then all sequences decoded together
    one token per step (the batched paged-attention path)."""
    import math

    from inference_engine.engine.block_allocator import BlockAllocator
    from inference_engine.engine.kv_cache_config import compute_kv_cache_config
    from inference_engine.engine.model_runner import ModelRunner, SequenceInput
    from inference_engine.engine.paged_kv_cache import PagedKVCacheManager

    cfg = Config(model_name="x", kv_block_size=16)
    num_blocks = sum(math.ceil(len(full) / 16) for _, full in sequences) + 1
    allocator = BlockAllocator(num_blocks, 16)
    pool = PagedKVCacheManager(compute_kv_cache_config(model, cfg), allocator, cfg)
    runner = ModelRunner(model, pool)
    for i, (_, full) in enumerate(sequences):
        allocator.allocate(str(i), math.ceil(len(full) / 16))
    tables = [allocator.get_blocks(str(i)) for i in range(len(sequences))]

    steps = [runner.forward_logits([
        SequenceInput(str(i), full[:plen], 0, tables[i]) for i, (plen, full) in enumerate(sequences)
    ]).float().cpu()]
    n_new = len(sequences[0][1]) - sequences[0][0]
    for t in range(n_new - 1):
        steps.append(runner.forward_logits([
            SequenceInput(str(i), [full[plen + t]], plen + t, tables[i])
            for i, (plen, full) in enumerate(sequences)
        ]).float().cpu())
    return [torch.stack([step[i] for step in steps]) for i in range(len(sequences))]


def _accuracy(logits, reference) -> tuple:
    agree = sum((a.argmax(-1) == b.argmax(-1)).sum().item() for a, b in zip(logits, reference))
    total = sum(len(b) for b in reference)
    mean_err = sum((a - b).abs().mean().item() for a, b in zip(logits, reference)) / len(reference)
    return agree / total, mean_err


def real_model_check(model_name: str, device: str) -> bool:
    """Numerical accuracy of the engine vs HuggingFace on the real model.

    Comparing free-running greedy outputs proves little in half precision:
    any two implementations round differently, and one flipped near-tie
    changes everything after it.  Instead both score the *same* tokens
    (teacher forcing) and each is compared with an fp32 reference; the engine
    passes if it is about as accurate as HuggingFace's own half-precision path.
    """
    print(f"\n[2/2] Real model {model_name} on {device}")
    model, tokenizer, _ = load_model_and_tokenizer(Config(model_name=model_name, device=device))
    half_dtype = next(model.parameters()).dtype
    n = 48

    reference_model = None
    if device == "cuda":
        free, _ = torch.cuda.mem_get_info()
        fp32_bytes = sum(p.numel() for p in model.parameters()) * 4
        if free > fp32_bytes * 1.3:
            reference_model = load_model(model_name, device, "float32")
    ref_label = "fp32" if reference_model is not None else f"HF {half_dtype}"
    generator = reference_model or model

    sequences = []
    for p in PROMPTS:
        ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            out = generator.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n,
                                     min_new_tokens=n, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
        sequences.append((ids.shape[1], out[0].tolist()))

    engine_logits = _engine_teacher_forced_logits(model, sequences)
    hf_logits = _teacher_forced_logits(model, sequences, device)
    if reference_model is not None:
        reference = _teacher_forced_logits(reference_model, sequences, device)
        del reference_model
        torch.cuda.empty_cache()
        hf_agree, hf_err = _accuracy(hf_logits, reference)
        eng_agree, eng_err = _accuracy(engine_logits, reference)
        print(f"  teacher-forced vs {ref_label} over {len(PROMPTS)}×{n} tokens:")
        print(f"    HF {str(half_dtype)[6:]:8s} top-1 agreement {hf_agree:6.1%}   mean |Δlogit| {hf_err:.4f}")
        print(f"    engine {str(half_dtype)[6:]:4s} top-1 agreement {eng_agree:6.1%}   mean |Δlogit| {eng_err:.4f}")
        ok = eng_agree >= hf_agree - 0.02 and eng_err <= 1.5 * hf_err + 0.02
    else:
        eng_agree, eng_err = _accuracy(engine_logits, hf_logits)
        print(f"  (not enough free memory for an fp32 copy — comparing with HF {half_dtype} directly)")
        print(f"    engine vs HF top-1 agreement {eng_agree:6.1%}   mean |Δlogit| {eng_err:.4f}")
        ok = eng_agree >= 0.97
    print(f"  accuracy → {'PASS' if ok else 'FAIL'}")

    # Throughput (informational): HF one-by-one vs the engine batching all prompts.
    t0 = time.perf_counter()
    for p in PROMPTS:
        ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n,
                           min_new_tokens=n, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    hf_s = time.perf_counter() - t0
    scheduler = ContinuousBatchingScheduler(model, tokenizer, Config(model_name=model_name, device=device))

    async def go():
        scheduler.start()
        _, f = await scheduler.add_request("warm up", sampling=SamplingParams(max_new_tokens=4))
        await f
        t = time.perf_counter()
        reqs = [await scheduler.add_request(p, sampling=SamplingParams(max_new_tokens=n, ignore_eos=True))
                for p in PROMPTS]
        await asyncio.gather(*[f for _, f in reqs])
        elapsed = time.perf_counter() - t
        await scheduler.stop()
        return elapsed

    engine_s = asyncio.run(go())
    total = len(PROMPTS) * n
    print(f"  throughput, {len(PROMPTS)} requests × {n} tokens: HF sequential {total / hf_s:.1f} tok/s"
          f" | engine {total / engine_s:.1f} tok/s ({hf_s / engine_s:.1f}×)")
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
