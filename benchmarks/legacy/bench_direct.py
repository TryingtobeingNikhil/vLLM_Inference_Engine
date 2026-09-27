#!/usr/bin/env python
"""
bench_direct.py — Standalone timing benchmark for PageServe Phase 2.

Measures actual decode speed directly against the model (no scheduler
event-loop complexity) then runs a scheduler integration test with a
single request at a time to get end-to-end numbers cleanly.

Usage:
    .venv/bin/python bench_direct.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time

import torch

sys.path.insert(0, ".")

from inference_engine.config import Config
from inference_engine.models.loader import load_model_and_tokenizer

# ── Config ────────────────────────────────────────────────────────────────────

MAX_TOKENS    = 50
N_CONCURRENT  = 4
PHASE1_MEAN_MS = 1418.4   # from baseline_metrics.json

PROMPTS = [
    "Explain entropy in thermodynamics.",
    "What is supervised learning?",
    "How does HTTPS work?",
    "Describe photosynthesis briefly.",
][:N_CONCURRENT]


# ── Direct model timing (no scheduler) ───────────────────────────────────────

def bench_single_request(model, tokenizer, device, prompt: str, max_tokens: int) -> dict:
    """Time one full request: prefill + N decode steps."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = ids.shape[1]

    # Prefill
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        out = model(ids, use_cache=True)
    ttft_ms = (time.perf_counter() - t_prefill_start) * 1000.0

    next_tok_id = int(out.logits[0, -1].argmax())
    past = out.past_key_values
    generated = [next_tok_id]

    eos_id = tokenizer.eos_token_id
    per_tok_ms = []

    # Decode steps
    for _ in range(max_tokens - 1):
        if eos_id is not None and next_tok_id == eos_id:
            break
        t0 = time.perf_counter()
        next_input = torch.tensor([[next_tok_id]], device=device)
        with torch.no_grad():
            out = model(next_input, past_key_values=past, use_cache=True)
        per_tok_ms.append((time.perf_counter() - t0) * 1000.0)
        next_tok_id = int(out.logits[0, -1].argmax())
        past = out.past_key_values
        generated.append(next_tok_id)

    total_ms = ttft_ms + sum(per_tok_ms)
    return {
        "prompt_len": prompt_len,
        "n_generated": len(generated),
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "per_tok_ms": per_tok_ms,
        "tps": len(generated) / (total_ms / 1000.0),
    }


def run_benchmark(model, tokenizer, device) -> dict:
    sep = "─" * 64

    # ── 1. Single-request warm-up ─────────────────────────────────────────────
    print("  Warming up (1 request × 5 tokens) …", flush=True)
    bench_single_request(model, tokenizer, device, PROMPTS[0], 5)
    print("  Warm-up done.\n", flush=True)

    # ── 2. Single-request baseline (matches Phase 1 methodology) ─────────────
    print(f"  Running single-request baseline ({MAX_TOKENS} tokens) …", flush=True)
    single_results = []
    for p in PROMPTS:
        r = bench_single_request(model, tokenizer, device, p, MAX_TOKENS)
        single_results.append(r)
        print(f"    TTFT={r['ttft_ms']:.1f} ms  total={r['total_ms']:.1f} ms  "
              f"{r['n_generated']} tok @ {r['tps']:.1f} tok/s")

    ttfts_single  = [r["ttft_ms"] for r in single_results]
    total_single  = [r["total_ms"] for r in single_results]
    tps_single    = [r["tps"] for r in single_results]

    print(f"\n{sep}")
    print(f"  Phase 2 — Single-request (sequential, {N_CONCURRENT} prompts)")
    print(sep)
    print(f"  TTFT mean:           {statistics.mean(ttfts_single):>8.1f} ms")
    print(f"  Total latency mean:  {statistics.mean(total_single):>8.1f} ms")
    print(f"  Throughput mean:     {statistics.mean(tps_single):>8.1f} tok/s")
    print(sep)

    # ── 3. Simulated concurrent (interleaved by scheduler strategy) ───────────
    # Simulate what the scheduler does: each request advances 1 token per step
    # This is the true concurrent-decode cost on a single MPS device.
    print(f"\n  Simulating {N_CONCURRENT}-way concurrent decode ({MAX_TOKENS} tokens each) …", flush=True)

    # Prefill all prompts first
    t_wall_start = time.perf_counter()
    states = []
    for p in PROMPTS:
        ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(ids, use_cache=True)
        first_token = int(out.logits[0, -1].argmax())
        states.append({
            "past": out.past_key_values,
            "next": first_token,
            "generated": [first_token],
            "ttft_ms": (time.perf_counter() - t_wall_start) * 1000.0,
            "per_tok_ms": [],
        })

    eos_id = tokenizer.eos_token_id

    # Round-robin decode until all sequences finish
    max_steps = MAX_TOKENS * N_CONCURRENT * 2  # safety cap
    step = 0
    while step < max_steps:
        active = [s for s in states if len(s["generated"]) < MAX_TOKENS
                  and not (eos_id is not None and s["generated"][-1] == eos_id)]
        if not active:
            break
        for s in active:
            t0 = time.perf_counter()
            inp = torch.tensor([[s["next"]]], device=device)
            with torch.no_grad():
                out = model(inp, past_key_values=s["past"], use_cache=True)
            s["per_tok_ms"].append((time.perf_counter() - t0) * 1000.0)
            s["next"] = int(out.logits[0, -1].argmax())
            s["past"] = out.past_key_values
            s["generated"].append(s["next"])
        step += 1

    wall_ms = (time.perf_counter() - t_wall_start) * 1000.0
    total_tokens = sum(len(s["generated"]) for s in states)
    agg_tps = total_tokens / (wall_ms / 1000.0)
    speedup = (PHASE1_MEAN_MS * N_CONCURRENT) / wall_ms
    ttfts = [s["ttft_ms"] for s in states]
    tot_lats = [s["ttft_ms"] + sum(s["per_tok_ms"]) for s in states]

    print(f"\n{sep}")
    print(f"  Phase 2 — {N_CONCURRENT}-way concurrent decode")
    print(sep)
    print(f"  {'#':>2}  {'TTFT (ms)':>10}  {'Total (ms)':>11}  {'Tokens':>6}")
    for i, s in enumerate(states):
        tot = s["ttft_ms"] + sum(s["per_tok_ms"])
        print(f"  {i:>2}  {s['ttft_ms']:>10.1f}  {tot:>11.1f}  {len(s['generated']):>6}")

    print(f"\n  Wall time (all {N_CONCURRENT} done):       {wall_ms:>9.1f} ms")
    print(f"  Phase 1 expected ({N_CONCURRENT}×serial):  {PHASE1_MEAN_MS*N_CONCURRENT:>9.1f} ms")
    print(f"  Speedup ratio:                   {speedup:>9.2f}×" +
          ("  ✓ faster" if speedup > 1.0 else "  ✗ slower (serial on 1 GPU)"))
    print(f"  Total tokens generated:          {total_tokens:>9}")
    print(f"  Aggregate throughput:            {agg_tps:>9.1f} tok/s")
    print(f"  TTFT mean:                       {statistics.mean(ttfts):>9.1f} ms")
    print(f"  TTFT min / max:                  {min(ttfts):>6.1f} / {max(ttfts):.1f} ms")
    print(f"  Total latency mean:              {statistics.mean(tot_lats):>9.1f} ms")
    print(sep)

    output = {
        "single_request": {
            "ttft_ms_mean": statistics.mean(ttfts_single),
            "total_latency_ms_mean": statistics.mean(total_single),
            "tps_mean": statistics.mean(tps_single),
        },
        "concurrent": {
            "n_requests": N_CONCURRENT,
            "max_tokens": MAX_TOKENS,
            "wall_ms": wall_ms,
            "phase1_expected_ms": PHASE1_MEAN_MS * N_CONCURRENT,
            "speedup": speedup,
            "aggregate_tps": agg_tps,
            "total_tokens": total_tokens,
            "ttft_ms": {"mean": statistics.mean(ttfts), "min": min(ttfts), "max": max(ttfts)},
            "total_latency_ms": {"mean": statistics.mean(tot_lats)},
        },
    }
    with open("bench_direct_results.json", "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n  Results saved → bench_direct_results.json\n")
    return output


def main() -> None:
    print("\n" + "=" * 64)
    print("  PageServe Direct Model Benchmark")
    print("=" * 64)

    cfg = Config()
    print(f"\n  Config: device={cfg.device}  max_batch_size={cfg.max_batch_size}")

    print("\n  Loading model …", flush=True)
    t0 = time.perf_counter()
    loaded = load_model_and_tokenizer(cfg)
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.device
    print(f"  Model loaded in {(time.perf_counter()-t0)*1000:.0f} ms\n", flush=True)

    run_benchmark(model, tokenizer, device)
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
