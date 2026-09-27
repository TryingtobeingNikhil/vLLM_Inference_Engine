#!/usr/bin/env python
"""
bench_batched_decode.py — True batched decode benchmark for PageServe.

This benchmark is meant for CUDA/T4 validation, but it also runs on MPS/CPU.
It compares three execution styles using the same model and prompts:

1. Serial baseline:
   Each request runs to completion before the next starts.

2. Round-robin interleaving:
   Each request advances one token at a time, but every sequence still uses a
   separate model forward. This is close to the older PageServe direct benchmark.

3. True batched decode:
   All active sequences advance together using one batched model forward per
   token step. This is the GPU path that should expose real batching efficiency.

Usage:
    DEVICE=cuda python bench_batched_decode.py

Output:
    bench_batched_decode_results.json
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from typing import Any, Dict, List

import torch

sys.path.insert(0, ".")

from inference_engine.config import Config
from inference_engine.models.loader import load_model_and_tokenizer


MAX_TOKENS = 50
N_REQUESTS = 4
OUTPUT_PATH = "bench_batched_decode_results.json"

PROMPTS = [
    "Explain entropy in thermodynamics.",
    "What is supervised learning?",
    "How does HTTPS work?",
    "Describe photosynthesis briefly.",
][:N_REQUESTS]


def sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def gpu_memory_mb(device: str) -> Dict[str, float]:
    if device != "cuda" or not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "max_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }


def timed_forward(device: str, fn) -> Any:
    sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = fn()
    sync(device)
    return out, (time.perf_counter() - t0) * 1000.0


def run_one_serial(
    model,
    tokenizer,
    device: str,
    prompt: str,
    max_tokens: int = MAX_TOKENS,
) -> Dict[str, Any]:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    out, ttft_ms = timed_forward(device, lambda: model(ids, use_cache=True))
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    tok_ms: List[float] = []
    for _ in range(max_tokens - 1):
        out, step_ms = timed_forward(
            device,
            lambda next_tok=next_tok, past=past: model(
                next_tok,
                past_key_values=past,
                use_cache=True,
            ),
        )
        tok_ms.append(step_ms)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    total_ms = ttft_ms + sum(tok_ms)
    return {
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "decode_ms": sum(tok_ms),
        "tokens": max_tokens,
        "tps": max_tokens / (total_ms / 1000.0),
    }


def bench_serial(model, tokenizer, device: str) -> Dict[str, Any]:
    sync(device)
    t0 = time.perf_counter()
    results = [run_one_serial(model, tokenizer, device, p) for p in PROMPTS]
    sync(device)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    total_tokens = sum(r["tokens"] for r in results)
    return {
        "n_requests": len(PROMPTS),
        "wall_ms": wall_ms,
        "total_tokens": total_tokens,
        "aggregate_tps": total_tokens / (wall_ms / 1000.0),
        "ttft_ms_mean": statistics.mean(r["ttft_ms"] for r in results),
        "latency_ms_mean": statistics.mean(r["total_ms"] for r in results),
        "per_request": results,
    }


def bench_round_robin(model, tokenizer, device: str) -> Dict[str, Any]:
    states = []
    prefill_ms = []
    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        out, ttft_ms = timed_forward(device, lambda ids=ids: model(ids, use_cache=True))
        states.append(
            {
                "past": out.past_key_values,
                "next": out.logits[:, -1, :].argmax(dim=-1, keepdim=True),
                "tok_ms": [],
            }
        )
        prefill_ms.append(ttft_ms)

    sync(device)
    t0 = time.perf_counter()
    for _ in range(MAX_TOKENS - 1):
        for state in states:
            out, step_ms = timed_forward(
                device,
                lambda state=state: model(
                    state["next"],
                    past_key_values=state["past"],
                    use_cache=True,
                ),
            )
            state["tok_ms"].append(step_ms)
            state["past"] = out.past_key_values
            state["next"] = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    sync(device)
    decode_wall_ms = (time.perf_counter() - t0) * 1000.0

    total_tokens = len(PROMPTS) * MAX_TOKENS
    decode_tokens = len(PROMPTS) * (MAX_TOKENS - 1)
    end_to_end_ms = sum(prefill_ms) + decode_wall_ms
    latencies = [prefill_ms[i] + sum(states[i]["tok_ms"]) for i in range(len(states))]
    return {
        "n_requests": len(PROMPTS),
        "decode_wall_ms": decode_wall_ms,
        "end_to_end_ms": end_to_end_ms,
        "total_tokens": total_tokens,
        "aggregate_tps_decode_only": decode_tokens / (decode_wall_ms / 1000.0),
        "aggregate_tps_end_to_end": total_tokens / (end_to_end_ms / 1000.0),
        "ttft_ms_mean": statistics.mean(prefill_ms),
        "latency_ms_mean": statistics.mean(latencies),
    }


def bench_true_batched(model, tokenizer, device: str) -> Dict[str, Any]:
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        batch = tokenizer(PROMPTS, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = old_padding_side

    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)

    sync(device)
    t0 = time.perf_counter()
    out, prefill_ms = timed_forward(
        device,
        lambda: model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True),
    )
    next_tokens = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past = out.past_key_values

    step_ms: List[float] = []
    for _ in range(MAX_TOKENS - 1):
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)],
            dim=1,
        )
        out, ms = timed_forward(
            device,
            lambda next_tokens=next_tokens, past=past, attention_mask=attention_mask: model(
                input_ids=next_tokens,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            ),
        )
        step_ms.append(ms)
        past = out.past_key_values
        next_tokens = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    sync(device)
    end_to_end_ms = (time.perf_counter() - t0) * 1000.0

    total_tokens = len(PROMPTS) * MAX_TOKENS
    decode_tokens = len(PROMPTS) * (MAX_TOKENS - 1)
    decode_ms = sum(step_ms)
    return {
        "n_requests": len(PROMPTS),
        "prefill_ms": prefill_ms,
        "decode_wall_ms": decode_ms,
        "end_to_end_ms": end_to_end_ms,
        "total_tokens": total_tokens,
        "aggregate_tps_decode_only": decode_tokens / (decode_ms / 1000.0),
        "aggregate_tps_end_to_end": total_tokens / (end_to_end_ms / 1000.0),
        "mean_batched_step_ms": statistics.mean(step_ms),
        "tokens_per_batched_step": len(PROMPTS),
    }


def main() -> None:
    print("\n" + "=" * 72)
    print("  PageServe True Batched Decode Benchmark")
    print("=" * 72)

    cfg = Config()
    print(f"\n  Model: {cfg.model_name}")
    print(f"  Device: {cfg.device}")
    print(f"  Requests: {N_REQUESTS} | max_new_tokens/request: {MAX_TOKENS}")

    if cfg.device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")

    print("\n  Loading model ...", flush=True)
    t0 = time.perf_counter()
    loaded = load_model_and_tokenizer(cfg)
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.device
    print(f"  Model loaded in {(time.perf_counter() - t0) * 1000:.0f} ms")

    print("\n  Warming up ...", flush=True)
    run_one_serial(model, tokenizer, device, "Hi", max_tokens=3)
    print("  Warm-up done.")

    print("\n  Running serial baseline ...", flush=True)
    serial = bench_serial(model, tokenizer, device)

    print("  Running round-robin interleaving ...", flush=True)
    round_robin = bench_round_robin(model, tokenizer, device)

    print("  Running true batched decode ...", flush=True)
    true_batched = bench_true_batched(model, tokenizer, device)

    serial_wall = serial["wall_ms"]
    rr_wall = round_robin["end_to_end_ms"]
    batched_wall = true_batched["end_to_end_ms"]

    comparisons = {
        "round_robin_vs_serial_speedup": serial_wall / rr_wall,
        "true_batched_vs_serial_speedup": serial_wall / batched_wall,
        "true_batched_vs_round_robin_speedup": rr_wall / batched_wall,
    }

    results = {
        "config": {
            "model": cfg.model_name,
            "device": device,
            "n_requests": N_REQUESTS,
            "max_tokens": MAX_TOKENS,
        },
        "serial": serial,
        "round_robin": round_robin,
        "true_batched": true_batched,
        "comparisons": comparisons,
        "gpu_memory": gpu_memory_mb(device),
    }

    print("\n" + "-" * 72)
    print("  Results")
    print("-" * 72)
    print(f"  Serial wall:                  {serial_wall:>9.1f} ms  | {serial['aggregate_tps']:>7.1f} tok/s")
    print(f"  Round-robin wall:             {rr_wall:>9.1f} ms  | {round_robin['aggregate_tps_end_to_end']:>7.1f} tok/s")
    print(f"  True batched wall:            {batched_wall:>9.1f} ms  | {true_batched['aggregate_tps_end_to_end']:>7.1f} tok/s")
    print(f"  True batched decode-only tps: {'':>9}     {true_batched['aggregate_tps_decode_only']:>7.1f} tok/s")
    print()
    print(f"  Round-robin vs serial:        {comparisons['round_robin_vs_serial_speedup']:>9.2f}x")
    print(f"  True batched vs serial:       {comparisons['true_batched_vs_serial_speedup']:>9.2f}x")
    print(f"  True batched vs round-robin:  {comparisons['true_batched_vs_round_robin_speedup']:>9.2f}x")
    if results["gpu_memory"]:
        mem = results["gpu_memory"]
        print()
        print(f"  CUDA max allocated:           {mem['max_allocated_mb']:>9.1f} MB")
        print(f"  CUDA max reserved:            {mem['max_reserved_mb']:>9.1f} MB")

    with open(OUTPUT_PATH, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  Results saved -> {OUTPUT_PATH}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
