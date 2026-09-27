#!/usr/bin/env python
"""
benchmarks/bench_offline.py — What does each optimisation actually buy?

Runs the same reproducible workload through several systems *in-process*
(no HTTP), all requests submitted at t=0, and reports throughput, latency
percentiles, GPU memory and utilisation.  Each suite isolates one idea:

  batching   random prompts, fixed output lengths
             hf_sequential      HF generate(), one request at a time (Phase 1)
             hf_static_batch    HF generate() on padded batches of B requests
                                (the classic "static batching" baseline)
             engine_no_batching this engine with max_batch_size=1
             engine             continuous batching + paged KV cache
  prefix     long shared system prompt + short unique questions
             engine             prefix caching off
             engine+prefix      automatic prefix caching on
  spec       copy-heavy prompts (and random prompts, to show the cost when
             drafts are bad)
             engine             no speculation
             engine+ngram       n-gram / prompt-lookup speculation
             engine+draft       draft-model speculation (--draft-model)

Examples
--------
  python -m benchmarks.bench_offline                    # everything, model sized to the GPU
  python -m benchmarks.bench_offline --suites batching --num-requests 256
  python -m benchmarks.bench_offline --suites spec --draft-model auto
  python -m benchmarks.bench_offline --quick            # small smoke run (CPU/MPS ok)

Results: results/offline_<gpu>_<time>.json and .md
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from benchmarks.common import (
    DRAFT_MODEL,
    RESULTS_DIR,
    gpu_tag,
    markdown_table,
    pcts,
    pick_model,
    save_results,
    system_info,
)
from inference_engine.config import Config
from inference_engine.engine.scheduler import ContinuousBatchingScheduler
from inference_engine.engine.sequence import SamplingParams
from inference_engine.engine.sequential import generate as hf_generate_one
from inference_engine.models.loader import load_model, load_model_and_tokenizer
from load_test.gpu_monitor import GPUMonitor
from load_test.workloads import WorkloadItem, make_workload

# ── Systems ───────────────────────────────────────────────────────────────────

ENGINE_SYSTEMS: Dict[str, dict] = {
    "engine_no_batching": {"max_batch_size": 1, "enable_prefix_caching": False},
    "engine": {"enable_prefix_caching": False},
    "engine+prefix": {"enable_prefix_caching": True},
    "engine+ngram": {"enable_prefix_caching": False, "speculative_method": "ngram"},
    "engine+draft": {"enable_prefix_caching": False, "speculative_method": "draft"},
}

SUITES = {
    "batching": [("random", ["hf_sequential", "hf_static_batch", "engine_no_batching", "engine"])],
    "prefix": [("shared_prefix", ["engine", "engine+prefix"])],
    "spec": [
        ("repetitive", ["engine", "engine+ngram", "engine+draft"]),
        ("random", ["engine", "engine+ngram", "engine+draft"]),
    ],
}

# Systems too slow to push the whole workload through: they run a prefix of
# it and report throughput from that subset.
SUBSET_SYSTEMS = {"hf_sequential", "engine_no_batching"}


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _reset_peak(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_mb(device: str) -> Optional[float]:
    return torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None


def _summarise(items: List[WorkloadItem], wall_s: float, input_tokens: int,
               output_tokens: int, **extra) -> dict:
    return {
        "num_requests": len(items),
        "wall_s": wall_s,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "output_tok_s": output_tokens / wall_s,
        "total_tok_s": (input_tokens + output_tokens) / wall_s,
        "req_s": len(items) / wall_s,
        **extra,
    }


# ── Baselines ─────────────────────────────────────────────────────────────────


def run_hf_sequential(model, tokenizer, device, items) -> dict:
    """Phase 1: one request at a time, HF KV cache, greedy."""
    _reset_peak(device)
    results = []
    with GPUMonitor() as gpu:
        _sync(device)
        t0 = time.perf_counter()
        for item in items:
            results.append(hf_generate_one(model, tokenizer, item.prompt, item.max_new_tokens,
                                           device, eos_token_id=None))
        _sync(device)
        wall = time.perf_counter() - t0
    return _summarise(
        items, wall,
        input_tokens=sum(r.prompt_tokens for r in results),
        output_tokens=sum(r.generated_tokens for r in results),
        ttft_ms=pcts([r.ttft_ms for r in results]),
        tpot_ms=pcts([sum(r.per_token_latencies_ms) / max(1, len(r.per_token_latencies_ms))
                      for r in results]),
        itl_ms=pcts([x for r in results for x in r.per_token_latencies_ms]),
        e2e_ms=pcts([r.total_latency_ms for r in results]),
        peak_gpu_mem_mb=_peak_mb(device),
        gpu=gpu.summary(),
        note="requests run back-to-back; TTFT/E2E exclude time spent waiting for earlier requests",
    )


def run_hf_static_batch(model, tokenizer, device, items, batch_size: int) -> dict:
    """Static batching: pad B prompts together, decode until the longest is done.

    Every request in a batch generates as many tokens as the longest request
    in it; only the requested tokens count as useful output.
    """
    _reset_peak(device)
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    input_tokens = 0
    with GPUMonitor() as gpu:
        _sync(device)
        t0 = time.perf_counter()
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            enc = tokenizer([it.prompt for it in batch], return_tensors="pt", padding=True).to(device)
            input_tokens += int(enc["attention_mask"].sum())
            n = max(it.max_new_tokens for it in batch)
            with torch.inference_mode():
                model.generate(**enc, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id)
        _sync(device)
        wall = time.perf_counter() - t0
    tokenizer.padding_side = old_side
    useful = sum(it.max_new_tokens for it in items)
    return _summarise(items, wall, input_tokens, useful, batch_size=batch_size,
                      peak_gpu_mem_mb=_peak_mb(device), gpu=gpu.summary())


# ── Engine ────────────────────────────────────────────────────────────────────


def run_engine(model, tokenizer, device, items, overrides: dict, base: dict,
               draft_model=None) -> dict:
    config = Config(**{**base, **overrides})
    _reset_peak(device)
    scheduler = ContinuousBatchingScheduler(model, tokenizer, config, draft_model=draft_model)

    async def go():
        scheduler.start()
        # Warm-up (kernel selection, allocator warm-up) — not timed.
        _, fut = await scheduler.add_request(
            "Warm up the engine.", sampling=SamplingParams(max_new_tokens=8, ignore_eos=True))
        await fut
        counters_before = scheduler.spec_decode_stats()
        with GPUMonitor() as gpu:
            t0 = time.perf_counter()
            pending = [
                await scheduler.add_request(
                    it.prompt, sampling=SamplingParams(max_new_tokens=it.max_new_tokens,
                                                       ignore_eos=True))
                for it in items
            ]
            seqs = await asyncio.gather(*[f for _, f in pending])
            wall = time.perf_counter() - t0
        await scheduler.stop()
        return seqs, wall, gpu.summary(), counters_before

    all_seqs, wall, gpu, before = asyncio.run(go())
    # Only completed requests count; an engine error (e.g. CUDA OOM in a step)
    # fails a step's sequences instead of raising — surface that, don't hide it.
    seqs = [s for s in all_seqs if s.finish_reason in ("length", "eos", "stop")]
    num_failed = len(all_seqs) - len(seqs)
    spec = scheduler.spec_decode_stats()
    drafted = spec["draft_tokens"] - before["draft_tokens"]
    accepted = spec["accepted_tokens"] - before["accepted_tokens"]
    alloc = scheduler.block_allocator.stats(include_per_sequence=False)
    result = _summarise(
        items, wall,
        input_tokens=sum(len(s.prompt_token_ids) for s in seqs),
        output_tokens=sum(len(s.generated_token_ids) for s in seqs),
        ttft_ms=pcts([s.ttft_ms for s in seqs]),
        tpot_ms=pcts([s.tpot_ms for s in seqs if len(s.generated_token_ids) > 1]),
        itl_ms=pcts([x for s in seqs for x in s.itl_ms]),
        e2e_ms=pcts([s.e2e_latency_ms for s in seqs]),
        peak_gpu_mem_mb=_peak_mb(device),
        gpu=gpu,
        kv_blocks=alloc["num_blocks"],
        kv_pool_mb=scheduler.paged_kv_cache.pool_size_mb(),
        max_batch_size=config.max_batch_size,
        prefix_cache_hit_rate=alloc["prefix_cache_hit_rate"],
        cached_prompt_tokens=sum(s.num_cached_tokens for s in seqs),
        spec_acceptance_rate=(accepted / drafted) if drafted else None,
        preemptions=scheduler.num_preemptions_swap + scheduler.num_preemptions_recompute,
        num_failed=num_failed,
        finish_reasons=sorted({s.finish_reason for s in all_seqs}),
    )
    del scheduler
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


# ── Reporting ─────────────────────────────────────────────────────────────────


def suite_markdown(title: str, runs: List[dict]) -> str:
    base = runs[0]["metrics"]["output_tok_s"] if runs else 1.0
    rows = []
    for run in runs:
        m = run["metrics"]
        extra = []
        if m.get("prefix_cache_hit_rate"):
            extra.append(f"prefix hit {m['prefix_cache_hit_rate']:.0%}")
        if m.get("spec_acceptance_rate") is not None:
            extra.append(f"accept {m['spec_acceptance_rate']:.0%}")
        if m.get("preemptions"):
            extra.append(f"{m['preemptions']} preemptions")
        if m.get("num_failed"):
            extra.append(f"⚠ {m['num_failed']} FAILED (excluded)")
        if run["system"] in SUBSET_SYSTEMS:
            extra.append(f"subset of {m['num_requests']}")
        gpu = m.get("gpu", {})
        rows.append([
            run["system"],
            m["output_tok_s"],
            m["output_tok_s"] / base,
            m.get("ttft_ms", {}).get("p50"),
            m.get("ttft_ms", {}).get("p99"),
            m.get("tpot_ms", {}).get("p50"),
            m.get("itl_ms", {}).get("p99"),
            m.get("peak_gpu_mem_mb"),
            gpu.get("util_mean_pct") if gpu.get("available") else None,
            ", ".join(extra),
        ])
    headers = ["system", "output tok/s", "speedup", "TTFT p50 ms", "TTFT p99 ms",
               "TPOT p50 ms", "ITL p99 ms", "peak mem MB", "GPU util %", "notes"]
    return f"### {title}\n\n" + markdown_table(headers, rows) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="auto")
    p.add_argument("--draft-model", default=None,
                   help=f"Draft model for engine+draft ('auto' = {DRAFT_MODEL})")
    p.add_argument("--suites", default="batching,prefix,spec")
    p.add_argument("--num-requests", type=int, default=128)
    p.add_argument("--sequential-requests", type=int, default=12,
                   help="requests for the slow one-at-a-time systems")
    p.add_argument("--input-len", type=int, nargs=2, default=(256, 512))
    p.add_argument("--output-len", type=int, nargs=2, default=(128, 256))
    p.add_argument("--max-batch-size", type=int, default=0, help="0 = engine default")
    p.add_argument("--static-batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=RESULTS_DIR)
    p.add_argument("--quick", action="store_true", help="tiny sizes for a smoke test")
    args = p.parse_args()

    if args.quick:
        args.num_requests, args.sequential_requests = 16, 4
        args.input_len, args.output_len = (32, 64), (16, 32)
        args.static_batch_size = 8

    base_config = Config()
    device = base_config.device
    model_name = pick_model(args.model)
    print(f"Loading {model_name} on {device} …", flush=True)
    model, tokenizer, _ = load_model_and_tokenizer(Config(model_name=model_name))

    draft_model = None
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    if "spec" in suites and args.draft_model:
        draft_name = DRAFT_MODEL if args.draft_model == "auto" else args.draft_model
        if draft_name == model_name:
            print("Draft model equals target; skipping engine+draft")
        else:
            print(f"Loading draft model {draft_name} …", flush=True)
            draft_model = load_model(draft_name, device, base_config.dtype)

    base = {
        "model_name": model_name,
        "request_timeout_ms": 1e9,          # offline: nothing should expire
        "max_queue_size": args.num_requests + 64,
    }
    if args.max_batch_size:
        base["max_batch_size"] = args.max_batch_size
    if draft_model is not None:
        base["draft_model_name"] = draft_name

    info = system_info()
    info["model"] = model_name
    payload = {"system": info, "args": vars(args), "runs": []}
    markdown = [f"## Offline benchmark — {info.get('gpu', info['device'].upper())} — {model_name}\n",
                f"torch {info['torch']}, transformers {info['transformers']}, "
                f"commit {info.get('git_commit', '?')}, seed {args.seed}\n"]

    for suite in suites:
        for workload_name, systems in SUITES[suite]:
            kwargs = {"output_len": tuple(args.output_len)}
            if workload_name == "random":
                kwargs["input_len"] = tuple(args.input_len)
            if workload_name == "repetitive":
                kwargs = {"passage_len": tuple(args.output_len)}
            if workload_name == "shared_prefix":
                kwargs["prefix_len"] = 1024 if not args.quick else 128
            items = make_workload(workload_name, args.num_requests, seed=args.seed,
                                  tokenizer=tokenizer, **kwargs)
            runs = []
            for system in systems:
                if system == "engine+draft" and draft_model is None:
                    continue
                subset = items[:args.sequential_requests] if system in SUBSET_SYSTEMS else items
                print(f"[{suite}/{workload_name}] {system}: {len(subset)} requests …", flush=True)
                try:
                    if system == "hf_sequential":
                        metrics = run_hf_sequential(model, tokenizer, device, subset)
                    elif system == "hf_static_batch":
                        metrics = run_hf_static_batch(model, tokenizer, device, subset,
                                                      args.static_batch_size)
                    else:
                        metrics = run_engine(model, tokenizer, device, subset,
                                             ENGINE_SYSTEMS[system], base,
                                             draft_model if system == "engine+draft" else None)
                except torch.cuda.OutOfMemoryError as exc:
                    print(f"  OOM: {exc}")
                    metrics = {"error": "out of memory", "output_tok_s": 0.0}
                    torch.cuda.empty_cache()
                print(f"  → {metrics['output_tok_s']:.1f} output tok/s", flush=True)
                runs.append({"suite": suite, "workload": workload_name, "system": system,
                             "metrics": metrics})
            payload["runs"].extend(runs)
            ok_runs = [r for r in runs if "error" not in r["metrics"]]
            markdown.append(suite_markdown(f"{suite}: {workload_name} workload", ok_runs))

    stem = f"offline_{gpu_tag()}_{time.strftime('%Y%m%d-%H%M%S')}"
    json_path = save_results(payload, args.output_dir, stem)
    md_path = os.path.join(args.output_dir, stem + ".md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(markdown))
    print("\n" + "\n".join(markdown))
    print(f"Saved {json_path} and {md_path}")


if __name__ == "__main__":
    main()
