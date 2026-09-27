#!/usr/bin/env python
"""
run_load_test.py — CLI entry point for the PageServe load tester.

Usage examples
--------------
# Poisson traffic at 4 req/s, streaming, fixed-length outputs:
  python run_load_test.py --server phase2 --profile poisson --rps 4 \\
         --workload random --num-requests 200 --stream --ignore-eos

# Closed loop: 32 concurrent users hammering the server:
  python run_load_test.py --server phase2 --concurrency 32 --num-requests 256 --stream

# Prefix-caching workload (long shared system prompt):
  python run_load_test.py --server phase2 --workload shared_prefix --rps 8 --stream

# Any OpenAI-compatible server (e.g. vLLM) with the same client:
  python run_load_test.py --server custom --url http://localhost:8000 --api openai \\
         --model Qwen/Qwen2.5-1.5B-Instruct --stream

# Legacy profiles and Phase 1 vs Phase 2 comparison still work:
  python run_load_test.py --server phase1 --output p1.json
  python run_load_test.py --server phase2 --output p2.json --compare-with p1.json

With no flags: Phase 2 server, 50 requests of the "chat" workload at a
constant 5 req/s.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

# Ensure project root is importable when running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_test.gpu_monitor import GPUMonitor
from load_test.profiles import (
    burst_load,
    constant_load,
    poisson_arrivals,
    ramp_load,
    schedule_workload,
)
from load_test.report import (
    build_report,
    compare_reports,
    engine_summary,
    print_comparison_table,
    print_report_table,
    save_report_json,
    scrape_metrics,
)
from load_test.runner import LoadTestRunner
from load_test.workloads import WORKLOADS, load_tokenizer, make_workload


# ── Argument parsing ──────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PageServe load tester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server selection
    p.add_argument("--server", choices=["phase1", "phase2", "custom"], default="phase2",
                   help="phase1 → :8000, phase2 → :8001, custom → --url")
    p.add_argument("--url", default=None, help="Base URL when --server custom")
    p.add_argument("--api", choices=["native", "openai"], default="native",
                   help="native /generate or OpenAI /v1/completions")
    p.add_argument("--model", default=None,
                   help="Model name (OpenAI API field; also used to load a "
                        "tokenizer for exact prompt lengths)")

    # Workload
    p.add_argument("--workload", choices=sorted(WORKLOADS), default="chat")
    p.add_argument("--num-requests", type=int, default=50)
    p.add_argument("--input-len", type=int, nargs=2, default=None, metavar=("MIN", "MAX"),
                   help="random workload: prompt token range")
    p.add_argument("--output-len", type=int, nargs=2, default=None, metavar=("MIN", "MAX"),
                   help="random/shared_prefix/chat: output token range")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Fix every request's output length (overrides --output-len)")
    p.add_argument("--seed", type=int, default=0)

    # Traffic
    p.add_argument("--profile", choices=["constant", "poisson", "ramp", "burst"],
                   default="constant")
    p.add_argument("--rps", type=float, default=5.0,
                   help="Request rate (constant/poisson); 'inf' sends all at once")
    p.add_argument("--start-rps", type=float, default=2.0, help="ramp profile")
    p.add_argument("--end-rps", type=float, default=10.0, help="ramp profile")
    p.add_argument("--duration", type=float, default=30.0, help="ramp duration (s)")
    p.add_argument("--num-bursts", type=int, default=5)
    p.add_argument("--requests-per-burst", type=int, default=10)
    p.add_argument("--burst-interval", type=float, default=5.0)
    p.add_argument("--concurrency", type=int, default=None,
                   help="Closed loop with N concurrent users (ignores --profile)")
    p.add_argument("--max-concurrent", type=int, default=None,
                   help="Cap in-flight requests in open-loop mode (default: none)")

    # Client behaviour
    p.add_argument("--stream", action="store_true",
                   help="Stream tokens and measure TTFT/ITL on the client")
    p.add_argument("--ignore-eos", action="store_true",
                   help="Always generate the requested number of tokens")
    p.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout (s)")
    p.add_argument("--slo-ttft-ms", type=float, default=2000.0)
    p.add_argument("--slo-tpot-ms", type=float, default=100.0)

    # Output
    p.add_argument("--output", default="load_test_report.json")
    p.add_argument("--compare-with", default=None, metavar="PATH")
    return p


def build_requests(args):
    tokenizer = load_tokenizer(args.model)
    kwargs = {}
    if args.input_len and args.workload == "random":
        kwargs["input_len"] = tuple(args.input_len)
    if args.output_len and args.workload in ("random", "shared_prefix", "chat"):
        kwargs["output_len"] = tuple(args.output_len)
    items = make_workload(args.workload, args.num_requests, seed=args.seed,
                          tokenizer=tokenizer, **kwargs)
    if args.max_new_tokens:
        for item in items:
            item.max_new_tokens = args.max_new_tokens

    if args.profile == "poisson":
        return schedule_workload(items, poisson_arrivals(len(items), args.rps, args.seed))
    prompts = [item.prompt for item in items]
    if args.profile == "constant":
        reqs = constant_load(len(items), args.rps, prompts) if args.rps != float("inf") \
            else schedule_workload(items, [0.0] * len(items))
    elif args.profile == "ramp":
        reqs = ramp_load(len(items), args.start_rps, args.end_rps, args.duration, prompts)
    else:
        reqs = burst_load(args.num_bursts, args.requests_per_burst, args.burst_interval, prompts)
    # Carry per-request output lengths over from the workload.
    for req, item in zip(reqs, items * (len(reqs) // max(1, len(items)) + 1)):
        req.max_new_tokens = item.max_new_tokens
    return reqs


async def run(args) -> dict:
    if args.server == "phase1":
        base_url, label = "http://localhost:8000", "phase1"
    elif args.server == "phase2":
        base_url, label = "http://localhost:8001", "phase2"
    else:
        if not args.url:
            raise SystemExit("--url is required when --server custom")
        base_url, label = args.url.rstrip("/"), "custom"

    load_requests = build_requests(args)
    mode = f"closed loop, concurrency={args.concurrency}" if args.concurrency else \
        f"open loop, {args.profile} @ {args.rps} req/s"
    print(f"\nPageServe load test → {base_url} ({args.api} API)\n"
          f"  workload={args.workload} requests={len(load_requests)} {mode}"
          f" stream={args.stream} ignore_eos={args.ignore_eos}\n")

    runner = LoadTestRunner(
        base_url=base_url,
        max_concurrent=args.max_concurrent,
        request_timeout_s=args.timeout,
        api=args.api,
        stream=args.stream,
        ignore_eos=args.ignore_eos,
        model=args.model,
    )

    metrics_before = await scrape_metrics(base_url)
    with GPUMonitor() as gpu:
        test_start = time.perf_counter()
        if args.concurrency:
            results = await runner.run_closed_loop(load_requests, args.concurrency)
        else:
            results = await runner.run(load_requests)
        test_duration_s = time.perf_counter() - test_start

    report = build_report(results, test_duration_s, server_label=label,
                          slo_ttft_ms=args.slo_ttft_ms, slo_tpot_ms=args.slo_tpot_ms)
    report["config"] = {k: v for k, v in vars(args).items() if k not in ("output", "compare_with")}
    report["gpu"] = gpu.summary()
    report["engine"] = engine_summary(await scrape_metrics(base_url), before=metrics_before)
    return report


def main() -> None:
    args = build_arg_parser().parse_args()
    report = asyncio.run(run(args))
    print()
    print_report_table(report)
    if report["gpu"].get("available"):
        g = report["gpu"]
        print(f"  GPU {g['gpu_name']}: util mean {g['util_mean_pct']:.0f}% "
              f"(max {g['util_max_pct']:.0f}%), peak mem {g['mem_used_max_mb']:.0f} MB")
    if report["engine"]:
        print(f"  Engine: {json.dumps({k: v for k, v in report['engine'].items() if k != 'gpu_memory'})}")
    print()
    save_report_json(report, args.output)

    if args.compare_with:
        try:
            with open(args.compare_with, encoding="utf-8") as fh:
                prior_report = json.load(fh)
            print()
            print_comparison_table(compare_reports(prior_report, report))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: could not load comparison report: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
