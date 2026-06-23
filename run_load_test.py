#!/usr/bin/env python
"""
run_load_test.py — Phase 11: CLI entry point for the PageServe load tester.

Usage examples
--------------
# Run against Phase 2 server with constant load:
  python run_load_test.py --server phase2 --profile constant --num-requests 20 --rps 3.0

# Ramp load on Phase 1:
  python run_load_test.py --server phase1 --profile ramp --num-requests 30 \
         --start-rps 1.0 --end-rps 10.0 --duration 30.0

# Burst load and save result:
  python run_load_test.py --server phase2 --profile burst --num-bursts 4 \
         --requests-per-burst 5 --burst-interval 3.0 --output burst_report.json

# Compare Phase 1 vs Phase 2:
  python run_load_test.py --server phase1 --output p1.json
  python run_load_test.py --server phase2 --output p2.json --compare-with p1.json

All arguments have sensible defaults so `python run_load_test.py` works with
no flags (targets Phase 2 server, 50 requests at 5 rps, constant profile).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

# Ensure project root is importable when running directly
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_test.profiles import (
    LoadRequest,
    burst_load,
    constant_load,
    default_prompt_pool,
    ramp_load,
)
from load_test.report import (
    build_report,
    compare_reports,
    print_comparison_table,
    print_report_table,
    save_report_json,
)
from load_test.runner import LoadTestRunner


# ── Argument parsing ──────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PageServe load tester — Phase 11",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server selection
    p.add_argument(
        "--server",
        choices=["phase1", "phase2", "custom"],
        default="phase2",
        help=(
            "Which server to target. "
            "phase1 → http://localhost:8000, "
            "phase2 → http://localhost:8001, "
            "custom → use --url"
        ),
    )
    p.add_argument(
        "--url",
        default=None,
        help="Base URL when --server custom (e.g. http://myhost:8080)",
    )

    # Load profile
    p.add_argument(
        "--profile",
        choices=["constant", "ramp", "burst"],
        default="constant",
        help="Load profile to apply",
    )
    p.add_argument("--num-requests", type=int, default=50,
                   help="Total requests (constant/ramp profiles)")
    p.add_argument("--rps", type=float, default=5.0,
                   help="Requests per second (constant profile)")
    p.add_argument("--start-rps", type=float, default=2.0,
                   help="Starting RPS (ramp profile)")
    p.add_argument("--end-rps", type=float, default=10.0,
                   help="Ending RPS (ramp profile)")
    p.add_argument("--duration", type=float, default=30.0,
                   help="Target ramp duration in seconds (ramp profile)")
    p.add_argument("--num-bursts", type=int, default=5,
                   help="Number of bursts (burst profile)")
    p.add_argument("--requests-per-burst", type=int, default=10,
                   help="Requests per burst (burst profile)")
    p.add_argument("--burst-interval", type=float, default=5.0,
                   help="Seconds between bursts (burst profile)")

    # Common generation params
    p.add_argument("--max-new-tokens", type=int, default=50,
                   help="Max tokens to generate per request")

    # Runner config
    p.add_argument("--max-concurrent", type=int, default=50,
                   help="Max in-flight requests at once")

    # Output
    p.add_argument("--output", default="load_test_report.json",
                   help="Path to write the JSON report")
    p.add_argument("--compare-with", default=None, metavar="PATH",
                   help="Path to a prior report JSON to compare against")

    return p


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Resolve base URL
    if args.server == "phase1":
        base_url = "http://localhost:8000"
        server_label = "phase1"
    elif args.server == "phase2":
        base_url = "http://localhost:8001"
        server_label = "phase2"
    else:
        if not args.url:
            parser.error("--url is required when --server custom")
        base_url = args.url
        server_label = "custom"

    prompts = default_prompt_pool()

    # Build load request list from selected profile
    if args.profile == "constant":
        load_requests = constant_load(
            num_requests=args.num_requests,
            requests_per_second=args.rps,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.profile == "ramp":
        load_requests = ramp_load(
            num_requests=args.num_requests,
            start_rps=args.start_rps,
            end_rps=args.end_rps,
            duration_seconds=args.duration,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
        )
    else:  # burst
        load_requests = burst_load(
            num_bursts=args.num_bursts,
            requests_per_burst=args.requests_per_burst,
            burst_interval_seconds=args.burst_interval,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
        )

    total = len(load_requests)
    print(
        f"\nPageServe Load Test — Phase 11\n"
        f"  Target: {base_url}\n"
        f"  Profile: {args.profile}\n"
        f"  Requests: {total}\n"
        f"  Max concurrent: {args.max_concurrent}\n"
    )

    runner = LoadTestRunner(
        base_url=base_url,
        max_concurrent=args.max_concurrent,
        request_timeout_s=120.0,
    )

    test_start = time.perf_counter()
    print("Starting load test …")
    results = await runner.run(load_requests)
    test_duration_s = time.perf_counter() - test_start

    report = build_report(results, test_duration_s, server_label=server_label)

    print()
    print_report_table(report)
    print()

    save_report_json(report, args.output)

    # Optional comparison
    if args.compare_with:
        try:
            with open(args.compare_with, encoding="utf-8") as fh:
                prior_report = json.load(fh)
            delta = compare_reports(prior_report, report)
            print()
            print_comparison_table(delta)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: could not load comparison report: {exc}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
