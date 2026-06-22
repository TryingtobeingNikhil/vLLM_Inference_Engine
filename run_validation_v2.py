"""
run_validation_v2.py — Phase 2 continuous batching benchmark.

Sends 8 concurrent requests to the Phase 2 server (port 8001) using
asyncio.gather + httpx async client, then compares total wall time and
aggregate throughput against the Phase 1 sequential baseline stored in
baseline_metrics.json.

Comparison metrics (Gap 6 resolution)
---------------------------------------
Three numbers are computed and reported:

1. Phase 1 expected wall time (ms):
   `mean_single_request_latency_ms × 8`
   Source: baseline_metrics.json → summary.total_latency_ms.mean

2. Phase 2 actual wall time (ms):
   asyncio.gather wall clock from first dispatch to last response.

3. Speedup ratio:
   phase1_expected / phase2_actual  (> 1.0 means Phase 2 is faster)

Additionally:
- TTFT per request (from response JSON .ttft_ms)
- Aggregate throughput: sum(generated_tokens) / phase2_wall_s  (tokens/s)

Plots
-----
- Batch size over time from /metrics endpoint  (requires matplotlib)
- Saved as batch_size_over_time.png in the script directory

Output
------
- continuous_batching_metrics.json

Usage
-----
    # Terminal 1 (Phase 2 server must be running):
    cd inference_engine
    uvicorn inference_engine.server.app_v2:app --host 0.0.0.0 --port 8001

    # Terminal 2:
    cd /Users/nikhilmourya/Desktop/PageServe
    python run_validation_v2.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Config ────────────────────────────────────────────────────────────────────

PHASE2_BASE_URL = "http://127.0.0.1:8001"
BASELINE_METRICS_PATH = Path(__file__).parent / "baseline_metrics.json"
OUTPUT_PATH = Path(__file__).parent / "continuous_batching_metrics.json"
PLOT_PATH = Path(__file__).parent / "batch_size_over_time.png"

N_CONCURRENT_REQUESTS = 8
MAX_NEW_TOKENS = 50

PROMPTS = [
    "Explain the concept of entropy in thermodynamics.",
    "What are the key differences between supervised and unsupervised learning?",
    "Describe the role of the prefrontal cortex in decision making.",
    "Summarize the causes of World War I in three sentences.",
    "How does a transformer architecture work in natural language processing?",
    "What is the significance of the Pythagorean theorem in geometry?",
    "Explain how HTTPS certificates establish trust on the internet.",
    "Describe the process of photosynthesis at the molecular level.",
]

assert len(PROMPTS) == N_CONCURRENT_REQUESTS, "Must have exactly N_CONCURRENT_REQUESTS prompts"


# ── HTTP helpers ──────────────────────────────────────────────────────────────

async def _send_request(
    client: Any,  # httpx.AsyncClient
    prompt: str,
    idx: int,
) -> Dict[str, Any]:
    """Send one /generate request; return the JSON body with added timing."""
    t_send = time.perf_counter()
    resp = await client.post(
        f"{PHASE2_BASE_URL}/generate",
        json={"prompt": prompt, "max_new_tokens": MAX_NEW_TOKENS},
        timeout=300.0,
    )
    t_recv = time.perf_counter()

    resp.raise_for_status()
    data = resp.json()
    data["_wall_ms"] = (t_recv - t_send) * 1000.0
    data["_idx"] = idx
    return data


async def _fetch_metrics(client: Any) -> Dict[str, Any]:
    resp = await client.get(f"{PHASE2_BASE_URL}/metrics", timeout=10.0)
    resp.raise_for_status()
    return resp.json()


# ── Phase 1 baseline loader ───────────────────────────────────────────────────

def _load_phase1_baseline() -> Optional[float]:
    """Load mean total_latency_ms from baseline_metrics.json.

    Returns None if the file is missing or malformed.
    """
    if not BASELINE_METRICS_PATH.exists():
        print(f"[WARN] baseline_metrics.json not found at {BASELINE_METRICS_PATH}")
        print("       Run Phase 1 server first and hit /generate to populate it.")
        return None
    try:
        with open(BASELINE_METRICS_PATH) as fh:
            data = json.load(fh)
        mean_ms: float = data["summary"]["total_latency_ms"]["mean"]
        print(f"[Phase 1] Mean single-request latency: {mean_ms:.1f} ms")
        return mean_ms
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"[WARN] Could not parse baseline_metrics.json: {exc}")
        return None


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_batch_size(batch_size_over_time: List[Tuple[float, int]]) -> None:
    """Plot batch_size_over_time if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping batch size plot.")
        return

    if not batch_size_over_time:
        print("[INFO] No batch_size_over_time data to plot.")
        return

    # Normalise timestamps: t=0 is the first scheduler step
    t0 = batch_size_over_time[0][0]
    xs = [(t - t0) for t, _ in batch_size_over_time]
    ys = [bs for _, bs in batch_size_over_time]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(xs, ys, where="post", linewidth=1.5, color="#2196F3")
    ax.fill_between(xs, ys, step="post", alpha=0.15, color="#2196F3")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Active batch size")
    ax.set_title("Phase 2: Batch Size Over Time")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"[Plot] Saved to {PLOT_PATH}")
    plt.close(fig)


# ── Main benchmark ────────────────────────────────────────────────────────────

async def main() -> None:
    try:
        import httpx
    except ImportError:
        print("ERROR: httpx is required.  Install with: pip install httpx")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Phase 2 Continuous Batching Validation")
    print(f"  Server: {PHASE2_BASE_URL}")
    print(f"  Concurrent requests: {N_CONCURRENT_REQUESTS}")
    print(f"  max_new_tokens: {MAX_NEW_TOKENS}")
    print(f"{'='*60}\n")

    # ── Health check ──────────────────────────────────────────────────────────
    async with httpx.AsyncClient() as client:
        try:
            health = await client.get(f"{PHASE2_BASE_URL}/health", timeout=5.0)
            health.raise_for_status()
            h = health.json()
            print(f"[Health] model={h.get('model')} device={h.get('device')} "
                  f"max_batch_size={h.get('max_batch_size')}\n")
        except Exception as exc:
            print(f"ERROR: Phase 2 server not reachable at {PHASE2_BASE_URL}: {exc}")
            print("Start it with:")
            print("  uvicorn inference_engine.server.app_v2:app --host 0.0.0.0 --port 8001")
            sys.exit(1)

        # ── Load Phase 1 baseline ─────────────────────────────────────────────
        phase1_mean_ms = _load_phase1_baseline()
        phase1_expected_ms: Optional[float] = (
            phase1_mean_ms * N_CONCURRENT_REQUESTS if phase1_mean_ms else None
        )

        # ── Send 8 concurrent requests ────────────────────────────────────────
        print(f"[Benchmark] Sending {N_CONCURRENT_REQUESTS} concurrent requests …")
        t_wall_start = time.perf_counter()

        results = await asyncio.gather(
            *[_send_request(client, prompt, i) for i, prompt in enumerate(PROMPTS)],
            return_exceptions=False,
        )

        phase2_wall_ms = (time.perf_counter() - t_wall_start) * 1000.0

        # ── Fetch scheduler metrics ───────────────────────────────────────────
        metrics = await _fetch_metrics(client)

    # ── Compute comparison numbers ────────────────────────────────────────────
    ttft_values = [r["ttft_ms"] for r in results]
    total_latencies = [r["total_latency_ms"] for r in results]
    generated_tokens_total = sum(r["generated_tokens"] for r in results)
    aggregate_tps = generated_tokens_total / (phase2_wall_ms / 1000.0)

    speedup_ratio: Optional[float] = None
    if phase1_expected_ms:
        speedup_ratio = phase1_expected_ms / phase2_wall_ms

    # ── Print results table ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Per-request results")
    print(f"{'─'*60}")
    print(f"  {'#':>2}  {'TTFT (ms)':>10}  {'Total (ms)':>12}  {'Tokens':>6}  {'Finish':>8}")
    print(f"  {'─'*2}  {'─'*10}  {'─'*12}  {'─'*6}  {'─'*8}")
    for r in sorted(results, key=lambda x: x["_idx"]):
        print(
            f"  {r['_idx']:>2}  {r['ttft_ms']:>10.1f}  "
            f"{r['total_latency_ms']:>12.1f}  {r['generated_tokens']:>6}  "
            f"{r.get('finish_reason', '?'):>8}"
        )

    print(f"\n{'─'*60}")
    print(f"  Summary")
    print(f"{'─'*60}")
    print(f"  Phase 2 actual wall time:          {phase2_wall_ms:>10.1f} ms")
    if phase1_expected_ms:
        print(f"  Phase 1 expected wall time:        {phase1_expected_ms:>10.1f} ms  "
              f"({N_CONCURRENT_REQUESTS} × {phase1_mean_ms:.1f} ms)")
        print(f"  Speedup ratio:                     {speedup_ratio:>10.2f}×"
              + (" ✓ faster" if speedup_ratio > 1.0 else " ✗ no improvement"))
    else:
        print("  Phase 1 baseline: not available")
    print(f"  Total tokens generated:            {generated_tokens_total:>10}")
    print(f"  Aggregate throughput:              {aggregate_tps:>10.1f} tokens/s")
    print(f"  TTFT mean:                         {sum(ttft_values)/len(ttft_values):>10.1f} ms")
    print(f"  TTFT min / max:                    {min(ttft_values):>6.1f} / {max(ttft_values):.1f} ms")

    sched = metrics.get("scheduler", {})
    bsot: List = sched.get("batch_size_over_time", [])
    step_lats: List = sched.get("scheduler_step_latency_ms", [])
    print(f"\n  Scheduler steps recorded:          {len(step_lats):>10}")
    if step_lats:
        import statistics
        print(f"  Scheduler step latency mean:       {statistics.mean(step_lats):>10.2f} ms")
        print(f"  Scheduler step latency p99:        "
              f"{sorted(step_lats)[int(0.99 * len(step_lats))]:>10.2f} ms")
    if bsot:
        max_bs = max(bs for _, bs in bsot)
        print(f"  Max observed batch size:           {max_bs:>10}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot_batch_size(bsot)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "phase": 2,
        "n_requests": N_CONCURRENT_REQUESTS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "phase2_wall_time_ms": phase2_wall_ms,
        "phase1_expected_wall_time_ms": phase1_expected_ms,
        "speedup_ratio": speedup_ratio,
        "aggregate_tokens_per_second": aggregate_tps,
        "total_tokens_generated": generated_tokens_total,
        "ttft_ms": {
            "mean": sum(ttft_values) / len(ttft_values),
            "min": min(ttft_values),
            "max": max(ttft_values),
            "values": ttft_values,
        },
        "total_latency_ms": {
            "mean": sum(total_latencies) / len(total_latencies),
            "min": min(total_latencies),
            "max": max(total_latencies),
            "values": total_latencies,
        },
        "scheduler_metrics": metrics.get("scheduler", {}),
        "per_sequence_stats": metrics.get("sequences", []),
        "summary_from_server": metrics.get("summary", {}),
        "per_request_results": [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in results
        ],
    }

    with open(OUTPUT_PATH, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n[Output] Saved to {OUTPUT_PATH}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
