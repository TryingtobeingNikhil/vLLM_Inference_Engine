#!/usr/bin/env python3
"""
run_validation.py — Sends 5 sequential requests to the inference server
and produces a summary table + baseline_metrics.json.

Wall-clock timestamps are recorded per request to verify that requests
are truly sequential: request N+1 must start only after request N finishes.

Usage
-----
# Terminal 1 — start the server
    cd inference_engine
    uvicorn server.app:app --host 0.0.0.0 --port 8000

# Terminal 2 — run validation
    cd inference_engine
    python ../run_validation.py [--host HOST] [--port PORT] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import List

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required.  Run: pip install httpx")
    sys.exit(1)


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_OUTPUT = "baseline_metrics.json"
REQUEST_TIMEOUT = 300.0  # seconds — generous for slow CPU inference

# ~100-token prompt (measured with a typical BPE tokenizer).
VALIDATION_PROMPT = (
    "In the field of machine learning, the transformer architecture has revolutionized "
    "natural language processing. Originally introduced in the paper 'Attention Is All "
    "You Need' by Vaswani et al., transformers rely on self-attention mechanisms to model "
    "long-range dependencies in sequences without recurrence. This has enabled the "
    "development of large language models such as GPT, BERT, and their successors, which "
    "have achieved state-of-the-art performance on a wide range of benchmarks. Describe "
    "the key components of the transformer architecture."
)

NUM_REQUESTS = 5
MAX_NEW_TOKENS = 50


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class RequestRecord:
    request_index: int       # 1-based
    wall_start: float        # time.time() before sending request
    wall_end: float          # time.time() after response received
    ttft_ms: float
    total_latency_ms: float
    tokens_per_second: float
    prompt_tokens: int
    generated_tokens: int
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float
    generated_text: str


# ── Main validation logic ─────────────────────────────────────────────────────


def wait_for_server(base_url: str, timeout: float = 60.0) -> None:
    """Poll /health until the server responds 200 or timeout expires."""
    deadline = time.time() + timeout
    print(f"Waiting for server at {base_url}/health ...", end="", flush=True)
    with httpx.Client(timeout=5.0) as client:
        while time.time() < deadline:
            try:
                resp = client.get(f"{base_url}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    print(f" ready! (model={data.get('model')}, device={data.get('device')})")
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            print(".", end="", flush=True)
            time.sleep(2)
    print()
    print("ERROR: Server did not become ready within timeout.")
    sys.exit(1)


def send_request(client: httpx.Client, base_url: str, idx: int) -> RequestRecord:
    """Send one /generate request and return a RequestRecord with timing."""
    payload = {"prompt": VALIDATION_PROMPT, "max_new_tokens": MAX_NEW_TOKENS}

    print(f"\n[Request {idx}/{NUM_REQUESTS}] sending ...", end="", flush=True)

    wall_start = time.time()
    resp = client.post(f"{base_url}/generate", json=payload, timeout=REQUEST_TIMEOUT)
    wall_end = time.time()

    if resp.status_code != 200:
        print(f" ERROR {resp.status_code}: {resp.text}")
        sys.exit(1)

    data = resp.json()
    print(
        f" done in {(wall_end - wall_start) * 1000:.0f} ms "
        f"(TTFT={data['ttft_ms']:.1f} ms, {data['tokens_per_second']:.1f} tok/s)"
    )

    return RequestRecord(
        request_index=idx,
        wall_start=wall_start,
        wall_end=wall_end,
        ttft_ms=data["ttft_ms"],
        total_latency_ms=data["total_latency_ms"],
        tokens_per_second=data["tokens_per_second"],
        prompt_tokens=data["prompt_tokens"],
        generated_tokens=data["generated_tokens"],
        gpu_memory_allocated_mb=data["gpu_memory_allocated_mb"],
        gpu_memory_reserved_mb=data["gpu_memory_reserved_mb"],
        generated_text=data["generated_text"],
    )


def verify_sequential(records: List[RequestRecord]) -> bool:
    """
    Assert that each request started AFTER the previous one finished.

    A small grace of 10 ms is allowed for networking / scheduling jitter.
    Returns True if all checks pass, False otherwise.
    """
    GRACE_S = 0.010  # 10 ms
    passed = True
    for i in range(1, len(records)):
        prev = records[i - 1]
        curr = records[i]
        gap = curr.wall_start - prev.wall_end
        ok = gap >= -GRACE_S
        status = "✓" if ok else "✗"
        print(
            f"  Seq check {prev.request_index}→{curr.request_index}: "
            f"gap={gap * 1000:+.1f} ms  {status}"
        )
        if not ok:
            passed = False
    return passed


def print_summary_table(records: List[RequestRecord]) -> None:
    """Print a formatted summary table to stdout."""
    col_w = [4, 13, 14, 12, 15, 15, 14, 14]
    headers = ["Req", "TTFT (ms)", "Total (ms)", "tok/s", "Prompt tok", "Gen tok", "Alloc MB", "Resv MB"]

    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"

    print("\n" + "=" * 80)
    print("  SEQUENTIAL SERVING BASELINE — VALIDATION RESULTS")
    print("=" * 80)
    print(sep)
    print(fmt.format(*headers))
    print(sep)

    for r in records:
        print(fmt.format(
            str(r.request_index),
            f"{r.ttft_ms:.1f}",
            f"{r.total_latency_ms:.1f}",
            f"{r.tokens_per_second:.2f}",
            str(r.prompt_tokens),
            str(r.generated_tokens),
            f"{r.gpu_memory_allocated_mb:.1f}",
            f"{r.gpu_memory_reserved_mb:.1f}",
        ))

    print(sep)

    # Summary row
    n = len(records)
    avg_ttft = sum(r.ttft_ms for r in records) / n
    avg_total = sum(r.total_latency_ms for r in records) / n
    avg_tps = sum(r.tokens_per_second for r in records) / n

    print(fmt.format(
        "avg",
        f"{avg_ttft:.1f}",
        f"{avg_total:.1f}",
        f"{avg_tps:.2f}",
        "-", "-", "-", "-",
    ))
    print(sep)
    print()


def save_results(records: List[RequestRecord], output_path: str) -> None:
    """Save raw results and wall-clock log to JSON."""
    wall_log = []
    for i in range(len(records) - 1):
        gap_ms = (records[i + 1].wall_start - records[i].wall_end) * 1000
        wall_log.append({
            "from_request": records[i].request_index,
            "to_request": records[i + 1].request_index,
            "gap_between_ms": round(gap_ms, 3),
            "sequential": gap_ms >= -10,  # 10 ms grace
        })

    payload = {
        "validation_config": {
            "num_requests": NUM_REQUESTS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "prompt_length_chars": len(VALIDATION_PROMPT),
        },
        "wall_clock_log": wall_log,
        "results": [
            {
                "request_index": r.request_index,
                "wall_start": r.wall_start,
                "wall_end": r.wall_end,
                "wall_duration_ms": (r.wall_end - r.wall_start) * 1000,
                "ttft_ms": r.ttft_ms,
                "total_latency_ms": r.total_latency_ms,
                "tokens_per_second": r.tokens_per_second,
                "prompt_tokens": r.prompt_tokens,
                "generated_tokens": r.generated_tokens,
                "gpu_memory_allocated_mb": r.gpu_memory_allocated_mb,
                "gpu_memory_reserved_mb": r.gpu_memory_reserved_mb,
                "generated_text": r.generated_text,
            }
            for r in records
        ],
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Results saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential inference validation script")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    wait_for_server(base_url)

    records: List[RequestRecord] = []
    with httpx.Client() as client:
        for i in range(1, NUM_REQUESTS + 1):
            record = send_request(client, base_url, i)
            records.append(record)

    print_summary_table(records)

    print("Sequential ordering verification:")
    passed = verify_sequential(records)
    if passed:
        print("\n  ✓ PASS — all requests were served sequentially\n")
    else:
        print("\n  ✗ FAIL — overlap detected; the inference lock may not be working\n")

    save_results(records, args.output)


if __name__ == "__main__":
    main()
