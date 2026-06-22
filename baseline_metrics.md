# Phase 1 — Sequential Serving Baseline Metrics

Captured on Apple M2, Qwen/Qwen2-0.5B, float16, MPS backend.

## Baseline Numbers

| Metric | Baseline Value |
|---|---|
| Steady-state TTFT | ~16 ms |
| Total latency (50 tokens) | ~1180 ms |
| Throughput | ~42 tok/s |
| Model memory | 950.5 MB |
| Concurrency model | 1 request at a time |

## What This Baseline Hides

The validation script sends requests **sequentially from a single client**.
It does not surface what happens under real concurrency:

| Concurrent users | User N waits... |
|---|---|
| 5 | ~6 s |
| 10 | ~12 s |
| 20 | ~24 s |

The `asyncio.Lock` serialises correctly — but that means every waiting
request adds one full `total_latency_ms` to the queue. This will be
surfaced properly in **Phase 10: Load Testing**.

## Phase Comparison Table

| Phase | Description | TTFT | Total Latency | Throughput | Notes |
|---|---|---|---|---|---|
| **1** | Sequential baseline | ~16 ms | ~1180 ms | ~42 tok/s | ← you are here |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| 6 | | | | | |
| 7 | | | | | |
| 8 | | | | | |
| 9 | | | | | |
| 10 | Load testing | | | | Concurrency surfaced |
