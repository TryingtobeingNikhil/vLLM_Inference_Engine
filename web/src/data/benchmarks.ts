/**
 * benchmarks.ts — Real measured numbers from PageServe benchmark runs.
 *
 * Sources:
 *   bench_direct_results.json   — single-request + 4-way concurrent
 *   bench_phases_results.json   — phase-by-phase comparison
 *   baseline_metrics.json       — sequential baseline per-request trace
 *   benchmark_comparison.md     — Before/After PR comparison
 *   baseline_metrics.md         — narrative summary
 *
 * Hardware: Apple M2 (MPS), Qwen/Qwen2-0.5B, float16
 */

// ── Single request (bench_direct_results.json) ──────────────────────────────

export const SINGLE_REQUEST = {
  ttft_ms: 16.52,
  total_latency_ms: 514.4,
  tps: 97.25,
} as const;

// ── 4-way concurrent (bench_direct_results.json) ─────────────────────────────

export const CONCURRENT = {
  n_requests: 4,
  max_tokens: 50,
  wall_ms: 3990.4,
  /** Extrapolated: 4 × Phase 1 avg 1418ms per request TTFT wait */
  phase1_expected_ms: 5673.6,
  speedup: 1.42,
  aggregate_tps: 50.1,
  total_tokens: 200,
  ttft: {
    mean_ms: 52.9,
    min_ms: 21.9,
    max_ms: 83.7,
  },
  total_latency_mean_ms: 541.4,
} as const;

// ── Phase-level results (bench_phases_results.json) ──────────────────────────

export const PHASE1 = {
  ttft_ms: 168.7,
  total_ms: 668.1,
  tps: 83.2,
} as const;

export const PHASE2 = {
  wall_ms: 4017.4,
  speedup: 1.41,
  agg_tps: 49.8,
  ttft_ms: 52.6,
  total_ms: 555.2,
} as const;

export const CHUNKED_PREFILL = {
  full_prefill_ms: 415.1,
  first_chunk_ms: 306.6,
  decode_step_ms: 11.1,
  prompt_len: 402,
  n_chunks: 4,
  delay_saved_ms: 108.5,
} as const;

export const PHASE8_DECODE = {
  live_ms_per_step: 10.0,
  live_tps: 99.6,
  reconstruct_ms_per_step: 0.66,
  live_tps_before: 85.9,
} as const;

// ── Sequential baseline per-request trace (baseline_metrics.json) ────────────

export const SEQUENTIAL_BASELINE = [
  {
    request_index: 1,
    wall_duration_ms: 2384.7,
    ttft_ms: 1209.6,
    total_latency_ms: 2357.0,
    tps: 21.2,
    gpu_memory_allocated_mb: 950.5,
  },
  {
    request_index: 2,
    wall_duration_ms: 1182.5,
    ttft_ms: 16.6,
    total_latency_ms: 1181.0,
    tps: 42.3,
    gpu_memory_allocated_mb: 950.5,
  },
  {
    request_index: 3,
    wall_duration_ms: 1197.1,
    ttft_ms: 15.8,
    total_latency_ms: 1195.5,
    tps: 41.8,
    gpu_memory_allocated_mb: 950.5,
  },
  {
    request_index: 4,
    wall_duration_ms: 1180.6,
    ttft_ms: 16.1,
    total_latency_ms: 1179.1,
    tps: 42.4,
    gpu_memory_allocated_mb: 950.5,
  },
  {
    request_index: 5,
    wall_duration_ms: 1181.3,
    ttft_ms: 16.1,
    total_latency_ms: 1179.7,
    tps: 42.4,
    gpu_memory_allocated_mb: 950.5,
  },
] as const;

// ── Before/After PR comparison (benchmark_comparison.md) ─────────────────────

export const PR_COMPARISON = {
  tests: { total: 84, passing: 84 },
  single_req: {
    ttft_before_ms: 19.8,
    ttft_after_ms: 16.5,
    ttft_delta_pct: -17,
    latency_before_ms: 620.1,
    latency_after_ms: 514.4,
    latency_delta_pct: -17,
    tps_before: 80.8,
    tps_after: 97.3,
    tps_delta_pct: 20,
  },
  concurrent_4way: {
    wall_before_ms: 4285,
    wall_after_ms: 3990,
    speedup_before: 1.32,
    speedup_after: 1.42,
    agg_tps_before: 46.7,
    agg_tps_after: 50.1,
  },
} as const;

// ── Engine configuration constants (inference_engine/config.py) ──────────────

export const ENGINE_CONFIG = {
  model_name: 'Qwen/Qwen2-0.5B',
  max_batch_size: 4,
  kv_block_size: 16,
  kv_num_blocks: 256,
  kv_num_cpu_blocks: 128,
  prefill_budget_tokens: 512,
  decode_batch_limit: 8,
  prefill_chunk_size: 128,
  request_timeout_ms: 30_000,
  scheduler_poll_interval_ms: 1.0,
} as const;

// ── TTFT comparison — the headline numbers ────────────────────────────────────

/** TTFT experienced by a late-arriving request under 4-way load in Phase 1 */
export const TTFT_SEQUENTIAL_UNDER_LOAD_MS = 1418.0;
/** TTFT under same load with continuous batching */
export const TTFT_BATCHED_UNDER_LOAD_MS = 20.9;
export const TTFT_IMPROVEMENT_PCT = Math.round(
  ((TTFT_SEQUENTIAL_UNDER_LOAD_MS - TTFT_BATCHED_UNDER_LOAD_MS) /
    TTFT_SEQUENTIAL_UNDER_LOAD_MS) *
    100,
);
