/**
 * phases.ts — 11-phase build log content.
 * Derived from README.md "Development Phases" section and benchmark data.
 */

export interface PhaseEntry {
  phase: number;
  title: string;
  file: string;
  problem: string;
  solution: string;
  metricBefore?: string;
  metricAfter?: string;
  metricLabel?: string;
  tag: 'scheduling' | 'memory' | 'observability' | 'testing';
}

export const PHASES: PhaseEntry[] = [
  {
    phase: 1,
    title: 'Sequential Serving Baseline',
    file: 'engine/sequential.py',
    problem: 'One request at a time. asyncio.Lock serializes all generation. New arrivals block until prior request fully completes.',
    solution: 'Establishes the ground truth baseline: a clean single-request HuggingFace generate() wrapper with per-token timing.',
    metricLabel: 'TTFT (warm)',
    metricAfter: '~32 ms',
    tag: 'scheduling',
  },
  {
    phase: 2,
    title: 'Continuous Batching Scheduler',
    file: 'engine/scheduler.py',
    problem: 'Head-of-line blocking: a 500-token request starves a 5-token request for its entire duration.',
    solution: 'Background asyncio task runs a scheduler loop. Each iteration advances all active sequences by one decode token. No request monopolizes the GPU.',
    metricLabel: 'TTFT under 4-way load',
    metricBefore: '1,418 ms',
    metricAfter: '20.9 ms',
    tag: 'scheduling',
  },
  {
    phase: 3,
    title: 'Request Queue',
    file: 'engine/request_queue.py',
    problem: 'Concurrent clients overwhelm the scheduler — connections drop or race.',
    solution: 'FIFO RequestQueue with configurable maxsize and per-request timeout. Returns HTTP 429 when full, TimeoutError on expiry.',
    metricAfter: 'maxsize: 32 · timeout: 30s',
    tag: 'scheduling',
  },
  {
    phase: 4,
    title: 'Prefill / Decode Separation',
    file: 'engine/scheduler.py',
    problem: 'Large prompts (prefill) block all decode steps — Time-to-First-Token spikes for every queued request.',
    solution: 'Independent budgets: ≤512 prefill tokens and ≤8 decode sequences per scheduler step. Chunked prefill splits long prompts across multiple iterations.',
    metricLabel: 'Decode stall saved (402-token prompt)',
    metricBefore: '415 ms (blocking)',
    metricAfter: '307 ms (first chunk)',
    tag: 'scheduling',
  },
  {
    phase: 5,
    title: 'KV Cache Memory Tracking',
    file: 'engine/kv_cache_tracker.py',
    problem: 'No visibility into how much GPU memory KV caches consume during execution.',
    solution: 'KVCacheTracker monitors physical KV memory footprint per sequence. Feeds into block eviction decisions.',
    tag: 'memory',
  },
  {
    phase: 6,
    title: 'Block Allocator',
    file: 'engine/block_allocator.py',
    problem: 'Standard Transformers pre-allocate max_sequence_length tensors per request — up to 60-80% wasted GPU memory.',
    solution: 'Thread-safe BlockAllocator maps each sequence\'s tokens into logical blocks of 16 slots. allocate() / write_token() / free() interface. Eliminates external fragmentation.',
    metricAfter: 'block_size: 16 tokens · pool: 256 blocks',
    tag: 'memory',
  },
  {
    phase: 7,
    title: 'Paged KV Cache',
    file: 'engine/paged_kv_cache.py',
    problem: 'Logical block IDs have no physical storage — KV tensors still live as contiguous per-sequence arrays.',
    solution: 'PagedKVCacheManager maps logical block IDs to pre-allocated physical GPU tensor slots. write_kv() / read_kv() per token position. Enables eviction and swap.',
    tag: 'memory',
  },
  {
    phase: 8,
    title: 'Batch Attention Integration',
    file: 'engine/attention_wrapper.py',
    problem: 'Each decode step rebuilds the full KV cache from the paged pool — expensive Metal dispatch overhead on MPS.',
    solution: 'Keep HuggingFace past_key_values live on the sequence during decode (skip pool read). Write new token into pool each step for eviction accuracy.',
    metricLabel: 'Decode step throughput',
    metricBefore: '85.9 tok/s (pool-reconstruct)',
    metricAfter: '99.6 tok/s (live KV)',
    tag: 'memory',
  },
  {
    phase: 9,
    title: 'CPU Staging Pool & Swapping',
    file: 'engine/cpu_swap_manager.py',
    problem: 'GPU block exhaustion causes OOM crashes or silent request drops under burst load.',
    solution: 'CPUSwapManager copies KV tensors of the largest preempted sequence to host RAM, freeing GPU blocks. Sequence resumes after blocks are available. 0 HTTP 500s.',
    metricAfter: 'CPU pool: 128 blocks · 0 OOM crashes',
    tag: 'memory',
  },
  {
    phase: 10,
    title: 'Unified Metrics Aggregation',
    file: 'engine/metrics_aggregator.py',
    problem: 'No unified telemetry surface — throughput, latency, SLO, and cache stats are scattered across components.',
    solution: 'MetricsAggregator consolidates: P50/P95/P99 TTFT, aggregate tok/s, queue depth, block utilization, swap counts, and SLO compliance into a single /metrics endpoint.',
    metricAfter: 'P50 TTFT · P95 TTFT · P99 TTFT · SLO %',
    tag: 'observability',
  },
  {
    phase: 11,
    title: 'Load Testing Tool',
    file: 'run_load_test.py',
    problem: 'No way to characterize engine behavior under realistic traffic patterns.',
    solution: 'CLI load tester with three profiles: constant (fixed RPS), ramp (linear ramp from start-rps to end-rps), burst (spike). Outputs JSON for cross-phase comparison.',
    metricAfter: 'constant · ramp · burst profiles',
    tag: 'testing',
  },
];
