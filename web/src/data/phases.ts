/**
 * phases.ts — 12-phase build log content.
 * Derived from README.md "Development phases" and "How it works" (v3).
 * Metrics tagged "v2, M2" are legacy measurements; "M2 smoke" are small local
 * v3 runs (see gpuBenchmarks.ts).
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
  isNew?: boolean;
}

export const PHASES: PhaseEntry[] = [
  {
    phase: 1,
    title: 'Sequential Serving Baseline',
    file: 'engine/sequential.py',
    problem: 'One request at a time. A lock serializes all generation, so new arrivals block until the previous request fully completes.',
    solution: 'The ground-truth baseline: a naive prefill → decode loop around HuggingFace, with streaming. Its TTFT now includes the time spent waiting for the lock.',
    metricLabel: 'Serves on',
    metricAfter: ':8000 · app.py',
    tag: 'scheduling',
  },
  {
    phase: 2,
    title: 'Continuous Batching Scheduler',
    file: 'engine/scheduler.py',
    problem: 'Head-of-line blocking: a 500-token request starves a 5-token request for its entire duration.',
    solution: 'A background loop does iteration-level scheduling: plan → execute → apply. Requests join and leave the running batch between steps.',
    metricLabel: 'TTFT under 4-way load (v2, M2)',
    metricBefore: '1,418 ms',
    metricAfter: '20.9 ms',
    tag: 'scheduling',
  },
  {
    phase: 3,
    title: 'Request Queue',
    file: 'engine/request_queue.py',
    problem: 'Concurrent clients overwhelm the scheduler: connections drop or race.',
    solution: 'FIFO RequestQueue with timeouts, cancellation and backpressure. Returns 503 when the queue is full and 504 when a request waits too long.',
    metricAfter: 'queue: 16 × batch · timeout: 60 s',
    tag: 'scheduling',
  },
  {
    phase: 4,
    title: 'Prefill / Decode Separation',
    file: 'engine/scheduler.py',
    problem: 'Long prompts stall every running decode, so time-to-first-token spikes for everyone.',
    solution: 'A per-step prefill token budget plus chunked prefill: long prompts are split across steps so they cannot stall running decodes.',
    metricLabel: 'Budget / chunk',
    metricAfter: '2048 / 512 tok (CUDA) · 512 / 128 (MPS, CPU)',
    tag: 'scheduling',
  },
  {
    phase: 5,
    title: 'KV Memory Tracking',
    file: 'engine/kv_cache_config.py',
    problem: 'No visibility into how much memory KV caches really need.',
    solution: 'Bytes-per-token sizing from the model config (with the correct head_dim per family) and logical KV accounting per sequence.',
    tag: 'memory',
  },
  {
    phase: 6,
    title: 'Block Allocator',
    file: 'engine/block_allocator.py',
    problem: 'Reserving max_length of KV per sequence leaves most of it unused, so few sequences fit.',
    solution: 'Fixed 16-token blocks and per-sequence block tables, with ref counts and content hashes so full blocks can be shared.',
    metricAfter: 'block_size: 16 tokens · ref-counted',
    tag: 'memory',
  },
  {
    phase: 7,
    title: 'Paged KV Cache',
    file: 'engine/paged_kv_cache.py',
    problem: 'Logical blocks need physical storage that every layer can address.',
    solution: 'One pre-allocated device pool shaped [layers, blocks, 16, H, D]. On CUDA it is sized automatically by profiling free GPU memory.',
    metricAfter: 'KV_NUM_BLOCKS=auto',
    tag: 'memory',
  },
  {
    phase: 8,
    title: 'Paged Attention Integration',
    file: 'engine/attention_wrapper.py',
    problem: 'In v2 each sequence kept its own HF cache and ran its own forward pass. The pool was only a shadow copy, so there was no real GPU batching.',
    solution: 'A custom attention backend registered with HuggingFace writes new K/V into the pool and attends through block tables. Every step is one packed [1, T] forward pass mixing prefill chunks and decode tokens.',
    metricLabel: 'Continuous batching vs HF sequential (M2 smoke)',
    metricBefore: '27.3 tok/s',
    metricAfter: '99.1 tok/s · 3.6×',
    tag: 'memory',
  },
  {
    phase: 9,
    title: 'Preemption: Swap or Recompute',
    file: 'engine/cpu_swap_manager.py',
    problem: 'When the pool is full, a running sequence can’t grow. Crashing or dropping requests is not acceptable.',
    solution: 'FCFS admission, LIFO preemption: the most recently arrived sequence is preempted. If it is decoding it is swapped to pinned CPU memory, otherwise its blocks are dropped and it re-prefills later.',
    metricAfter: 'PREEMPTION_MODE=swap → falls back to recompute',
    tag: 'memory',
  },
  {
    phase: 10,
    title: 'Unified Metrics',
    file: 'engine/metrics_aggregator.py',
    problem: 'v2 metrics flattered the engine: TTFT excluded queueing, and "per-token latency" was forward time, not inter-token latency.',
    solution: 'One /metrics surface: TTFT (including queueing), TPOT, real ITL, E2E p50–p99, prefix hit rate, speculation acceptance, step stats, KV utilisation and SLO compliance.',
    metricAfter: 'TTFT · TPOT · ITL · E2E · p50–p99',
    tag: 'observability',
  },
  {
    phase: 11,
    title: 'Load Testing',
    file: 'run_load_test.py',
    problem: 'Measuring after a client-side semaphore hides queueing (coordinated omission).',
    solution: 'Open-loop Poisson or closed-loop load with a streaming client. Latency is measured from the scheduled send time, so a server that falls behind is charged for it.',
    metricAfter: 'Poisson · closed-loop · seeded workloads',
    tag: 'testing',
  },
  {
    phase: 12,
    title: 'Prefix Caching, Speculation, Streaming & GPU Benchmarks',
    file: 'engine/spec_decode.py',
    problem: 'Shared prompts were recomputed, decode stayed bandwidth-bound, and there was no streaming or standard API, or GPU numbers.',
    solution: 'Hash-chained prefix caching; n-gram and draft-model speculative decoding (identical to greedy); SSE streaming and an OpenAI-compatible /v1/completions; an offline + serving benchmark suite, a Colab notebook and a GPU smoke test.',
    metricLabel: 'M2 smoke',
    metricAfter: 'prefix 2.0× · n-gram 2.9× (1 request)',
    tag: 'testing',
    isNew: true,
  },
];

export const NEXT_STEPS = [
  { title: 'Fused paged-attention kernel', desc: 'Triton or FlashInfer instead of gather + matmul, reading the pool in place.' },
  { title: 'CUDA graphs for decode', desc: 'Remove Python and launch overhead at small batch sizes.' },
  { title: 'Rejection sampling', desc: 'So speculative decoding also applies to sampled (non-greedy) requests.' },
  { title: 'Tensor parallelism', desc: 'For models that don’t fit on one GPU.' },
];
