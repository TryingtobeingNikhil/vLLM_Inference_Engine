/**
 * gpuBenchmarks.ts — Where PageServe v3 has been tested, and what it measured.
 *
 * ┌─ Adding Colab results (one-file edit) ─────────────────────────────────────┐
 * │ 1. Run colab/PageServe_Colab.ipynb on a T4 / L4 / A100 runtime.            │
 * │ 2. Open results/offline_<gpu>_<time>.json: paste its `runs` array into     │
 * │    that platform's `offline.runs` (and `system` into `system`).            │
 * │ 3. Open results/serving_<gpu>_<time>.json: paste its `configs` object into │
 * │    `serving.configs`, and set `serving.workload` to the --workload used.   │
 * │ 4. Set `status: 'verified'` and `date`.                                    │
 * │ The JSON can be pasted as-is: field names below mirror what                │
 * │ benchmarks/bench_offline.py and benchmarks/bench_serving.py write, and    │
 * │ extra keys are allowed. Python's `Infinity` (the "inf" rate) is valid JS. │
 * └────────────────────────────────────────────────────────────────────────────┘
 *
 * Rules: never invent numbers. A platform without a metric block renders
 * placeholders; a missing field inside a block renders "–".
 */

type Extra = { [key: string]: unknown };

/** `pcts()` in benchmarks/common.py */
export interface Percentiles extends Extra {
  p50: number;
  p90?: number;
  p95?: number;
  p99: number;
  mean?: number;
}

/** `GPUMonitor.summary()` in load_test/gpu_monitor.py */
export interface GpuMonitorSummary extends Extra {
  available: boolean;
  gpu_name?: string;
  util_mean_pct?: number;
  util_max_pct?: number;
  mem_used_max_mb?: number;
  mem_total_mb?: number;
}

/** `runs[].metrics` in results/offline_*.json (bench_offline.py `_summarise` / `run_engine`) */
export interface OfflineMetrics extends Extra {
  num_requests?: number;
  wall_s?: number;
  output_tok_s: number;
  total_tok_s?: number;
  req_s?: number;
  ttft_ms?: Percentiles;
  tpot_ms?: Percentiles;
  itl_ms?: Percentiles;
  e2e_ms?: Percentiles;
  peak_gpu_mem_mb?: number | null;
  gpu?: GpuMonitorSummary;
  prefix_cache_hit_rate?: number;
  spec_acceptance_rate?: number | null;
  preemptions?: number;
  num_failed?: number;
  error?: string;
}

/** One entry of `runs` in results/offline_*.json */
export interface OfflineRun extends Extra {
  suite: string;     // 'batching' | 'prefix' | 'spec' | …
  workload: string;  // 'random' | 'shared_prefix' | 'repetitive' | …
  system: string;    // 'hf_sequential' | 'hf_static_batch' | 'engine_no_batching' | 'engine' | 'engine+prefix' | 'engine+ngram' | 'engine+draft'
  metrics: OfflineMetrics;
}

/** `configs[name].levels[].report` in results/serving_*.json (load_test/report.py `build_report`) */
export interface ServingReport extends Extra {
  total_requests?: number;
  failed?: number;
  throughput_requests_per_sec: number;
  throughput_tokens_per_sec: number;
  goodput: Extra & { pct: number; slo_ttft_ms?: number; slo_tpot_ms?: number };
  latency: Extra & {
    ttft_ms: Percentiles;
    tpot_ms: Percentiles;
    itl_ms: Percentiles;
    total_latency_ms: Percentiles; // E2E
  };
  gpu?: GpuMonitorSummary;
}

export interface ServingLevel extends Extra {
  label: string;               // '2/s', 'inf', 'c=32'
  rate?: number | null;        // Poisson rate (req/s); Infinity = all at once
  concurrency?: number | null; // closed-loop level
  report: ServingReport;
}

export interface ServingConfig extends Extra {
  levels: ServingLevel[];
  error?: string;
}

/** Top-level `system` block written by both scripts (benchmarks/common.py `system_info`) */
export interface SystemInfo extends Extra {
  torch?: string;
  transformers?: string;
  device?: string;
  gpu?: string;
  gpu_memory_gb?: number;
  cuda?: string;
  git_commit?: string;
  model?: string;
}

export interface PlatformEntry {
  id: string;
  status: 'verified' | 'pending';
  gpu: string;
  device: 'cpu' | 'mps' | 'cuda';
  where: 'Local' | 'Colab';
  model: string;
  dtype: string;
  date: string | null;
  /** One line shown on the card. */
  summary: string;
  /** What was verified (correctness checks, test counts…). */
  checks?: string[];
  /** Small local runs — never presented as headline GPU results. */
  smokeRun?: boolean;
  system?: SystemInfo;
  offline?: {
    runs: OfflineRun[];
    /** Workload description per `suite/workload` key, shown above each table. */
    workloads?: Record<string, string>;
    /** Extra caveats per `suite/workload` key. */
    notes?: Record<string, string>;
  };
  serving?: {
    workload?: string;
    configs: Record<string, ServingConfig>;
  };
}

export const SOFTWARE_TESTED = [
  { transformers: '4.51', torch: '2.6' },
  { transformers: '5.12', torch: '2.12' },
];

export const PLATFORMS: PlatformEntry[] = [
  {
    id: 'm2-cpu',
    status: 'verified',
    gpu: 'Apple M2',
    device: 'cpu',
    where: 'Local',
    model: 'Tiny test models',
    dtype: 'fp64',
    date: '2026-09-27', // v3 engine commit
    summary: '141 tests pass. Output is token-for-token identical to HF generate() under every feature combination.',
    checks: [
      '141 tests passing',
      'Token-for-token match vs HF generate()',
      'Chunked prefill · batching · prefix caching',
      'Swap + recompute preemption',
      'n-gram + draft speculation',
    ],
  },
  {
    id: 'm2-mps',
    status: 'verified',
    gpu: 'Apple M2',
    device: 'mps',
    where: 'Local',
    model: 'Qwen/Qwen2-0.5B',
    dtype: 'fp16',
    date: '2026-09-27', // v3 engine commit
    summary: 'Runs on the Apple GPU. Small local smoke runs below — not headline GPU results.',
    checks: ['Real-model runs on MPS', 'Offline ablation smoke run'],
    smokeRun: true,
    offline: {
      workloads: {
        'concurrency/concurrent': '8 concurrent requests × 64 output tokens',
        'batching/random': '16 requests · prompts 32–64 tokens · outputs 16–32 tokens',
        'prefix/shared_prefix': 'Shared-prompt workload',
        'spec/single_request': 'n-gram speculation · one request running',
      },
      notes: {
        'spec/single_request':
          'At batch size 8 there is no gain (188.5 vs 192.6 tok/s): speculation helps most when few requests are running.',
      },
      runs: [
        { suite: 'concurrency', workload: 'concurrent', system: 'hf_sequential', metrics: { num_requests: 8, output_tok_s: 44.1 } },
        { suite: 'concurrency', workload: 'concurrent', system: 'engine', metrics: { num_requests: 8, output_tok_s: 239.0 } },

        { suite: 'batching', workload: 'random', system: 'hf_sequential', metrics: { num_requests: 16, output_tok_s: 27.3 } },
        { suite: 'batching', workload: 'random', system: 'hf_static_batch', metrics: { num_requests: 16, output_tok_s: 30.3 } },
        { suite: 'batching', workload: 'random', system: 'engine_no_batching', metrics: { num_requests: 16, output_tok_s: 31.6 } },
        { suite: 'batching', workload: 'random', system: 'engine', metrics: { num_requests: 16, output_tok_s: 99.1 } },

        { suite: 'prefix', workload: 'shared_prefix', system: 'engine', metrics: { output_tok_s: 69.4 } },
        { suite: 'prefix', workload: 'shared_prefix', system: 'engine+prefix', metrics: { output_tok_s: 139.1, prefix_cache_hit_rate: 0.6 } },

        { suite: 'spec', workload: 'single_request', system: 'engine', metrics: { num_requests: 1, output_tok_s: 38.7 } },
        {
          suite: 'spec', workload: 'single_request', system: 'engine+ngram',
          metrics: { num_requests: 1, output_tok_s: 113.1, spec_acceptance_rate: 0.88, tokens_per_step: 4.5 },
        },
      ],
    },
  },
  {
    id: 't4',
    status: 'pending',
    gpu: 'NVIDIA T4',
    device: 'cuda',
    where: 'Colab',
    model: 'Qwen/Qwen2.5-1.5B-Instruct',
    dtype: 'fp16',
    date: null,
    summary: 'Benchmarks coming. T4 has no bf16, so it runs fp16.',
  },
  {
    id: 'l4',
    status: 'pending',
    gpu: 'NVIDIA L4',
    device: 'cuda',
    where: 'Colab',
    model: 'Qwen/Qwen2.5-3B-Instruct',
    dtype: 'bf16',
    date: null,
    summary: 'Benchmarks coming.',
  },
  {
    id: 'a100',
    status: 'pending',
    gpu: 'NVIDIA A100',
    device: 'cuda',
    where: 'Colab',
    model: 'Qwen/Qwen2.5-7B-Instruct',
    dtype: 'bf16',
    date: null,
    summary: 'Benchmarks coming.',
  },
];

// ── Helpers used by the UI ────────────────────────────────────────────────────

export const SYSTEM_LABELS: Record<string, string> = {
  hf_sequential: 'HF sequential',
  hf_static_batch: 'HF static batching',
  engine_no_batching: 'Engine, no batching',
  engine: 'Continuous batching',
  'engine+prefix': '+ prefix caching',
  'engine+ngram': '+ n-gram speculation',
  'engine+draft': '+ draft-model speculation',
  sequential: 'Sequential (Phase 1)',
  no_batching: 'Engine, no batching',
  continuous: 'Continuous batching',
  prefix: '+ prefix caching',
  ngram: '+ n-gram speculation',
  draft: '+ draft-model speculation',
};

export const labelFor = (system: string) => SYSTEM_LABELS[system] ?? system;

export interface OfflineRow {
  system: string;
  outputTokS: number | null;
  speedup: number | null;
  ttftP50: number | null;
  ttftP99: number | null;
  tpotP50: number | null;
  itlP99: number | null;
  peakMemMb: number | null;
  gpuUtilPct: number | null;
  notes: string;
}

export interface OfflineGroup {
  key: string;
  suite: string;
  workload: string;
  description?: string;
  note?: string;
  rows: OfflineRow[];
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** Groups runs by suite/workload and derives the same columns as bench_offline.py's markdown. */
export function offlineGroups(entry: PlatformEntry): OfflineGroup[] {
  if (!entry.offline) return [];
  const groups = new Map<string, OfflineGroup>();
  for (const run of entry.offline.runs) {
    const key = `${run.suite}/${run.workload}`;
    if (!groups.has(key)) {
      groups.set(key, {
        key,
        suite: run.suite,
        workload: run.workload,
        description: entry.offline.workloads?.[key],
        note: entry.offline.notes?.[key],
        rows: [],
      });
    }
    const m = run.metrics;
    const notes: string[] = [];
    if (m.prefix_cache_hit_rate) notes.push(`prefix hit ${Math.round(m.prefix_cache_hit_rate * 100)}%`);
    if (typeof m.spec_acceptance_rate === 'number') notes.push(`accept ${Math.round(m.spec_acceptance_rate * 100)}%`);
    if (typeof m.tokens_per_step === 'number') notes.push(`${m.tokens_per_step} tok/step`);
    if (m.preemptions) notes.push(`${m.preemptions} preemptions`);
    if (m.num_failed) notes.push(`${m.num_failed} failed (excluded)`);
    if (m.error) notes.push(m.error);
    groups.get(key)!.rows.push({
      system: run.system,
      outputTokS: m.error ? null : num(m.output_tok_s),
      speedup: null,
      ttftP50: num(m.ttft_ms?.p50),
      ttftP99: num(m.ttft_ms?.p99),
      tpotP50: num(m.tpot_ms?.p50),
      itlP99: num(m.itl_ms?.p99),
      peakMemMb: num(m.peak_gpu_mem_mb),
      gpuUtilPct: m.gpu?.available ? num(m.gpu.util_mean_pct) : null,
      notes: notes.join(', '),
    });
  }
  // Speedup is relative to the first system in each group, as in the script.
  for (const g of Array.from(groups.values())) {
    const base = g.rows[0]?.outputTokS;
    for (const r of g.rows) r.speedup = base && r.outputTokS !== null ? r.outputTokS / base : null;
  }
  return Array.from(groups.values());
}

export interface ServingRow {
  config: string;
  load: string;
  reqS: number | null;
  outTokS: number | null;
  ttftP50: number | null;
  ttftP99: number | null;
  tpotP50: number | null;
  tpotP99: number | null;
  itlP99: number | null;
  e2eP50: number | null;
  goodputPct: number | null;
  gpuUtilPct: number | null;
  gpuMemMb: number | null;
}

/** Flattens `serving.configs` into one row per config × load level. */
export function servingRows(entry: PlatformEntry): ServingRow[] {
  if (!entry.serving) return [];
  const rows: ServingRow[] = [];
  for (const [config, cfg] of Object.entries(entry.serving.configs)) {
    for (const level of cfg.levels ?? []) {
      const r = level.report;
      const lat = r.latency;
      const g = r.gpu;
      rows.push({
        config,
        load: level.label,
        reqS: num(r.throughput_requests_per_sec),
        outTokS: num(r.throughput_tokens_per_sec),
        ttftP50: num(lat?.ttft_ms?.p50),
        ttftP99: num(lat?.ttft_ms?.p99),
        tpotP50: num(lat?.tpot_ms?.p50),
        tpotP99: num(lat?.tpot_ms?.p99),
        itlP99: num(lat?.itl_ms?.p99),
        e2eP50: num(lat?.total_latency_ms?.p50),
        goodputPct: num(r.goodput?.pct),
        gpuUtilPct: g?.available ? num(g.util_mean_pct) : null,
        gpuMemMb: g?.available ? num(g.mem_used_max_mb) : null,
      });
    }
  }
  return rows;
}

export const platform = (id: string) => PLATFORMS.find((p) => p.id === id);

/** Throughput of `to` vs `from` inside one suite/workload of a platform's offline runs. */
export function compareSystems(platformId: string, groupKey: string, from: string, to: string) {
  const entry = platform(platformId);
  const group = entry ? offlineGroups(entry).find((g) => g.key === groupKey) : undefined;
  const a = group?.rows.find((r) => r.system === from)?.outputTokS ?? null;
  const b = group?.rows.find((r) => r.system === to)?.outputTokS ?? null;
  return { from: a, to: b, ratio: a && b ? b / a : null };
}

/** The four headline comparisons from the local v3 smoke run (Apple M2, MPS). */
export const SMOKE_HIGHLIGHTS = [
  { label: 'vs HF sequential · 8 concurrent requests', ...compareSystems('m2-mps', 'concurrency/concurrent', 'hf_sequential', 'engine') },
  { label: 'continuous batching · offline ablation', ...compareSystems('m2-mps', 'batching/random', 'hf_sequential', 'engine') },
  { label: 'prefix caching · shared-prompt workload', ...compareSystems('m2-mps', 'prefix/shared_prefix', 'engine', 'engine+prefix') },
  { label: 'n-gram speculation · one request', ...compareSystems('m2-mps', 'spec/single_request', 'engine', 'engine+ngram') },
];
