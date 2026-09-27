'use client';

import { useState, type ReactNode } from 'react';
import { SectionHeader, Accent } from '@/components/ui/SectionHeader';
import { Card } from '@/components/ui/Card';
import { Reveal } from '@/components/ui/Reveal';
import {
  PLATFORMS,
  labelFor,
  offlineGroups,
  servingRows,
  type OfflineGroup,
  type PlatformEntry,
  type ServingRow,
  type ServingSweep,
} from '@/data/gpuBenchmarks';
import {
  SINGLE_REQUEST,
  CONCURRENT,
  PHASE1,
  PHASE2,
  CHUNKED_PREFILL,
  PR_COMPARISON,
  TTFT_SEQUENTIAL_UNDER_LOAD_MS,
  TTFT_BATCHED_UNDER_LOAD_MS,
} from '@/data/benchmarks';

// ── Formatting ────────────────────────────────────────────────────────────────

const dash = <span className="text-fg-4">–</span>;
const f1 = (v: number | null) =>
  v === null ? dash : v >= 1000 ? Math.round(v).toLocaleString() : v < 1 ? v.toFixed(2) : v.toFixed(1);
const fx = (v: number | null) => (v === null ? dash : `${v.toFixed(2)}×`);
const fpct = (v: number | null) => (v === null ? dash : `${Math.round(v)}%`);

// ── Generic table ─────────────────────────────────────────────────────────────

function Table({ head, rows, minW = 480 }: { head: string[]; rows: ReactNode[][]; minW?: number }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full font-mono text-[12px]" style={{ minWidth: minW }}>
        <thead>
          <tr className="text-fg-4">
            {head.map((h, i) => (
              <th key={h} className={`whitespace-nowrap px-4 py-2.5 text-[10px] font-normal uppercase tracking-[0.1em] ${i === 0 ? 'text-left' : 'text-right'}`}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((cells, r) => (
            <tr key={r} className="border-t border-line transition-colors hover:bg-white/[0.02]">
              {cells.map((c, i) => (
                <td key={i} className={`whitespace-nowrap px-4 py-3 tabular-nums ${i === 0 ? 'text-left text-fg-2' : 'text-right text-fg-3'}`}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Column headers mirror the markdown written by bench_offline.py / bench_serving.py.
const OFFLINE_HEAD = ['system', 'output tok/s', 'speedup', 'TTFT p50 ms', 'TTFT p99 ms', 'TPOT p50 ms', 'ITL p99 ms', 'peak mem MB', 'GPU util %', 'notes'];
const SERVING_HEAD = ['config · load', 'req/s', 'out tok/s', 'TTFT p50', 'TTFT p99', 'TPOT p50', 'TPOT p99', 'ITL p99', 'E2E p50', 'goodput %', 'GPU util %', 'GPU mem MB'];

function ThroughputCell({ value, max }: { value: number | null; max: number }) {
  if (value === null) return dash;
  return (
    <span className="inline-flex items-center justify-end gap-2">
      <span className="hidden h-1.5 w-16 overflow-hidden rounded-full bg-white/[0.06] sm:inline-block">
        <span className="block h-full rounded-full bg-gradient-to-r from-mint/50 to-mint" style={{ width: `${(value / max) * 100}%` }} />
      </span>
      <span className="text-fg">{f1(value)}</span>
    </span>
  );
}

function OfflineTable({ group }: { group: OfflineGroup }) {
  const max = Math.max(1, ...group.rows.map((r) => r.outputTokS ?? 0));
  const cells = group.rows.map((r) => [
    { v: r.system, node: labelFor(r.system) },
    { v: r.outputTokS, node: <ThroughputCell key="t" value={r.outputTokS} max={max} /> },
    {
      v: r.speedup,
      node: r.speedup !== null && r.speedup > 1.05 ? <span key="s" className="rounded-full bg-mint/10 px-2 py-0.5 text-mint">{fx(r.speedup)}</span> : fx(r.speedup),
    },
    { v: r.ttftP50, node: f1(r.ttftP50) },
    { v: r.ttftP99, node: f1(r.ttftP99) },
    { v: r.tpotP50, node: f1(r.tpotP50) },
    { v: r.itlP99, node: f1(r.itlP99) },
    { v: r.peakMemMb, node: f1(r.peakMemMb) },
    { v: r.gpuUtilPct, node: fpct(r.gpuUtilPct) },
    { v: r.notes || null, node: r.notes ? <span key="n" className="text-fg-3">{r.notes}</span> : dash },
  ]);
  // Always show system / tok/s / speedup; collapse metric columns this run didn't record.
  const keep = OFFLINE_HEAD.map((_, i) => i < 3 || cells.some((row) => row[i].v !== null));
  const missing = OFFLINE_HEAD.filter((_, i) => !keep[i] && i !== OFFLINE_HEAD.length - 1);

  return (
    <div className="overflow-hidden rounded-xl border border-line">
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-line bg-white/[0.015] px-4 py-2.5">
        <p className="font-mono text-[11px] text-fg-2">
          <span className="text-fg">{group.suite}</span> <span className="text-fg-4">/</span> {group.workload}
        </p>
        {group.description && <p className="text-[12px] text-fg-3">{group.description}</p>}
      </div>
      <Table
        head={OFFLINE_HEAD.filter((_, i) => keep[i])}
        minW={keep.filter(Boolean).length > 5 ? 920 : 520}
        rows={cells.map((row) => row.filter((_, i) => keep[i]).map((c) => c.node))}
      />
      {missing.length > 0 && (
        <p className="border-t border-line px-4 py-2 font-mono text-[10.5px] text-fg-4">
          Not recorded in this run: {missing.join(' · ')}
        </p>
      )}
      {group.note && (
        <p className="border-t border-line bg-amber/[0.05] px-4 py-2.5 text-[12.5px] text-fg-2">
          <span className="text-amber">Note · </span>
          {group.note}
        </p>
      )}
    </div>
  );
}

function ServingTable({ rows, sweep }: { rows: ServingRow[]; sweep: ServingSweep }) {
  return (
    <div className="overflow-hidden rounded-xl border border-line">
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-line bg-white/[0.015] px-4 py-2.5">
        <p className="font-mono text-[11px] text-fg-2">
          <span className="text-fg">workload</span> <span className="text-fg-4">/</span> {sweep.workload}
          <span className="text-fg-4"> · {sweep.natural_stop ? 'outputs end at EOS' : 'fixed output lengths'}</span>
        </p>
        {sweep.description && <p className="text-[12px] text-fg-3">{sweep.description}</p>}
      </div>
      <Table
        head={SERVING_HEAD}
        minW={1040}
        rows={rows.map((r) => [
          <span key="c"><span className="text-fg">{labelFor(r.config)}</span> <span className="text-fg-4">· {r.load}</span></span>,
          f1(r.reqS), f1(r.outTokS), f1(r.ttftP50), f1(r.ttftP99), f1(r.tpotP50), f1(r.tpotP99), f1(r.itlP99), f1(r.e2eP50),
          fpct(r.goodputPct), fpct(r.gpuUtilPct), f1(r.gpuMemMb),
        ])}
      />
      {sweep.note && (
        <p className="border-t border-line bg-amber/[0.05] px-4 py-2.5 text-[12.5px] text-fg-2">
          <span className="text-amber">Note · </span>
          {sweep.note}
        </p>
      )}
    </div>
  );
}

/** Headers with shimmering empty rows: what the table will look like once numbers land. */
function PlaceholderTable({ head, minW, message }: { head: string[]; minW: number; message: string }) {
  return (
    <div className="relative overflow-hidden rounded-xl border border-dashed border-white/[0.12]">
      <Table
        head={head}
        minW={minW}
        rows={Array.from({ length: 3 }).map((_, r) =>
          head.map((_, i) => (
            <span key={i} className={`relative inline-block h-2 overflow-hidden rounded-full bg-white/[0.05] ${i === 0 ? 'w-28' : 'w-10'}`}>
              <span className="shimmer absolute inset-0" style={{ animationDelay: `${(r * 3 + i) * 90}ms` }} />
            </span>
          )),
        )}
      />
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
        <span className="rounded-full border border-line-2 bg-ink-950/90 px-3 py-1.5 font-mono text-[11px] text-fg-2 backdrop-blur">{message}</span>
      </div>
    </div>
  );
}

// ── Platform switcher ─────────────────────────────────────────────────────────

function PlatformResults() {
  const [id, setId] = useState('t4');
  const p: PlatformEntry = PLATFORMS.find((x) => x.id === id) ?? PLATFORMS[0];
  const groups = offlineGroups(p);
  const sweeps = p.serving ?? [];
  const pending = p.status === 'pending';

  return (
    <Card pad={false} glow={false} className="overflow-hidden">
      <div className="flex gap-1 overflow-x-auto border-b border-line p-2" role="tablist" aria-label="Platform">
        {PLATFORMS.map((x) => {
          const on = x.id === id;
          return (
            <button
              key={x.id}
              role="tab"
              aria-selected={on}
              onClick={() => setId(x.id)}
              className={`flex shrink-0 items-center gap-2 rounded-full px-3.5 py-1.5 text-[12.5px] transition-all duration-300 ${
                on ? 'bg-white/[0.08] text-fg shadow-[inset_0_0_0_1px_rgba(255,255,255,0.1)]' : 'text-fg-3 hover:text-fg'
              }`}
            >
              <span className={`h-1.5 w-1.5 rounded-full ${x.status === 'verified' ? 'bg-mint' : 'bg-amber'}`} />
              {x.gpu.replace('NVIDIA ', '')}
              <span className="font-mono text-[10px] text-fg-4">{x.device.toUpperCase()}</span>
            </button>
          );
        })}
      </div>

      <div key={id} className="space-y-5 p-4 [animation:log-in_0.35s_var(--ease-out)_both] sm:p-6">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-[17px] font-semibold tracking-tight text-fg">
              {p.gpu} <span className="text-fg-3">· {p.model} · {p.dtype}</span>
            </p>
            <p className="mt-1 text-[13px] text-fg-3">{p.summary}</p>
          </div>
          {p.smokeRun && (
            <span className="rounded-full border border-amber/30 bg-amber/10 px-2.5 py-1 font-mono text-[10.5px] text-amber">
              local smoke run · not a headline GPU result
            </span>
          )}
        </div>

        <div>
          <p className="mb-2.5 font-mono text-[10.5px] uppercase tracking-[0.16em] text-fg-3">Offline ablation · bench_offline.py</p>
          {groups.length ? (
            <div className="space-y-4">
              {groups.map((g) => <OfflineTable key={g.key} group={g} />)}
              <p className="font-mono text-[10.5px] text-fg-4">
                Speedup is relative to the first system in each table, as in bench_offline.py.
              </p>
            </div>
          ) : (
            <PlaceholderTable
              head={OFFLINE_HEAD}
              minW={920}
              message={pending ? 'benchmarks coming · run the Colab notebook' : 'correctness platform · no throughput runs'}
            />
          )}
        </div>

        <div>
          <p className="mb-2.5 font-mono text-[10.5px] uppercase tracking-[0.16em] text-fg-3">
            Serving sweeps · bench_serving.py · streaming · Poisson arrivals · latencies in ms
          </p>
          {sweeps.length ? (
            <div className="space-y-4">
              {sweeps.map((sw) => <ServingTable key={sw.workload} sweep={sw} rows={servingRows(sw)} />)}
            </div>
          ) : (
            <PlaceholderTable head={SERVING_HEAD} minW={1040} message={pending ? 'benchmarks coming' : 'no serving sweep recorded yet'} />
          )}
        </div>
      </div>
    </Card>
  );
}

// ── Metric definitions ────────────────────────────────────────────────────────

const DEFINITIONS = [
  { k: 'TTFT', v: 'Scheduled send time → first streamed token. Includes queueing.' },
  { k: 'TPOT', v: '(last token − first token) / (tokens − 1), per request.' },
  { k: 'ITL', v: 'Every gap between streamed chunks, pooled across requests.' },
  { k: 'E2E', v: 'Scheduled send time → last token.' },
  { k: 'Goodput', v: 'Share of requests meeting TTFT ≤ 2 s and TPOT ≤ 100 ms.' },
  { k: 'GPU util', v: 'NVML “time a kernel was running”: how busy the GPU is, not FLOP efficiency.' },
];

// ── v2 legacy ─────────────────────────────────────────────────────────────────

const good = (v: ReactNode) => <span className="text-mint">{v}</span>;

function RaceBar({ label, value, pct, color, note }: { label: string; value: string; pct: number; color: string; note?: string }) {
  return (
    <div className="grid grid-cols-[88px_1fr] items-center gap-4 sm:grid-cols-[128px_1fr]">
      <span className="text-[13px] text-fg-2">{label}</span>
      <div className="flex items-center gap-3">
        <div className="h-8 flex-1">
          <div
            className="flex h-full items-center rounded-lg"
            style={{
              width: `${Math.max(pct, 1.2)}%`,
              background: `linear-gradient(90deg, ${color}26, ${color}99)`,
              boxShadow: `inset 0 0 0 1px ${color}66`,
            }}
          />
        </div>
        <span className="w-[76px] text-right font-mono text-sm tabular-nums sm:w-[92px]" style={{ color }}>
          {value}
        </span>
      </div>
      {note && <span className="col-start-2 -mt-2 font-mono text-[10.5px] text-fg-4">{note}</span>}
    </div>
  );
}

function LegacyV2() {
  const [open, setOpen] = useState(false);
  const ttftImprovement = (((TTFT_SEQUENTIAL_UNDER_LOAD_MS - TTFT_BATCHED_UNDER_LOAD_MS) / TTFT_SEQUENTIAL_UNDER_LOAD_MS) * 100).toFixed(1);

  return (
    <div className="rounded-2xl border border-line bg-white/[0.01]">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-4 px-5 py-4 text-left"
      >
        <span>
          <span className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-line-2 px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.12em] text-fg-3">v2 (legacy)</span>
            <span className="text-[14.5px] font-medium text-fg">Apple M2 · the engine before v3</span>
          </span>
          <span className="mt-1 block text-[12.5px] text-fg-3">
            Kept for history. v2 ran one forward pass per request, so these numbers don’t describe the current engine.
          </span>
        </span>
        <span
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-line-2 text-fg-3 transition-transform duration-500 [transition-timing-function:var(--ease-spring)]"
          style={{ transform: open ? 'rotate(45deg)' : 'none' }}
          aria-hidden="true"
        >
          +
        </span>
      </button>

      <div className="collapse-grid" data-open={open}>
        <div>
          <div className="space-y-5 border-t border-line p-5">
            <div className="space-y-4">
              <p className="font-mono text-[10.5px] uppercase tracking-[0.16em] text-fg-3">TTFT · 4 concurrent requests (v2)</p>
              <RaceBar label="Sequential" value={`${TTFT_SEQUENTIAL_UNDER_LOAD_MS.toLocaleString()} ms`} pct={100} color="#FB7185" note="Phase 1 · lock-serialized" />
              <RaceBar label="Batched (v2)" value={`${TTFT_BATCHED_UNDER_LOAD_MS} ms`} pct={(TTFT_BATCHED_UNDER_LOAD_MS / TTFT_SEQUENTIAL_UNDER_LOAD_MS) * 100} color="#4ADE80" />
            </div>

            <div className="overflow-hidden rounded-xl border border-line">
              <Table
                head={['Metric (v2, M2)', 'Phase 1 · sequential', 'v2 · batched', 'Δ']}
                rows={[
                  ['TTFT under 4-way load', `${TTFT_SEQUENTIAL_UNDER_LOAD_MS.toLocaleString()} ms`, good(`${TTFT_BATCHED_UNDER_LOAD_MS} ms`), `−${ttftImprovement}%`],
                  ['Wall clock (all done)', `${CONCURRENT.phase1_expected_ms.toLocaleString()} ms`, good(`${CONCURRENT.wall_ms.toLocaleString()} ms`), '−29.6%'],
                  ['Aggregate throughput', '~42.0 tok/s', good(`${CONCURRENT.aggregate_tps} tok/s`), '+19.3%'],
                  ['Speedup vs serial', '1.00×', good(`${CONCURRENT.speedup}×`), ''],
                  ['Single-request TTFT / decode', '—', `${SINGLE_REQUEST.ttft_ms} ms / ${SINGLE_REQUEST.tps} tok/s`, ''],
                ]}
              />
            </div>

            <div className="grid gap-5 lg:grid-cols-2 [&>*]:min-w-0">
              <div className="overflow-hidden rounded-xl border border-line">
                <Table
                  head={['bench_phases.py (v2)', 'Phase 1', 'Phase 2']}
                  rows={[
                    ['TTFT (mean)', `${PHASE1.ttft_ms} ms`, good(`${PHASE2.ttft_ms} ms`)],
                    ['Total latency', `${PHASE1.total_ms} ms`, good(`${PHASE2.total_ms} ms`)],
                    ['Throughput', `${PHASE1.tps} tok/s`, good(`${PHASE2.agg_tps} tok/s`)],
                    ['Speedup', '1.00×', good(`${PHASE2.speedup}×`)],
                  ]}
                />
              </div>
              <div className="overflow-hidden rounded-xl border border-line">
                <Table
                  head={['Chunked prefill (v2)', 'value']}
                  minW={320}
                  rows={[
                    [`${CHUNKED_PREFILL.prompt_len}-token prompt`, `${CHUNKED_PREFILL.n_chunks} × 128-token chunks`],
                    ['Full prefill (blocking)', `${CHUNKED_PREFILL.full_prefill_ms} ms`],
                    ['First chunk only', good(`${CHUNKED_PREFILL.first_chunk_ms} ms`)],
                    ['Decode stall saved', good(`${CHUNKED_PREFILL.delay_saved_ms} ms`)],
                  ]}
                />
              </div>
            </div>

            <div className="overflow-hidden rounded-xl border border-line">
              <Table
                head={['PR #1 fixes (v2)', 'Before', 'After', 'Δ']}
                rows={[
                  ['TTFT (single req)', `${PR_COMPARISON.single_req.ttft_before_ms} ms`, good(`${PR_COMPARISON.single_req.ttft_after_ms} ms`), `${PR_COMPARISON.single_req.ttft_delta_pct}%`],
                  ['Throughput (single req)', `${PR_COMPARISON.single_req.tps_before} tok/s`, good(`${PR_COMPARISON.single_req.tps_after} tok/s`), `+${PR_COMPARISON.single_req.tps_delta_pct}%`],
                  ['4-way concurrent speedup', `${PR_COMPARISON.concurrent_4way.speedup_before}×`, good(`${PR_COMPARISON.concurrent_4way.speedup_after}×`), '+0.10×'],
                  ['Tests passing', 'unknown', good(`${PR_COMPARISON.tests.passing} / ${PR_COMPARISON.tests.total}`), ''],
                ]}
              />
            </div>
            <p className="font-mono text-[10.5px] text-fg-4">
              Sources: benchmarks/legacy/bench_direct_results.json, bench_phases_results.json, baseline_metrics.json,
              benchmark_comparison.md · Apple M2 (MPS) · Qwen2-0.5B · fp16.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Section ───────────────────────────────────────────────────────────────────

export function BenchmarksSection() {
  return (
    <section id="benchmarks" className="relative px-5 py-24 sm:px-6 sm:py-32">
      <div className="mx-auto max-w-5xl">
        <SectionHeader
          index="08"
          label="Measured results"
          title={<>Numbers you can <Accent gradient>reproduce.</Accent></>}
          subtitle={
            <>
              Seeded workloads; every result records GPU, versions and git commit. Tables mirror what{' '}
              <code className="font-mono text-[0.9em] text-sky">bench_offline.py</code> and{' '}
              <code className="font-mono text-[0.9em] text-sky">bench_serving.py</code> write. Pending GPUs show empty tables
              until real Colab runs land. The NVIDIA T4 results come from a full notebook run; the raw files are in{' '}
              <code className="font-mono text-[0.9em] text-sky">benchmarks/published/</code>.
            </>
          }
        />

        <Reveal>
          <PlatformResults />
        </Reveal>

        <Reveal>
          <div className="mt-5 grid gap-px overflow-hidden rounded-2xl border border-line bg-line sm:grid-cols-2 lg:grid-cols-3">
            {DEFINITIONS.map((d) => (
              <div key={d.k} className="bg-ink-950 px-5 py-4">
                <p className="font-mono text-[11px] text-fg">{d.k}</p>
                <p className="mt-1 text-[12.5px] leading-relaxed text-fg-3">{d.v}</p>
              </div>
            ))}
          </div>
          <p className="mt-2 font-mono text-[10.5px] text-fg-4">
            Latency is measured from the scheduled send time, so a server that falls behind is charged for it. Batching and
            prefix-caching runs use ignore_eos so every system generates the same number of tokens; speculative-decoding runs let
            outputs end at EOS instead (greedy speculation is exact, so every config emits the same text, while forced output
            past EOS makes models loop and would overstate the gain).
          </p>
        </Reveal>

        <Reveal>
          <div className="mt-8">
            <LegacyV2 />
          </div>
        </Reveal>
      </div>
    </section>
  );
}
