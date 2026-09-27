import { type ReactNode } from 'react';
import { SectionHeader, Accent } from '@/components/ui/SectionHeader';
import { MetricCard } from '@/components/ui/MetricCard';
import { Card } from '@/components/ui/Card';
import { Reveal } from '@/components/ui/Reveal';
import { CountUp } from '@/components/ui/CountUp';
import {
  SINGLE_REQUEST,
  CONCURRENT,
  PHASE1,
  PHASE2,
  CHUNKED_PREFILL,
  PHASE8_DECODE,
  PR_COMPARISON,
  TTFT_SEQUENTIAL_UNDER_LOAD_MS,
  TTFT_BATCHED_UNDER_LOAD_MS,
} from '@/data/benchmarks';

const good = (v: ReactNode) => <span className="text-mint">{v}</span>;
const chip = (v: ReactNode) => (
  <span className="rounded-full bg-mint/10 px-2 py-0.5 text-mint">{v}</span>
);

function Table({ title, head, rows }: { title: string; head: string[]; rows: ReactNode[][] }) {
  return (
    <Card pad={false} glow={false} className="overflow-hidden">
      <div className="border-b border-line px-5 py-3.5">
        <p className="font-mono text-[10.5px] uppercase tracking-[0.16em] text-fg-3">{title}</p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[480px] font-mono text-[12px]">
          <thead>
            <tr className="text-fg-4">
              {head.map((h, i) => (
                <th key={h} className={`px-5 py-2.5 text-[10.5px] font-normal uppercase tracking-[0.1em] ${i === 0 ? 'text-left' : 'text-right'}`}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((cells, r) => (
              <tr key={r} className="border-t border-line transition-colors hover:bg-white/[0.02]">
                {cells.map((c, i) => (
                  <td key={i} className={`px-5 py-3 tabular-nums ${i === 0 ? 'text-left text-fg-2' : 'text-right text-fg-3'}`}>
                    {c}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function RaceBar({ label, value, pct, color, note }: { label: string; value: string; pct: number; color: string; note?: string }) {
  return (
    <div className="grid grid-cols-[88px_1fr] items-center gap-4 sm:grid-cols-[128px_1fr]">
      <span className="text-[13px] text-fg-2">{label}</span>
      <div className="flex items-center gap-3">
        <div className="h-9 flex-1">
          <div
            className="bar-grow flex h-full items-center rounded-lg"
            style={{
              ['--w' as string]: `${Math.max(pct, 1.2)}%`,
              background: `linear-gradient(90deg, ${color}26, ${color}99)`,
              boxShadow: `inset 0 0 0 1px ${color}66, 0 0 24px -6px ${color}`,
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

export function BenchmarksSection() {
  const ttftImprovement = (
    ((TTFT_SEQUENTIAL_UNDER_LOAD_MS - TTFT_BATCHED_UNDER_LOAD_MS) / TTFT_SEQUENTIAL_UNDER_LOAD_MS) * 100
  ).toFixed(1);
  const speedup = Math.round(TTFT_SEQUENTIAL_UNDER_LOAD_MS / TTFT_BATCHED_UNDER_LOAD_MS);

  return (
    <section id="benchmarks" className="relative px-5 py-24 sm:px-6 sm:py-32">
      <div className="mx-auto max-w-5xl">
        <SectionHeader
          index="06"
          label="Measured results"
          title={<>Measured on a laptop, <Accent gradient>not a slide deck.</Accent></>}
          subtitle={
            <>
              Apple M2 (MPS) · Qwen/Qwen2-0.5B · float16. Every number comes from{' '}
              <code className="font-mono text-[0.9em] text-sky">bench_direct_results.json</code> and{' '}
              <code className="font-mono text-[0.9em] text-sky">bench_phases_results.json</code>.
            </>
          }
        />

        {/* Headline numbers */}
        <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Reveal delay={0}>
            <MetricCard label="Single-req TTFT" value={<CountUp value={SINGLE_REQUEST.ttft_ms} decimals={1} />} unit="ms" />
          </Reveal>
          <Reveal delay={70}>
            <MetricCard label="Single-req decode" value={<CountUp value={SINGLE_REQUEST.tps} decimals={1} />} unit="tok/s" accent="#60A5FA" />
          </Reveal>
          <Reveal delay={140}>
            <MetricCard
              label="TTFT · 4-way load"
              value={<CountUp value={TTFT_BATCHED_UNDER_LOAD_MS} decimals={1} />}
              unit="ms"
              delta={`−${ttftImprovement}% vs sequential`}
              deltaPositive
              accent="#A78BFA"
            />
          </Reveal>
          <Reveal delay={210}>
            <MetricCard
              label="Tests passing"
              value={<>{PR_COMPARISON.tests.passing}<span className="text-fg-4">/{PR_COMPARISON.tests.total}</span></>}
              delta="all green"
              deltaPositive
              accent="#FBBF24"
            />
          </Reveal>
        </div>

        {/* The race */}
        <Reveal>
          <Card className="mb-5 p-6 sm:p-8" glow={false}>
            <div className="mb-7 flex flex-wrap items-end justify-between gap-4">
              <div>
                <p className="font-mono text-[10.5px] uppercase tracking-[0.16em] text-fg-3">Time to first token · 4 concurrent requests</p>
                <p className="mt-2 text-[15px] text-fg-2">The late arrival no longer waits for everyone else to finish.</p>
              </div>
              <p className="text-5xl font-semibold tracking-tight sm:text-6xl">
                <span className="text-gradient">{speedup}×</span>
                <span className="accent-serif ml-2 text-2xl text-fg-3">faster</span>
              </p>
            </div>
            <div className="space-y-4">
              <RaceBar label="Sequential" value={`${TTFT_SEQUENTIAL_UNDER_LOAD_MS.toLocaleString()} ms`} pct={100} color="#FB7185" note="Phase 1 · asyncio.Lock" />
              <RaceBar
                label="Batched"
                value={`${TTFT_BATCHED_UNDER_LOAD_MS} ms`}
                pct={(TTFT_BATCHED_UNDER_LOAD_MS / TTFT_SEQUENTIAL_UNDER_LOAD_MS) * 100}
                color="#4ADE80"
                note="Phase 2–9 · continuous batching ← that sliver is the whole wait"
              />
            </div>
          </Card>
        </Reveal>

        <Reveal>
          <div className="mb-5">
            <Table
              title="4-way concurrent load — Phase 1 vs Phase 2–9"
              head={['Metric', 'Phase 1 · sequential', 'Phase 2–9 · batched', 'Δ']}
              rows={[
                ['TTFT under load (mean)', `${TTFT_SEQUENTIAL_UNDER_LOAD_MS.toLocaleString()} ms`, good(`${TTFT_BATCHED_UNDER_LOAD_MS} ms`), chip(`−${ttftImprovement}%`)],
                ['Wall clock (all done)', `${CONCURRENT.phase1_expected_ms.toLocaleString()} ms`, good(`${CONCURRENT.wall_ms.toLocaleString()} ms`), chip('−29.6%')],
                ['Aggregate throughput', '~42.0 tok/s', good(`${CONCURRENT.aggregate_tps} tok/s`), chip('+19.3%')],
                ['Speedup vs serial', '1.00×', good(`${CONCURRENT.speedup}×`), chip(`+${((CONCURRENT.speedup - 1) * 100).toFixed(0)}%`)],
                ['Total latency mean', '—', <span key="lat" className="text-fg-2">{CONCURRENT.total_latency_mean_ms} ms</span>, ''],
              ]}
            />
          </div>
        </Reveal>

        <div className="mb-5 grid gap-5 lg:grid-cols-2 [&>*]:min-w-0">
          <Reveal>
            <Table
              title="Phase 1 vs Phase 2 · bench_phases.py"
              head={['Metric', 'Phase 1', 'Phase 2']}
              rows={[
                ['TTFT (mean)', `${PHASE1.ttft_ms} ms`, good(`${PHASE2.ttft_ms} ms`)],
                ['Total latency', `${PHASE1.total_ms} ms`, good(`${PHASE2.total_ms} ms`)],
                ['Throughput', `${PHASE1.tps} tok/s`, good(`${PHASE2.agg_tps} tok/s`)],
                ['Speedup', '1.00×', good(`${PHASE2.speedup}×`)],
              ]}
            />
          </Reveal>

          <Reveal delay={80}>
            <Card className="h-full" glow={false}>
              <p className="mb-5 font-mono text-[10.5px] uppercase tracking-[0.16em] text-fg-3">Chunked prefill · Phase 4 / 9</p>

              {/* 402-token prompt, split into 128-token chunks */}
              <div className="mb-2 flex gap-1">
                {Array.from({ length: CHUNKED_PREFILL.n_chunks }).map((_, i) => {
                  const w = i < CHUNKED_PREFILL.n_chunks - 1 ? 128 : CHUNKED_PREFILL.prompt_len - 128 * (CHUNKED_PREFILL.n_chunks - 1);
                  return (
                    <div
                      key={i}
                      className={`flex h-8 items-center justify-center rounded-md font-mono text-[10px] ${i === 0 ? 'bg-mint/25 text-mint ring-1 ring-mint/50' : 'hatch-amber text-amber/80'}`}
                      style={{ flexGrow: w }}
                    >
                      {w}
                    </div>
                  );
                })}
              </div>
              <p className="mb-5 font-mono text-[10.5px] text-fg-4">
                {CHUNKED_PREFILL.prompt_len}-token prompt → {CHUNKED_PREFILL.n_chunks} chunks × 128 tok · decode runs between chunks
              </p>

              <div className="space-y-2.5">
                <MetricRow label="Full prefill (blocking)" value={`${CHUNKED_PREFILL.full_prefill_ms} ms`} bad />
                <MetricRow label="First chunk only" value={`${CHUNKED_PREFILL.first_chunk_ms} ms`} good />
                <MetricRow label="Decode stall saved" value={`${CHUNKED_PREFILL.delay_saved_ms} ms`} good />
                <MetricRow label="Co-running decode step" value={`${CHUNKED_PREFILL.decode_step_ms} ms`} />
                <div className="my-1 border-t border-line" />
                <MetricRow label="Live KV decode (Phase 8)" value={`${PHASE8_DECODE.live_tps} tok/s`} good />
                <MetricRow label="Pool-reconstruct (naive)" value={`${PHASE8_DECODE.live_tps_before} tok/s`} bad />
              </div>
            </Card>
          </Reveal>
        </div>

        <Reveal>
          <Table
            title="Before / after — fix(engine): correct scheduler and KV-cache lifecycle bugs"
            head={['Metric', 'Before', 'After', 'Δ']}
            rows={[
              ['TTFT (single req)', `${PR_COMPARISON.single_req.ttft_before_ms} ms`, good(`${PR_COMPARISON.single_req.ttft_after_ms} ms`), chip(`${PR_COMPARISON.single_req.ttft_delta_pct}%`)],
              ['Throughput (single req)', `${PR_COMPARISON.single_req.tps_before} tok/s`, good(`${PR_COMPARISON.single_req.tps_after} tok/s`), chip(`+${PR_COMPARISON.single_req.tps_delta_pct}%`)],
              ['4-way concurrent speedup', `${PR_COMPARISON.concurrent_4way.speedup_before}×`, good(`${PR_COMPARISON.concurrent_4way.speedup_after}×`), chip('+0.10×')],
              ['Tests passing', 'unknown', good('84 / 84 ✓'), chip('all green')],
            ]}
          />
        </Reveal>
      </div>
    </section>
  );
}

function MetricRow({ label, value, good, bad }: { label: string; value: string; good?: boolean; bad?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-[13px] text-fg-3">{label}</span>
      <span className={`font-mono text-[12.5px] tabular-nums ${good ? 'text-mint' : bad ? 'text-rose' : 'text-fg-2'}`}>{value}</span>
    </div>
  );
}
