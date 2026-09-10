import { SectionHeader } from '@/components/ui/SectionHeader';
import { MetricCard } from '@/components/ui/MetricCard';
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

function TableRow({ cells, header = false }: { cells: (string | React.ReactNode)[]; header?: boolean }) {
  return (
    <div
      className={`grid font-mono text-[11px] ${
        header
          ? 'border-b border-[#2a2a2a] text-[#444444]'
          : 'border-b border-[#1a1a1a] text-[#888888] last:border-0'
      }`}
      style={{ gridTemplateColumns: `repeat(${cells.length}, 1fr)` }}
    >
      {cells.map((cell, i) => (
        <div
          key={i}
          className={`px-3 py-2 ${i > 0 ? 'text-right' : ''}`}
        >
          {cell}
        </div>
      ))}
    </div>
  );
}

export function BenchmarksSection() {
  const ttftImprovement = Math.round(
    ((TTFT_SEQUENTIAL_UNDER_LOAD_MS - TTFT_BATCHED_UNDER_LOAD_MS) /
      TTFT_SEQUENTIAL_UNDER_LOAD_MS) *
      100,
  );

  return (
    <section id="benchmarks" className="border-b border-[#1e1e1e] px-6 py-20 sm:px-10 lg:px-16">
      <div className="mx-auto max-w-5xl">
        <SectionHeader
          label="// measured results"
          title="Benchmark Results"
        />

        {/* Hardware badge */}
        <div className="mb-8 border border-[#2a2a2a] bg-[#111111] px-4 py-2 font-mono text-[10px] text-[#444444]">
          Apple M2 (MPS) · Qwen/Qwen2-0.5B · float16 · All numbers from{' '}
          <span className="text-[#60A5FA]">bench_direct_results.json</span> /{' '}
          <span className="text-[#60A5FA]">bench_phases_results.json</span>
        </div>

        {/* ── Headline numbers ──────────────────────────────── */}
        <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <MetricCard
            label="Single-req TTFT"
            value={`${SINGLE_REQUEST.ttft_ms} ms`}
            highlight
          />
          <MetricCard
            label="Single-req throughput"
            value={`${SINGLE_REQUEST.tps} tok/s`}
            highlight
          />
          <MetricCard
            label="TTFT under 4-way load"
            value={`${TTFT_BATCHED_UNDER_LOAD_MS} ms`}
            delta={`vs ${TTFT_SEQUENTIAL_UNDER_LOAD_MS.toLocaleString()} ms sequential (−${ttftImprovement}%)`}
            deltaPositive
            highlight
          />
          <MetricCard
            label="Tests passing"
            value={`${PR_COMPARISON.tests.passing} / ${PR_COMPARISON.tests.total}`}
            delta="84/84 ✓"
            deltaPositive
          />
        </div>

        {/* ── Concurrency comparison table ─────────────────── */}
        <div className="mb-6 border border-[#2a2a2a] bg-[#0d0d0d]">
          <div className="border-b border-[#2a2a2a] px-4 py-2.5">
            <span className="font-mono text-[10px] uppercase tracking-widest text-[#333333]">
              4-Way Concurrent Load — Phase 1 vs Phase 2–9
            </span>
          </div>
          <div className="p-2">
            <TableRow
              header
              cells={['Metric', 'Phase 1 (Sequential)', 'Phase 2–9 (Batched)', 'Delta']}
            />
            <TableRow
              cells={[
                'TTFT under load (mean)',
                `${TTFT_SEQUENTIAL_UNDER_LOAD_MS.toLocaleString()} ms`,
                <span key="ttft2" className="text-[#4ADE80]">{TTFT_BATCHED_UNDER_LOAD_MS} ms</span>,
                <span key="ttft-d" className="text-[#4ADE80]">−{ttftImprovement}%</span>,
              ]}
            />
            <TableRow
              cells={[
                'Wall clock (all done)',
                `${CONCURRENT.phase1_expected_ms.toLocaleString()} ms`,
                <span key="wc2" className="text-[#4ADE80]">{CONCURRENT.wall_ms.toLocaleString()} ms</span>,
                <span key="wc-d" className="text-[#4ADE80]">−29.6%</span>,
              ]}
            />
            <TableRow
              cells={[
                'Aggregate throughput',
                `~42.0 tok/s`,
                <span key="tps2" className="text-[#4ADE80]">{CONCURRENT.aggregate_tps} tok/s</span>,
                <span key="tps-d" className="text-[#4ADE80]">+19.3%</span>,
              ]}
            />
            <TableRow
              cells={[
                'Speedup vs serial',
                '1.00×',
                <span key="sp2" className="text-[#4ADE80]">{CONCURRENT.speedup}×</span>,
                <span key="sp-d" className="text-[#4ADE80]">+{((CONCURRENT.speedup - 1) * 100).toFixed(0)}%</span>,
              ]}
            />
            <TableRow
              cells={[
                'Total latency mean',
                '—',
                <span key="lat2" className="text-[#aaa]">{CONCURRENT.total_latency_mean_ms} ms</span>,
                '',
              ]}
            />
          </div>
        </div>

        {/* ── Phase-level + chunked prefill ────────────────── */}
        <div className="mb-6 grid gap-4 sm:grid-cols-2">
          {/* Phase results */}
          <div className="border border-[#2a2a2a] bg-[#0d0d0d]">
            <div className="border-b border-[#2a2a2a] px-4 py-2.5">
              <span className="font-mono text-[10px] uppercase tracking-widest text-[#333333]">
                Phase 1 vs Phase 2 (bench_phases.py)
              </span>
            </div>
            <div className="p-2">
              <TableRow header cells={['Metric', 'Phase 1', 'Phase 2']} />
              <TableRow
                cells={[
                  'TTFT (mean)',
                  `${PHASE1.ttft_ms} ms`,
                  <span key="p2ttft" className="text-[#4ADE80]">{PHASE2.ttft_ms} ms</span>,
                ]}
              />
              <TableRow
                cells={[
                  'Total latency',
                  `${PHASE1.total_ms} ms`,
                  <span key="p2lat" className="text-[#4ADE80]">{PHASE2.total_ms} ms</span>,
                ]}
              />
              <TableRow
                cells={[
                  'Throughput',
                  `${PHASE1.tps} tok/s`,
                  <span key="p2tps" className="text-[#4ADE80]">{PHASE2.agg_tps} tok/s</span>,
                ]}
              />
              <TableRow
                cells={[
                  'Speedup',
                  '1.00×',
                  <span key="p2sp" className="text-[#4ADE80]">{PHASE2.speedup}×</span>,
                ]}
              />
            </div>
          </div>

          {/* Chunked prefill */}
          <div className="border border-[#2a2a2a] bg-[#0d0d0d]">
            <div className="border-b border-[#2a2a2a] px-4 py-2.5">
              <span className="font-mono text-[10px] uppercase tracking-widest text-[#333333]">
                Chunked Prefill (Phase 4/9)
              </span>
            </div>
            <div className="p-4 space-y-2">
              <MetricRow label="Prompt length" value={`${CHUNKED_PREFILL.prompt_len} tokens`} />
              <MetricRow label="Chunks" value={`${CHUNKED_PREFILL.n_chunks} × 128 tok`} />
              <MetricRow
                label="Full prefill (blocking)"
                value={`${CHUNKED_PREFILL.full_prefill_ms} ms`}
                bad
              />
              <MetricRow
                label="First chunk only"
                value={`${CHUNKED_PREFILL.first_chunk_ms} ms`}
                good
              />
              <MetricRow
                label="Delay saved"
                value={`${CHUNKED_PREFILL.delay_saved_ms} ms`}
                good
              />
              <MetricRow
                label="Co-running decode step"
                value={`${CHUNKED_PREFILL.decode_step_ms} ms`}
              />
              <div className="border-t border-[#1e1e1e] pt-2 mt-2">
                <MetricRow
                  label="Live KV decode (Phase 8)"
                  value={`${PHASE8_DECODE.live_tps} tok/s`}
                  good
                />
                <MetricRow
                  label="Pool-reconstruct (naive)"
                  value={`${PHASE8_DECODE.live_tps_before} tok/s`}
                  bad
                />
              </div>
            </div>
          </div>
        </div>

        {/* ── PR Before/After ───────────────────────────────── */}
        <div className="border border-[#2a2a2a] bg-[#0d0d0d]">
          <div className="border-b border-[#2a2a2a] px-4 py-2.5">
            <span className="font-mono text-[10px] uppercase tracking-widest text-[#333333]">
              Before / After — fix(engine): correct scheduler and KV-cache lifecycle bugs
            </span>
          </div>
          <div className="p-2">
            <TableRow header cells={['Metric', 'Before', 'After', 'Δ']} />
            <TableRow
              cells={[
                'TTFT (single req)',
                `${PR_COMPARISON.single_req.ttft_before_ms} ms`,
                <span key="ttft-a" className="text-[#4ADE80]">{PR_COMPARISON.single_req.ttft_after_ms} ms</span>,
                <span key="ttft-d2" className="text-[#4ADE80]">{PR_COMPARISON.single_req.ttft_delta_pct}%</span>,
              ]}
            />
            <TableRow
              cells={[
                'Throughput (single req)',
                `${PR_COMPARISON.single_req.tps_before} tok/s`,
                <span key="tps-a" className="text-[#4ADE80]">{PR_COMPARISON.single_req.tps_after} tok/s</span>,
                <span key="tps-d2" className="text-[#4ADE80]">+{PR_COMPARISON.single_req.tps_delta_pct}%</span>,
              ]}
            />
            <TableRow
              cells={[
                '4-way concurrent speedup',
                `${PR_COMPARISON.concurrent_4way.speedup_before}×`,
                <span key="sp-a" className="text-[#4ADE80]">{PR_COMPARISON.concurrent_4way.speedup_after}×</span>,
                <span key="sp-d2" className="text-[#4ADE80]">+0.10×</span>,
              ]}
            />
            <TableRow
              cells={[
                'Tests passing',
                'unknown',
                <span key="test-a" className="text-[#4ADE80]">84 / 84 ✓</span>,
                <span key="test-d" className="text-[#4ADE80]">all green</span>,
              ]}
            />
          </div>
        </div>
      </div>
    </section>
  );
}

function MetricRow({
  label,
  value,
  good,
  bad,
}: {
  label: string;
  value: string;
  good?: boolean;
  bad?: boolean;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="font-mono text-[10px] text-[#444444]">{label}</span>
      <span
        className="font-mono text-[11px]"
        style={{ color: good ? '#4ADE80' : bad ? '#F87171' : '#888888' }}
      >
        {value}
      </span>
    </div>
  );
}
