'use client';

import { useEffect, useRef, useState } from 'react';
import { useSchedulerSim } from '@/lib/useSchedulerSim';
import { SEQ_COLORS } from '@/lib/motion';
import { Badge } from '@/components/ui/Badge';
import { Window } from '@/components/ui/Card';
import { CountUp } from '@/components/ui/CountUp';
import { StarButton } from '@/components/GitHubStats';
import { BlockField } from '@/components/BlockField';
import { ENGINE_CONFIG, SINGLE_REQUEST, PR_COMPARISON, TTFT_BATCHED_UNDER_LOAD_MS, TTFT_SEQUENTIAL_UNDER_LOAD_MS } from '@/data/benchmarks';
import type { SimSequence, SeqState } from '@/data/simulation';

const STATE_LABELS: Record<SeqState, string> = {
  waiting: 'queued',
  prefill: 'prefill',
  chunked_prefilling: 'prefill',
  decoding: 'decoding',
  swapped: 'swapped',
  finished: 'done',
};

const STATE_COLORS: Record<SeqState, string> = {
  waiting: '#767680',
  prefill: '#FBBF24',
  chunked_prefilling: '#FBBF24',
  decoding: '#4ADE80',
  swapped: '#FB7185',
  finished: '#52525B',
};

function SequenceLane({ seq }: { seq: SimSequence }) {
  const isPrefill = seq.state === 'prefill' || seq.state === 'chunked_prefilling';
  const isDecode = seq.state === 'decoding' || seq.state === 'finished';
  const progress = isDecode
    ? (seq.tokensGenerated / seq.maxNewTokens) * 100
    : isPrefill
    ? (seq.prefillChunksDone / seq.prefillChunksTotal) * 100
    : 0;
  const pct = Math.max(0, Math.min(100, progress));
  const color = STATE_COLORS[seq.state];
  const idColor = SEQ_COLORS[seq.shortId] ?? '#A6A6B0';

  return (
    <div className="grid grid-cols-[64px_1fr_76px] items-center gap-3 py-[7px] sm:grid-cols-[84px_1fr_92px_44px]">
      <span className="flex items-center gap-2 font-mono text-[11px] text-fg-2">
        <span className="h-2 w-2 rounded-[3px]" style={{ backgroundColor: idColor, boxShadow: `0 0 10px ${idColor}80` }} />
        {seq.shortId}
      </span>

      <div className="relative h-6 overflow-hidden rounded-md border border-line bg-white/[0.02]">
        {seq.state === 'waiting' && <div className="hatch absolute inset-0 opacity-70" />}
        <div
          className="absolute inset-y-0 left-0 rounded-md transition-[width] duration-200 ease-linear"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}14, ${color}55)`,
          }}
        >
          {seq.state === 'decoding' && (
            <span
              className="absolute right-0 top-1/2 h-4 w-[3px] -translate-y-1/2 rounded-full"
              style={{ backgroundColor: color, boxShadow: `0 0 12px 2px ${color}` }}
            />
          )}
          {isPrefill && <span className="shimmer absolute inset-0" />}
        </div>
        {/* KV block boundaries every 16 tokens */}
        {Array.from({ length: 7 }).map((_, i) => (
          <span
            key={i}
            className="absolute inset-y-1.5 w-px bg-white/[0.06]"
            style={{ left: `${(i + 1) * 12.5}%` }}
          />
        ))}
      </div>

      <span
        className="justify-self-end rounded-full px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.1em]"
        style={{ color, backgroundColor: `${color}14`, border: `1px solid ${color}33` }}
      >
        {STATE_LABELS[seq.state]}
      </span>

      <span className="hidden text-right font-mono text-[11px] tabular-nums text-fg-3 sm:block">
        {isDecode ? `${seq.tokensGenerated}t` : '—'}
      </span>
    </div>
  );
}

function Sparkline({ values, max }: { values: number[]; max: number }) {
  const W = 72, H = 18;
  if (values.length < 2) return <svg width={W} height={H} />;
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * W},${H - (v / max) * (H - 2) - 1}`).join(' ');
  return (
    <svg width={W} height={H} className="overflow-visible" aria-hidden="true">
      <defs>
        <linearGradient id="spark-fill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="#4ADE80" stopOpacity="0.35" />
          <stop offset="1" stopColor="#4ADE80" stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={`0,${H} ${pts} ${W},${H}`} fill="url(#spark-fill)" />
      <polyline points={pts} fill="none" stroke="#4ADE80" strokeWidth="1.25" strokeLinejoin="round" />
    </svg>
  );
}

function SchedulerWindow() {
  const { tick, sequences, batchSize, queueDepth, tokPerSec, freeBlocks, allocatedBlocks, isRunning, pause, resume } =
    useSchedulerSim(180);
  const history = useRef<number[]>([]);
  const [, force] = useState(0);

  useEffect(() => {
    history.current = [...history.current, tokPerSec].slice(-32);
    force((n) => n + 1);
  }, [tick, tokPerSec]);

  const displaySeqs = sequences.slice(0, 5);
  const emptySlots = Math.max(0, 5 - displaySeqs.length);
  const totalBlocks = freeBlocks + allocatedBlocks;
  const maxBatch = ENGINE_CONFIG.max_batch_size;

  return (
    <Window
      title={<><span className="text-fg-2">scheduler.py</span> <span className="text-fg-4">— continuous batching</span></>}
      right={
        <>
          <Badge label="Demo data" variant="demo" className="hidden sm:inline-flex" />
          <button
            type="button"
            onClick={isRunning ? pause : resume}
            className="inline-flex h-6 items-center gap-1.5 rounded-full border border-line-2 bg-white/[0.03] px-2.5 font-mono text-[10px] uppercase tracking-[0.12em] text-fg-2 transition-colors hover:border-white/25 hover:text-fg"
            aria-label={isRunning ? 'Pause simulation' : 'Resume simulation'}
          >
            {isRunning ? (
              <><span className="relative flex h-1.5 w-1.5"><span className="ping-soft absolute inset-0 rounded-full bg-mint" /><span className="relative h-1.5 w-1.5 rounded-full bg-mint" /></span>Live</>
            ) : (
              <><span className="h-1.5 w-1.5 rounded-full bg-amber" />Paused</>
            )}
          </button>
        </>
      }
    >
      <div className="px-4 py-3 sm:px-5">
        {displaySeqs.map((seq) => (
          <SequenceLane key={seq.id} seq={seq} />
        ))}
        {Array.from({ length: emptySlots }).map((_, i) => (
          <div key={`empty-${i}`} className="grid grid-cols-[64px_1fr_76px] items-center gap-3 py-[7px] sm:grid-cols-[84px_1fr_92px_44px]">
            <span className="font-mono text-[11px] text-fg-4">·</span>
            <div className="h-6 rounded-md border border-dashed border-line" />
            <span className="justify-self-end font-mono text-[10px] uppercase tracking-[0.1em] text-fg-4">free slot</span>
            <span className="hidden sm:block" />
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-px border-t border-line bg-line sm:grid-cols-5">
        <Stat label="iteration" value={String(tick).padStart(4, '0')} />
        <Stat
          label="batch"
          value={
            <span className="flex items-center gap-2">
              {batchSize}/{maxBatch}
              <span className="flex gap-0.5">
                {Array.from({ length: maxBatch }).map((_, i) => (
                  <span key={i} className={`h-2.5 w-1.5 rounded-sm transition-colors duration-300 ${i < batchSize ? 'bg-mint' : 'bg-white/10'}`} />
                ))}
              </span>
            </span>
          }
        />
        <Stat label="queue" value={String(queueDepth)} />
        <Stat
          label="tok/s"
          value={
            <span className="flex items-center gap-2">
              <span className="w-8 text-mint">{tokPerSec}</span>
              <Sparkline values={history.current} max={maxBatch * 91} />
            </span>
          }
        />
        <Stat
          label="kv blocks"
          className="col-span-2 sm:col-span-1"
          value={
            <span className="flex items-center gap-2">
              {allocatedBlocks}/{totalBlocks}
              <span className="h-1.5 w-12 overflow-hidden rounded-full bg-white/10">
                <span className="block h-full rounded-full bg-sky transition-[width] duration-300" style={{ width: `${Math.max(4, (allocatedBlocks / totalBlocks) * 100)}%` }} />
              </span>
            </span>
          }
        />
      </div>
    </Window>
  );
}

function Stat({ label, value, className = '' }: { label: string; value: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-ink-900 px-4 py-3 ${className}`}>
      <p className="mb-1 font-mono text-[9.5px] uppercase tracking-[0.16em] text-fg-4">{label}</p>
      <div className="font-mono text-[13px] tabular-nums text-fg">{value}</div>
    </div>
  );
}

const PILLARS = [
  { label: 'Continuous batching', color: '#4ADE80' },
  { label: 'Paged KV-cache', color: '#60A5FA' },
  { label: 'Chunked prefill', color: '#FBBF24' },
  { label: 'CPU swap pool', color: '#FB7185' },
];

export function HeroSection() {
  const speedup = TTFT_SEQUENTIAL_UNDER_LOAD_MS / TTFT_BATCHED_UNDER_LOAD_MS;

  return (
    <section id="top" className="relative overflow-hidden pb-20 pt-32 sm:pt-40">
      {/* Interactive KV-block backdrop */}
      <div className="absolute inset-0 [mask-image:radial-gradient(ellipse_75%_60%_at_50%_28%,#000_35%,transparent_80%)]">
        <BlockField />
      </div>
      {/* Aurora */}
      <div className="pointer-events-none absolute left-1/2 top-[-18rem] h-[36rem] w-[60rem] -translate-x-1/2 rounded-full bg-[radial-gradient(closest-side,rgba(74,222,128,0.16),transparent)] blur-2xl" />
      <div className="pointer-events-none absolute right-[-10rem] top-[10rem] h-[26rem] w-[30rem] rounded-full bg-[radial-gradient(closest-side,rgba(167,139,250,0.12),transparent)] blur-2xl" />

      <div className="relative mx-auto max-w-5xl px-5 text-center sm:px-6">
        <a
          href="#phases"
          className="group mb-8 inline-flex items-center gap-2 rounded-full border border-line-2 bg-ink-900/60 py-1 pl-1 pr-3 text-[12.5px] text-fg-2 backdrop-blur transition-colors hover:border-white/20 hover:text-fg"
        >
          <span className="whitespace-nowrap rounded-full bg-mint/15 px-2 py-0.5 font-mono text-[10.5px] text-mint">11 phases</span>
          <span className="hidden sm:inline">Built from scratch · </span>{PR_COMPARISON.tests.passing}/{PR_COMPARISON.tests.total} tests green
          <span className="transition-transform duration-300 group-hover:translate-x-0.5">→</span>
        </a>

        <h1 className="text-balance text-[2.6rem] font-semibold leading-[1.02] tracking-[-0.035em] text-fg sm:text-6xl lg:text-[5.2rem]">
          Every token served.
          <br />
          <span className="accent-serif text-gradient pr-2 text-[1.1em] leading-[0.9]">No page wasted.</span>
        </h1>

        <p className="mx-auto mt-7 max-w-2xl text-pretty text-base leading-relaxed text-fg-2 sm:text-lg">
          <span className="text-fg">PageServe</span> is an LLM inference engine built from first principles —
          continuous batching, a paged KV-cache, chunked prefill and a CPU swap pool. No vLLM. No{' '}
          <code className="rounded-md border border-line bg-white/[0.04] px-1.5 py-0.5 font-mono text-[0.85em] text-fg">generate()</code>.
        </p>

        <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
          <StarButton label="View source" />
          <a
            href="#scheduler"
            className="group inline-flex items-center gap-2 rounded-full border border-line-2 bg-white/[0.03] px-5 py-2.5 text-sm text-fg-2 backdrop-blur transition-all duration-300 hover:border-white/25 hover:text-fg"
          >
            How it works
            <span className="transition-transform duration-300 group-hover:translate-y-0.5">↓</span>
          </a>
        </div>

        <ul className="mt-10 flex flex-wrap items-center justify-center gap-x-5 gap-y-2">
          {PILLARS.map((p) => (
            <li key={p.label} className="flex items-center gap-2 font-mono text-[11.5px] text-fg-3">
              <span className="h-1.5 w-1.5 rounded-[2px]" style={{ backgroundColor: p.color }} />
              {p.label}
            </li>
          ))}
        </ul>
      </div>

      {/* Live scheduler window */}
      <div className="relative mx-auto mt-16 max-w-5xl px-5 sm:px-6">
        <div className="pointer-events-none absolute -inset-x-4 -inset-y-6 rounded-[32px] bg-[radial-gradient(60%_50%_at_50%_50%,rgba(74,222,128,0.10),transparent)] blur-xl" />
        <div className="tilt-in relative origin-bottom">
          <SchedulerWindow />
        </div>
        <p className="mt-4 hidden text-right font-mono text-[11px] text-fg-4 sm:block">
          <span className="accent-serif text-[15px] text-fg-3">psst —</span> the background is a block pool. Move your cursor to allocate, click to burst.
        </p>
      </div>

      {/* Headline numbers */}
      <div className="relative mx-5 mt-14 grid max-w-5xl grid-cols-2 gap-px overflow-hidden rounded-2xl border border-line bg-line sm:mx-6 sm:grid-cols-4 lg:mx-auto">
        <HeroStat value={<CountUp value={TTFT_BATCHED_UNDER_LOAD_MS} decimals={1} suffix=" ms" />} label="TTFT under 4-way load" />
        <HeroStat value={<CountUp value={speedup} decimals={0} suffix="×" />} label="faster first token vs. sequential" accent />
        <HeroStat value={<CountUp value={SINGLE_REQUEST.tps} decimals={1} />} unit="tok/s" label="single-stream decode" />
        <HeroStat value="0" unit="OOM crashes" label="under burst load, via CPU swap" />
      </div>
    </section>
  );
}

function HeroStat({ value, unit, label, accent = false }: { value: React.ReactNode; unit?: string; label: string; accent?: boolean }) {
  return (
    <div className="bg-ink-950/80 px-5 py-6 backdrop-blur sm:px-6">
      <p className={`text-3xl font-semibold tracking-tight sm:text-[2.1rem] ${accent ? 'text-gradient' : 'text-fg'}`}>
        {value}
        {unit && <span className="ml-1.5 text-sm font-normal text-fg-3">{unit}</span>}
      </p>
      <p className="mt-1.5 text-[12.5px] leading-snug text-fg-3">{label}</p>
    </div>
  );
}
