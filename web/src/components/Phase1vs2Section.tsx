'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { SectionHeader, Accent } from '@/components/ui/SectionHeader';
import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { Reveal } from '@/components/ui/Reveal';
import { useInView, usePrefersReducedMotion } from '@/lib/motion';
import { PHASE1_TIMELINE, PHASE2_TIMELINE, type TimelineBar } from '@/data/simulation';
import { TTFT_BATCHED_UNDER_LOAD_MS } from '@/data/benchmarks';
import { compareSystems } from '@/data/gpuBenchmarks';

type Mode = 'phase1' | 'phase2';

// Both views share the Phase 1 wall clock so the comparison is visually honest.
const SCALE_MS = 9161;
const SWEEP_MS = 3400; // real time for the playhead to cross the chart
const REQ_COLORS = ['#4ADE80', '#60A5FA', '#FBBF24', '#A78BFA'];

const pct = (ms: number) => (ms / SCALE_MS) * 100;
const fmtMs = (ms: number) => (ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(2)} s`);

function TimelineRow({ bar, t, color }: { bar: TimelineBar; t: number; color: string }) {
  const queued = pct(bar.prefillStartMs - bar.waitStartMs);
  const prefill = pct(bar.decodeStartMs - bar.prefillStartMs);
  const decode = pct(bar.finishMs - bar.decodeStartMs);
  const firstToken = t >= bar.decodeStartMs;
  const done = t >= bar.finishMs;

  return (
    <div className="grid grid-cols-[52px_1fr] items-center gap-3 py-1.5 sm:grid-cols-[60px_1fr_104px]">
      <span className="flex items-center gap-2 font-mono text-[11px] text-fg-2">
        <span className="h-2 w-2 rounded-[3px]" style={{ backgroundColor: color }} />
        {bar.label}
      </span>

      <div className="relative h-7 rounded-lg border border-line bg-white/[0.015]">
        <div
          className="absolute inset-0 flex overflow-hidden rounded-lg"
          style={{ clipPath: `inset(0 ${100 - pct(t)}% 0 0)` }}
        >
          <div style={{ width: `${pct(bar.waitStartMs)}%` }} />
          {queued > 0.05 && <div className="hatch h-full" style={{ width: `${queued}%` }} />}
          <div className="h-full bg-amber/30" style={{ width: `${prefill}%` }} />
          <div
            className="h-full rounded-r-md"
            style={{ width: `${decode}%`, background: 'linear-gradient(90deg, rgba(74,222,128,0.28), rgba(74,222,128,0.5))' }}
          />
        </div>
        {/* First-token flag */}
        <div
          className="absolute -top-1 bottom-[-4px] w-px bg-mint transition-opacity duration-300"
          style={{ left: `${pct(bar.decodeStartMs)}%`, opacity: firstToken ? 1 : 0 }}
        >
          <span
            className="absolute -top-1 left-0 h-2 w-2 -translate-x-1/2 rounded-full bg-mint shadow-[0_0_10px_#4ADE80] transition-transform duration-500 [transition-timing-function:var(--ease-spring)]"
            style={{ transform: `translateX(-50%) scale(${firstToken ? 1 : 0})` }}
          />
        </div>
        {done && (
          <span className="absolute right-1 top-1/2 -translate-y-1/2 rounded bg-ink-950/85 px-1 font-mono text-[9.5px] text-fg-2 sm:hidden">
            {fmtMs(bar.ttftMs)}
          </span>
        )}
      </div>

      <span className="hidden text-right font-mono text-[11px] sm:block">
        <span className="text-fg-4">TTFT </span>
        <span
          className="tabular-nums transition-opacity duration-300"
          style={{ color: bar.ttftMs > 500 ? '#FB7185' : '#4ADE80', opacity: firstToken ? 1 : 0.15 }}
        >
          {fmtMs(bar.ttftMs)}
        </span>
      </span>
    </div>
  );
}

export function Phase1vs2Section() {
  const [mode, setMode] = useState<Mode>('phase1');
  const [t, setT] = useState(0);
  const [ref, inView] = useInView<HTMLDivElement>({ threshold: 0.35 });
  const reduced = usePrefersReducedMotion();
  const raf = useRef(0);

  const play = useCallback(() => {
    cancelAnimationFrame(raf.current);
    if (reduced) { setT(SCALE_MS); return; }
    const start = performance.now();
    const step = (now: number) => {
      const p = Math.min((now - start) / SWEEP_MS, 1);
      setT(p * SCALE_MS);
      if (p < 1) raf.current = requestAnimationFrame(step);
    };
    setT(0);
    raf.current = requestAnimationFrame(step);
  }, [reduced]);

  useEffect(() => { if (inView) play(); }, [inView, play]);
  useEffect(() => () => cancelAnimationFrame(raf.current), []);

  const handleModeChange = (m: Mode) => {
    if (m === mode) return;
    setMode(m);
    play();
  };

  const bars = mode === 'phase1' ? PHASE1_TIMELINE : PHASE2_TIMELINE;
  const allDoneAt = Math.max(...bars.map((b) => b.finishMs));
  const finished = t >= allDoneAt;

  return (
    <section id="comparison" className="relative px-5 py-24 sm:px-6 sm:py-32">
      <div className="mx-auto max-w-5xl">
        <SectionHeader
          index="02"
          label="Phase comparison"
          title={<>Four requests. <Accent gradient>Two very different waits.</Accent></>}
          subtitle={
            <>
              4 concurrent requests, 50 tokens each, replayed from the{' '}
              <span className="text-fg">v2 (legacy) engine on an Apple M2</span>. Both views share one wall-clock scale: watch
              where the playhead is when the last request finishes. v3 numbers from the same laptop are below the chart.
            </>
          }
        />

        <Reveal>
          <div ref={ref} className="surface overflow-hidden">
            {/* Controls */}
            <div className="flex flex-wrap items-center gap-3 border-b border-line px-4 py-3 sm:px-5">
              <div className="relative grid grid-cols-2 rounded-full border border-line-2 bg-ink-950 p-1" role="tablist">
                <span
                  className="absolute inset-y-1 left-1 w-[calc(50%-4px)] rounded-full transition-all duration-500 [transition-timing-function:var(--ease-spring)]"
                  style={{
                    transform: mode === 'phase2' ? 'translateX(100%)' : 'none',
                    background: mode === 'phase1' ? 'rgba(251,113,133,0.14)' : 'rgba(74,222,128,0.14)',
                    boxShadow: `inset 0 0 0 1px ${mode === 'phase1' ? 'rgba(251,113,133,0.35)' : 'rgba(74,222,128,0.35)'}`,
                  }}
                  aria-hidden="true"
                />
                {[
                  { id: 'phase1' as Mode, prefix: 'Phase 1 · ', label: 'Sequential', on: 'text-rose' },
                  { id: 'phase2' as Mode, prefix: 'Phase 2+ · ', label: 'Batched', on: 'text-mint' },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    role="tab"
                    aria-selected={mode === tab.id}
                    onClick={() => handleModeChange(tab.id)}
                    className={`relative z-10 whitespace-nowrap rounded-full px-3 py-1.5 font-mono text-[11px] transition-colors duration-300 sm:px-4 sm:text-xs ${
                      mode === tab.id ? tab.on : 'text-fg-3 hover:text-fg'
                    }`}
                  >
                    <span className="hidden sm:inline">{tab.prefix}</span>
                    {tab.label}
                  </button>
                ))}
              </div>
              <div className="ml-auto flex items-center gap-2">
                <Badge label="v2 (legacy) · Apple M2" variant="demo" className="hidden sm:inline-flex" />
                <button
                  onClick={play}
                  className="group inline-flex items-center gap-1.5 rounded-full border border-line-2 px-3 py-1.5 font-mono text-[11px] text-fg-2 transition-colors hover:border-white/25 hover:text-fg"
                >
                  <span className="inline-block transition-transform duration-500 group-hover:-rotate-180">↺</span> Replay
                </button>
              </div>
            </div>

            {/* Chart */}
            <div className="px-4 pb-5 pt-6 sm:px-5">
              <div className="mb-3 grid grid-cols-[52px_1fr] gap-3 sm:grid-cols-[60px_1fr_104px]">
                <span />
                <div className="relative h-4 font-mono text-[9.5px] text-fg-4">
                  {[0, 2000, 4000, 6000, 8000].map((ms) => (
                    <span key={ms} className="absolute -translate-x-1/2 first:translate-x-0" style={{ left: `${pct(ms)}%` }}>
                      {ms / 1000}s
                    </span>
                  ))}
                </div>
              </div>

              <div className="relative">
                {bars.map((bar, i) => (
                  <TimelineRow key={bar.reqId} bar={bar} t={t} color={REQ_COLORS[i]} />
                ))}

                {/* Playhead */}
                <div className="pointer-events-none absolute inset-y-0 left-[64px] right-0 sm:left-[72px] sm:right-[116px]">
                  <div
                    className="absolute -bottom-1 -top-1 w-px bg-white/40"
                    style={{ left: `${pct(t)}%`, opacity: t >= SCALE_MS ? 0 : 1 }}
                  >
                    <span className="absolute -bottom-6 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-full bg-white/10 px-1.5 py-0.5 font-mono text-[9.5px] tabular-nums text-fg">
                      {(t / 1000).toFixed(2)}s
                    </span>
                  </div>
                </div>
              </div>

              <div className="mt-9 flex flex-wrap items-center justify-between gap-3">
                <div className="flex flex-wrap gap-4 font-mono text-[10.5px] text-fg-3">
                  <span className="flex items-center gap-1.5"><span className="hatch h-3 w-5 rounded-sm border border-line" />queued</span>
                  <span className="flex items-center gap-1.5"><span className="h-3 w-5 rounded-sm bg-amber/30" />prefill</span>
                  <span className="flex items-center gap-1.5"><span className="h-3 w-5 rounded-sm bg-mint/40" />decoding</span>
                  <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-mint" />first token</span>
                </div>
                <p
                  className={`font-mono text-[11px] transition-all duration-500 ${finished ? 'opacity-100' : 'translate-y-1 opacity-0'} ${
                    mode === 'phase1' ? 'text-rose' : 'text-mint'
                  }`}
                >
                  {mode === 'phase1'
                    ? `Req 4 waited ${fmtMs(PHASE1_TIMELINE[3].ttftMs)} for its first token.`
                    : `All 4 finished at ${fmtMs(allDoneAt)} — first token in ${TTFT_BATCHED_UNDER_LOAD_MS} ms.`}
                </p>
              </div>
            </div>
          </div>
        </Reveal>

        {/* v3 comparison cards — local smoke runs */}
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
          {V3_CARDS.map((s, i) => (
            <Reveal key={s.label} delay={i * 90}>
              <StatDiff {...s} />
            </Reveal>
          ))}
        </div>
        <p className="mt-2 font-mono text-[10.5px] text-fg-4">
          v3 cards: small local smoke runs · Apple M2 (MPS) · Qwen2-0.5B fp16 · output tokens/s
        </p>

        <Reveal>
          <AddRequestDemo mode={mode} />
        </Reveal>
      </div>
    </section>
  );
}

const tokS = (v: number | null) => (v === null ? '–' : `${v.toFixed(1)} tok/s`);
const times = (v: number | null) => (v === null ? '–' : `${v.toFixed(1)}×`);

const V3_CARDS = [
  { label: 'v3 · 8 concurrent × 64 tokens', cmp: compareSystems('m2-mps', 'concurrency/concurrent', 'hf_sequential', 'engine'), beforeLabel: 'HF sequential' },
  { label: 'v3 · vs HF sequential (16 req)', cmp: compareSystems('m2-mps', 'batching/random', 'hf_sequential', 'engine'), beforeLabel: 'HF sequential' },
  { label: 'v3 · vs HF static batching', cmp: compareSystems('m2-mps', 'batching/random', 'hf_static_batch', 'engine'), beforeLabel: 'static batching' },
].map((c) => ({ label: c.label, before: `${c.beforeLabel} ${tokS(c.cmp.from)}`, after: tokS(c.cmp.to), delta: times(c.cmp.ratio) }));

function StatDiff({ label, before, after, delta }: { label: string; before: string; after: string; delta: string }) {
  return (
    <Card className="h-full">
      <p className="mb-5 font-mono text-[10px] uppercase tracking-[0.16em] text-fg-3">{label}</p>
      <div className="flex items-end justify-between gap-3">
        <div>
          <p className="font-mono text-[12px] text-fg-4 line-through decoration-rose/60">{before}</p>
          <p className="mt-1 text-2xl font-semibold tracking-tight text-fg">{after}</p>
        </div>
        <span className="rounded-full bg-mint/10 px-2.5 py-1 font-mono text-xs font-medium text-mint">{delta}</span>
      </div>
    </Card>
  );
}

// ── "Add Request" interactive demo ────────────────────────────────────────────

type ReqState = 'waiting' | 'admitted' | 'decoding' | 'done';
const STEPS: { id: ReqState; label: string; color: string }[] = [
  { id: 'waiting', label: 'Waiting', color: '#FBBF24' },
  { id: 'admitted', label: 'Admitted', color: '#60A5FA' },
  { id: 'decoding', label: 'Decoding', color: '#4ADE80' },
  { id: 'done', label: 'Done', color: '#A6A6B0' },
];

function AddRequestDemo({ mode }: { mode: Mode }) {
  const [req, setReq] = useState<{ id: string; state: ReqState } | null>(null);
  const [step, setStep] = useState(0);

  const inject = () => {
    const id = Math.random().toString(36).slice(2, 6);
    setReq({ id, state: 'waiting' });
    setStep(1);
  };

  // Reset when switching modes so the demo always matches the chart.
  useEffect(() => { setReq(null); setStep(0); }, [mode]);

  useEffect(() => {
    if (!req || step === 0) return;
    const delays = mode === 'phase1' ? [0, 5000, 600, 2400] : [0, 400, 800, 1800];
    const timer = setTimeout(() => {
      if (step === 1) setReq((s) => s && { ...s, state: 'admitted' });
      if (step === 2) setReq((s) => s && { ...s, state: 'decoding' });
      if (step === 3) {
        setReq((s) => s && { ...s, state: 'done' });
        setTimeout(() => { setReq(null); setStep(0); }, 1600);
      }
      setStep((s) => s + 1);
    }, delays[step]);
    return () => clearTimeout(timer);
  }, [step, req, mode]);

  const activeIdx = req ? STEPS.findIndex((s) => s.id === req.state) : -1;
  const hint = !req
    ? mode === 'phase1'
      ? 'In sequential mode it has to wait for the whole running generation.'
      : 'In batched mode it slides into the running batch at the next iteration.'
    : req.state === 'waiting' && mode === 'phase1'
    ? 'Blocked behind asyncio.Lock… still waiting…'
    : req.state === 'admitted'
    ? 'Interleaved with the running batch.'
    : req.state === 'decoding'
    ? 'Streaming tokens.'
    : req.state === 'done'
    ? 'Finished — blocks returned to the pool.'
    : 'Queued for the next iteration.';

  return (
    <div className="mt-4 flex flex-col gap-4 rounded-2xl border border-dashed border-line-2 p-4 sm:flex-row sm:items-center sm:p-5">
      <div className="min-w-0 flex-1">
        <p className="text-[14px] font-medium text-fg">
          Try it: inject a 5th request{req && <span className="ml-2 font-mono text-[11px] text-fg-3">seq-{req.id}</span>}
        </p>
        <p className="mt-1 text-[13px] text-fg-3">{hint}</p>
        <div className="mt-3 flex items-center gap-1.5">
          {STEPS.map((s, i) => {
            const reached = i <= activeIdx;
            const current = i === activeIdx;
            return (
              <div key={s.id} className="flex items-center gap-1.5">
                <span
                  className="rounded-full border px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.1em] transition-all duration-500"
                  style={{
                    color: reached ? s.color : '#52525B',
                    borderColor: reached ? `${s.color}55` : 'rgba(255,255,255,0.07)',
                    backgroundColor: current ? `${s.color}18` : 'transparent',
                    boxShadow: current ? `0 0 16px -2px ${s.color}66` : 'none',
                  }}
                >
                  {s.label}
                </span>
                {i < STEPS.length - 1 && (
                  <span className="h-px w-3 transition-colors duration-500 sm:w-5" style={{ backgroundColor: i < activeIdx ? s.color : 'rgba(255,255,255,0.1)' }} />
                )}
              </div>
            );
          })}
        </div>
      </div>
      <button
        onClick={inject}
        disabled={!!req}
        className="group inline-flex shrink-0 items-center justify-center gap-2 rounded-full bg-fg px-5 py-2.5 text-sm font-medium text-ink-950 transition-all duration-300 hover:-translate-y-0.5 hover:shadow-[0_10px_30px_-10px_rgba(255,255,255,0.5)] disabled:translate-y-0 disabled:cursor-not-allowed disabled:opacity-30 disabled:shadow-none"
      >
        <span className="text-base leading-none transition-transform duration-300 group-hover:rotate-90">+</span>
        Add request
      </button>
    </div>
  );
}
