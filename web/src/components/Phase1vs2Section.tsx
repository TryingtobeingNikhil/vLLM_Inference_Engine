'use client';

import { useState, useEffect, useRef } from 'react';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { Badge } from '@/components/ui/Badge';
import { PHASE1_TIMELINE, PHASE2_TIMELINE, type TimelineBar } from '@/data/simulation';
import { TTFT_SEQUENTIAL_UNDER_LOAD_MS, TTFT_BATCHED_UNDER_LOAD_MS } from '@/data/benchmarks';

type Mode = 'phase1' | 'phase2';

// Scale timeline bars to fit in the visual width
// Phase 1 max = ~9161ms; Phase 2 max = ~541ms.
// We want both to use the same relative scale (total wall-clock time)
const P1_TOTAL_WALL = 9161;
const P2_TOTAL_WALL = 3990;   // actual 4-way wall clock from benchmark data
// Use the Phase 1 duration as the shared scale so comparison is visually honest
const SCALE_MS = P1_TOTAL_WALL;

function pct(ms: number) {
  return `${Math.max(0, Math.min(100, (ms / SCALE_MS) * 100)).toFixed(2)}%`;
}

function TimelineRow({
  bar,
  animate,
  showTtftLabel,
}: {
  bar: TimelineBar;
  animate: boolean;
  showTtftLabel: boolean;
}) {
  const waitPct = (bar.waitStartMs / SCALE_MS) * 100;
  const prefillPct = ((bar.prefillStartMs - bar.waitStartMs) / SCALE_MS) * 100;
  const decodePct = ((bar.decodeStartMs - bar.prefillStartMs) / SCALE_MS) * 100;
  const runPct = ((bar.finishMs - bar.decodeStartMs) / SCALE_MS) * 100;
  const ttftLinePct = (bar.decodeStartMs / SCALE_MS) * 100;

  return (
    <div className="flex items-center gap-3 py-1">
      <span className="w-12 font-mono text-[11px] text-[#555555]">{bar.label}</span>
      <div className="relative h-6 flex-1" style={{ backgroundColor: '#0d0d0d', border: '1px solid #1a1a1a' }}>
        {/* Waiting/queued (amber dim) */}
        {prefillPct > 0.1 && (
          <div
            className={`absolute inset-y-0 transition-all ${animate ? 'duration-[2000ms]' : 'duration-0'} ease-out`}
            style={{
              left: `${waitPct}%`,
              width: animate ? `${prefillPct}%` : '0%',
              backgroundColor: '#3d2e0a',
              opacity: 0.7,
            }}
          />
        )}
        {/* Prefill stage */}
        {decodePct > 0.1 && (
          <div
            className={`absolute inset-y-0 transition-all ${animate ? 'duration-[2000ms] delay-300' : 'duration-0'} ease-out`}
            style={{
              left: `${waitPct + prefillPct}%`,
              width: animate ? `${decodePct}%` : '0%',
              backgroundColor: '#3d2e0a',
            }}
          />
        )}
        {/* Decoding (green) */}
        <div
          className={`absolute inset-y-0 transition-all ${animate ? 'duration-[2500ms] delay-500' : 'duration-0'} ease-out`}
          style={{
            left: `${waitPct + prefillPct + decodePct}%`,
            width: animate ? `${runPct}%` : '0%',
            backgroundColor: '#1a3d27',
          }}
        />
        {/* TTFT line */}
        <div
          className="absolute inset-y-0 w-px"
          style={{ left: `${ttftLinePct}%`, backgroundColor: '#4ADE80', opacity: 0.7 }}
        />
      </div>
      {/* TTFT label */}
      <span className="w-24 text-right font-mono text-[10px] text-[#555555]">
        TTFT{' '}
        <span className={bar.ttftMs > 500 ? 'text-[#F87171]' : 'text-[#4ADE80]'}>
          {bar.ttftMs < 1000 ? `${bar.ttftMs.toFixed(0)}ms` : `${(bar.ttftMs / 1000).toFixed(2)}s`}
        </span>
      </span>
    </div>
  );
}

export function Phase1vs2Section() {
  const [mode, setMode] = useState<Mode>('phase1');
  const [animated, setAnimated] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);
  const sectionRef = useRef<HTMLDivElement>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);

  // Auto-animate once when section enters viewport
  useEffect(() => {
    observerRef.current = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && !hasStarted) {
          setHasStarted(true);
          setAnimated(true);
        }
      },
      { threshold: 0.3 },
    );
    if (sectionRef.current) observerRef.current.observe(sectionRef.current);
    return () => observerRef.current?.disconnect();
  }, [hasStarted]);

  const handleReplay = () => {
    setAnimated(false);
    setTimeout(() => setAnimated(true), 50);
  };

  const handleModeChange = (m: Mode) => {
    setMode(m);
    setAnimated(false);
    setTimeout(() => setAnimated(true), 50);
  };

  const bars = mode === 'phase1' ? PHASE1_TIMELINE : PHASE2_TIMELINE;

  return (
    <section
      id="comparison"
      ref={sectionRef}
      className="border-b border-[#1e1e1e] px-6 py-20 sm:px-10 lg:px-16"
    >
      <div className="mx-auto max-w-5xl">
        <SectionHeader
          label="// phase comparison"
          title="Sequential vs Continuous Batching"
          subtitle="4 concurrent requests, 50 tokens each. The wall clock scale is identical across both views."
        />

        {/* ── Tab switcher ─────────────────────────────────────── */}
        <div className="mb-6 flex items-center gap-0">
          {[
            { id: 'phase1' as Mode, label: 'Phase 1 — Sequential' },
            { id: 'phase2' as Mode, label: 'Phase 2–9 — Batched' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => handleModeChange(tab.id)}
              className={`border px-4 py-2 font-mono text-xs transition-colors ${
                mode === tab.id
                  ? 'border-[#4ADE80] bg-[#1a3d27] text-[#4ADE80]'
                  : 'border-[#2a2a2a] bg-[#111111] text-[#555555] hover:text-[#888888]'
              }`}
            >
              {tab.label}
            </button>
          ))}
          <div className="ml-auto flex items-center gap-3">
            <Badge label="Benchmark Replay" variant="demo" />
            <button
              onClick={handleReplay}
              className="border border-[#2a2a2a] px-3 py-1.5 font-mono text-[11px] text-[#555555] transition-colors hover:text-[#888888]"
            >
              ↺ Replay
            </button>
          </div>
        </div>

        {/* ── Timeline chart ───────────────────────────────────── */}
        <div className="border border-[#2a2a2a] bg-[#0d0d0d] p-4">
          {/* Legend */}
          <div className="mb-4 flex flex-wrap gap-4">
            <LegendItem color="#3d2e0a" label="Waiting / Prefill" />
            <LegendItem color="#1a3d27" label="Decoding" />
            <LegendItem color="#4ADE80" label="TTFT marker" line />
          </div>

          {/* Scale label */}
          <div className="mb-2 flex justify-between font-mono text-[9px] text-[#333333]">
            <span>0ms</span>
            <span>{mode === 'phase1' ? '~9,161ms (wall clock)' : '~9,161ms (same scale)'}</span>
          </div>

          {bars.map((bar, i) => (
            <TimelineRow key={bar.reqId} bar={bar} animate={animated} showTtftLabel={i === 0} />
          ))}

          {mode === 'phase1' && (
            <p className="mt-3 font-mono text-[10px] text-[#444444]">
              ↑ Requests 2–4 wait behind a complete generation. TTFT = wall-clock wait + prefill.
            </p>
          )}
          {mode === 'phase2' && (
            <p className="mt-3 font-mono text-[10px] text-[#4ADE80]/60">
              ↑ All 4 requests admitted concurrently. First token arrives within{' '}
              {TTFT_BATCHED_UNDER_LOAD_MS} ms for Req 1.
            </p>
          )}
        </div>

        {/* ── Stat comparison cards ────────────────────────────── */}
        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
          <StatDiff
            label="TTFT under 4-way load"
            before={`${TTFT_SEQUENTIAL_UNDER_LOAD_MS.toLocaleString()} ms`}
            after={`${TTFT_BATCHED_UNDER_LOAD_MS} ms`}
            delta="−98.5%"
            positive
          />
          <StatDiff
            label="Wall clock (all done)"
            before="5,673 ms"
            after="3,990 ms"
            delta="−29.6%"
            positive
          />
          <StatDiff
            label="Aggregate throughput"
            before="42.0 tok/s"
            after="50.1 tok/s"
            delta="+19.3%"
            positive
          />
        </div>

        {/* ── "Add Request" interactive demo ───────────────────── */}
        <AddRequestDemo mode={mode} />
      </div>
    </section>
  );
}

function LegendItem({
  color,
  label,
  line = false,
}: {
  color: string;
  label: string;
  line?: boolean;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <div
        className="h-3 w-5"
        style={{
          backgroundColor: line ? 'transparent' : color,
          border: line ? `1px solid ${color}` : 'none',
        }}
      />
      <span className="font-mono text-[10px] text-[#555555]">{label}</span>
    </div>
  );
}

function StatDiff({
  label,
  before,
  after,
  delta,
  positive,
}: {
  label: string;
  before: string;
  after: string;
  delta: string;
  positive: boolean;
}) {
  return (
    <div className="border border-[#2a2a2a] bg-[#111111] p-3">
      <p className="mb-2 font-mono text-[9px] uppercase tracking-widest text-[#444444]">{label}</p>
      <p className="font-mono text-[11px] text-[#555555] line-through">{before}</p>
      <p className="font-mono text-sm text-[#e8e8e8]">{after}</p>
      <p className={`mt-1 font-mono text-xs font-semibold ${positive ? 'text-[#4ADE80]' : 'text-[#F87171]'}`}>
        {delta}
      </p>
    </div>
  );
}

// ── "Add Request" interactive demo ────────────────────────────────────────────

interface InjectedReq {
  id: string;
  state: 'waiting' | 'admitted' | 'decoding' | 'done';
}

function AddRequestDemo({ mode }: { mode: Mode }) {
  const [injected, setInjected] = useState<InjectedReq | null>(null);
  const [step, setStep] = useState(0);

  const inject = () => {
    const id = Math.random().toString(36).slice(2, 6);
    setInjected({ id, state: 'waiting' });
    setStep(1);
  };

  // Advance state machine on each step
  useEffect(() => {
    if (!injected) return;
    if (step === 0) return;
    const delays =
      mode === 'phase1'
        ? [0, 0, 0, 5000] // Phase 1: stays waiting a long time
        : [0, 400, 1200, 3000]; // Phase 2: admitted quickly
    const timer = setTimeout(() => {
      if (step === 1) setInjected((s) => s && { ...s, state: 'admitted' });
      if (step === 2) setInjected((s) => s && { ...s, state: 'decoding' });
      if (step === 3) {
        setInjected((s) => s && { ...s, state: 'done' });
        setTimeout(() => { setInjected(null); setStep(0); }, 1500);
      }
      setStep((s) => s + 1);
    }, delays[step]);
    return () => clearTimeout(timer);
  }, [step, injected, mode]);

  const stateColors: Record<string, string> = {
    waiting:  '#FBBF24',
    admitted: '#60A5FA',
    decoding: '#4ADE80',
    done:     '#555555',
  };

  return (
    <div className="mt-4 border border-[#2a2a2a] bg-[#111111] p-4">
      <div className="flex items-center justify-between">
        <p className="font-mono text-[11px] text-[#555555]">
          Inject a synthetic 5th request →
        </p>
        <button
          onClick={inject}
          disabled={!!injected}
          className="border border-[#3a3a3a] px-3 py-1.5 font-mono text-[11px] text-[#888888] transition-colors hover:border-[#666666] hover:text-[#cccccc] disabled:cursor-not-allowed disabled:opacity-40"
        >
          + Add Request
        </button>
      </div>
      {injected && (
        <div className="mt-3 flex items-center gap-2 font-mono text-[11px]">
          <span className="text-[#444444]">seq-{injected.id}</span>
          <span className="text-[#333333]">→</span>
          <span style={{ color: stateColors[injected.state] }}>
            {injected.state.toUpperCase()}
          </span>
          {mode === 'phase1' && injected.state === 'waiting' && (
            <span className="text-[#444444]">
              &nbsp;(blocked behind active generation…)
            </span>
          )}
          {mode === 'phase2' && injected.state === 'admitted' && (
            <span className="text-[#444444]">
              &nbsp;(interleaved with running batch)
            </span>
          )}
        </div>
      )}
    </div>
  );
}
