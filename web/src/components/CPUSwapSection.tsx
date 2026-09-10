'use client';

import { useState, useEffect, useRef } from 'react';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { CPU_SWAP_EVENT_LOG, type SwapEventLogEntry } from '@/data/simulation';
import { ENGINE_CONFIG } from '@/data/benchmarks';

const EVENT_COLORS: Record<SwapEventLogEntry['type'], string> = {
  admit:    '#60A5FA',
  oom:      '#F87171',
  swap_out: '#FBBF24',
  alloc:    '#4ADE80',
  resume:   '#a78bfa',
  decode:   '#4ADE80',
  done:     '#555555',
};

const EVENT_PREFIX: Record<SwapEventLogEntry['type'], string> = {
  admit:    '[ADMIT  ]',
  oom:      '[OOM    ]',
  swap_out: '[SWAPOUT]',
  alloc:    '[ALLOC  ]',
  resume:   '[SWAP IN]',
  decode:   '[DECODE ]',
  done:     '[DONE   ]',
};

export function CPUSwapSection() {
  const [visibleCount, setVisibleCount] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [gpuFill, setGpuFill] = useState(60);       // % of GPU blocks used
  const [cpuFill, setCpuFill] = useState(0);        // % of CPU blocks used
  const logRef = useRef<HTMLDivElement>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const stepRef = useRef(0);

  const startReplay = () => {
    setVisibleCount(0);
    setGpuFill(60);
    setCpuFill(0);
    stepRef.current = 0;
    setIsRunning(true);
  };

  useEffect(() => {
    if (!isRunning) return;

    const events = CPU_SWAP_EVENT_LOG;
    intervalRef.current = setInterval(() => {
      const idx = stepRef.current;
      if (idx >= events.length) {
        clearInterval(intervalRef.current!);
        setIsRunning(false);
        return;
      }

      const ev = events[idx];
      setVisibleCount(idx + 1);

      // Update fill animations based on event type
      if (ev.type === 'oom')      setGpuFill(98);
      if (ev.type === 'swap_out') { setGpuFill(65); setCpuFill(40); }
      if (ev.type === 'alloc')    setGpuFill(72);
      if (ev.type === 'decode')   setGpuFill(75);
      if (ev.type === 'resume')   { setCpuFill(0); setGpuFill(68); }
      if (ev.type === 'done')     setGpuFill(60);

      stepRef.current += 1;

      // Auto-scroll log
      if (logRef.current) {
        logRef.current.scrollTop = logRef.current.scrollHeight;
      }
    }, 600);

    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [isRunning]);

  return (
    <section id="swapping" className="border-b border-[#1e1e1e] px-6 py-20 sm:px-10 lg:px-16">
      <div className="mx-auto max-w-5xl">
        <SectionHeader
          label="// phase 9: cpu swap manager"
          title="CPU Swap Pool"
          subtitle="GPU blocks exhausted? Preempt the largest sequence, not the request."
        />

        <div className="grid gap-6 lg:grid-cols-2">
          {/* ── Left: Animated memory zones ─────────────────── */}
          <div className="border border-[#2a2a2a] bg-[#0d0d0d]">
            <div className="flex items-center justify-between border-b border-[#1e1e1e] px-4 py-2.5">
              <span className="font-mono text-[10px] uppercase tracking-widest text-[#333333]">
                Burst Scenario Replay
              </span>
              <Badge label="Demo Data" variant="demo" />
            </div>
            <div className="p-5">
              {/* GPU pool */}
              <div className="mb-6">
                <div className="mb-2 flex justify-between font-mono text-[11px]">
                  <span className="text-[#888888]">
                    GPU Block Pool ({ENGINE_CONFIG.kv_num_blocks} blocks)
                  </span>
                  <span style={{ color: gpuFill > 90 ? '#F87171' : '#4ADE80' }}>
                    {Math.round(gpuFill)}% used
                  </span>
                </div>
                <div className="h-8 w-full" style={{ backgroundColor: '#0d0d0d', border: '1px solid #1e1e1e' }}>
                  <div
                    className="h-full transition-all duration-500"
                    style={{
                      width: `${gpuFill}%`,
                      backgroundColor: gpuFill > 90 ? '#3d1a1a' : '#1a3d27',
                    }}
                  />
                </div>
                <div className="mt-1 flex justify-between font-mono text-[9px] text-[#333333]">
                  <span>0</span>
                  <span>{ENGINE_CONFIG.kv_num_blocks} blocks</span>
                </div>
              </div>

              {/* Transfer arrows */}
              <div className="mb-6 flex items-center justify-center gap-4">
                <div className="text-center">
                  <p className="font-mono text-[9px] text-[#F87171]">swap_out ▼</p>
                  <p className="font-mono text-[8px] text-[#333333]">(OOM pressure)</p>
                </div>
                <div
                  className="h-0.5 flex-1"
                  style={{
                    backgroundColor: cpuFill > 0 ? '#F87171' : '#222222',
                    transition: 'background-color 0.5s',
                  }}
                />
                <div className="h-4 w-4 flex-shrink-0 font-mono text-[11px] text-[#333333]">⇄</div>
                <div
                  className="h-0.5 flex-1"
                  style={{
                    backgroundColor: cpuFill > 0 ? '#a78bfa' : '#222222',
                    transition: 'background-color 0.5s',
                  }}
                />
                <div className="text-center">
                  <p className="font-mono text-[9px] text-[#a78bfa]">▲ swap_in</p>
                  <p className="font-mono text-[8px] text-[#333333]">(blocks freed)</p>
                </div>
              </div>

              {/* CPU pool */}
              <div className="mb-5">
                <div className="mb-2 flex justify-between font-mono text-[11px]">
                  <span className="text-[#888888]">
                    CPU Staging Pool ({ENGINE_CONFIG.kv_num_cpu_blocks} blocks)
                  </span>
                  <span style={{ color: cpuFill > 0 ? '#FBBF24' : '#444444' }}>
                    {Math.round(cpuFill)}% used
                  </span>
                </div>
                <div className="h-8 w-full" style={{ backgroundColor: '#0d0d0d', border: '1px solid #1e1e1e' }}>
                  <div
                    className="h-full transition-all duration-500"
                    style={{
                      width: `${cpuFill}%`,
                      backgroundColor: '#3d2e0a',
                    }}
                  />
                </div>
                <div className="mt-1 flex justify-between font-mono text-[9px] text-[#333333]">
                  <span>0</span>
                  <span>{ENGINE_CONFIG.kv_num_cpu_blocks} blocks</span>
                </div>
              </div>

              <button
                onClick={startReplay}
                disabled={isRunning}
                className="w-full border border-[#2a2a2a] py-2 font-mono text-xs text-[#666666] transition-colors hover:border-[#3a3a3a] hover:text-[#888888] disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isRunning ? '● Replaying…' : '↺ Replay Burst Scenario'}
              </button>
            </div>
          </div>

          {/* ── Right: Event log + decision cards ────────────── */}
          <div className="flex flex-col gap-4">
            {/* Event log */}
            <div className="border border-[#2a2a2a] bg-[#0d0d0d]">
              <div className="border-b border-[#1e1e1e] px-4 py-2.5">
                <span className="font-mono text-[10px] uppercase tracking-widest text-[#333333]">
                  Scheduler Event Log
                </span>
              </div>
              <div
                ref={logRef}
                className="log-panel h-48 overflow-y-auto p-3"
              >
                {visibleCount === 0 && (
                  <p className="font-mono text-[10px] text-[#2a2a2a]">
                    Press Replay to start burst scenario…
                  </p>
                )}
                {CPU_SWAP_EVENT_LOG.slice(0, visibleCount).map((ev, i) => (
                  <div key={i} className="mb-0.5 flex items-start gap-2">
                    <span
                      className="flex-shrink-0 font-mono text-[10px]"
                      style={{ color: EVENT_COLORS[ev.type] }}
                    >
                      {EVENT_PREFIX[ev.type]}
                    </span>
                    <span className="font-mono text-[10px] text-[#666666]">
                      {ev.message}
                    </span>
                    {ev.blocks && (
                      <span className="ml-auto flex-shrink-0 font-mono text-[9px] text-[#444444]">
                        {ev.blocks}blk
                      </span>
                    )}
                  </div>
                ))}
                {/* Cursor blink when running */}
                {isRunning && (
                  <span className="animate-blink font-mono text-[11px] text-[#4ADE80]">█</span>
                )}
              </div>
            </div>

            {/* Without vs With */}
            <div className="grid grid-cols-2 gap-3">
              <Card className="border-[#F87171]/30">
                <p className="mb-2 font-mono text-[9px] uppercase tracking-widest text-[#F87171]">
                  Without Swap
                </p>
                <ul className="space-y-1">
                  {['HTTP 500 on OOM', 'Request dropped', 'GPU crash', 'Data lost'].map((l) => (
                    <li key={l} className="font-mono text-[10px] text-[#555555]">
                      ✗ {l}
                    </li>
                  ))}
                </ul>
              </Card>
              <Card className="border-[#4ADE80]/20">
                <p className="mb-2 font-mono text-[9px] uppercase tracking-widest text-[#4ADE80]">
                  With PageServe
                </p>
                <ul className="space-y-1">
                  {['Victim preempted', 'KV → CPU RAM', 'New req admitted', 'Victim resumes'].map((l) => (
                    <li key={l} className="font-mono text-[10px] text-[#555555]">
                      ✓ {l}
                    </li>
                  ))}
                </ul>
              </Card>
            </div>

            {/* Selection policy */}
            <div className="border-l-2 border-[#FBBF24] bg-[#111111] px-4 py-3">
              <p className="font-mono text-[10px] uppercase tracking-widest text-[#FBBF24]">
                Victim Selection Policy
              </p>
              <p className="mt-1 font-mono text-[11px] leading-relaxed text-[#666666]">
                Pick the sequence with the{' '}
                <span className="text-[#e8e8e8]">most allocated blocks</span> (not LRU). Freeing one
                large sequence is more efficient than evicting several small ones.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
