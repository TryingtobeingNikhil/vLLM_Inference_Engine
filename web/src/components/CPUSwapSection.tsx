'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { SectionHeader, Accent } from '@/components/ui/SectionHeader';
import { Badge } from '@/components/ui/Badge';
import { Card, Window } from '@/components/ui/Card';
import { Reveal } from '@/components/ui/Reveal';
import { SEQ_COLORS, useInView, usePrefersReducedMotion } from '@/lib/motion';
import { CPU_SWAP_EVENT_LOG, type SwapEventLogEntry } from '@/data/simulation';
import { ENGINE_CONFIG } from '@/data/benchmarks';

const EVENT_COLORS: Record<SwapEventLogEntry['type'], string> = {
  admit:    '#60A5FA',
  oom:      '#FB7185',
  swap_out: '#FBBF24',
  alloc:    '#4ADE80',
  resume:   '#A78BFA',
  decode:   '#4ADE80',
  done:     '#A6A6B0',
};

const EVENT_PREFIX: Record<SwapEventLogEntry['type'], string> = {
  admit:    'ADMIT',
  oom:      'OOM',
  swap_out: 'SWAPOUT',
  alloc:    'ALLOC',
  resume:   'SWAP IN',
  decode:   'DECODE',
  done:     'DONE',
};

// Pools drawn at 1 cell = 8 blocks: 32 GPU cells = 256 blocks, 16 CPU cells = 128 blocks.
const GPU_CELLS = ENGINE_CONFIG.kv_num_blocks / 8;
const CPU_CELLS = ENGINE_CONFIG.kv_num_cpu_blocks / 8;

interface PoolFrame {
  gpu: (string | null)[];
  cpu: (string | null)[];
  ghost: number;          // cells requested by the incoming sequence that don't fit
  victim: boolean;
  transfer: 'down' | 'up' | null;
}

function fill(arr: (string | null)[], from: number, n: number, owner: string) {
  for (let i = from; i < from + n && i < arr.length; i++) arr[i] = owner;
}

/** Pool state after the first `k` log events have played — mirrors the block counts in the log. */
function frameFor(k: number): PoolFrame {
  const gpu: (string | null)[] = Array(GPU_CELLS).fill(null);
  const cpu: (string | null)[] = Array(CPU_CELLS).fill(null);
  fill(gpu, 0, 8, 'a1b2');
  fill(gpu, 8, 7, 'g7h8');
  let ghost = 0, victim = false, transfer: PoolFrame['transfer'] = null;

  if (k < 5 || k === 8) fill(gpu, 15, 12, 'c3d4');       // victim resident on GPU
  if (k === 3) ghost = 7;                                 // OOM: 7 requested, 5 free
  if (k === 4) victim = true;
  if (k >= 5 && k < 8) { fill(cpu, 0, 12, 'c3d4'); }      // parked in host RAM
  if (k === 5) transfer = 'down';
  if (k === 6) fill(gpu, 15, 7, 'e5f6');
  if (k === 7) fill(gpu, 15, 8, 'e5f6');
  if (k === 8) transfer = 'up';
  if (k === 9) fill(gpu, 15, 13, 'c3d4');                 // k ≥ 10: finished, blocks freed
  return { gpu, cpu, ghost, victim, transfer };
}

function Pool({ cells, label, capacity, ghost = 0, victim = false }: { cells: (string | null)[]; label: string; capacity: number; ghost?: number; victim?: boolean }) {
  const used = cells.filter(Boolean).length;
  const pct = Math.round((used / cells.length) * 100);
  const free = cells.length - used;
  const hot = ghost > 0;
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between font-mono text-[11px]">
        <span className="text-fg-2">{label} <span className="text-fg-4">· {capacity} blocks</span></span>
        <span className="tabular-nums transition-colors duration-300" style={{ color: hot ? '#FB7185' : pct > 0 ? '#EDEDEF' : '#52525B' }}>
          {hot ? 'OUT OF BLOCKS' : `${pct}% used`}
        </span>
      </div>
      <div
        className={`grid gap-[3px] rounded-xl border p-2 transition-colors duration-300 ${hot ? 'border-rose/50 bg-rose/[0.06]' : 'border-line bg-ink-950'}`}
        style={{ gridTemplateColumns: `repeat(${Math.min(cells.length, 16)}, 1fr)` }}
      >
        {cells.map((owner, i) => {
          const isGhost = !owner && hot && i >= used && i < used + Math.min(ghost, free);
          const isVictim = victim && owner === 'c3d4';
          const color = owner ? SEQ_COLORS[owner] : null;
          return (
            <div
              key={i}
              className={`aspect-square rounded-[4px] transition-all duration-500 ${isVictim ? 'animate-pulse' : ''}`}
              style={{
                backgroundColor: color ? `${color}${isVictim ? '55' : '40'}` : isGhost ? 'rgba(251,113,133,0.18)' : 'rgba(255,255,255,0.03)',
                boxShadow: color
                  ? `inset 0 0 0 1px ${isVictim ? '#FB7185' : `${color}99`}`
                  : isGhost
                  ? 'inset 0 0 0 1px rgba(251,113,133,0.7)'
                  : 'inset 0 0 0 1px rgba(255,255,255,0.04)',
                transform: color ? 'scale(1)' : 'scale(0.86)',
              }}
            />
          );
        })}
      </div>
      {hot && (
        <p className="mt-1.5 text-right font-mono text-[10px] text-rose">seq-e5f6 needs 7 · only {free} free</p>
      )}
    </div>
  );
}

export function CPUSwapSection() {
  const [visibleCount, setVisibleCount] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);
  const [ref, inView] = useInView<HTMLDivElement>({ threshold: 0.35 });
  const reduced = usePrefersReducedMotion();

  const startReplay = useCallback(() => {
    if (reduced) { setVisibleCount(CPU_SWAP_EVENT_LOG.length); return; }
    setVisibleCount(0);
    setIsRunning(true);
  }, [reduced]);

  // Autoplay the first time the scenario scrolls into view.
  useEffect(() => { if (inView) startReplay(); }, [inView, startReplay]);

  useEffect(() => {
    if (!isRunning) return;
    if (visibleCount >= CPU_SWAP_EVENT_LOG.length) {
      setIsRunning(false);
      return;
    }
    const id = setTimeout(() => setVisibleCount((n) => n + 1), visibleCount === 0 ? 300 : 900);
    return () => clearTimeout(id);
  }, [isRunning, visibleCount]);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }, [visibleCount]);

  const frame = frameFor(visibleCount);
  const done = visibleCount >= CPU_SWAP_EVENT_LOG.length;

  return (
    <section id="swapping" className="relative px-5 py-24 sm:px-6 sm:py-32">
      <div className="mx-auto max-w-5xl">
        <SectionHeader
          index="04"
          label="Phase 9 · CPU swap manager"
          title={<>Out of blocks? <Accent gradient>Swap, don&apos;t drop.</Accent></>}
          subtitle="When GPU blocks run out, PageServe preempts the largest sequence and parks its KV-cache in host RAM — instead of failing the request."
        />

        <div ref={ref} className="grid gap-5 lg:grid-cols-2 [&>*]:min-w-0">
          {/* ── Memory pools ─────────────────── */}
          <Reveal>
            <Window
              className="flex h-full flex-col"
              bodyClassName="flex flex-1 flex-col"
              title={<>burst scenario <span className="text-fg-4">— replay</span></>}
              right={<Badge label="Demo data" variant="demo" />}
            >
              <div className="flex flex-1 flex-col p-5">
                <Pool cells={frame.gpu} label="GPU block pool" capacity={ENGINE_CONFIG.kv_num_blocks} ghost={frame.ghost} victim={frame.victim} />

                {/* Transfer lane */}
                <div className="relative my-3 flex h-14 items-center justify-center">
                  <div className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-gradient-to-b from-line-2 via-line to-line-2" />
                  {frame.transfer &&
                    Array.from({ length: 4 }).map((_, i) => (
                      <span
                        key={`${frame.transfer}-${visibleCount}-${i}`}
                        className="absolute left-1/2 top-0 h-2 w-2 -translate-x-1/2 rounded-[2px] bg-sky"
                        style={{
                          marginLeft: (i - 1.5) * 12,
                          animation: `${frame.transfer === 'down' ? 'swap-down' : 'swap-up'} 0.9s var(--ease-out) ${i * 0.08}s both`,
                          boxShadow: '0 0 8px #60A5FA',
                        }}
                      />
                    ))}
                  <span
                    className={`relative rounded-full border px-3 py-1 font-mono text-[10.5px] transition-all duration-300 ${
                      frame.transfer === 'down'
                        ? 'border-amber/40 bg-ink-900 text-amber'
                        : frame.transfer === 'up'
                        ? 'border-violet/40 bg-ink-900 text-violet'
                        : 'border-line bg-ink-900 text-fg-4'
                    }`}
                  >
                    {frame.transfer === 'down' ? '▼ swap_out · 12 blocks' : frame.transfer === 'up' ? '▲ swap_in · 12 blocks' : 'swap_out ▼ ▲ swap_in'}
                  </span>
                </div>

                <Pool cells={frame.cpu} label="CPU staging pool" capacity={ENGINE_CONFIG.kv_num_cpu_blocks} />

                <p className="mb-5 mt-3 font-mono text-[10px] text-fg-4">1 cell = 8 blocks</p>

                <button
                  onClick={startReplay}
                  disabled={isRunning}
                  className="group mt-auto flex w-full items-center justify-center gap-2 rounded-full border border-line-2 bg-white/[0.02] py-2.5 font-mono text-xs text-fg-2 transition-all duration-300 hover:border-white/25 hover:text-fg disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isRunning ? (
                    <><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-mint" /> Replaying… {visibleCount}/{CPU_SWAP_EVENT_LOG.length}</>
                  ) : (
                    <><span className="inline-block transition-transform duration-500 group-hover:-rotate-180">↺</span> Replay burst scenario</>
                  )}
                </button>
              </div>
            </Window>
          </Reveal>

          {/* ── Log + policy ─────────────────── */}
          <div className="flex flex-col gap-5">
            <Reveal delay={80}>
              <Window title={<>scheduler.log <span className="text-fg-4">— tail -f</span></>}>
                <div ref={logRef} className="h-56 overflow-y-auto px-4 py-3 font-mono text-[11px] leading-[1.9]">
                  {visibleCount === 0 && <p className="text-fg-4">$ waiting for burst…</p>}
                  {CPU_SWAP_EVENT_LOG.slice(0, visibleCount).map((ev, i) => (
                    <div key={i} className="flex items-start gap-3 [animation:log-in_0.4s_var(--ease-out)_both]">
                      <span className="shrink-0 tabular-nums text-fg-4">+{String(ev.timeMs).padStart(4, '0')}ms</span>
                      <span className="w-[58px] shrink-0 font-medium" style={{ color: EVENT_COLORS[ev.type] }}>
                        {EVENT_PREFIX[ev.type]}
                      </span>
                      <span className="min-w-0 flex-1 text-fg-2">{ev.message}</span>
                    </div>
                  ))}
                  {isRunning && <span className="animate-blink text-mint">▍</span>}
                  {done && !isRunning && (
                    <p className="mt-1 text-mint">✓ 0 requests dropped · 0 HTTP 500s</p>
                  )}
                </div>
              </Window>
            </Reveal>

            <div className="grid grid-cols-2 gap-3">
              <Reveal delay={140}>
                <Card className="h-full" glow={false}>
                  <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.14em] text-rose">Without swap</p>
                  <ul className="space-y-1.5">
                    {['HTTP 500 on OOM', 'Request dropped', 'GPU crash', 'Work lost'].map((l) => (
                      <li key={l} className="flex items-center gap-2 text-[13px] text-fg-3">
                        <span className="text-rose">✕</span> {l}
                      </li>
                    ))}
                  </ul>
                </Card>
              </Reveal>
              <Reveal delay={200}>
                <Card className="h-full border-mint/20 bg-mint/[0.03]" glow={false}>
                  <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.14em] text-mint">With PageServe</p>
                  <ul className="space-y-1.5">
                    {['Victim preempted', 'KV → CPU RAM', 'New req admitted', 'Victim resumes'].map((l) => (
                      <li key={l} className="flex items-center gap-2 text-[13px] text-fg-2">
                        <span className="text-mint">✓</span> {l}
                      </li>
                    ))}
                  </ul>
                </Card>
              </Reveal>
            </div>

            <Reveal delay={240}>
              <div className="flex gap-4 rounded-2xl border border-amber/20 bg-gradient-to-r from-amber/[0.07] to-transparent p-5">
                <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-amber/15 font-mono text-sm text-amber">⚖</span>
                <div>
                  <p className="text-[14px] font-medium text-fg">Victim selection: biggest first, not LRU</p>
                  <p className="mt-1 text-[13.5px] leading-relaxed text-fg-2">
                    Pick the sequence holding the <span className="text-fg">most allocated blocks</span>. Freeing one large
                    sequence is cheaper than evicting several small ones.
                  </p>
                </div>
              </div>
            </Reveal>
          </div>
        </div>
      </div>
    </section>
  );
}
