'use client';

import { useEffect, useState } from 'react';
import { SectionHeader, Accent } from '@/components/ui/SectionHeader';
import { Card } from '@/components/ui/Card';
import { Reveal } from '@/components/ui/Reveal';
import { useInView, usePrefersReducedMotion } from '@/lib/motion';
import { ENGINE_CONFIG } from '@/data/benchmarks';

const LOOP_STEPS = [
  {
    step: '01',
    label: 'Admit',
    title: 'RequestQueue',
    lines: [
      `FIFO ordering`,
      `maxsize: ${ENGINE_CONFIG.max_batch_size * 8}`,
      `timeout: 30s`,
      `→ 429 when full`,
    ],
    color: '#60A5FA',
  },
  {
    step: '02',
    label: 'Prefill',
    title: 'Chunked Prefill',
    lines: [
      `budget: ${ENGINE_CONFIG.prefill_budget_tokens} tok/step`,
      `chunk: ${ENGINE_CONFIG.prefill_chunk_size} tok`,
      `allocate blocks`,
      `write KV pool`,
    ],
    color: '#FBBF24',
  },
  {
    step: '03',
    label: 'Decode',
    title: 'Decode Loop',
    lines: [
      `limit: ${ENGINE_CONFIG.decode_batch_limit} seqs`,
      `1 token per seq`,
      `live KV cache`,
      `~11 ms/step`,
    ],
    color: '#4ADE80',
  },
  {
    step: '04',
    label: 'Evict',
    title: 'Block Reclaim',
    lines: [
      `free block alloc`,
      `clear KV pool`,
      `update metrics`,
      `admit next seq`,
    ],
    color: '#A78BFA',
  },
];

function LoopSteps() {
  const [ref, inView] = useInView<HTMLDivElement>({ once: false, threshold: 0.3 });
  const reduced = usePrefersReducedMotion();
  const [active, setActive] = useState(0);

  useEffect(() => {
    if (!inView || reduced) return;
    const id = setInterval(() => setActive((a) => (a + 1) % LOOP_STEPS.length), 1500);
    return () => clearInterval(id);
  }, [inView, reduced]);

  return (
    <div ref={ref} className="relative mb-6">
      {/* The loop rail: a track the "iteration" pip travels along */}
      <div className="relative mb-4 hidden h-8 lg:block" aria-hidden="true">
        <div className="absolute inset-x-[12.5%] top-1/2 h-px -translate-y-1/2 bg-gradient-to-r from-sky/40 via-mint/40 to-violet/40" />
        <div
          className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full transition-all duration-700 [transition-timing-function:var(--ease-spring)]"
          style={{
            left: `${12.5 + active * 25}%`,
            backgroundColor: LOOP_STEPS[active].color,
            boxShadow: `0 0 0 4px ${LOOP_STEPS[active].color}26, 0 0 18px ${LOOP_STEPS[active].color}`,
          }}
        />
        <span className="absolute right-0 top-1/2 -translate-y-1/2 font-mono text-[10px] text-fg-4">↻ repeat every iteration</span>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {LOOP_STEPS.map((step, i) => {
          const on = i === active;
          return (
            <Reveal key={step.step} delay={i * 80}>
              <Card
                className="h-full transition-all duration-500"
                style={{
                  borderColor: on ? `${step.color}55` : undefined,
                  boxShadow: on ? `0 0 0 1px ${step.color}22, 0 20px 50px -24px ${step.color}66` : undefined,
                  transform: on ? 'translateY(-3px)' : undefined,
                }}
              >
                <div className="mb-5 flex items-center justify-between">
                  <span
                    className="font-mono text-[2.2rem] font-semibold leading-none tracking-tight transition-colors duration-500"
                    style={{ color: on ? step.color : 'transparent', WebkitTextStroke: `1px ${step.color}` }}
                  >
                    {step.step}
                  </span>
                  <span
                    className="rounded-full px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em]"
                    style={{ color: step.color, backgroundColor: `${step.color}14` }}
                  >
                    {step.label}
                  </span>
                </div>
                <p className="mb-3 text-[15px] font-medium text-fg">{step.title}</p>
                <ul className="space-y-1.5">
                  {step.lines.map((line) => (
                    <li key={line} className="flex items-center gap-2 font-mono text-[11.5px] text-fg-3">
                      <span className="h-1 w-1 rounded-full bg-fg-4" />
                      {line}
                    </li>
                  ))}
                </ul>
              </Card>
            </Reveal>
          );
        })}
      </div>
    </div>
  );
}

// ── Architecture diagram ──────────────────────────────────────────────────────

type Node = { x: number; y: number; w: number; label: string; sub: string; color?: string; strong?: boolean };
const H = 40;
const NODES: Record<string, Node> = {
  req:   { x: 12,  y: 130, w: 104, label: 'Request',        sub: 'POST /generate' },
  queue: { x: 150, y: 130, w: 150, label: 'RequestQueue',   sub: 'FIFO · 429 when full' },
  sched: { x: 334, y: 130, w: 156, label: 'Scheduler Loop', sub: 'asyncio task', strong: true },
  pre:   { x: 540, y: 40,  w: 156, label: 'Prefill Stage',  sub: '≤512 tok / iter', color: '#FBBF24' },
  dec:   { x: 540, y: 220, w: 156, label: 'Decode Stage',   sub: '1 tok / seq / iter', color: '#4ADE80' },
  kv:    { x: 746, y: 130, w: 168, label: 'PagedKVCache',   sub: '256 × 16-tok blocks', color: '#60A5FA' },
  cpu:   { x: 746, y: 250, w: 168, label: 'CPUSwapPool',    sub: '128 host blocks', color: '#FB7185' },
};

const EDGES = [
  { id: 'e1', d: 'M116,150 L148,150', color: '#767680' },
  { id: 'e2', d: 'M300,150 L332,150', color: '#767680' },
  { id: 'e3', d: 'M490,142 C515,142 512,60 538,60', color: '#FBBF24' },
  { id: 'e4', d: 'M490,158 C515,158 512,240 538,240', color: '#4ADE80' },
  { id: 'e5', d: 'M696,60 C724,60 718,140 744,140', color: '#767680' },
  { id: 'e6', d: 'M696,240 C724,240 718,160 744,160', color: '#767680' },
  { id: 'swapout', d: 'M810,170 L810,248', color: '#FB7185' },
  { id: 'swapin', d: 'M914,270 C952,270 952,150 916,150', color: '#FB7185' },
  { id: 'loop', d: 'M540,252 C470,300 420,260 412,172', color: '#A78BFA' },
];

// Packets that ride the edges — the request's journey through one iteration.
const PACKETS = [
  { path: 'M116,150 L332,150', color: '#EDEDEF', dur: 2.4, begin: 0 },
  { path: EDGES[2].d, color: '#FBBF24', dur: 1.6, begin: 0.4 },
  { path: EDGES[3].d, color: '#4ADE80', dur: 1.6, begin: 1.2 },
  { path: EDGES[4].d, color: '#FBBF24', dur: 1.6, begin: 1.6 },
  { path: EDGES[5].d, color: '#4ADE80', dur: 1.6, begin: 2.2 },
  { path: EDGES[8].d, color: '#A78BFA', dur: 2.2, begin: 0.8 },
  { path: EDGES[6].d, color: '#FB7185', dur: 2.6, begin: 1.0 },
  { path: EDGES[7].d, color: '#FB7185', dur: 2.6, begin: 2.3 },
];

function ArchitectureDiagram() {
  const reduced = usePrefersReducedMotion();
  const [ref, inView] = useInView<HTMLDivElement>({ once: false, threshold: 0.1 });

  return (
    <div ref={ref} className="overflow-x-auto">
      <svg viewBox="0 0 980 320" className="min-w-[760px] w-full" style={{ fontFamily: 'var(--font-mono), monospace' }} role="img" aria-label="System architecture: requests flow through the queue into the scheduler loop, which drives prefill and decode stages backed by a paged KV cache and a CPU swap pool.">
        <defs>
          {['#767680', '#FBBF24', '#4ADE80', '#FB7185', '#A78BFA'].map((c) => (
            <marker key={c} id={`arr-${c.slice(1)}`} markerWidth="8" markerHeight="8" refX="6.5" refY="4" orient="auto">
              <path d="M1,1 L7,4 L1,7" fill="none" stroke={c} strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
            </marker>
          ))}
          <linearGradient id="sched-grad" x1="0" x2="1">
            <stop offset="0" stopColor="#4ADE80" />
            <stop offset="0.5" stopColor="#22D3EE" />
            <stop offset="1" stopColor="#A78BFA" />
          </linearGradient>
          <filter id="soft-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" />
          </filter>
        </defs>

        {/* Edges */}
        {EDGES.map((e) => (
          <path
            key={e.id}
            d={e.d}
            fill="none"
            stroke={e.color}
            strokeOpacity={e.color === '#767680' ? 0.7 : 0.75}
            strokeWidth="1.3"
            className={e.id === 'loop' || e.id.startsWith('swap') ? 'flow-slow' : e.color !== '#767680' ? 'flow' : undefined}
            markerEnd={`url(#arr-${e.color.slice(1)})`}
          />
        ))}

        {/* Edge labels */}
        <text x="800" y="214" fill="#FB7185" fontSize="10" textAnchor="end">swap_out</text>
        <text x="954" y="214" fill="#FB7185" fontSize="10" textAnchor="middle">swap_in</text>
        <text x="448" y="292" fill="#A78BFA" fontSize="10" textAnchor="middle">next iteration ↺</text>

        {/* Nodes */}
        {Object.entries(NODES).map(([id, n]) => {
          const c = n.color ?? '#EDEDEF';
          return (
            <g key={id}>
              {n.strong && (
                <rect x={n.x - 2} y={n.y - 2} width={n.w + 4} height={H + 4} rx="13" fill="url(#sched-grad)" opacity="0.35" filter="url(#soft-glow)" />
              )}
              <rect
                x={n.x}
                y={n.y}
                width={n.w}
                height={H}
                rx="11"
                fill={n.color ? `${n.color}12` : '#101216'}
                stroke={n.strong ? 'url(#sched-grad)' : n.color ? `${n.color}80` : 'rgba(255,255,255,0.14)'}
                strokeWidth={n.strong ? 1.5 : 1}
              />
              <text x={n.x + n.w / 2} y={n.y + 17} fill={n.color ?? '#EDEDEF'} fontSize="11.5" fontWeight={n.strong ? 600 : 500} textAnchor="middle">
                {n.label}
              </text>
              <text x={n.x + n.w / 2} y={n.y + 31} fill={c} fillOpacity="0.5" fontSize="9" textAnchor="middle">
                {n.sub}
              </text>
            </g>
          );
        })}

        {/* Travelling packets */}
        {!reduced && inView &&
          PACKETS.map((p, i) => (
            <circle key={i} r="3.2" fill={p.color} style={{ filter: `drop-shadow(0 0 4px ${p.color})` }}>
              <animateMotion dur={`${p.dur}s`} begin={`${p.begin}s`} repeatCount="indefinite" path={p.path} keyPoints="0;1" keyTimes="0;1" calcMode="spline" keySplines="0.45 0 0.35 1" />
            </circle>
          ))}
      </svg>
    </div>
  );
}

export function SchedulerLiveSection() {
  return (
    <section id="scheduler" className="relative px-5 py-24 sm:px-6 sm:py-32">
      <div className="mx-auto max-w-5xl">
        <SectionHeader
          index="01"
          label="Scheduler loop"
          title={<>One token per sequence, <Accent gradient>every iteration.</Accent></>}
          subtitle="At every iteration, all active sequences advance by exactly one decode token. No single request monopolizes the GPU."
        />

        <LoopSteps />

        <Reveal>
          <Card pad={false} glow={false} className="p-5 sm:p-7">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
              <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-fg-3">System architecture</p>
              <div className="flex flex-wrap gap-4 font-mono text-[10.5px] text-fg-3">
                <span className="flex items-center gap-1.5"><span className="h-px w-4 bg-amber" />prefill path</span>
                <span className="flex items-center gap-1.5"><span className="h-px w-4 bg-mint" />decode path</span>
                <span className="flex items-center gap-1.5"><span className="h-px w-4 bg-rose" />memory pressure</span>
              </div>
            </div>
            <ArchitectureDiagram />
          </Card>
        </Reveal>

        <Reveal>
          <div className="mt-6 flex gap-4 rounded-2xl border border-mint/20 bg-gradient-to-r from-mint/[0.07] to-transparent p-5">
            <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-mint/15 text-mint">
              <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M8 1.5v3M8 11.5v3M1.5 8h3M11.5 8h3M3.4 3.4l2.1 2.1M10.5 10.5l2.1 2.1M3.4 12.6l2.1-2.1M10.5 5.5l2.1-2.1" strokeLinecap="round" /></svg>
            </span>
            <p className="text-[14px] leading-relaxed text-fg-2">
              <span className="font-medium text-fg">Iteration-level scheduling.</span> Unlike Phase 1&apos;s{' '}
              <code className="rounded bg-white/5 px-1 font-mono text-[0.85em] text-fg">asyncio.Lock</code>, which blocks every
              request until generation completes, the scheduler advances <span className="text-fg">every sequence</span> by
              exactly one token per iteration, then yields. TTFT under 4-way load drops from{' '}
              <span className="font-mono text-rose line-through decoration-rose/50">1,418 ms</span> to{' '}
              <span className="font-mono text-mint">20.9 ms</span>.
            </p>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
