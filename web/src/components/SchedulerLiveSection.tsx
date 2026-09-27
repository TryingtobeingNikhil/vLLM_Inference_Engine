'use client';

import { useEffect, useState } from 'react';
import { SectionHeader, Accent } from '@/components/ui/SectionHeader';
import { Card } from '@/components/ui/Card';
import { Reveal } from '@/components/ui/Reveal';
import { useInView, usePrefersReducedMotion } from '@/lib/motion';

// One scheduler step (README "How it works"): plan → pack → execute → apply.
const LOOP_STEPS = [
  {
    step: '01',
    label: 'Plan',
    title: 'Pick this step’s work',
    lines: [
      `decodes: 1 token each (+k spec)`,
      `prefill chunks ≤ token budget`,
      `admit if blocks allow (FCFS)`,
      `prefix-cache hits reused`,
    ],
    color: '#60A5FA',
  },
  {
    step: '02',
    label: 'Pack',
    title: 'One [1, T] batch',
    lines: [
      `concat every scheduled token`,
      `explicit position_ids`,
      `slot mapping per token`,
      `no padding through MLPs`,
    ],
    color: '#FBBF24',
  },
  {
    step: '03',
    label: 'Execute',
    title: 'One forward pass',
    lines: [
      `HF model, paged attention`,
      `writes K/V into the pool`,
      `reads context via block tables`,
      `logits only where sampled`,
    ],
    color: '#4ADE80',
  },
  {
    step: '04',
    label: 'Apply',
    title: 'Tokens out, blocks back',
    lines: [
      `append / verify tokens`,
      `stream them (SSE)`,
      `publish full blocks to cache`,
      `finish + free`,
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
  client: { x: 12,  y: 130, w: 120, label: 'Client',           sub: 'SSE · OpenAI API' },
  queue:  { x: 160, y: 130, w: 140, label: 'RequestQueue',     sub: 'FIFO · backpressure' },
  sched:  { x: 330, y: 130, w: 170, label: 'Scheduler',        sub: 'plan → execute → apply', strong: true },
  runner: { x: 540, y: 130, w: 190, label: 'ModelRunner',      sub: 'ONE packed pass · [1, T]', color: '#4ADE80' },
  attn:   { x: 540, y: 230, w: 190, label: 'Paged attention',  sub: 'HF attention backend', color: '#4ADE80' },
  alloc:  { x: 770, y: 40,  w: 190, label: 'BlockAllocator',   sub: 'block tables · prefix cache', color: '#22D3EE' },
  kv:     { x: 770, y: 130, w: 190, label: 'PagedKVCache',     sub: '[layers, blocks, 16, H, D]', color: '#60A5FA' },
  cpu:    { x: 770, y: 230, w: 190, label: 'CPUSwapManager',   sub: 'pinned host RAM', color: '#FB7185' },
};

const GREY = '#767680';
const EDGES = [
  { id: 'in1',     d: 'M132,150 L158,150', color: GREY },
  { id: 'in2',     d: 'M300,150 L328,150', color: GREY },
  { id: 'pack',    d: 'M500,150 L538,150', color: '#4ADE80' },
  { id: 'fwd',     d: 'M635,170 L635,228', color: '#4ADE80' },
  { id: 'kvio',    d: 'M730,250 C752,250 748,160 768,160', color: '#60A5FA' },
  { id: 'plan',    d: 'M415,130 C415,72 440,60 500,60 L768,60', color: '#22D3EE' },
  { id: 'tables',  d: 'M865,80 L865,128', color: '#22D3EE' },
  { id: 'swapout', d: 'M865,170 L865,228', color: '#FB7185' },
  { id: 'swapin',  d: 'M960,250 C990,250 990,150 962,150', color: '#FB7185' },
  { id: 'sampled', d: 'M560,170 C540,212 440,212 420,172', color: '#A78BFA' },
  { id: 'stream',  d: 'M340,170 C300,228 130,228 72,172', color: '#A78BFA' },
];
const edge = (id: string) => EDGES.find((e) => e.id === id)!.d;

// Packets that ride the edges: one request's trip through a scheduler step.
const PACKETS = [
  { path: 'M132,150 L328,150', color: '#EDEDEF', dur: 2.4, begin: 0 },
  { path: edge('plan'), color: '#22D3EE', dur: 2.2, begin: 0.3 },
  { path: edge('pack'), color: '#4ADE80', dur: 1.2, begin: 0.6 },
  { path: edge('fwd'), color: '#4ADE80', dur: 1.2, begin: 1.1 },
  { path: edge('kvio'), color: '#60A5FA', dur: 1.4, begin: 1.6 },
  { path: edge('sampled'), color: '#A78BFA', dur: 1.8, begin: 1.4 },
  { path: edge('stream'), color: '#A78BFA', dur: 2.4, begin: 2.0 },
  { path: edge('swapout'), color: '#FB7185', dur: 2.6, begin: 1.0 },
  { path: edge('swapin'), color: '#FB7185', dur: 2.6, begin: 2.3 },
];

function ArchitectureDiagram() {
  const reduced = usePrefersReducedMotion();
  const [ref, inView] = useInView<HTMLDivElement>({ once: false, threshold: 0.1 });

  return (
    <div ref={ref} className="overflow-x-auto">
      <svg viewBox="0 0 1000 310" className="w-full min-w-[780px]" style={{ fontFamily: 'var(--font-mono), monospace' }} role="img" aria-label="System architecture: requests flow through the queue into the scheduler, which packs all work into one forward pass; a paged attention backend reads and writes the KV pool through block tables, with a CPU swap pool for preemption.">
        <defs>
          {[GREY, '#4ADE80', '#60A5FA', '#22D3EE', '#FB7185', '#A78BFA'].map((c) => (
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
            strokeOpacity={e.color === GREY ? 0.7 : 0.75}
            strokeWidth="1.3"
            className={e.id === 'stream' || e.id === 'sampled' || e.id.startsWith('swap') ? 'flow-slow' : e.color !== GREY ? 'flow' : undefined}
            markerEnd={`url(#arr-${e.color.slice(1)})`}
          />
        ))}

        {/* Edge labels */}
        <text x="600" y="52" fill="#22D3EE" fontSize="10" textAnchor="middle">plan: allocate blocks · reuse prefix hits</text>
        <text x="873" y="108" fill="#22D3EE" fontSize="10">block tables</text>
        <text x="857" y="204" fill="#FB7185" fontSize="10" textAnchor="end">preempt: swap</text>
        <text x="978" y="205" fill="#FB7185" fontSize="10" textAnchor="middle">swap_in</text>
        <text x="635" y="292" fill="#60A5FA" fontSize="10" textAnchor="middle">write K/V · read context via block tables</text>
        <text x="484" y="224" fill="#A78BFA" fontSize="10" textAnchor="middle">sampled tokens</text>
        <text x="205" y="238" fill="#A78BFA" fontSize="10" textAnchor="middle">stream tokens (SSE)</text>

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
          title={<>All the work, <Accent gradient>one forward pass.</Accent></>}
          subtitle="Every scheduler step packs the prefill chunks of new prompts and one decode token per running sequence into a single forward pass. Requests join and leave the batch between steps."
        />

        <LoopSteps />

        <Reveal>
          <Card pad={false} glow={false} className="p-5 sm:p-7">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
              <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-fg-3">System architecture</p>
              <div className="flex flex-wrap gap-4 font-mono text-[10.5px] text-fg-3">
                <span className="flex items-center gap-1.5"><span className="h-px w-4 bg-mint" />packed forward pass</span>
                <span className="flex items-center gap-1.5"><span className="h-px w-4 bg-sky" />KV via block tables</span>
                <span className="flex items-center gap-1.5"><span className="h-px w-4 bg-violet" />tokens back</span>
                <span className="flex items-center gap-1.5"><span className="h-px w-4 bg-rose" />preemption</span>
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
              <span className="font-medium text-fg">One number drives scheduling.</span> Each sequence tracks{' '}
              <code className="rounded bg-white/5 px-1 font-mono text-[0.85em] text-fg">num_computed_tokens</code>, the tokens that
              already have K/V in the pool. A prefill chunk, a decode step and a recompute are the{' '}
              <span className="text-fg">same operation</span> over the uncomputed tokens, which is why one forward pass can mix
              them. Attention finds each token&apos;s slot as{' '}
              <code className="rounded bg-white/5 px-1 font-mono text-[0.85em] text-fg">block_table[pos // 16] * 16 + pos % 16</code>.
            </p>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
