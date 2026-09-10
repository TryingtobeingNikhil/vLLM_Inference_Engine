'use client';

import { SectionHeader } from '@/components/ui/SectionHeader';
import { Card } from '@/components/ui/Card';
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
    color: '#a78bfa',
  },
];

const ARCHITECTURE_NODES = [
  { id: 'req',       label: 'Request',         x: 5,   y: 42, w: 80 },
  { id: 'queue',     label: 'RequestQueue',     x: 110, y: 30, w: 110 },
  { id: 'scheduler', label: 'Scheduler Loop',   x: 255, y: 30, w: 110 },
  { id: 'prefill',   label: 'Prefill Stage',    x: 400, y: 5,  w: 100 },
  { id: 'decode',    label: 'Decode Stage',     x: 400, y: 55, w: 100 },
  { id: 'kvcache',   label: 'PagedKVCache',     x: 535, y: 30, w: 100 },
  { id: 'swap',      label: 'CPUSwapPool',      x: 535, y: 75, w: 100 },
];

export function SchedulerLiveSection() {
  return (
    <section id="scheduler" className="border-b border-[#1e1e1e] px-6 py-20 sm:px-10 lg:px-16">
      <div className="mx-auto max-w-5xl">
        <SectionHeader
          label="// scheduler loop"
          title="The Scheduler Loop"
          subtitle="At every iteration, all active sequences advance by exactly one decode token. No single request monopolizes the GPU."
        />

        {/* ── Four step cards ──────────────────────────────────── */}
        <div className="mb-12 grid grid-cols-2 gap-3 lg:grid-cols-4">
          {LOOP_STEPS.map((step) => (
            <Card key={step.step} className="relative">
              <div
                className="mb-3 flex items-baseline gap-2"
              >
                <span
                  className="font-mono text-2xl font-bold"
                  style={{ color: step.color }}
                >
                  {step.step}
                </span>
                <span className="font-mono text-[10px] uppercase tracking-widest text-[#444444]">
                  {step.label}
                </span>
              </div>
              <p className="mb-2 font-mono text-sm text-[#cccccc]">{step.title}</p>
              <div className="space-y-1">
                {step.lines.map((line) => (
                  <p key={line} className="font-mono text-[11px] text-[#555555]">
                    {line}
                  </p>
                ))}
              </div>
            </Card>
          ))}
        </div>

        {/* ── Architecture flow SVG ────────────────────────────── */}
        <div className="border border-[#2a2a2a] bg-[#0d0d0d] p-6">
          <p className="mb-4 font-mono text-[10px] uppercase tracking-widest text-[#333333]">
            System Architecture
          </p>
          <div className="overflow-x-auto">
            <svg
              viewBox="0 0 680 120"
              className="w-full min-w-[580px]"
              style={{ fontFamily: 'JetBrains Mono, monospace' }}
            >
              {/* Arrows */}
              {/* Request → Queue */}
              <line x1="85" y1="50" x2="108" y2="50" stroke="#333" strokeWidth="1" markerEnd="url(#arrow)" />
              {/* Queue → Scheduler */}
              <line x1="220" y1="50" x2="253" y2="50" stroke="#333" strokeWidth="1" markerEnd="url(#arrow)" />
              {/* Scheduler → Prefill */}
              <line x1="310" y1="38" x2="398" y2="20" stroke="#FBBF24" strokeWidth="1" strokeDasharray="3,2" markerEnd="url(#arrow-amber)" />
              {/* Scheduler → Decode */}
              <line x1="310" y1="62" x2="398" y2="68" stroke="#4ADE80" strokeWidth="1" strokeDasharray="3,2" markerEnd="url(#arrow-green)" />
              {/* Prefill → KVCache */}
              <line x1="500" y1="18" x2="533" y2="38" stroke="#333" strokeWidth="1" markerEnd="url(#arrow)" />
              {/* Decode → KVCache */}
              <line x1="500" y1="68" x2="533" y2="52" stroke="#333" strokeWidth="1" markerEnd="url(#arrow)" />
              {/* KVCache → Swap */}
              <line x1="585" y1="68" x2="585" y2="73" stroke="#F87171" strokeWidth="1" strokeDasharray="2,2" markerEnd="url(#arrow-red)" />
              {/* Swap back label */}
              <text x="638" y="85" fill="#F87171" fontSize="7" textAnchor="middle">swap_in</text>
              <line x1="635" y1="78" x2="635" y2="55" stroke="#F87171" strokeWidth="1" strokeDasharray="2,2" />
              <line x1="635" y1="55" x2="636" y2="55" stroke="#F87171" strokeWidth="1" markerEnd="url(#arrow-red)" />

              {/* Arrow markers */}
              <defs>
                <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#444" />
                </marker>
                <marker id="arrow-amber" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#FBBF24" />
                </marker>
                <marker id="arrow-green" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#4ADE80" />
                </marker>
                <marker id="arrow-red" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#F87171" />
                </marker>
              </defs>

              {/* Nodes */}
              {[
                { x: 5,   y: 38, w: 80,  h: 22, label: 'Request',        color: '#333' },
                { x: 110, y: 30, w: 110, h: 22, label: 'RequestQueue',   color: '#333' },
                { x: 255, y: 30, w: 110, h: 22, label: 'Scheduler Loop', color: '#333' },
                { x: 400, y: 5,  w: 100, h: 22, label: 'Prefill Stage',  color: '#3d2e0a', textColor: '#FBBF24' },
                { x: 400, y: 55, w: 100, h: 22, label: 'Decode Stage',   color: '#1a3d27', textColor: '#4ADE80' },
                { x: 535, y: 30, w: 100, h: 22, label: 'PagedKVCache',   color: '#333' },
                { x: 535, y: 73, w: 100, h: 22, label: 'CPUSwapPool',    color: '#3d1a1a', textColor: '#F87171' },
              ].map((node) => (
                <g key={node.label}>
                  <rect
                    x={node.x} y={node.y} width={node.w} height={node.h}
                    fill={node.color}
                    stroke="#2a2a2a"
                    strokeWidth="1"
                  />
                  <text
                    x={node.x + node.w / 2}
                    y={node.y + 14}
                    fill={(node as { textColor?: string }).textColor ?? '#888888'}
                    fontSize="8"
                    textAnchor="middle"
                  >
                    {node.label}
                  </text>
                </g>
              ))}
            </svg>
          </div>
        </div>

        {/* ── Key insight callout ──────────────────────────────── */}
        <div className="mt-6 border-l-2 border-[#4ADE80] bg-[#111111] px-4 py-3">
          <p className="font-mono text-[11px] leading-relaxed text-[#888888]">
            <span className="text-[#4ADE80]">Iteration-level scheduling</span> — unlike Phase 1's{' '}
            <code className="text-[#aaa]">asyncio.Lock</code> which blocks all requests until
            generation completes, the scheduler advances{' '}
            <span className="text-[#e8e8e8]">every sequence</span> by exactly one token per
            iteration, then yields back. Result: TTFT drops from{' '}
            <span className="text-[#F87171]">1,418 ms</span> to{' '}
            <span className="text-[#4ADE80]">20.9 ms</span> under 4-way concurrent load.
          </p>
        </div>
      </div>
    </section>
  );
}
