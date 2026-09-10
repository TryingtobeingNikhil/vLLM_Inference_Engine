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
              viewBox="0 0 820 140"
              className="w-full min-w-[640px]"
              style={{ fontFamily: 'JetBrains Mono, monospace' }}
            >
              <defs>
                <marker id="arrow"       markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#444" />
                </marker>
                <marker id="arrow-amber" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#FBBF24" />
                </marker>
                <marker id="arrow-green" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#4ADE80" />
                </marker>
                <marker id="arrow-red"   markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#F87171" />
                </marker>
              </defs>

              {/* ── Nodes ─────────────────────────────────────────── */}
              {/* Request */}
              <rect x="8"   y="46" width="88"  height="24" fill="#1a1a1a" stroke="#2a2a2a" strokeWidth="1" />
              <text x="52"  y="62" fill="#888" fontSize="9" textAnchor="middle">Request</text>

              {/* RequestQueue */}
              <rect x="118" y="34" width="118" height="24" fill="#1a1a1a" stroke="#2a2a2a" strokeWidth="1" />
              <text x="177" y="50" fill="#888" fontSize="9" textAnchor="middle">RequestQueue</text>

              {/* Scheduler Loop */}
              <rect x="268" y="34" width="118" height="24" fill="#1a1a1a" stroke="#2a2a2a" strokeWidth="1" />
              <text x="327" y="50" fill="#888" fontSize="9" textAnchor="middle">Scheduler Loop</text>

              {/* Prefill Stage */}
              <rect x="420" y="8"  width="112" height="24" fill="#3d2e0a" stroke="#FBBF24" strokeWidth="1" />
              <text x="476" y="24" fill="#FBBF24" fontSize="9" textAnchor="middle">Prefill Stage</text>

              {/* Decode Stage */}
              <rect x="420" y="68" width="112" height="24" fill="#1a3d27" stroke="#4ADE80" strokeWidth="1" />
              <text x="476" y="84" fill="#4ADE80" fontSize="9" textAnchor="middle">Decode Stage</text>

              {/* PagedKVCache */}
              <rect x="566" y="34" width="112" height="24" fill="#1a1a1a" stroke="#2a2a2a" strokeWidth="1" />
              <text x="622" y="50" fill="#888" fontSize="9" textAnchor="middle">PagedKVCache</text>

              {/* CPUSwapPool */}
              <rect x="566" y="90" width="112" height="24" fill="#3d1a1a" stroke="#F87171" strokeWidth="1" />
              <text x="622" y="106" fill="#F87171" fontSize="9" textAnchor="middle">CPUSwapPool</text>

              {/* ── Arrows ────────────────────────────────────────── */}
              {/* Request → RequestQueue */}
              <line x1="96"  y1="58" x2="116"  y2="46" stroke="#444" strokeWidth="1" markerEnd="url(#arrow)" />

              {/* RequestQueue → Scheduler Loop */}
              <line x1="236" y1="46" x2="266"  y2="46" stroke="#444" strokeWidth="1" markerEnd="url(#arrow)" />

              {/* Scheduler → Prefill (upper dashed amber) */}
              <line x1="386" y1="40" x2="418"  y2="20" stroke="#FBBF24" strokeWidth="1" strokeDasharray="4,3" markerEnd="url(#arrow-amber)" />

              {/* Scheduler → Decode (lower dashed green) */}
              <line x1="386" y1="52" x2="418"  y2="72" stroke="#4ADE80" strokeWidth="1" strokeDasharray="4,3" markerEnd="url(#arrow-green)" />

              {/* Prefill → PagedKVCache */}
              <line x1="532" y1="20" x2="564"  y2="38" stroke="#444" strokeWidth="1" markerEnd="url(#arrow)" />

              {/* Decode → PagedKVCache */}
              <line x1="532" y1="74" x2="564"  y2="56" stroke="#444" strokeWidth="1" markerEnd="url(#arrow)" />

              {/* PagedKVCache → CPUSwapPool (swap_out, vertical) */}
              <line x1="622" y1="58" x2="622"  y2="88" stroke="#F87171" strokeWidth="1" strokeDasharray="3,2" markerEnd="url(#arrow-red)" />
              <text x="634"  y="76" fill="#F87171" fontSize="7" textAnchor="start">swap_out</text>

              {/* CPUSwapPool → PagedKVCache (swap_in, arc back on left side) */}
              <path
                d="M566,102 Q530,102 530,46 Q530,46 564,46"
                fill="none"
                stroke="#F87171"
                strokeWidth="1"
                strokeDasharray="3,2"
                markerEnd="url(#arrow-red)"
              />
              <text x="498"  y="118" fill="#F87171" fontSize="7" textAnchor="middle">swap_in</text>
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
