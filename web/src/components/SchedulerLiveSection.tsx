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
            {/*
              Layout (all y values are tops of 24-tall boxes):
                Request       x=8,   y=44  → center=(52,  56)
                RequestQueue  x=118, y=32  → center=(177, 44)
                SchedulerLoop x=268, y=32  → center=(327, 44)
                PrefillStage  x=420, y=6   → center=(476, 18)
                DecodeStage   x=420, y=62  → center=(476, 74)
                PagedKVCache  x=566, y=32  → center=(622, 44)
                CPUSwapPool   x=566, y=82  → center=(622, 94) bottom=106

              swap_out: vertical line 622,56 → 622,80   label at x=634 y=70
              swap_in:  cubic bezier M566,94 C518,94 518,44 566,44  label at x=500 y=72
              Total height needed: 106 + 10 padding = 116 → viewBox height = 125
            */}
            <svg
              viewBox="0 0 820 125"
              className="w-full min-w-[640px]"
              style={{ fontFamily: 'JetBrains Mono, monospace' }}
            >
              <defs>
                <marker id="arr"       markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#444" /></marker>
                <marker id="arr-amber" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#FBBF24" /></marker>
                <marker id="arr-green" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#4ADE80" /></marker>
                <marker id="arr-red"   markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#F87171" /></marker>
              </defs>

              {/* ── Nodes ── */}
              <rect x="8"   y="44" width="88"  height="24" fill="#1a1a1a" stroke="#2a2a2a" strokeWidth="1" />
              <text x="52"  y="60" fill="#888" fontSize="9" textAnchor="middle">Request</text>

              <rect x="118" y="32" width="118" height="24" fill="#1a1a1a" stroke="#2a2a2a" strokeWidth="1" />
              <text x="177" y="48" fill="#888" fontSize="9" textAnchor="middle">RequestQueue</text>

              <rect x="268" y="32" width="118" height="24" fill="#1a1a1a" stroke="#2a2a2a" strokeWidth="1" />
              <text x="327" y="48" fill="#888" fontSize="9" textAnchor="middle">Scheduler Loop</text>

              <rect x="420" y="6"  width="112" height="24" fill="#3d2e0a" stroke="#FBBF24" strokeWidth="1" />
              <text x="476" y="22" fill="#FBBF24" fontSize="9" textAnchor="middle">Prefill Stage</text>

              <rect x="420" y="62" width="112" height="24" fill="#1a3d27" stroke="#4ADE80" strokeWidth="1" />
              <text x="476" y="78" fill="#4ADE80" fontSize="9" textAnchor="middle">Decode Stage</text>

              <rect x="566" y="32" width="112" height="24" fill="#1a1a1a" stroke="#2a2a2a" strokeWidth="1" />
              <text x="622" y="48" fill="#888" fontSize="9" textAnchor="middle">PagedKVCache</text>

              <rect x="566" y="82" width="112" height="24" fill="#3d1a1a" stroke="#F87171" strokeWidth="1" />
              <text x="622" y="98" fill="#F87171" fontSize="9" textAnchor="middle">CPUSwapPool</text>

              {/* ── Arrows ── */}
              {/* Request → RequestQueue */}
              <line x1="96"  y1="56" x2="116"  y2="44" stroke="#444" strokeWidth="1" markerEnd="url(#arr)" />
              {/* RequestQueue → SchedulerLoop */}
              <line x1="236" y1="44" x2="266"  y2="44" stroke="#444" strokeWidth="1" markerEnd="url(#arr)" />
              {/* Scheduler → Prefill */}
              <line x1="386" y1="38" x2="418"  y2="18" stroke="#FBBF24" strokeWidth="1" strokeDasharray="4,3" markerEnd="url(#arr-amber)" />
              {/* Scheduler → Decode */}
              <line x1="386" y1="50" x2="418"  y2="68" stroke="#4ADE80" strokeWidth="1" strokeDasharray="4,3" markerEnd="url(#arr-green)" />
              {/* Prefill → PagedKVCache */}
              <line x1="532" y1="18" x2="564"  y2="36" stroke="#444" strokeWidth="1" markerEnd="url(#arr)" />
              {/* Decode → PagedKVCache */}
              <line x1="532" y1="74" x2="564"  y2="54" stroke="#444" strokeWidth="1" markerEnd="url(#arr)" />

              {/* swap_out: PagedKVCache ↓ CPUSwapPool */}
              <line x1="622" y1="56" x2="622"  y2="80" stroke="#F87171" strokeWidth="1" strokeDasharray="3,2" markerEnd="url(#arr-red)" />
              <text x="630"  y="71" fill="#F87171" fontSize="7" textAnchor="start">swap_out</text>

              {/* swap_in: CPUSwapPool → PagedKVCache via left-side Bezier arc */}
              <path
                d="M566,94 C518,94 518,44 566,44"
                fill="none" stroke="#F87171" strokeWidth="1" strokeDasharray="3,2"
                markerEnd="url(#arr-red)"
              />
              <text x="503"  y="73" fill="#F87171" fontSize="7" textAnchor="middle">swap_in</text>
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
