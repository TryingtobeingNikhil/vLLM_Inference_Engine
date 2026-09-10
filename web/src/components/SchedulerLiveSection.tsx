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
            <svg
              viewBox="0 0 900 200"
              className="w-full"
              style={{ fontFamily: 'JetBrains Mono, monospace' }}
            >
              <defs>
                <marker id="arr"       markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#555" /></marker>
                <marker id="arr-amber" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#FBBF24" /></marker>
                <marker id="arr-green" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#4ADE80" /></marker>
                <marker id="arr-red"   markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#F87171" /></marker>
              </defs>

              {/* ════════════════ NODES ════════════════ */}

              {/* Request  x=15 y=82 w=100 h=30  center=(65,97) */}
              <rect x="15"  y="82" width="100" height="30" fill="#141414" stroke="#2a2a2a" strokeWidth="1" rx="2" />
              <text x="65"  y="101" fill="#777" fontSize="10" textAnchor="middle">Request</text>

              {/* RequestQueue  x=140 y=70 w=150 h=30  center=(215,85) */}
              <rect x="140" y="70" width="150" height="30" fill="#141414" stroke="#2a2a2a" strokeWidth="1" rx="2" />
              <text x="215" y="89"  fill="#777" fontSize="10" textAnchor="middle">RequestQueue</text>

              {/* Scheduler Loop  x=320 y=70 w=150 h=30  center=(395,85) */}
              <rect x="320" y="70" width="150" height="30" fill="#141414" stroke="#3a3a3a" strokeWidth="1.5" rx="2" />
              <text x="395" y="89"  fill="#aaa" fontSize="10" fontWeight="bold" textAnchor="middle">Scheduler Loop</text>

              {/* Prefill Stage  x=510 y=18 w=150 h=30  center=(585,33) */}
              <rect x="510" y="18" width="150" height="30" fill="#3d2e0a" stroke="#FBBF24" strokeWidth="1" rx="2" />
              <text x="585" y="37"  fill="#FBBF24" fontSize="10" textAnchor="middle">Prefill Stage</text>

              {/* Decode Stage  x=510 y=142 w=150 h=30  center=(585,157) */}
              <rect x="510" y="142" width="150" height="30" fill="#1a3d27" stroke="#4ADE80" strokeWidth="1" rx="2" />
              <text x="585" y="161" fill="#4ADE80" fontSize="10" textAnchor="middle">Decode Stage</text>

              {/* PagedKVCache  x=700 y=70 w=150 h=30  center=(775,85) */}
              <rect x="700" y="70" width="165" height="30" fill="#141414" stroke="#3a3a3a" strokeWidth="1" rx="2" />
              <text x="782" y="89"  fill="#888" fontSize="10" textAnchor="middle">PagedKVCache</text>

              {/* CPUSwapPool  x=700 y=148 w=150 h=30  center=(775,163) bottom=178 */}
              <rect x="700" y="148" width="165" height="30" fill="#3d1a1a" stroke="#F87171" strokeWidth="1" rx="2" />
              <text x="782" y="167" fill="#F87171" fontSize="10" textAnchor="middle">CPUSwapPool</text>

              {/* ════════════════ ARROWS ════════════════ */}

              {/* Request → RequestQueue */}
              <line x1="115" y1="97" x2="138" y2="85" stroke="#555" strokeWidth="1.2" markerEnd="url(#arr)" />

              {/* RequestQueue → Scheduler Loop */}
              <line x1="290" y1="85" x2="318" y2="85" stroke="#555" strokeWidth="1.2" markerEnd="url(#arr)" />

              {/* Scheduler → Prefill (amber dashed, upper) */}
              <line x1="470" y1="76" x2="508" y2="40" stroke="#FBBF24" strokeWidth="1.2" strokeDasharray="5,3" markerEnd="url(#arr-amber)" />

              {/* Scheduler → Decode (green dashed, lower) */}
              <line x1="470" y1="94" x2="508" y2="150" stroke="#4ADE80" strokeWidth="1.2" strokeDasharray="5,3" markerEnd="url(#arr-green)" />

              {/* Prefill → PagedKVCache */}
              <line x1="660" y1="33" x2="698" y2="76" stroke="#555" strokeWidth="1.2" markerEnd="url(#arr)" />

              {/* Decode → PagedKVCache */}
              <line x1="660" y1="157" x2="698" y2="94" stroke="#555" strokeWidth="1.2" markerEnd="url(#arr)" />

              {/* swap_out: PagedKVCache ↓ CPUSwapPool  (vertical, center x=775) */}
              <line x1="775" y1="100" x2="775" y2="146" stroke="#F87171" strokeWidth="1.2" strokeDasharray="4,3" markerEnd="url(#arr-red)" />
              <text x="790" y="127" fill="#F87171" fontSize="8" textAnchor="start">swap_out</text>

              {/* swap_in: CPUSwapPool → PagedKVCache  cubic Bezier on left side */}
              <path d="M700,163 C648,163 648,85 700,85"
                fill="none" stroke="#F87171" strokeWidth="1.2" strokeDasharray="4,3"
                markerEnd="url(#arr-red)"
              />
              <text x="632" y="128" fill="#F87171" fontSize="8" textAnchor="middle">swap_in</text>
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
