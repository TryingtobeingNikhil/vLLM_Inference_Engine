'use client';

import { useState, useEffect, useRef } from 'react';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { ENGINE_CONFIG } from '@/data/benchmarks';
import { buildBlockGridTrace, type BlockGridState } from '@/data/simulation';

const TRACE = buildBlockGridTrace(80);

const OWNER_COLORS: Record<string, string> = {
  a1b2: '#4ADE80',
  c3d4: '#60A5FA',
  e5f6: '#FBBF24',
  g7h8: '#a78bfa',
};

export function KVCacheSection() {
  const [frameIdx, setFrameIdx] = useState(0);
  const [isRunning, setIsRunning] = useState(true);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (isRunning) {
      intervalRef.current = setInterval(() => {
        setFrameIdx((i) => (i + 1) % TRACE.length);
      }, 150);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isRunning]);

  const frame: BlockGridState = TRACE[frameIdx];
  const allocatedCount = frame.blocks.filter((b) => b.state !== 'free').length;
  const freeCount = ENGINE_CONFIG.kv_num_blocks - allocatedCount;
  const utilPct = Math.round((allocatedCount / ENGINE_CONFIG.kv_num_blocks) * 100);

  return (
    <section id="kvcache" className="border-b border-[#1e1e1e] px-6 py-20 sm:px-10 lg:px-16">
      <div className="mx-auto max-w-5xl">
        <SectionHeader
          label="// phase 6–7: paged kv cache"
          title="Paged KV-Cache"
          subtitle={`${ENGINE_CONFIG.kv_num_blocks} blocks × ${ENGINE_CONFIG.kv_block_size} token slots. Allocated on demand. No external fragmentation.`}
        />

        <div className="grid gap-6 lg:grid-cols-2">
          {/* ── Left: Block grid ─────────────────────────────── */}
          <div className="border border-[#2a2a2a] bg-[#0d0d0d]">
            <div className="flex items-center justify-between border-b border-[#1e1e1e] px-4 py-2.5">
              <span className="font-mono text-[10px] uppercase tracking-widest text-[#333333]">
                Block Pool — {ENGINE_CONFIG.kv_num_blocks} Blocks
              </span>
              <div className="flex items-center gap-2">
                <Badge label="Demo Data" variant="demo" />
                <button
                  onClick={() => setIsRunning((r) => !r)}
                  className="font-mono text-[10px] text-[#444444] hover:text-[#888888]"
                >
                  {isRunning ? '⏸' : '▶'}
                </button>
              </div>
            </div>
            <div className="p-4">
              {/* 16×16 grid */}
              <div
                className="grid gap-0.5"
                style={{ gridTemplateColumns: `repeat(16, 1fr)` }}
              >
                {frame.blocks.map((block) => {
                  const color =
                    block.state === 'free'
                      ? '#1a1a1a'
                      : block.ownerShortId
                      ? OWNER_COLORS[block.ownerShortId] ?? '#4ADE80'
                      : '#3a3a3a';
                  const opacity =
                    block.state === 'free'
                      ? 0.4
                      : block.state === 'dirty'
                      ? 1
                      : 0.7 + 0.3 * (block.tokensUsed / ENGINE_CONFIG.kv_block_size);
                  return (
                    <div
                      key={block.blockId}
                      className="aspect-square transition-colors duration-100"
                      style={{
                        backgroundColor: color,
                        opacity,
                        border: '1px solid #111',
                      }}
                      title={`Block ${block.blockId}: ${block.state}${block.ownerShortId ? ` (seq-${block.ownerShortId})` : ''} — ${block.tokensUsed}/${ENGINE_CONFIG.kv_block_size} tokens`}
                    />
                  );
                })}
              </div>

              {/* Legend */}
              <div className="mt-3 flex flex-wrap gap-3">
                <LegendDot color="#1a1a1a" label="free" opacity={0.4} />
                {Object.entries(OWNER_COLORS).map(([id, color]) => (
                  <LegendDot key={id} color={color} label={`seq-${id}`} opacity={0.7} />
                ))}
              </div>

              {/* Stats bar */}
              <div className="mt-3 flex items-center gap-4 border-t border-[#1e1e1e] pt-2">
                <StatPill label="allocated" value={String(allocatedCount)} color="#4ADE80" />
                <StatPill label="free" value={String(freeCount)} color="#444444" />
                <StatPill label="util" value={`${utilPct}%`} color="#FBBF24" />
              </div>
            </div>
          </div>

          {/* ── Right: Fragmentation comparison + explanation ─── */}
          <div className="flex flex-col gap-4">
            {/* Fragmentation comparison */}
            <Card>
              <p className="mb-4 font-mono text-[10px] uppercase tracking-widest text-[#444444]">
                Memory Efficiency Comparison
              </p>

              <p className="mb-1 font-mono text-[11px] text-[#666666]">
                Naïve (max_seq_len pre-allocated)
              </p>
              <div className="mb-1 h-6 w-full" style={{ backgroundColor: '#0d0d0d', border: '1px solid #1e1e1e' }}>
                <div className="h-full" style={{ width: '45%', backgroundColor: '#3a2a0a' }} />
              </div>
              <div className="mb-4 flex justify-between font-mono text-[10px]">
                <span className="text-[#FBBF24]">45% used</span>
                <span className="text-[#F87171]">55% wasted (padding)</span>
              </div>

              <p className="mb-1 font-mono text-[11px] text-[#666666]">
                PageServe (paged, demand-allocated)
              </p>
              <div className="mb-1 h-6 w-full" style={{ backgroundColor: '#0d0d0d', border: '1px solid #1e1e1e' }}>
                <div
                  className="h-full transition-all duration-150"
                  style={{
                    width: `${utilPct}%`,
                    backgroundColor: '#1a3d27',
                  }}
                />
              </div>
              <div className="flex justify-between font-mono text-[10px]">
                <span className="text-[#4ADE80]">{utilPct}% used</span>
                <span className="text-[#444444]">0% wasted</span>
              </div>
            </Card>

            {/* Engineering decision */}
            <Card>
              <p className="mb-3 font-mono text-[10px] uppercase tracking-widest text-[#444444]">
                Engineering Decision
              </p>
              <p className="mb-3 text-sm leading-relaxed text-[#888888]">
                Standard Transformers allocate memory for{' '}
                <code className="text-[#aaa]">max_sequence_length</code> upfront — regardless of
                actual output length. Because output lengths are unpredictable, this causes{' '}
                <span className="text-[#FBBF24]">60–80% internal fragmentation</span> under typical
                workloads.
              </p>
              <p className="text-sm leading-relaxed text-[#888888]">
                PageServe's{' '}
                <code className="text-[#aaa]">BlockAllocator</code> maps each sequence's tokens
                into{' '}
                <span className="text-[#4ADE80]">
                  {ENGINE_CONFIG.kv_block_size}-token logical blocks
                </span>{' '}
                backed by pre-allocated physical tensor slots in{' '}
                <code className="text-[#aaa]">PagedKVCacheManager</code>. Blocks are freed
                immediately on completion.
              </p>
            </Card>

            {/* Code snippet */}
            <Card pad={false} className="overflow-hidden">
              <div className="border-b border-[#2a2a2a] px-4 py-2">
                <span className="font-mono text-[10px] uppercase tracking-widest text-[#333333]">
                  block_allocator.py
                </span>
              </div>
              <pre className="overflow-x-auto p-4 text-[0.72rem] leading-relaxed text-[#666666]">
                <span className="text-[#a78bfa]">class</span>{' '}
                <span className="text-[#60A5FA]">BlockAllocator</span>:{'\n'}
                {'    '}<span className="text-[#888]">def</span>{' '}
                <span className="text-[#4ADE80]">allocate</span>
                (seq_id, n_blocks) → list[int]{'\n'}
                {'    '}<span className="text-[#888]">def</span>{' '}
                <span className="text-[#4ADE80]">write_token</span>
                (seq_id, count=1) → None{'\n'}
                {'    '}<span className="text-[#888]">def</span>{' '}
                <span className="text-[#4ADE80]">free</span>
                (seq_id) → None{'\n'}
                {'    '}<span className="text-[#444]"># Thread-safe via threading.Lock</span>
              </pre>
            </Card>
          </div>
        </div>
      </div>
    </section>
  );
}

function LegendDot({
  color,
  label,
  opacity,
}: {
  color: string;
  label: string;
  opacity: number;
}) {
  return (
    <div className="flex items-center gap-1">
      <div
        className="h-2.5 w-2.5"
        style={{ backgroundColor: color, opacity, border: '1px solid #222' }}
      />
      <span className="font-mono text-[9px] text-[#444444]">{label}</span>
    </div>
  );
}

function StatPill({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <span className="font-mono text-[11px]">
      <span className="text-[#333333]">{label}: </span>
      <span style={{ color }}>{value}</span>
    </span>
  );
}
