'use client';

import { useSchedulerSim } from '@/lib/useSchedulerSim';
import { Badge } from '@/components/ui/Badge';
import type { SimSequence, SeqState } from '@/data/simulation';

const STATE_LABELS: Record<SeqState, string> = {
  waiting: 'WAITING',
  prefill: 'PREFILL',
  chunked_prefilling: 'PREFILL',
  decoding: 'DECODING',
  swapped: 'SWAPPED',
  finished: 'DONE',
};

const STATE_COLORS: Record<SeqState, string> = {
  waiting: '#444444',
  prefill: '#FBBF24',
  chunked_prefilling: '#FBBF24',
  decoding: '#4ADE80',
  swapped: '#F87171',
  finished: '#333333',
};

const BAR_BG_COLORS: Record<SeqState, string> = {
  waiting: '#181818',
  prefill: '#3d2e0a',
  chunked_prefilling: '#3d2e0a',
  decoding: '#1a3d27',
  swapped: '#3d1a1a',
  finished: '#111111',
};

function SequenceBar({ seq }: { seq: SimSequence }) {
  const progress =
    seq.state === 'decoding' || seq.state === 'finished'
      ? (seq.tokensGenerated / seq.maxNewTokens) * 100
      : seq.state === 'prefill' || seq.state === 'chunked_prefilling'
      ? (seq.prefillChunksDone / seq.prefillChunksTotal) * 100
      : 0;

  const color = STATE_COLORS[seq.state];
  const bgColor = BAR_BG_COLORS[seq.state];
  const label = STATE_LABELS[seq.state];

  return (
    <div className="flex items-center gap-3 py-1.5">
      {/* Sequence ID */}
      <span className="w-16 font-mono text-[11px] text-[#555555]">
        seq-{seq.shortId}
      </span>

      {/* Progress bar */}
      <div
        className="relative h-5 flex-1 overflow-hidden"
        style={{ backgroundColor: '#0d0d0d', border: '1px solid #1e1e1e' }}
      >
        <div
          className="absolute inset-y-0 left-0 transition-all duration-150"
          style={{
            width: `${Math.max(0, Math.min(100, progress))}%`,
            backgroundColor: bgColor,
          }}
        />
        {/* Block ticks every 16 tokens */}
        {Array.from({ length: 8 }).map((_, i) => (
          <div
            key={i}
            className="absolute inset-y-0 w-px"
            style={{ left: `${(i + 1) * 12.5}%`, backgroundColor: '#1a1a1a' }}
          />
        ))}
      </div>

      {/* State label */}
      <span
        className="w-20 font-mono text-[11px] text-right"
        style={{ color }}
      >
        {label}
      </span>

      {/* Token count */}
      <span className="w-8 font-mono text-[11px] text-right text-[#444444]">
        {seq.state === 'decoding' || seq.state === 'finished'
          ? `${seq.tokensGenerated}t`
          : '—'}
      </span>
    </div>
  );
}

export function HeroSection() {
  const {
    tick,
    sequences,
    batchSize,
    queueDepth,
    tokPerSec,
    freeBlocks,
    allocatedBlocks,
  } = useSchedulerSim(180);

  // Show up to 5 sequences in the panel
  const displaySeqs = sequences.slice(0, 5);
  // Pad with empty slots if fewer
  const emptySlots = Math.max(0, 5 - displaySeqs.length);

  return (
    <section className="relative border-b border-[#1e1e1e] px-6 py-20 sm:px-10 lg:px-16">
      {/* ── Header text ─────────────────────────────────────────── */}
      <div className="mx-auto max-w-5xl">
        <p className="mb-3 font-mono text-[11px] uppercase tracking-[0.25em] text-[#444444]">
          An inference engine built from first principles
        </p>
        <h1 className="mb-4 text-4xl font-semibold tracking-tight text-[#e8e8e8] sm:text-5xl lg:text-6xl">
          PageServe
        </h1>
        <div className="mb-12 flex flex-wrap items-center gap-3">
          {['Continuous Batching', 'Paged KV Cache', 'CPU Swapping'].map((tag) => (
            <span
              key={tag}
              className="border border-[#2a2a2a] px-3 py-1 font-mono text-xs text-[#666666]"
            >
              {tag}
            </span>
          ))}
        </div>

        {/* ── Scheduler panel ────────────────────────────────────── */}
        <div className="border border-[#2a2a2a] bg-[#0d0d0d]">
          {/* Panel header */}
          <div className="flex items-center justify-between border-b border-[#1e1e1e] px-4 py-2.5">
            <span className="font-mono text-[11px] uppercase tracking-widest text-[#333333]">
              Scheduler — Continuous Batching
            </span>
            <div className="flex items-center gap-3">
              <Badge label="Demo Data" variant="demo" />
              <Badge label="Running" variant="live" dot />
            </div>
          </div>

          {/* Sequence lanes */}
          <div className="px-4 py-3">
            {displaySeqs.map((seq) => (
              <SequenceBar key={seq.id} seq={seq} />
            ))}
            {/* Empty placeholder rows */}
            {Array.from({ length: emptySlots }).map((_, i) => (
              <div key={`empty-${i}`} className="flex items-center gap-3 py-1.5">
                <span className="w-16 font-mono text-[11px] text-[#222222]">—</span>
                <div
                  className="h-5 flex-1"
                  style={{ backgroundColor: '#0d0d0d', border: '1px solid #1a1a1a' }}
                />
                <span className="w-20 font-mono text-[11px] text-right text-[#222222]">EMPTY</span>
                <span className="w-8 font-mono text-[11px] text-[#222222]">—</span>
              </div>
            ))}
          </div>

          {/* Status bar */}
          <div className="flex flex-wrap items-center gap-x-6 gap-y-1 border-t border-[#1e1e1e] px-4 py-2.5">
            <StatusItem label="iter" value={String(tick).padStart(4, '0')} />
            <StatusItem label="batch" value={String(batchSize)} />
            <StatusItem label="queue" value={String(queueDepth)} />
            <StatusItem label="tok/s" value={String(tokPerSec)} highlight />
            <StatusItem label="blocks" value={`${allocatedBlocks}/${freeBlocks + allocatedBlocks}`} />
            <span className="ml-auto font-mono text-[10px] text-[#333333]">
              block_size=16 · max_batch=4
            </span>
          </div>
        </div>

        {/* ── CTA row ────────────────────────────────────────────── */}
        <div className="mt-8 flex flex-wrap items-center gap-4">
          <a
            href="https://github.com/TryingtobeingNikhil/vLLM_Inference_Engine"
            target="_blank"
            rel="noopener noreferrer"
            className="border border-[#e8e8e8] px-5 py-2.5 font-mono text-sm text-[#e8e8e8] transition-colors hover:bg-[#e8e8e8] hover:text-[#0a0a0a]"
          >
            View Source →
          </a>
          <a
            href="#scheduler"
            className="border border-[#2a2a2a] px-5 py-2.5 font-mono text-sm text-[#666666] transition-colors hover:border-[#3a3a3a] hover:text-[#888888]"
          >
            How it works ↓
          </a>
        </div>
      </div>
    </section>
  );
}

function StatusItem({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <span className="font-mono text-[11px]">
      <span className="text-[#444444]">{label}: </span>
      <span className={highlight ? 'text-[#4ADE80]' : 'text-[#888888]'}>{value}</span>
    </span>
  );
}
