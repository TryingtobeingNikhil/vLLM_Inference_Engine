'use client';

import { useState, useEffect } from 'react';
import { SectionHeader, Accent } from '@/components/ui/SectionHeader';
import { Badge } from '@/components/ui/Badge';
import { Card, Window } from '@/components/ui/Card';
import { Reveal } from '@/components/ui/Reveal';
import { CodeBlock } from '@/components/ui/CodeBlock';
import { SEQ_COLORS, useInView, usePrefersReducedMotion } from '@/lib/motion';
import { DEMO_CONFIG as ENGINE_CONFIG } from '@/data/engine';
import { buildBlockGridTrace, type BlockGridState, type BlockState } from '@/data/simulation';

const TRACE = buildBlockGridTrace(80);
const OWNERS = ['a1b2', 'c3d4', 'e5f6', 'g7h8'];
const BS = ENGINE_CONFIG.kv_block_size;

// Slot utilisation under paging for the demo sequences (prompt + max_new tokens):
// only each sequence's last block can be partially empty.
const DEMO_SEQ_TOKENS = [98, 72, 99, 69];
const PAGED_UTIL = Math.round(
  (DEMO_SEQ_TOKENS.reduce((a, t) => a + t, 0) /
    DEMO_SEQ_TOKENS.reduce((a, t) => a + Math.ceil(t / BS) * BS, 0)) *
    100,
);

const ALLOCATOR_SRC = `class BlockAllocator:  # block tables, ref counts, prefix cache
    def allocate(seq_id, num_blocks=1) -> list[int]
    def ensure_capacity(seq_id, total_tokens) -> list[int]
    def register_computed_blocks(seq_id, token_ids, num_computed_tokens)
    def free(seq_id) -> int

# attention_wrapper.py — where each token's K/V lives in the pool
slot = block_table[pos // 16] * 16 + pos % 16`;

export function KVCacheSection() {
  const [frameIdx, setFrameIdx] = useState(0);
  const [isRunning, setIsRunning] = useState(true);
  const [hovered, setHovered] = useState<number | null>(null);
  const [ref, inView] = useInView<HTMLDivElement>({ once: false, threshold: 0.1 });
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    if (!isRunning || !inView || reduced) return;
    const id = setInterval(() => setFrameIdx((i) => (i + 1) % TRACE.length), 150);
    return () => clearInterval(id);
  }, [isRunning, inView, reduced]);

  // Reduced motion: park on a busy frame so the grid still tells the story.
  useEffect(() => { if (reduced) setFrameIdx(30); }, [reduced]);

  const frame: BlockGridState = TRACE[frameIdx];
  const allocatedCount = frame.blocks.filter((b) => b.state !== 'free').length;
  const freeCount = ENGINE_CONFIG.kv_num_blocks - allocatedCount;
  const poolPct = Math.round((allocatedCount / ENGINE_CONFIG.kv_num_blocks) * 100);
  const hoveredBlock: BlockState | null = hovered !== null ? frame.blocks[hovered] : null;

  return (
    <section id="kvcache" className="relative px-5 py-24 sm:px-6 sm:py-32">
      <div className="mx-auto max-w-5xl">
        <SectionHeader
          index="03"
          label="Phase 6–8 · Paged KV cache"
          title={<>Virtual memory for attention: <Accent gradient>pages, not slabs.</Accent></>}
          subtitle={`Fixed ${ENGINE_CONFIG.kv_block_size}-token blocks in one pre-allocated pool, and attention reads and writes it directly through per-sequence block tables. On CUDA the pool is sized from free GPU memory; the grid below is a small demo pool.`}
        />

        <div className="grid gap-5 lg:grid-cols-[1.05fr_1fr] [&>*]:min-w-0">
          {/* ── Block grid ─────────────────────────────── */}
          <Reveal>
            <div ref={ref}>
              <Window
                title={<>demo pool <span className="text-fg-4">— {ENGINE_CONFIG.kv_num_blocks} blocks</span></>}
                right={
                  <>
                    <Badge label="Demo data" variant="demo" className="hidden sm:inline-flex" />
                    <button
                      onClick={() => setIsRunning((r) => !r)}
                      className="flex h-6 w-6 items-center justify-center rounded-full border border-line-2 text-[10px] text-fg-2 transition-colors hover:border-white/25 hover:text-fg"
                      aria-label={isRunning ? 'Pause' : 'Play'}
                    >
                      {isRunning ? '❚❚' : '▶'}
                    </button>
                  </>
                }
              >
                <div className="p-4 sm:p-5">
                  <div
                    className="grid gap-[3px]"
                    style={{ gridTemplateColumns: 'repeat(16, 1fr)' }}
                    onMouseLeave={() => setHovered(null)}
                  >
                    {frame.blocks.map((block) => {
                      const owner = block.ownerShortId;
                      const color = owner ? SEQ_COLORS[owner] ?? '#4ADE80' : null;
                      const fill = block.tokensUsed / BS;
                      const sameOwner = hoveredBlock?.ownerShortId && hoveredBlock.ownerShortId === owner;
                      const dim = hoveredBlock?.ownerShortId && !sameOwner;
                      return (
                        <div
                          key={block.blockId}
                          onMouseEnter={() => setHovered(block.blockId)}
                          className="relative aspect-square overflow-hidden rounded-[3px] transition-all duration-150"
                          style={{
                            backgroundColor: color ? `${color}22` : 'rgba(255,255,255,0.035)',
                            boxShadow: color
                              ? `inset 0 0 0 1px ${color}${sameOwner ? 'cc' : '55'}${sameOwner ? `, 0 0 10px ${color}66` : ''}`
                              : hovered === block.blockId
                              ? 'inset 0 0 0 1px rgba(255,255,255,0.3)'
                              : 'none',
                            opacity: dim ? 0.35 : 1,
                          }}
                        >
                          {color && (
                            <div
                              className="absolute inset-x-0 bottom-0 transition-[height] duration-150"
                              style={{ height: `${fill * 100}%`, backgroundColor: color, opacity: 0.85 }}
                            />
                          )}
                        </div>
                      );
                    })}
                  </div>

                  {/* Inspector readout */}
                  <div className="mt-4 flex min-h-[34px] items-center gap-3 rounded-lg border border-line bg-ink-950 px-3 py-2 font-mono text-[11px]">
                    {hoveredBlock ? (
                      <>
                        <span className="text-fg-3">block</span>
                        <span className="text-fg">{String(hoveredBlock.blockId).padStart(3, '0')}</span>
                        <span className="text-fg-4">·</span>
                        {hoveredBlock.ownerShortId ? (
                          <>
                            <span style={{ color: SEQ_COLORS[hoveredBlock.ownerShortId] }}>seq-{hoveredBlock.ownerShortId}</span>
                            <span className="text-fg-4">·</span>
                            <span className="text-fg-2">{hoveredBlock.tokensUsed}/{BS} tokens</span>
                          </>
                        ) : (
                          <span className="text-fg-3">free</span>
                        )}
                      </>
                    ) : (
                      <span className="text-fg-4">↖ hover any block to inspect it</span>
                    )}
                  </div>

                  <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2">
                    {OWNERS.map((id) => (
                      <span key={id} className="flex items-center gap-1.5 font-mono text-[10.5px] text-fg-3">
                        <span className="h-2.5 w-2.5 rounded-[3px]" style={{ backgroundColor: SEQ_COLORS[id] }} />
                        seq-{id}
                      </span>
                    ))}
                    <span className="flex items-center gap-1.5 font-mono text-[10.5px] text-fg-3">
                      <span className="h-2.5 w-2.5 rounded-[3px] bg-white/[0.06]" />
                      free
                    </span>
                  </div>

                  <div className="mt-4 grid grid-cols-3 gap-px overflow-hidden rounded-xl border border-line bg-line">
                    <Mini label="allocated" value={allocatedCount} color="#4ADE80" />
                    <Mini label="free" value={freeCount} color="#A6A6B0" />
                    <Mini label="pool used" value={`${poolPct}%`} color="#FBBF24" />
                  </div>
                </div>
              </Window>
            </div>
          </Reveal>

          {/* ── Explanation column ─────────────────────── */}
          <div className="flex flex-col gap-5">
            <Reveal delay={80}>
              <Card>
                <p className="mb-5 font-mono text-[10px] uppercase tracking-[0.16em] text-fg-3">Memory efficiency</p>

                <MemBar label="Naïve · max_seq_len pre-allocated" used={45} usedColor="#FBBF24" wasteLabel="55% wasted padding" />
                <div className="h-5" />
                <MemBar label="PageServe · paged, demand-allocated" used={PAGED_UTIL} usedColor="#4ADE80" wasteLabel={`≤ ${BS - 1} empty slots / seq`} good />
              </Card>
            </Reveal>

            <Reveal delay={140}>
              <Card>
                <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.16em] text-fg-3">Engineering decision</p>
                <p className="mb-3 text-[14px] leading-relaxed text-fg-2">
                  Reserving <code className="rounded bg-white/5 px-1 font-mono text-[0.85em] text-fg">max_length</code> of KV per
                  sequence up front <span className="text-amber">leaves most of it unused</span>, because output lengths are
                  unpredictable, so few sequences fit.
                </p>
                <p className="text-[14px] leading-relaxed text-fg-2">
                  PageServe&apos;s <code className="rounded bg-white/5 px-1 font-mono text-[0.85em] text-fg">BlockAllocator</code> gives
                  each sequence a block table of <span className="text-mint">{BS}-token blocks</span> in one device pool. A custom
                  attention backend plugged into the HuggingFace model scatters new K/V into those slots and attends through the
                  table, so there is <span className="text-fg">no per-sequence past_key_values</span>. Full blocks are
                  content-hashed and ref-counted, so requests with a shared prefix share them.
                </p>
              </Card>
            </Reveal>

            <Reveal delay={200}>
              <Window title="engine/block_allocator.py">
                <CodeBlock code={ALLOCATOR_SRC} language="python" />
              </Window>
            </Reveal>
          </div>
        </div>
      </div>
    </section>
  );
}

function MemBar({ label, used, usedColor, wasteLabel, good = false }: { label: string; used: number; usedColor: string; wasteLabel: string; good?: boolean }) {
  return (
    <div>
      <p className="mb-2 text-[13px] text-fg-2">{label}</p>
      <div className={`flex h-7 overflow-hidden rounded-lg border border-line ${good ? 'bg-white/[0.02]' : 'hatch-rose'}`}>
        <div
          className="h-full rounded-r-md transition-[width] duration-300"
          style={{ width: `${used}%`, background: `linear-gradient(90deg, ${usedColor}33, ${usedColor}88)` }}
        />
      </div>
      <div className="mt-1.5 flex justify-between font-mono text-[10.5px]">
        <span style={{ color: usedColor }}>{used}% of reserved slots used</span>
        <span className={good ? 'text-fg-3' : 'text-rose'}>{wasteLabel}</span>
      </div>
    </div>
  );
}

function Mini({ label, value, color }: { label: string; value: number | string; color: string }) {
  return (
    <div className="bg-ink-900 px-3 py-2.5">
      <p className="font-mono text-[9.5px] uppercase tracking-[0.14em] text-fg-4">{label}</p>
      <p className="mt-0.5 font-mono text-sm tabular-nums" style={{ color }}>{value}</p>
    </div>
  );
}
