'use client';

import { useEffect, useState, type ReactNode } from 'react';
import { SectionHeader, Accent } from '@/components/ui/SectionHeader';
import { Card } from '@/components/ui/Card';
import { Reveal } from '@/components/ui/Reveal';
import { SEQ_COLORS, useInView, usePrefersReducedMotion } from '@/lib/motion';
import { compareSystems, platform } from '@/data/gpuBenchmarks';

// Figures: NVIDIA T4 (Colab), Qwen2.5-1.5B-Instruct fp16, offline ablation.
const prefix = compareSystems('t4', 'prefix/shared_prefix', 'engine', 'engine+prefix');
const spec = compareSystems('t4', 'spec/repetitive', 'engine', 'engine+ngram');
const t4 = platform('t4');
const prefixHit = t4?.offline?.runs.find((r) => r.suite === 'prefix' && r.system === 'engine+prefix')?.metrics.prefix_cache_hit_rate;
const specRun = t4?.offline?.runs.find((r) => r.workload === 'repetitive' && r.system === 'engine+ngram')?.metrics;
const specNote = t4?.offline?.notes?.['spec/chat'];

const fx = (v: number | null | undefined, d = 1) => (typeof v === 'number' ? v.toFixed(d) : '–');

function SmokeStat({ from, to, ratio, extra }: { from: number | null; to: number | null; ratio: number | null; extra?: ReactNode }) {
  return (
    <div className="mt-5 flex flex-wrap items-end justify-between gap-3 border-t border-line pt-4">
      <div>
        <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-fg-4">T4 · output tok/s</p>
        <p className="mt-1 font-mono text-[13px] text-fg-2">
          <span className="text-fg-3 line-through decoration-rose/50">{fx(from)}</span> → <span className="text-fg">{fx(to)}</span>
          {extra}
        </p>
      </div>
      <span className="text-gradient text-3xl font-semibold tracking-tight">{fx(ratio)}×</span>
    </div>
  );
}

// ── Prefix caching: a radix tree over blocks ─────────────────────────────────

function PrefixTree() {
  const shared = ['h₁', 'h₂', 'h₃'];
  const branches = [
    { id: 'a1b2', blocks: 2, label: 'req A' },
    { id: 'c3d4', blocks: 3, label: 'req B' },
  ];
  return (
    <div className="rounded-xl border border-line bg-ink-950 p-4">
      <div className="flex items-center gap-3">
        <div className="flex gap-1.5">
          {shared.map((h, i) => (
            <div
              key={h}
              className="flex h-10 w-12 flex-col items-center justify-center rounded-md border border-cyan/60 bg-cyan/15 font-mono text-[10px] text-cyan"
              style={{ animation: `float-y 3s ease-in-out ${i * 0.2}s infinite` }}
            >
              {h}
              <span className="text-[8.5px] text-cyan/70">ref 2</span>
            </div>
          ))}
        </div>
        <span className="hidden font-mono text-[10.5px] text-fg-3 sm:block">
          shared system prompt · computed once
        </span>
      </div>
      <div className="ml-6 space-y-1.5 border-l border-dashed border-cyan/40 pl-3 pt-2">
        {branches.map((b) => (
          <div key={b.id} className="flex items-center gap-1.5">
            {Array.from({ length: b.blocks }).map((_, i) => (
              <div
                key={i}
                className="h-6 w-8 rounded-[5px] border"
                style={{ borderColor: `${SEQ_COLORS[b.id]}88`, backgroundColor: `${SEQ_COLORS[b.id]}22` }}
              />
            ))}
            <span className="ml-1 font-mono text-[10px] text-fg-3">{b.label} · private blocks</span>
          </div>
        ))}
      </div>
      <p className="mt-3 font-mono text-[10.5px] text-fg-4">
        hₙ = hash(hₙ₋₁, 16 tokens) — a hash names the whole prefix up to that block
      </p>
    </div>
  );
}

// ── Speculative decoding: drafts verified in one pass ────────────────────────

const ROUNDS = [4, 3, 4, 2, 4, 4]; // accepted drafts per step (illustrative)
const K = 4;

function SpecDemo() {
  const [ref, inView] = useInView<HTMLDivElement>({ once: false, threshold: 0.4 });
  const reduced = usePrefersReducedMotion();
  const [round, setRound] = useState(0);
  const [tick, setTick] = useState(K + 2);

  useEffect(() => {
    if (!inView || reduced) return;
    const id = setInterval(() => {
      setTick((t) => {
        if (t >= K + 3) {
          setRound((r) => (r + 1) % ROUNDS.length);
          return 0;
        }
        return t + 1;
      });
    }, 420);
    return () => clearInterval(id);
  }, [inView, reduced]);

  const accepted = ROUNDS[round];
  return (
    <div ref={ref} className="rounded-xl border border-line bg-ink-950 p-4">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="rounded-md border border-line-2 px-2 py-1.5 font-mono text-[10.5px] text-fg-2">last</span>
        {Array.from({ length: K }).map((_, i) => {
          const shown = tick > i;
          const ok = i < accepted;
          const state = !shown ? 'pending' : ok ? 'accept' : 'reject';
          const c = state === 'accept' ? '#4ADE80' : state === 'reject' ? '#FB7185' : '#52525B';
          return (
            <span
              key={i}
              className="flex h-8 w-11 items-center justify-center rounded-md border font-mono text-[10.5px] transition-all duration-300"
              style={{ color: c, borderColor: `${c}88`, backgroundColor: state === 'pending' ? 'transparent' : `${c}18`, opacity: i > accepted && shown ? 0.35 : 1 }}
            >
              {state === 'accept' ? '✓' : state === 'reject' ? '✕' : `d${i + 1}`}
            </span>
          );
        })}
        <span
          className="flex h-8 items-center rounded-md border border-violet/60 bg-violet/15 px-2 font-mono text-[10.5px] text-violet transition-all duration-300"
          style={{ opacity: tick > K ? 1 : 0.15 }}
        >
          +1 bonus
        </span>
      </div>
      <p className="mt-3 font-mono text-[10.5px] text-fg-3">
        n-gram proposes {K} drafts → target scores [last, d1..d{K}] in one pass →{' '}
        <span className="text-mint">{accepted + 1} tokens</span> this step
      </p>
    </div>
  );
}

const SMALL = [
  {
    title: 'Streaming + OpenAI API',
    body: 'SSE on the native /generate, plus an OpenAI-compatible /v1/completions. A client disconnect frees its KV memory.',
    tag: 'SSE · /v1/completions',
    color: '#60A5FA',
  },
  {
    title: 'KV pool sized for you',
    body: 'On CUDA the engine profiles free GPU memory and sizes the block pool to GPU_MEMORY_UTILIZATION (0.85 by default).',
    tag: 'KV_NUM_BLOCKS=auto',
    color: '#22D3EE',
  },
  {
    title: 'Honest metrics',
    body: 'TTFT includes queueing, ITL is the real gap between streamed chunks, p50/p90/p95/p99. Load is Poisson or closed-loop, measured from the scheduled send time.',
    tag: 'no coordinated omission',
    color: '#FBBF24',
  },
];

export function V3FeaturesSection() {
  return (
    <section id="v3" className="relative px-5 py-24 sm:px-6 sm:py-32">
      <div className="mx-auto max-w-5xl">
        <SectionHeader
          index="05"
          label="New in v3"
          title={<>Reuse the past, <Accent gradient>guess the future.</Accent></>}
          subtitle="Prefix caching skips work the engine has already done. Speculative decoding does several tokens of work per step. Both return exactly what greedy decoding would."
        />

        <div className="grid gap-4 lg:grid-cols-2 [&>*]:min-w-0">
          <Reveal>
            <Card className="h-full p-6">
              <div className="mb-4 flex items-center justify-between gap-3">
                <h3 className="text-lg font-semibold text-fg">Automatic prefix caching</h3>
                <span className="rounded-full bg-cyan/10 px-2 py-0.5 font-mono text-[10px] text-cyan">radix tree over blocks</span>
              </div>
              <p className="mb-5 text-[14px] leading-relaxed text-fg-2">
                Full blocks are content-hashed and ref-counted. A new request walks its prompt block by block and shares every
                hit instead of recomputing it. Freed blocks keep their contents in an LRU pool until memory is needed.
              </p>
              <PrefixTree />
              <SmokeStat
                {...prefix}
                extra={typeof prefixHit === 'number' && <span className="ml-2 text-cyan">· {Math.round(prefixHit * 100)}% prefix hit rate</span>}
              />
            </Card>
          </Reveal>

          <Reveal delay={90}>
            <Card className="h-full p-6">
              <div className="mb-4 flex items-center justify-between gap-3">
                <h3 className="text-lg font-semibold text-fg">Speculative decoding</h3>
                <span className="rounded-full bg-violet/10 px-2 py-0.5 font-mono text-[10px] text-violet">n-gram · draft model</span>
              </div>
              <p className="mb-5 text-[14px] leading-relaxed text-fg-2">
                Decode is bandwidth-bound, so verifying 5 tokens costs about as much as generating 1. A proposer guesses k
                tokens and the target checks them in one pass. Output is identical to greedy decoding.
              </p>
              <SpecDemo />
              <SmokeStat
                {...spec}
                extra={
                  specRun && (
                    <span className="ml-2 text-violet">
                      · {Math.round((specRun.spec_acceptance_rate ?? 0) * 100)}% accepted · copy-heavy output
                    </span>
                  )
                }
              />
              {specNote && (
                <p className="mt-3 rounded-lg border border-amber/20 bg-amber/[0.06] px-3 py-2 text-[12.5px] leading-relaxed text-fg-2">
                  <span className="text-amber">Caveat · </span>
                  {specNote}
                </p>
              )}
            </Card>
          </Reveal>
        </div>

        <div className="mt-4 grid gap-4 sm:grid-cols-3">
          {SMALL.map((f, i) => (
            <Reveal key={f.title} delay={i * 80}>
              <Card className="h-full">
                <span className="mb-4 inline-block h-1.5 w-8 rounded-full" style={{ backgroundColor: f.color, boxShadow: `0 0 12px ${f.color}` }} />
                <h3 className="text-[15px] font-semibold text-fg">{f.title}</h3>
                <p className="mt-2 text-[13.5px] leading-relaxed text-fg-3">{f.body}</p>
                <p className="mt-4 font-mono text-[10.5px]" style={{ color: f.color }}>{f.tag}</p>
              </Card>
            </Reveal>
          ))}
        </div>
        <p className="mt-3 font-mono text-[10.5px] text-fg-4">
          Figures: NVIDIA T4 (Colab) · Qwen2.5-1.5B-Instruct fp16 · offline ablation. The draft-token animation is illustrative.
        </p>
      </div>
    </section>
  );
}
