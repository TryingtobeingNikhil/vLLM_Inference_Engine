'use client';

import { useState } from 'react';
import { SectionHeader, Accent } from '@/components/ui/SectionHeader';
import { Card } from '@/components/ui/Card';
import { Reveal } from '@/components/ui/Reveal';
import { PHASES, type PhaseEntry } from '@/data/phases';

type Tag = PhaseEntry['tag'];

const TAG_COLORS: Record<Tag, string> = {
  scheduling:    '#60A5FA',
  memory:        '#4ADE80',
  observability: '#A78BFA',
  testing:       '#FBBF24',
};

function PhaseRow({ phase, open, onToggle, muted }: { phase: PhaseEntry; open: boolean; onToggle: () => void; muted: boolean }) {
  const color = TAG_COLORS[phase.tag];

  return (
    <li className={`relative pl-12 transition-opacity duration-500 sm:pl-16 ${muted ? 'opacity-25' : 'opacity-100'}`}>
      {/* Node on the rail */}
      <span
        className="absolute left-[11px] top-[22px] flex h-[18px] w-[18px] items-center justify-center rounded-md border transition-all duration-500 sm:left-[19px]"
        style={{
          borderColor: `${color}99`,
          backgroundColor: open ? color : '#0B0C0F',
          boxShadow: open ? `0 0 0 4px ${color}22, 0 0 18px ${color}` : `0 0 0 4px #07080A`,
        }}
        aria-hidden="true"
      >
        <span className="h-1.5 w-1.5 rounded-[2px] transition-colors duration-300" style={{ backgroundColor: open ? '#07080A' : color }} />
      </span>

      <div
        className={`rounded-2xl border transition-all duration-500 ${
          open ? 'border-line-2 bg-ink-900' : 'border-transparent hover:border-line hover:bg-white/[0.015]'
        }`}
      >
        <button
          onClick={onToggle}
          aria-expanded={open}
          className="flex w-full flex-col gap-2 px-4 py-4 text-left sm:px-5"
        >
          <div className="flex w-full items-center gap-3">
            <span className="font-mono text-[11px] tabular-nums text-fg-4">{String(phase.phase).padStart(2, '0')}</span>
            <span className="flex-1 text-[15px] font-medium text-fg">{phase.title}</span>
            <span
              className="hidden rounded-full px-2 py-0.5 font-mono text-[9.5px] uppercase tracking-[0.14em] sm:inline"
              style={{ color, backgroundColor: `${color}14` }}
            >
              {phase.tag}
            </span>
            <span
              className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-line-2 text-fg-3 transition-transform duration-500 [transition-timing-function:var(--ease-spring)]"
              style={{ transform: open ? 'rotate(45deg)' : 'none' }}
              aria-hidden="true"
            >
              +
            </span>
          </div>

          {(phase.metricBefore || phase.metricAfter) && (
            <div className="flex flex-wrap items-center gap-2 pl-7 font-mono text-[11px]">
              <span className="text-fg-4">{phase.metricLabel ?? 'Result'}</span>
              {phase.metricBefore && (
                <>
                  <span className="text-fg-3 line-through decoration-rose/50">{phase.metricBefore}</span>
                  <span className="text-fg-4">→</span>
                </>
              )}
              <span className="rounded-md px-1.5 py-0.5" style={{ color, backgroundColor: `${color}12` }}>
                {phase.metricAfter}
              </span>
            </div>
          )}
        </button>

        <div className="collapse-grid" data-open={open}>
          <div>
            <div className="grid gap-4 px-4 pb-5 pl-11 sm:grid-cols-2 sm:px-5 sm:pl-12">
              <div>
                <p className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.16em] text-rose/80">Problem</p>
                <p className="text-[13.5px] leading-relaxed text-fg-2">{phase.problem}</p>
              </div>
              <div>
                <p className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.16em] text-mint/80">Solution</p>
                <p className="text-[13.5px] leading-relaxed text-fg-2">{phase.solution}</p>
              </div>
              <div className="flex items-center gap-2 sm:col-span-2">
                <span className="font-mono text-[10px] text-fg-4">source</span>
                <code className="rounded-md border border-line bg-ink-950 px-2 py-0.5 font-mono text-[11px] text-sky">{phase.file}</code>
              </div>
            </div>
          </div>
        </div>
      </div>
    </li>
  );
}

const NEXT_UP = [
  { title: 'Tensor-level batching', desc: 'FlashAttention-2 or custom CUDA kernels instead of thread-based serialization.', icon: '↯' },
  { title: 'Disconnect-aware cancellation', desc: 'Propagate client disconnects into admitted sequences to stop expensive decode work immediately.', icon: '✂' },
  { title: 'Speculative decoding', desc: 'Integrate a draft model to verify candidate tokens in parallel.', icon: '◎' },
  { title: 'Dynamic block sizing', desc: 'Experiment with block sizes 8 and 32 to study cache chunk overhead.', icon: '▦' },
];

export function PhaseBuildLogSection() {
  const [open, setOpen] = useState<number | null>(2);
  const [filter, setFilter] = useState<Tag | 'all'>('all');

  return (
    <section id="phases" className="relative px-5 py-24 sm:px-6 sm:py-32">
      <div className="mx-auto max-w-5xl">
        <SectionHeader
          index="05"
          label="Development history"
          title={<>Eleven phases, <Accent gradient>one engine.</Accent></>}
          subtitle="PageServe was built phase by phase, retracing how production inference engines evolved. Open any phase for the problem it hit and how it was solved."
        />

        <Reveal>
          <div className="mb-6 flex flex-wrap gap-2" role="group" aria-label="Filter phases by area">
            {(['all', ...Object.keys(TAG_COLORS)] as (Tag | 'all')[]).map((tag) => {
              const on = filter === tag;
              const color = tag === 'all' ? '#EDEDEF' : TAG_COLORS[tag];
              const count = tag === 'all' ? PHASES.length : PHASES.filter((p) => p.tag === tag).length;
              return (
                <button
                  key={tag}
                  onClick={() => setFilter(tag)}
                  aria-pressed={on}
                  className="flex items-center gap-2 rounded-full border px-3 py-1.5 font-mono text-[11px] capitalize transition-all duration-300"
                  style={{
                    color: on ? color : '#767680',
                    borderColor: on ? `${color}55` : 'rgba(255,255,255,0.08)',
                    backgroundColor: on ? `${color}12` : 'transparent',
                  }}
                >
                  {tag !== 'all' && <span className="h-1.5 w-1.5 rounded-[2px]" style={{ backgroundColor: color }} />}
                  {tag}
                  <span className="text-fg-4">{count}</span>
                </button>
              );
            })}
          </div>
        </Reveal>

        <Reveal>
          <ol className="relative">
            {/* Rail */}
            <span
              className="absolute bottom-6 left-[19px] top-6 w-px sm:left-[27px]"
              style={{ background: 'linear-gradient(to bottom, #60A5FA66, #4ADE8066 45%, #A78BFA66 85%, #FBBF2466)' }}
              aria-hidden="true"
            />
            {PHASES.map((phase) => (
              <PhaseRow
                key={phase.phase}
                phase={phase}
                open={open === phase.phase}
                muted={filter !== 'all' && phase.tag !== filter}
                onToggle={() => setOpen((o) => (o === phase.phase ? null : phase.phase))}
              />
            ))}
          </ol>
        </Reveal>

        <div className="mt-14">
          <Reveal>
            <p className="mb-5 flex items-center gap-3">
              <span className="accent-serif text-2xl text-fg">What I&apos;d do differently</span>
              <span className="h-px flex-1 bg-gradient-to-r from-line-2 to-transparent" />
            </p>
          </Reveal>
          <div className="grid gap-3 sm:grid-cols-2">
            {NEXT_UP.map((item, i) => (
              <Reveal key={item.title} delay={i * 70}>
                <Card className="group h-full">
                  <div className="flex gap-4">
                    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-line-2 bg-white/[0.03] text-fg-2 transition-transform duration-500 [transition-timing-function:var(--ease-spring)] group-hover:-rotate-6 group-hover:scale-110">
                      {item.icon}
                    </span>
                    <div>
                      <p className="text-[14.5px] font-medium text-fg">{item.title}</p>
                      <p className="mt-1 text-[13.5px] leading-relaxed text-fg-3">{item.desc}</p>
                    </div>
                  </div>
                </Card>
              </Reveal>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
