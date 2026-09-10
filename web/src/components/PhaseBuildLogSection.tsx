'use client';

import { useState } from 'react';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { PHASES, type PhaseEntry } from '@/data/phases';

const TAG_COLORS = {
  scheduling:   { dot: '#60A5FA', dim: '#1a2540' },
  memory:       { dot: '#4ADE80', dim: '#1a3d27' },
  observability:{ dot: '#a78bfa', dim: '#2a1a40' },
  testing:      { dot: '#FBBF24', dim: '#3d2e0a' },
};

function PhaseRow({ phase, isLast }: { phase: PhaseEntry; isLast: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const colors = TAG_COLORS[phase.tag];

  return (
    <div className={`relative ${!isLast ? 'border-b border-[#1a1a1a]' : ''}`}>
      {/* Timeline connector */}
      <div
        className="absolute left-[19px] top-0 bottom-0 w-px"
        style={{ backgroundColor: '#1e1e1e' }}
      />

      <button
        onClick={() => setExpanded((e) => !e)}
        className="relative w-full px-4 py-4 text-left transition-colors hover:bg-[#111111]"
      >
        <div className="flex items-start gap-4">
          {/* Dot */}
          <div
            className="relative z-10 mt-0.5 h-4 w-4 flex-shrink-0 border"
            style={{
              backgroundColor: colors.dim,
              borderColor: colors.dot,
            }}
          />

          {/* Phase number + title */}
          <div className="flex flex-1 flex-wrap items-baseline gap-3">
            <span className="font-mono text-xs text-[#333333]">
              Phase {String(phase.phase).padStart(2, '0')}
            </span>
            <span className="font-mono text-sm text-[#cccccc]">{phase.title}</span>
            <span
              className="ml-auto font-mono text-[9px] uppercase tracking-widest"
              style={{ color: colors.dot }}
            >
              {phase.tag}
            </span>
            <span className="font-mono text-[11px] text-[#333333]">
              {expanded ? '▲' : '▼'}
            </span>
          </div>
        </div>

        {/* Inline metric delta (always visible) */}
        {(phase.metricBefore || phase.metricAfter) && (
          <div className="ml-8 mt-2 flex flex-wrap items-center gap-2">
            <span className="font-mono text-[10px] text-[#444444]">{phase.metricLabel ?? 'Result'}:</span>
            {phase.metricBefore && (
              <>
                <span className="font-mono text-[10px] text-[#555555] line-through">
                  {phase.metricBefore}
                </span>
                <span className="font-mono text-[10px] text-[#333333]">→</span>
              </>
            )}
            <span
              className="font-mono text-[10px]"
              style={{ color: colors.dot }}
            >
              {phase.metricAfter}
            </span>
          </div>
        )}
      </button>

      {/* Expanded detail */}
      {expanded && (
        <div className="ml-8 border-t border-[#1a1a1a] bg-[#0d0d0d] px-4 py-4">
          <div className="mb-3 flex items-center gap-2">
            <span className="font-mono text-[9px] text-[#333333]">source:</span>
            <code className="font-mono text-[10px] text-[#60A5FA]">{phase.file}</code>
          </div>
          <div className="space-y-2">
            <div>
              <p className="mb-1 font-mono text-[9px] uppercase tracking-widest text-[#333333]">Problem</p>
              <p className="text-xs leading-relaxed text-[#666666]">{phase.problem}</p>
            </div>
            <div>
              <p className="mb-1 font-mono text-[9px] uppercase tracking-widest text-[#333333]">Solution</p>
              <p className="text-xs leading-relaxed text-[#777777]">{phase.solution}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function PhaseBuildLogSection() {
  return (
    <section id="phases" className="border-b border-[#1e1e1e] px-6 py-20 sm:px-10 lg:px-16">
      <div className="mx-auto max-w-5xl">
        <SectionHeader
          label="// development history"
          title="11 Phases — Built Incrementally"
          subtitle="PageServe was constructed phase-by-phase to map the evolution of production inference engine design."
        />

        {/* Tag legend */}
        <div className="mb-6 flex flex-wrap gap-4">
          {Object.entries(TAG_COLORS).map(([tag, colors]) => (
            <div key={tag} className="flex items-center gap-1.5">
              <div
                className="h-2.5 w-2.5"
                style={{ backgroundColor: colors.dot }}
              />
              <span className="font-mono text-[10px] uppercase tracking-widest text-[#444444]">
                {tag}
              </span>
            </div>
          ))}
        </div>

        {/* Phase list */}
        <div className="border border-[#2a2a2a] bg-[#0a0a0a]">
          {PHASES.map((phase, i) => (
            <PhaseRow
              key={phase.phase}
              phase={phase}
              isLast={i === PHASES.length - 1}
            />
          ))}
        </div>

        {/* "What I'd do differently" callout */}
        <div className="mt-6 border border-[#2a2a2a] bg-[#111111] p-4">
          <p className="mb-3 font-mono text-[10px] uppercase tracking-widest text-[#444444]">
            What I'd Do Differently
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            {[
              { title: 'Tensor-Level Batching', desc: 'FlashAttention-2 or custom CUDA kernels instead of thread-based serialization.' },
              { title: 'Disconnect-Aware Cancellation', desc: 'Propagate client disconnects into admitted sequences to stop expensive decode work immediately.' },
              { title: 'Speculative Decoding', desc: 'Integrate a draft model to verify candidate tokens in parallel.' },
              { title: 'Dynamic Block Sizing', desc: 'Experiment with block sizes 8 and 32 to study cache chunk overhead.' },
            ].map((item) => (
              <div key={item.title} className="border-l border-[#2a2a2a] pl-3">
                <p className="mb-0.5 font-mono text-[11px] text-[#888888]">{item.title}</p>
                <p className="text-xs text-[#444444]">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
