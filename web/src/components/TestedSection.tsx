import { SectionHeader, Accent } from '@/components/ui/SectionHeader';
import { Card } from '@/components/ui/Card';
import { Reveal } from '@/components/ui/Reveal';
import { PLATFORMS, SOFTWARE_TESTED, type PlatformEntry } from '@/data/gpuBenchmarks';
import { COLAB_URL } from '@/data/engine';

const DEVICE_LABEL: Record<PlatformEntry['device'], string> = { cpu: 'CPU', mps: 'MPS · Apple GPU', cuda: 'CUDA' };

function StatusPill({ status }: { status: PlatformEntry['status'] }) {
  return status === 'verified' ? (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-mint/30 bg-mint/10 px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.12em] text-mint">
      <svg viewBox="0 0 16 16" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2.2"><path d="M3 8.5l3 3 7-7" strokeLinecap="round" strokeLinejoin="round" /></svg>
      Verified
    </span>
  ) : (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-amber/30 bg-amber/10 px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.12em] text-amber">
      <span className="relative flex h-1.5 w-1.5">
        <span className="ping-soft absolute inset-0 rounded-full bg-amber" />
        <span className="relative h-1.5 w-1.5 rounded-full bg-amber" />
      </span>
      Pending
    </span>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="font-mono text-[9.5px] uppercase tracking-[0.14em] text-fg-4">{label}</dt>
      <dd className="mt-0.5 truncate font-mono text-[11.5px] text-fg-2" title={value}>{value}</dd>
    </div>
  );
}

function PlatformCard({ p }: { p: PlatformEntry }) {
  const verified = p.status === 'verified';
  return (
    <Card
      glow={verified}
      className={`flex h-full flex-col ${verified ? '' : '!border-dashed !border-white/[0.12] !bg-transparent'}`}
    >
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <p className="text-[17px] font-semibold tracking-tight text-fg">{p.gpu}</p>
          <p className="mt-0.5 font-mono text-[10.5px] text-fg-3">
            {DEVICE_LABEL[p.device]} · {p.where}
          </p>
        </div>
        <StatusPill status={p.status} />
      </div>

      <dl className="mb-4 grid grid-cols-[1fr_auto] gap-x-4 gap-y-3">
        <Field label={verified ? 'Model' : 'Planned model'} value={p.model} />
        <Field label="dtype" value={p.dtype} />
        <Field label="Date" value={p.date ?? '—'} />
      </dl>

      <p className="text-[13px] leading-relaxed text-fg-2">{p.summary}</p>

      {verified && p.checks && (
        <ul className="mt-4 space-y-1.5">
          {p.checks.map((c) => (
            <li key={c} className="flex items-start gap-2 text-[12.5px] text-fg-3">
              <span className="mt-[1px] text-mint">✓</span>
              {c}
            </li>
          ))}
        </ul>
      )}

      {!verified && (
        <div className="mt-auto pt-5" aria-label="Benchmarks coming">
          <div className="space-y-2 rounded-xl border border-line bg-white/[0.015] p-3">
            {[72, 54, 64].map((w, i) => (
              <div key={i} className="relative h-2 overflow-hidden rounded-full bg-white/[0.05]" style={{ width: `${w}%` }}>
                <span className="shimmer absolute inset-0" />
              </div>
            ))}
            <p className="pt-1 font-mono text-[10.5px] text-fg-4">benchmarks coming · no numbers until they’re measured</p>
          </div>
        </div>
      )}
    </Card>
  );
}

export function TestedSection() {
  return (
    <section id="tested" className="relative px-5 py-24 sm:px-6 sm:py-32">
      <div className="mx-auto max-w-5xl">
        <SectionHeader
          index="07"
          label="Where it's been tested"
          title={<>Tested where it runs. <Accent gradient>Honest where it hasn’t.</Accent></>}
          subtitle="Verified means it ran and matched: tests, exact-match checks and real-model runs. Pending GPUs have a planned model and a notebook ready, but no numbers yet. Nothing on this site is extrapolated."
        />

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {PLATFORMS.map((p, i) => (
            <Reveal key={p.id} delay={i * 60}>
              <PlatformCard p={p} />
            </Reveal>
          ))}

          {/* Colab callout fills the sixth cell */}
          <Reveal delay={PLATFORMS.length * 60}>
            <a
              href={COLAB_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="group relative flex h-full min-h-[240px] flex-col overflow-hidden rounded-[18px] border border-amber/30 p-5 transition-all duration-500 hover:-translate-y-1 hover:border-amber/60"
              style={{ background: 'radial-gradient(120% 90% at 100% 0%, rgba(251,191,36,0.16), transparent 55%), linear-gradient(180deg, rgba(255,255,255,0.03), transparent), #0B0C0F' }}
            >
              <span className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-amber/15 text-lg text-amber transition-transform duration-500 [transition-timing-function:var(--ease-spring)] group-hover:rotate-[-8deg] group-hover:scale-110">
                ▶
              </span>
              <p className="text-[17px] font-semibold tracking-tight text-fg">Run it on Colab</p>
              <p className="mt-2 text-[13px] leading-relaxed text-fg-2">
                Pick a T4, L4 or A100 runtime and <span className="text-fg">Run all</span>: unit tests, the GPU smoke test
                (exact-match on the GPU), the offline ablation and the serving sweep, then charts and a zip of the results.
              </p>
              <span className="mt-auto inline-flex w-fit items-center gap-2 pt-5 font-mono text-[12px] text-amber">
                Open PageServe_Colab.ipynb
                <span className="transition-transform duration-300 group-hover:translate-x-1 group-hover:-translate-y-0.5">↗</span>
              </span>
            </a>
          </Reveal>
        </div>

        <Reveal>
          <div className="mt-6 flex flex-col gap-3 rounded-2xl border border-line bg-white/[0.015] p-4 sm:flex-row sm:items-center sm:justify-between sm:p-5">
            <div>
              <p className="text-[14px] font-medium text-fg">Software versions tested</p>
              <p className="mt-0.5 text-[13px] text-fg-3">The engine is tested against both stacks.</p>
            </div>
            <div className="flex flex-wrap gap-2">
              {SOFTWARE_TESTED.map((v) => (
                <span key={v.transformers} className="rounded-full border border-line-2 bg-ink-900 px-3 py-1.5 font-mono text-[11.5px] text-fg-2">
                  <span className="text-fg">transformers {v.transformers}</span>
                  <span className="text-fg-4"> + </span>
                  torch {v.torch}
                </span>
              ))}
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
