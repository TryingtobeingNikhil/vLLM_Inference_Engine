import { GitHubStats, StarButton, REPO_URL } from '@/components/GitHubStats';
import { LogoMark } from '@/components/Nav';
import { Reveal } from '@/components/ui/Reveal';

export function GitHubFooter() {
  const LINKS = [
    { label: 'Source',           href: REPO_URL },
    { label: 'Issues',           href: `${REPO_URL}/issues` },
    { label: 'README',           href: `${REPO_URL}/blob/main/README.md` },
    { label: 'bench_direct',     href: `${REPO_URL}/blob/main/bench_direct_results.json` },
    { label: 'bench_phases',     href: `${REPO_URL}/blob/main/bench_phases_results.json` },
    { label: 'baseline_metrics', href: `${REPO_URL}/blob/main/baseline_metrics.json` },
  ];

  const TECH_STACK = [
    'Python 3.11',
    'PyTorch / MPS',
    'FastAPI',
    'asyncio',
    'Qwen2-0.5B',
    'Next.js 14',
    'Tailwind CSS',
  ];

  return (
    <footer className="relative overflow-hidden border-t border-line px-5 pb-10 pt-24 sm:px-6">
      <div className="pointer-events-none absolute bottom-[-20rem] left-1/2 h-[34rem] w-[70rem] -translate-x-1/2 rounded-full bg-[radial-gradient(closest-side,rgba(74,222,128,0.10),transparent)] blur-2xl" />

      <div className="relative mx-auto max-w-5xl">
        {/* Closing CTA */}
        <Reveal className="mb-20 text-center">
          <p className="accent-serif text-2xl text-fg-3 sm:text-3xl">Curious how it works under the hood?</p>
          <h2 className="mt-2 text-balance text-4xl font-semibold tracking-tight text-fg sm:text-5xl">
            Read the source. <span className="text-gradient">Break the scheduler.</span>
          </h2>
          <div className="mt-8 flex flex-col items-center gap-4">
            <StarButton label="Star PageServe on GitHub" />
            <GitHubStats className="justify-center" />
          </div>
        </Reveal>

        <div className="grid gap-10 border-t border-line pt-10 sm:grid-cols-[1.3fr_1fr]">
          <div>
            <div className="mb-3 flex items-center gap-2.5">
              <LogoMark className="h-6 w-6" />
              <span className="text-lg font-semibold tracking-tight text-fg">PageServe</span>
              <span className="rounded-full border border-line-2 px-2 py-0.5 font-mono text-[10px] text-fg-3">v0.1.0</span>
            </div>
            <p className="mb-5 max-w-md text-[13.5px] leading-relaxed text-fg-3">
              An LLM inference engine built from first principles — continuous batching, paged KV cache, chunked
              prefill and CPU swap pool, implemented without vLLM or HuggingFace generate.
            </p>
            <div className="flex flex-wrap gap-1.5">
              {TECH_STACK.map((t) => (
                <span key={t} className="rounded-full border border-line px-2.5 py-1 font-mono text-[10.5px] text-fg-3">
                  {t}
                </span>
              ))}
            </div>
          </div>

          <div>
            <p className="mb-3 font-mono text-[10.5px] uppercase tracking-[0.16em] text-fg-4">Explore</p>
            <ul className="grid grid-cols-2 gap-x-6 gap-y-2">
              {LINKS.map((link) => (
                <li key={link.label}>
                  <a
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group inline-flex items-center gap-1 font-mono text-[12px] text-fg-3 transition-colors hover:text-fg"
                  >
                    {link.label}
                    <span className="translate-y-px opacity-0 transition-all duration-300 group-hover:translate-x-0.5 group-hover:opacity-100">↗</span>
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Oversized wordmark */}
        <div className="pointer-events-none mt-16 select-none overflow-hidden" aria-hidden="true">
          <p
            className="text-center text-[18vw] font-semibold leading-[0.8] tracking-[-0.06em] text-transparent lg:text-[11.5rem]"
            style={{
              WebkitTextStroke: '1px rgba(255,255,255,0.09)',
              backgroundImage: 'linear-gradient(180deg, rgba(255,255,255,0.06), transparent 75%)',
              WebkitBackgroundClip: 'text',
              backgroundClip: 'text',
            }}
          >
            PageServe
          </p>
        </div>

        <div className="mt-8 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-6 font-mono text-[10.5px] text-fg-4">
          <span>Apple M2 · Qwen/Qwen2-0.5B · float16 · MPS — all benchmarks measured on real hardware</span>
          <div className="flex items-center gap-4">
            <span>© {new Date().getFullYear()} PageServe</span>
            <a href="#top" className="rounded-full border border-line px-2.5 py-1 text-fg-3 transition-colors hover:border-line-2 hover:text-fg">
              ↑ Top
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
