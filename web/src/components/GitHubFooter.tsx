import { GitHubStats } from '@/components/GitHubStats';

export function GitHubFooter() {
  const REPO_URL = 'https://github.com/TryingtobeingNikhil/vLLM_Inference_Engine';

  const LINKS = [
    { label: 'Source',         href: REPO_URL },
    { label: 'Issues',         href: `${REPO_URL}/issues` },
    { label: 'bench_direct',   href: `${REPO_URL}/blob/main/bench_direct_results.json` },
    { label: 'bench_phases',   href: `${REPO_URL}/blob/main/bench_phases_results.json` },
    { label: 'baseline_metrics', href: `${REPO_URL}/blob/main/baseline_metrics.json` },
    { label: 'README',         href: `${REPO_URL}/blob/main/README.md` },
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
    <footer className="border-t border-[#1e1e1e] px-6 py-16 sm:px-10 lg:px-16">
      <div className="mx-auto max-w-5xl">
        {/* Top row */}
        <div className="mb-10 grid gap-8 sm:grid-cols-[1fr_auto]">
          {/* Left — project identity */}
          <div>
            <div className="mb-1 flex items-baseline gap-3">
              <span className="font-mono text-xl font-semibold text-[#e8e8e8]">PageServe</span>
              <span className="font-mono text-[10px] uppercase tracking-widest text-[#333333]">
                v0.1.0
              </span>
            </div>
            <p className="mb-4 max-w-md font-mono text-xs leading-relaxed text-[#444444]">
              An LLM inference engine built from first principles — continuous batching, paged KV
              cache, chunked prefill and CPU swap pool, implemented without vLLM or HuggingFace
              generate.
            </p>

            {/* Tech stack pills */}
            <div className="flex flex-wrap gap-2">
              {TECH_STACK.map((t) => (
                <span
                  key={t}
                  className="border border-[#1e1e1e] px-2 py-0.5 font-mono text-[10px] text-[#333333]"
                >
                  {t}
                </span>
              ))}
            </div>
          </div>

          {/* Right — CTA */}
          <div className="flex flex-col items-start gap-3 sm:items-end">
            <a
              href={REPO_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 border border-[#e8e8e8] px-5 py-2.5 font-mono text-sm text-[#e8e8e8] transition-colors hover:bg-[#e8e8e8] hover:text-[#0a0a0a]"
            >
              {/* GitHub icon (inline SVG, no external dep) */}
              <svg
                viewBox="0 0 24 24"
                fill="currentColor"
                className="h-4 w-4"
                aria-hidden="true"
              >
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.8 8.2 11.38.6.1.82-.26.82-.58v-2.03c-3.34.72-4.04-1.6-4.04-1.6-.55-1.39-1.34-1.76-1.34-1.76-1.09-.74.08-.73.08-.73 1.2.09 1.84 1.24 1.84 1.24 1.07 1.83 2.8 1.3 3.49 1 .1-.78.42-1.3.76-1.6-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.13-.3-.54-1.52.12-3.18 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 3-.4c1.02.01 2.04.14 3 .4 2.3-1.55 3.3-1.23 3.3-1.23.66 1.66.25 2.88.12 3.18.77.84 1.24 1.91 1.24 3.22 0 4.61-2.81 5.63-5.48 5.92.43.37.81 1.1.81 2.22v3.29c0 .32.22.69.82.57C20.56 21.8 24 17.3 24 12c0-6.63-5.37-12-12-12z" />
              </svg>
              View on GitHub
            </a>
            <GitHubStats />
            <span className="font-mono text-[10px] text-[#333333]">
              TryingtobeingNikhil / vLLM_Inference_Engine
            </span>
          </div>
        </div>

        {/* Divider */}
        <div className="mb-8 border-t border-[#1a1a1a]" />

        {/* Links row */}
        <div className="mb-8 flex flex-wrap gap-x-6 gap-y-2">
          {LINKS.map((link) => (
            <a
              key={link.label}
              href={link.href}
              target="_blank"
              rel="noopener noreferrer"
              className="font-mono text-[11px] text-[#444444] transition-colors hover:text-[#888888]"
            >
              {link.label}
            </a>
          ))}
        </div>

        {/* Bottom bar */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span className="font-mono text-[10px] text-[#2a2a2a]">
            Hardware: Apple M2 · Qwen/Qwen2-0.5B · float16 · MPS
          </span>
          <div className="flex items-center gap-4">
            <span className="font-mono text-[10px] text-[#2a2a2a]">
              All benchmark data measured on real hardware
            </span>
            <span className="font-mono text-[10px] text-[#2a2a2a]">
              © {new Date().getFullYear()} PageServe
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
