'use client';

import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { StarButton } from '@/components/GitHubStats';

const LINKS = [
  { id: 'scheduler',  label: 'Scheduler' },
  { id: 'comparison', label: 'Batching' },
  { id: 'kvcache',    label: 'KV Cache' },
  { id: 'swapping',   label: 'Preemption' },
  { id: 'v3',         label: 'New in v3' },
  { id: 'phases',     label: 'Build log' },
  { id: 'tested',     label: 'Tested' },
  { id: 'benchmarks', label: 'Results' },
  { id: 'api',        label: 'API' },
];

/** Four KV blocks — three owned, one being paged in. */
export function LogoMark({ className = 'h-5 w-5' }: { className?: string }) {
  return (
    <svg viewBox="0 0 20 20" className={className} aria-hidden="true">
      <rect x="1" y="1" width="8" height="8" rx="2.2" fill="#4ADE80" />
      <rect x="11" y="1" width="8" height="8" rx="2.2" fill="#60A5FA" />
      <rect x="1" y="11" width="8" height="8" rx="2.2" fill="#A78BFA" />
      <rect x="11.5" y="11.5" width="7" height="7" rx="1.8" fill="none" stroke="#FBBF24" strokeWidth="1" strokeDasharray="2 1.6">
        <animate attributeName="stroke-dashoffset" from="0" to="-7.2" dur="1.6s" repeatCount="indefinite" />
      </rect>
    </svg>
  );
}

export function Nav() {
  const [active, setActive] = useState<string | null>(null);
  const [scrolled, setScrolled] = useState(false);
  const [progress, setProgress] = useState(0);
  const [indicator, setIndicator] = useState<{ left: number; width: number } | null>(null);
  const linkRefs = useRef<Record<string, HTMLAnchorElement | null>>({});

  useEffect(() => {
    const onScroll = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      setProgress(max > 0 ? window.scrollY / max : 0);
      setScrolled(window.scrollY > 24);
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    const sections = LINKS.map((l) => document.getElementById(l.id)).filter(Boolean) as HTMLElement[];
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) if (e.isIntersecting) setActive(e.target.id);
      },
      { rootMargin: '-45% 0px -50% 0px' },
    );
    sections.forEach((s) => io.observe(s));
    // Above the first section → nothing active.
    const top = () => { if (window.scrollY < window.innerHeight * 0.5) setActive(null); };
    window.addEventListener('scroll', top, { passive: true });
    return () => { io.disconnect(); window.removeEventListener('scroll', top); };
  }, []);

  useLayoutEffect(() => {
    const el = active ? linkRefs.current[active] : null;
    setIndicator(el ? { left: el.offsetLeft, width: el.offsetWidth } : null);
  }, [active]);

  return (
    <header className="pointer-events-none fixed inset-x-0 top-3 z-50 flex justify-center px-3 sm:top-4">
      <nav
        className={`pointer-events-auto relative flex w-full max-w-5xl items-center gap-2 overflow-hidden rounded-full border py-1.5 pl-4 pr-1.5 transition-all duration-500 ${
          scrolled
            ? 'border-line-2 bg-ink-900/70 shadow-[0_10px_40px_-12px_rgba(0,0,0,0.8)] backdrop-blur-xl'
            : 'border-transparent bg-transparent'
        }`}
        aria-label="Primary"
      >
        <a href="#top" className="group flex items-center gap-2 pr-2" aria-label="PageServe — back to top">
          <LogoMark className="h-5 w-5 transition-transform duration-500 group-hover:rotate-90" />
          <span className="text-[15px] font-semibold tracking-tight text-fg">PageServe</span>
        </a>

        <div className="relative mx-auto hidden items-center lg:flex">
          {indicator && (
            <span
              className="absolute top-1/2 h-8 -translate-y-1/2 rounded-full bg-white/[0.07] transition-all duration-500 [transition-timing-function:var(--ease-spring)]"
              style={{ left: indicator.left, width: indicator.width }}
              aria-hidden="true"
            />
          )}
          {LINKS.map((l) => (
            <a
              key={l.id}
              ref={(el) => { linkRefs.current[l.id] = el; }}
              href={`#${l.id}`}
              className={`relative rounded-full px-3 py-1.5 text-[13px] transition-colors duration-300 ${
                active === l.id ? 'text-fg' : 'text-fg-3 hover:text-fg'
              }`}
            >
              {l.label}
            </a>
          ))}
        </div>

        <div className="ml-auto lg:ml-0">
          <StarButton label="Star" compact />
        </div>

        {/* Reading progress — a thin token stream along the bottom edge */}
        <span
          className="absolute bottom-0 left-0 h-px bg-gradient-to-r from-mint via-cyan to-violet transition-opacity duration-500"
          style={{ width: `${progress * 100}%`, opacity: scrolled ? 0.8 : 0 }}
          aria-hidden="true"
        />
      </nav>
    </header>
  );
}
