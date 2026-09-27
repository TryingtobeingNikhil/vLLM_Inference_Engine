'use client';

import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { REPO, type RepoStats } from '@/lib/github';

export { REPO };
export const REPO_URL = `https://github.com/${REPO}`;
const DURATION = 2800; // ms

// Last-resort fallback, only used if both the server and browser fetches fail.
const FALLBACK: RepoStats = { stars: 97, forks: 10, watchers: 97 };

// Stats fetched on the server at build/revalidate time, so the first paint is already correct.
const InitialStatsContext = createContext<RepoStats | null>(null);

export function RepoStatsProvider({ initial, children }: { initial: RepoStats | null; children: ReactNode }) {
  return <InitialStatsContext.Provider value={initial}>{children}</InitialStatsContext.Provider>;
}

// One browser request per page load, shared by every widget — refreshes the server value.
let statsPromise: Promise<RepoStats | null> | null = null;
function fetchStats(): Promise<RepoStats | null> {
  statsPromise ??= fetch(`https://api.github.com/repos/${REPO}`)
    .then((r) => (r.ok ? r.json() : null))
    .then((data) =>
      data && data.stargazers_count !== undefined
        ? { stars: data.stargazers_count, forks: data.forks_count, watchers: data.watchers_count }
        : null,
    )
    .catch(() => null);
  return statsPromise;
}

function useRepoStats() {
  const initial = useContext(InitialStatsContext);
  const [stats, setStats] = useState<RepoStats>(initial ?? FALLBACK);
  const [live, setLive] = useState(initial !== null);
  useEffect(() => {
    let alive = true;
    fetchStats().then((s) => {
      if (alive && s) {
        setStats(s);
        setLive(true);
      }
    });
    return () => { alive = false; };
  }, []);
  return { stats, live };
}

/**
 * Animates from the current displayed value → new target whenever target changes.
 * Uses ease-out cubic so it feels smooth and decelerates into the final number.
 */
function useCountUp(target: number): number {
  const [display, setDisplay] = useState(0);
  const rafRef    = useRef<number | null>(null);
  const startRef  = useRef<number | null>(null);
  const fromRef   = useRef<number>(0); // where this animation segment starts from

  useEffect(() => {
    if (target === 0) return;

    // Cancel any running animation
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    startRef.current = null;

    // Capture the current display value as the starting point
    const from = fromRef.current;
    const delta = target - from;

    const animate = (ts: number) => {
      if (!startRef.current) startRef.current = ts;
      const elapsed  = ts - startRef.current;
      const progress = Math.min(elapsed / DURATION, 1);
      // ease-out cubic: 1 - (1-t)^3
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = Math.round(from + eased * delta);
      setDisplay(current);
      fromRef.current = current;
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(animate);
      } else {
        fromRef.current = target; // lock to exact target at end
      }
    };

    rafRef.current = requestAnimationFrame(animate);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [target]); // eslint-disable-line react-hooks/exhaustive-deps

  return display;
}

export function GitHubIcon({ className = 'h-4 w-4' }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className={className} aria-hidden="true">
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.8 8.2 11.38.6.1.82-.26.82-.58v-2.03c-3.34.72-4.04-1.6-4.04-1.6-.55-1.39-1.34-1.76-1.34-1.76-1.09-.74.08-.73.08-.73 1.2.09 1.84 1.24 1.84 1.24 1.07 1.83 2.8 1.3 3.49 1 .1-.78.42-1.3.76-1.6-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.13-.3-.54-1.52.12-3.18 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 3-.4c1.02.01 2.04.14 3 .4 2.3-1.55 3.3-1.23 3.3-1.23.66 1.66.25 2.88.12 3.18.77.84 1.24 1.91 1.24 3.22 0 4.61-2.81 5.63-5.48 5.92.43.37.81 1.1.81 2.22v3.29c0 .32.22.69.82.57C20.56 21.8 24 17.3 24 12c0-6.63-5.37-12-12-12z" />
    </svg>
  );
}

function StarIcon({ className = 'h-3.5 w-3.5' }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" fill="currentColor" className={className} aria-hidden="true">
      <path d="M8 .25a.75.75 0 0 1 .673.418l1.882 3.815 4.21.612a.75.75 0 0 1 .416 1.279l-3.046 2.97.719 4.192a.751.751 0 0 1-1.088.791L8 12.347l-3.766 1.98a.75.75 0 0 1-1.088-.79l.72-4.194L.873 6.374a.75.75 0 0 1 .416-1.28l4.21-.611L7.327.668A.75.75 0 0 1 8 .25Z" />
    </svg>
  );
}

function ForkIcon({ className = 'h-3.5 w-3.5' }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" fill="currentColor" className={className} aria-hidden="true">
      <path d="M5 5.372v.878c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75v-.878a2.25 2.25 0 1 1 1.5 0v.878a2.25 2.25 0 0 1-2.25 2.25h-1.5v2.128a2.251 2.251 0 1 1-1.5 0V8.5h-1.5A2.25 2.25 0 0 1 3.5 6.25v-.878a2.25 2.25 0 1 1 1.5 0ZM5 3.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0Zm6.75.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Zm-3 8.75a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0Z" />
    </svg>
  );
}

/** Primary CTA: a white pill that doubles as a live star counter. */
export function StarButton({ label = 'Star on GitHub', compact = false }: { label?: string; compact?: boolean }) {
  const { stats } = useRepoStats();
  const stars = useCountUp(stats.stars);
  return (
    <a
      href={REPO_URL}
      target="_blank"
      rel="noopener noreferrer"
      className={`group relative inline-flex items-center overflow-hidden rounded-full bg-fg font-medium text-ink-950 shadow-[0_0_0_1px_rgba(255,255,255,0.1),0_8px_30px_-8px_rgba(74,222,128,0.45)] transition-all duration-300 hover:-translate-y-0.5 hover:shadow-[0_0_0_1px_rgba(255,255,255,0.2),0_14px_40px_-8px_rgba(74,222,128,0.6)] active:translate-y-0 ${
        compact ? 'gap-2 py-1.5 pl-3 pr-1.5 text-[13px]' : 'gap-2.5 py-2 pl-5 pr-2 text-sm'
      }`}
    >
      <span className="shimmer pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
      <GitHubIcon className={compact ? 'h-3.5 w-3.5' : 'h-4 w-4'} />
      {label}
      <span
        className={`inline-flex items-center gap-1 rounded-full bg-ink-950/90 font-mono text-fg ${
          compact ? 'px-2 py-0.5 text-[11px]' : 'px-2.5 py-1 text-xs'
        }`}
      >
        <StarIcon className="h-3 w-3 text-amber transition-transform duration-500 group-hover:rotate-[72deg] group-hover:scale-125" />
        <span className="tabular-nums">{stars}</span>
      </span>
    </a>
  );
}

export function GitHubStats({ className = '' }: { className?: string }) {
  const { stats, live } = useRepoStats();
  const displayStars = useCountUp(stats.stars);
  const displayForks = useCountUp(stats.forks);

  const pill =
    'group flex items-center gap-1.5 rounded-full border border-line-2 bg-white/[0.02] px-3 py-1.5 font-mono text-xs text-fg transition-all duration-300 hover:-translate-y-0.5';

  return (
    <div className={`flex flex-wrap items-center gap-2 ${className}`}>
      <a
        href={`${REPO_URL}/stargazers`}
        target="_blank"
        rel="noopener noreferrer"
        className={`${pill} hover:border-amber/50 hover:bg-amber/5`}
        title="GitHub Stars"
      >
        <StarIcon className="h-3.5 w-3.5 text-amber transition-transform duration-500 group-hover:rotate-[72deg]" />
        <span className="tabular-nums">{displayStars}</span>
        <span className="text-fg-3">stars</span>
      </a>
      <a
        href={`${REPO_URL}/forks`}
        target="_blank"
        rel="noopener noreferrer"
        className={`${pill} hover:border-mint/50 hover:bg-mint/5`}
        title="GitHub Forks"
      >
        <ForkIcon className="h-3.5 w-3.5 text-mint" />
        <span className="tabular-nums">{displayForks}</span>
        <span className="text-fg-3">forks</span>
      </a>
      <span className="flex items-center gap-1.5 pl-1 font-mono text-[10px] text-fg-4">
        <span className={`inline-block h-1.5 w-1.5 rounded-full ${live ? 'bg-mint' : 'bg-fg-4'}`} />
        {live ? 'live' : 'cached'}
      </span>
    </div>
  );
}
