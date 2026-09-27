'use client';

import { type CSSProperties, type MouseEvent, type ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
  pad?: boolean;
  glow?: boolean;
  style?: CSSProperties;
}

function track(e: MouseEvent<HTMLDivElement>) {
  const r = e.currentTarget.getBoundingClientRect();
  e.currentTarget.style.setProperty('--mx', `${e.clientX - r.left}px`);
  e.currentTarget.style.setProperty('--my', `${e.clientY - r.top}px`);
}

/** Rounded surface with a cursor-following spotlight and (optionally) a lit hairline border. */
export function Card({ children, className = '', pad = true, glow = true, style }: CardProps) {
  return (
    <div
      onMouseMove={track}
      style={style}
      className={`surface spotlight ${glow ? 'ring-glow' : ''} ${pad ? 'p-5' : ''} ${className}`}
    >
      {children}
    </div>
  );
}

interface WindowProps {
  title: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}

/** A macOS-ish app window — the frame for every live visualisation on the page. */
export function Window({ title, right, children, className = '', bodyClassName = '' }: WindowProps) {
  return (
    <div
      className={`overflow-hidden rounded-2xl border border-line-2 bg-ink-900/90 shadow-[0_30px_80px_-30px_rgba(0,0,0,0.8),inset_0_1px_0_rgba(255,255,255,0.05)] backdrop-blur ${className}`}
    >
      <div className="flex items-center gap-3 border-b border-line bg-white/[0.015] px-4 py-2.5">
        <div className="flex gap-1.5" aria-hidden="true">
          <span className="h-2.5 w-2.5 rounded-full bg-[#ff5f57]/80" />
          <span className="h-2.5 w-2.5 rounded-full bg-[#febc2e]/80" />
          <span className="h-2.5 w-2.5 rounded-full bg-[#28c840]/80" />
        </div>
        <div className="min-w-0 flex-1 truncate font-mono text-[11px] text-fg-3">{title}</div>
        {right && <div className="flex shrink-0 items-center gap-2">{right}</div>}
      </div>
      <div className={bodyClassName}>{children}</div>
    </div>
  );
}
