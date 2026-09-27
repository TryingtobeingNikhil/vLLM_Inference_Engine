import { type ReactNode } from 'react';
import { Reveal } from '@/components/ui/Reveal';

interface SectionHeaderProps {
  index: string;
  label: string;
  title: ReactNode;
  subtitle?: ReactNode;
  className?: string;
}

export function SectionHeader({ index, label, title, subtitle, className = '' }: SectionHeaderProps) {
  return (
    <Reveal className={`mb-12 max-w-3xl ${className}`}>
      <div className="mb-5 flex items-center gap-3 font-mono text-[11px] uppercase tracking-[0.2em] text-fg-3">
        <span className="rounded-full border border-line-2 bg-white/[0.03] px-2.5 py-1 text-fg-2">
          {index}
        </span>
        <span className="h-px w-8 bg-gradient-to-r from-line-2 to-transparent" />
        <span>{label}</span>
      </div>
      <h2 className="text-balance text-3xl font-semibold leading-[1.1] tracking-tight text-fg sm:text-[2.6rem]">
        {title}
      </h2>
      {subtitle && (
        <p className="mt-4 max-w-2xl text-pretty text-[15px] leading-relaxed text-fg-2">{subtitle}</p>
      )}
    </Reveal>
  );
}

/** The italic serif flourish used inside headings. */
export function Accent({ children, gradient = false }: { children: ReactNode; gradient?: boolean }) {
  return (
    <span className={`accent-serif pr-1 text-[1.08em] ${gradient ? 'text-gradient' : 'text-fg-2'}`}>
      {children}
    </span>
  );
}
