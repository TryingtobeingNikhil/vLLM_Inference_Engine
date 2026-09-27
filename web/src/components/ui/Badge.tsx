interface BadgeProps {
  label: string;
  variant?: 'demo' | 'live' | 'warn' | 'neutral' | 'green' | 'amber' | 'red';
  dot?: boolean;
  className?: string;
}

const variantStyles: Record<string, string> = {
  demo:    'border-line-2 bg-white/[0.03] text-fg-3',
  live:    'border-mint/30 bg-mint/10 text-mint',
  warn:    'border-amber/30 bg-amber/10 text-amber',
  neutral: 'border-line-2 bg-white/[0.03] text-fg-2',
  green:   'border-mint/30 bg-mint/10 text-mint',
  amber:   'border-amber/30 bg-amber/10 text-amber',
  red:     'border-rose/30 bg-rose/10 text-rose',
};

const dotColors: Record<string, string> = {
  demo:    'bg-fg-4',
  live:    'bg-mint',
  warn:    'bg-amber',
  neutral: 'bg-fg-3',
  green:   'bg-mint',
  amber:   'bg-amber',
  red:     'bg-rose',
};

export function Badge({ label, variant = 'demo', dot = false, className = '' }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] ${variantStyles[variant]} ${className}`}
    >
      {dot && (
        <span className="relative inline-flex h-1.5 w-1.5">
          <span className={`ping-soft absolute inset-0 rounded-full ${dotColors[variant]}`} />
          <span className={`relative inline-block h-1.5 w-1.5 rounded-full ${dotColors[variant]}`} />
        </span>
      )}
      {label}
    </span>
  );
}
