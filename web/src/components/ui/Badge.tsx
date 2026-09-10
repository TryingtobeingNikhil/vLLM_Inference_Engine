interface BadgeProps {
  label: string;
  variant?: 'demo' | 'live' | 'warn' | 'neutral' | 'green' | 'amber' | 'red';
  dot?: boolean;
  className?: string;
}

const variantStyles: Record<string, string> = {
  demo:    'border-[#3a3a3a] text-[#666666]',
  live:    'border-[#4ADE80] text-[#4ADE80]',
  warn:    'border-[#FBBF24] text-[#FBBF24]',
  neutral: 'border-[#3a3a3a] text-[#888888]',
  green:   'border-[#4ADE80] text-[#4ADE80]',
  amber:   'border-[#FBBF24] text-[#FBBF24]',
  red:     'border-[#F87171] text-[#F87171]',
};

const dotColors: Record<string, string> = {
  demo:    'bg-[#444444]',
  live:    'bg-[#4ADE80]',
  warn:    'bg-[#FBBF24]',
  neutral: 'bg-[#888888]',
  green:   'bg-[#4ADE80]',
  amber:   'bg-[#FBBF24]',
  red:     'bg-[#F87171]',
};

export function Badge({ label, variant = 'demo', dot = false, className = '' }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 border px-2 py-0.5 font-mono text-[10px] uppercase tracking-widest ${variantStyles[variant]} ${className}`}
    >
      {dot && (
        <span className={`inline-block h-1.5 w-1.5 ${dotColors[variant]} animate-pulse`} />
      )}
      {label}
    </span>
  );
}
