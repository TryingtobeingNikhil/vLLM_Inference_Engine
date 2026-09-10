interface SectionHeaderProps {
  label: string;
  title: string;
  subtitle?: string;
  className?: string;
}

export function SectionHeader({ label, title, subtitle, className = '' }: SectionHeaderProps) {
  return (
    <div className={`mb-10 ${className}`}>
      <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em] text-[#444444]">
        {label}
      </p>
      <h2 className="text-2xl font-semibold tracking-tight text-[#e8e8e8] sm:text-3xl">
        {title}
      </h2>
      {subtitle && (
        <p className="mt-3 max-w-2xl text-sm text-[#666666]">{subtitle}</p>
      )}
    </div>
  );
}
