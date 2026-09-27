import { type ReactNode } from 'react';
import { Card } from '@/components/ui/Card';

interface MetricCardProps {
  label: string;
  value: ReactNode;
  unit?: string;
  delta?: string;
  deltaPositive?: boolean;
  sub?: string;
  accent?: string;
}

export function MetricCard({ label, value, unit, delta, deltaPositive, sub, accent = '#4ADE80' }: MetricCardProps) {
  return (
    <Card className="flex h-full flex-col">
      <div className="mb-6 flex items-center gap-2">
        <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: accent, boxShadow: `0 0 12px ${accent}` }} />
        <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-fg-3">{label}</p>
      </div>
      <p className="mt-auto flex items-baseline gap-1.5 text-[2rem] font-semibold leading-none tracking-tight text-fg">
        <span>{value}</span>
        {unit && <span className="text-sm font-normal text-fg-3">{unit}</span>}
      </p>
      {delta && (
        <p
          className={`mt-3 inline-flex w-fit items-center rounded-full px-2 py-0.5 font-mono text-[10.5px] ${
            deltaPositive ? 'bg-mint/10 text-mint' : 'bg-rose/10 text-rose'
          }`}
        >
          {delta}
        </p>
      )}
      {sub && <p className="mt-2 font-mono text-[11px] text-fg-3">{sub}</p>}
    </Card>
  );
}
