interface MetricCardProps {
  label: string;
  value: string;
  delta?: string;
  deltaPositive?: boolean;
  sub?: string;
  highlight?: boolean;
}

export function MetricCard({
  label,
  value,
  delta,
  deltaPositive,
  sub,
  highlight = false,
}: MetricCardProps) {
  return (
    <div
      className={`border p-4 ${
        highlight
          ? 'border-[#4ADE80]/30 bg-[#1a3d27]/20'
          : 'border-[#2a2a2a] bg-[#111111]'
      }`}
    >
      <p className="mb-1 font-mono text-[10px] uppercase tracking-widest text-[#444444]">
        {label}
      </p>
      <p className="font-mono text-2xl font-semibold text-[#e8e8e8]">{value}</p>
      {delta && (
        <p
          className={`mt-1 font-mono text-xs ${
            deltaPositive ? 'text-[#4ADE80]' : 'text-[#F87171]'
          }`}
        >
          {delta}
        </p>
      )}
      {sub && <p className="mt-1 font-mono text-[11px] text-[#555555]">{sub}</p>}
    </div>
  );
}
