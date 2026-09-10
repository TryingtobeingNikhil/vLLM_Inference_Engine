import { type ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
  pad?: boolean;
}

export function Card({ children, className = '', pad = true }: CardProps) {
  return (
    <div
      className={`border border-[#2a2a2a] bg-[#111111] ${pad ? 'p-4' : ''} ${className}`}
    >
      {children}
    </div>
  );
}

export function Card2({ children, className = '', pad = true }: CardProps) {
  return (
    <div
      className={`border border-[#2a2a2a] bg-[#181818] ${pad ? 'p-4' : ''} ${className}`}
    >
      {children}
    </div>
  );
}
