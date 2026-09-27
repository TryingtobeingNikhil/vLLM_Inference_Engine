import type { Metadata, Viewport } from 'next';
import { Inter, JetBrains_Mono, Instrument_Serif } from 'next/font/google';
import './globals.css';

const sans = Inter({ subsets: ['latin'], variable: '--font-sans', display: 'swap' });
const mono = JetBrains_Mono({ subsets: ['latin'], variable: '--font-mono', display: 'swap' });
const serif = Instrument_Serif({
  subsets: ['latin'],
  weight: '400',
  style: ['normal', 'italic'],
  variable: '--font-serif',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'PageServe — LLM Inference Engine',
  description:
    'A production-grade LLM inference engine built from first principles: continuous batching, paged KV cache, chunked prefill, and CPU swap pool.',
  icons: {
    icon: '/favicon.png',
    shortcut: '/favicon.png',
    apple: '/favicon.png',
  },
  openGraph: {
    title: 'PageServe — LLM Inference Engine',
    description:
      'Continuous batching, paged KV cache, chunked prefill — built without vLLM or HuggingFace generate.',
    type: 'website',
    images: ['/favicon.png'],
  },
  twitter: {
    card: 'summary',
    title: 'PageServe',
    description: 'LLM inference engine built from first principles.',
    images: ['/favicon.png'],
  },
};

export const viewport: Viewport = {
  themeColor: '#07080A',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`dark ${sans.variable} ${mono.variable} ${serif.variable}`}>
      <body>{children}</body>
    </html>
  );
}
