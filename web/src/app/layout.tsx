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
    'An LLM inference engine built from scratch: continuous batching over a paged KV cache, prefix caching, preemption and speculative decoding, in readable PyTorch.',
  // Favicons come from the file conventions in this folder:
  // icon.svg (modern browsers), favicon.ico (fallback), apple-icon.png (iOS/Safari).
  openGraph: {
    title: 'PageServe — LLM Inference Engine',
    description:
      'One packed forward pass per step, paged attention through block tables, prefix caching, speculative decoding and an OpenAI-compatible API, built without vLLM.',
    type: 'website',
    images: ['/favicon.png'],
  },
  twitter: {
    card: 'summary',
    title: 'PageServe',
    description: 'LLM inference engine built from scratch: paged attention, prefix caching, speculative decoding.',
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
