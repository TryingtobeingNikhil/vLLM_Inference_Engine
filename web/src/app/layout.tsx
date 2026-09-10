import type { Metadata } from 'next';
import './globals.css';

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

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        {/* Preconnect to Google Fonts (already @imported in globals.css) */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>{children}</body>
    </html>
  );
}
