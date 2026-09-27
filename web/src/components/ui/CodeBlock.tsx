'use client';

import { useState, type ReactNode } from 'react';

type Lang = 'bash' | 'python' | 'typescript' | 'json' | 'text';

const KEYWORDS: Record<Lang, string[]> = {
  bash: ['curl', 'python', 'pip', 'git', 'cd', 'export', 'uvicorn', 'pytest', 'jq', 'install', 'clone'],
  python: [
    'def', 'class', 'import', 'from', 'return', 'if', 'else', 'elif', 'for', 'while', 'with', 'as',
    'not', 'and', 'or', 'in', 'is', 'None', 'True', 'False', 'self', 'async', 'await', 'raise',
  ],
  typescript: [
    'interface', 'async', 'function', 'const', 'let', 'await', 'return', 'if', 'throw', 'new',
    'string', 'number', 'Promise', 'export', 'import', 'from', 'type',
  ],
  json: ['true', 'false', 'null'],
  text: [],
};

function buildLexer(lang: Lang): RegExp {
  const comment = lang === 'typescript' ? String.raw`//[^\n]*` : String.raw`#[^\n]*`;
  const single = lang === 'bash' ? String.raw`'[^']*'` : String.raw`'(?:[^'\\\n]|\\.)*'`;
  const str = String.raw`"(?:[^"\\\n]|\\.)*"|${single}|` + '`(?:[^`\\\\]|\\\\.)*`';
  const kw = KEYWORDS[lang].length ? String.raw`\b(?:${KEYWORDS[lang].join('|')})\b` : '(?!)';
  const flag = lang === 'bash' ? String.raw`(?<=\s)--?[A-Za-z][\w-]*` : '(?!)';
  return new RegExp(
    `(?<comment>${comment})|(?<str>${str})|(?<flag>${flag})|(?<kw>${kw})|(?<num>\\b\\d+(?:\\.\\d+)?\\b)|(?<fn>\\b[A-Za-z_]\\w*(?=\\())`,
    'g',
  );
}

interface Token { text: string; cls?: string }

function tokenize(code: string, lang: Lang): Token[] {
  if (lang === 'text') return [{ text: code }];
  const re = buildLexer(lang);
  const out: Token[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(code))) {
    if (m[0].length === 0) { re.lastIndex++; continue; }
    if (m.index > last) out.push({ text: code.slice(last, m.index) });
    const g = m.groups ?? {};
    const cls =
      g.comment ? 'tok-comment' : g.str ? 'tok-str' : g.flag ? 'tok-flag' : g.kw ? 'tok-kw' : g.num ? 'tok-num' : 'tok-fn';
    out.push({ text: m[0], cls });
    last = m.index + m[0].length;
  }
  if (last < code.length) out.push({ text: code.slice(last) });
  return out;
}

/** Tokenises the whole snippet (so multi-line strings stay intact), then splits into lines. */
function highlightLines(code: string, lang: Lang): ReactNode[][] {
  const lines: ReactNode[][] = [[]];
  let key = 0;
  for (const tok of tokenize(code, lang)) {
    tok.text.split('\n').forEach((part, i) => {
      if (i > 0) lines.push([]);
      if (part) lines[lines.length - 1].push(tok.cls ? <span key={key++} className={tok.cls}>{part}</span> : part);
    });
  }
  return lines;
}

export function CopyButton({ text, className = '' }: { text: string; className?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={() => {
        navigator.clipboard?.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1600);
        });
      }}
      className={`inline-flex items-center gap-1.5 rounded-full border border-line-2 bg-white/[0.03] px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.12em] transition-all duration-300 hover:border-white/25 hover:text-fg ${copied ? 'text-mint' : 'text-fg-3'} ${className}`}
      aria-label="Copy code"
    >
      {copied ? (
        <svg viewBox="0 0 16 16" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 8.5l3 3 7-7" strokeLinecap="round" strokeLinejoin="round" /></svg>
      ) : (
        <svg viewBox="0 0 16 16" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="5" y="5" width="8.5" height="8.5" rx="2" /><path d="M10.5 5V3.5A1.5 1.5 0 009 2H3.5A1.5 1.5 0 002 3.5V9A1.5 1.5 0 003.5 10.5H5" /></svg>
      )}
      {copied ? 'Copied' : 'Copy'}
    </button>
  );
}

interface CodeBlockProps {
  code: string;
  language?: Lang;
  className?: string;
  lineNumbers?: boolean;
}

export function CodeBlock({ code, language = 'text', className = '', lineNumbers = false }: CodeBlockProps) {
  const lines = highlightLines(code, language);
  return (
    <div className={`relative ${className}`}>
      <pre className="overflow-x-auto p-5 font-mono text-[12px] leading-[1.75] text-fg-2">
        <code className="grid" style={{ gridTemplateColumns: lineNumbers ? 'auto 1fr' : '1fr' }}>
          {lines.map((line, i) => (
            <span key={i} className="contents">
              {lineNumbers && (
                <span className="select-none pr-5 text-right text-fg-4/70">{i + 1}</span>
              )}
              <span className="whitespace-pre">{line.length ? line : ' '}</span>
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}
