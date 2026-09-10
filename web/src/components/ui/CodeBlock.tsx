interface CodeBlockProps {
  code: string;
  language?: string;
  title?: string;
  className?: string;
}

// Minimal tokenizer for JSON and Python — no external dependency
function tokenizeJson(code: string): string {
  return code
    .replace(/("(?:[^"\\]|\\.)*")\s*:/g, '<span class="cb-key">$1</span>:')
    .replace(/:\s*("(?:[^"\\]|\\.)*")/g, ': <span class="cb-str">$1</span>')
    .replace(/:\s*(\d+\.?\d*)/g, ': <span class="cb-num">$1</span>')
    .replace(/:\s*(true|false|null)/g, ': <span class="cb-kw">$1</span>');
}

function tokenizePython(code: string): string {
  return code
    .replace(/(#[^\n]*)/g, '<span class="cb-comment">$1</span>')
    .replace(/\b(def|class|import|from|return|if|else|elif|for|while|with|as|not|and|or|in|is|None|True|False|self|async|await|raise)\b/g,
      '<span class="cb-kw">$1</span>')
    .replace(/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g, '<span class="cb-str">$1</span>');
}

function tokenizeBash(code: string): string {
  return code
    .replace(/(#[^\n]*)/g, '<span class="cb-comment">$1</span>')
    .replace(/\b(python|uvicorn|pytest|pip|export|cd|git)\b/g, '<span class="cb-kw">$1</span>');
}

export function CodeBlock({ code, language = 'text', title, className = '' }: CodeBlockProps) {
  let highlighted = code
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  if (language === 'json') highlighted = tokenizeJson(highlighted);
  else if (language === 'python') highlighted = tokenizePython(highlighted);
  else if (language === 'bash') highlighted = tokenizeBash(highlighted);

  return (
    <div className={`border border-[#2a2a2a] bg-[#0d0d0d] ${className}`}>
      {title && (
        <div className="flex items-center border-b border-[#2a2a2a] px-4 py-2">
          <span className="font-mono text-[10px] uppercase tracking-widest text-[#444444]">
            {title}
          </span>
          {language !== 'text' && (
            <span className="ml-auto font-mono text-[10px] text-[#333333]">
              .{language}
            </span>
          )}
        </div>
      )}
      <pre
        className="overflow-x-auto p-4 text-[0.78rem] leading-relaxed text-[#aaaaaa]"
        dangerouslySetInnerHTML={{ __html: highlighted }}
      />
      <style>{`
        .cb-key     { color: #60a5fa; }
        .cb-str     { color: #86efac; }
        .cb-num     { color: #fbbf24; }
        .cb-kw      { color: #a78bfa; }
        .cb-comment { color: #4a4a4a; font-style: italic; }
      `}</style>
    </div>
  );
}
