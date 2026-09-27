'use client';

import { useLayoutEffect, useRef, useState } from 'react';
import { SectionHeader, Accent } from '@/components/ui/SectionHeader';
import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { Reveal } from '@/components/ui/Reveal';
import { CodeBlock, CopyButton } from '@/components/ui/CodeBlock';
import { ENGINE_DEFAULTS, ENGINE_PORT, DEFAULT_QUICKSTART_MODEL, COLAB_URL } from '@/data/engine';

// ── Tab definitions ────────────────────────────────────────────────────────────

interface Tab {
  id: string;
  label: string;
  lang: 'bash' | 'python' | 'typescript';
  file: string;
  code: string;
}

const TABS: Tab[] = [
  {
    id: 'stream',
    label: 'cURL · SSE',
    lang: 'bash',
    file: 'stream.sh',
    code: `# Start the engine on :${ENGINE_PORT}
MODEL_NAME=${DEFAULT_QUICKSTART_MODEL} \\
  python -m uvicorn inference_engine.server.app_v2:app --port ${ENGINE_PORT}

# POST /generate with "stream": true → Server-Sent Events
curl -N localhost:${ENGINE_PORT}/generate -H 'content-type: application/json' \\
  -d '{"prompt": "Explain paged attention in one sentence.",
       "max_new_tokens": 64, "stream": true}'

# data: {"token_ids": [...], "text": "Paged"}        ← one event per step
# data: {"token_ids": [...], "text": " attention"}
# ...
# data: {"done": true, "ttft_ms": ..., "tpot_ms": ..., "finish_reason": ...}
# data: [DONE]`,
  },
  {
    id: 'openai',
    label: 'OpenAI',
    lang: 'python',
    file: 'openai_client.py',
    code: `from openai import OpenAI

# PageServe speaks the OpenAI completions API on /v1/completions
client = OpenAI(base_url="http://localhost:${ENGINE_PORT}/v1", api_key="unused")

stream = client.completions.create(
    model="${DEFAULT_QUICKSTART_MODEL}",
    prompt="Write a haiku about KV caches.",
    max_tokens=64,
    temperature=0.0,  # greedy (speculative decoding applies to greedy requests)
    stream=True,      # OpenAI chunks, ending with a usage chunk
)
for chunk in stream:
    print(chunk.choices[0].text, end="", flush=True)`,
  },
  {
    id: 'python',
    label: 'Python',
    lang: 'python',
    file: 'sse_client.py',
    code: `import asyncio, json, httpx

async def stream(prompt: str, max_new_tokens: int = 64) -> dict:
    body = {"prompt": prompt, "max_new_tokens": max_new_tokens, "stream": True}
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", "http://localhost:${ENGINE_PORT}/generate", json=body) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[len("data: "):])
                if event.get("done"):
                    return event          # full result: ttft_ms, tpot_ms, ...
                print(event["text"], end="", flush=True)

async def main():
    # Concurrent requests share every forward pass: continuous batching
    results = await asyncio.gather(*[stream("Explain KV caching:") for _ in range(8)])
    for r in results:
        print(f"\\nttft={r['ttft_ms']:.1f}ms  tpot={r['tpot_ms']:.1f}ms")

asyncio.run(main())`,
  },
  {
    id: 'health',
    label: 'Health · Metrics',
    lang: 'bash',
    file: 'observe.sh',
    code: `# GET /health: model, device, batch occupancy, queue depth,
#              KV block usage, enabled features
curl localhost:${ENGINE_PORT}/health

# GET /metrics: one JSON document with
#   system                 in-flight, waiting, throughput
#   e2e_latency            TTFT / TPOT / ITL / E2E / queue, p50–p99
#   engine                 KV blocks, prefix hit rate, preemptions
#   speculative_decoding   acceptance, tokens per step
#   engine_steps           avg batch size, tokens/step, KV util, step latency
#   gpu_memory · paged_kv_cache · cpu_swap · queue_stats · slo_compliance
curl localhost:${ENGINE_PORT}/metrics`,
  },
];

const QUICK_START = `git clone https://github.com/TryingtobeingNikhil/vLLM_Inference_Engine.git
cd vLLM_Inference_Engine
pip install -r requirements.txt
MODEL_NAME=${DEFAULT_QUICKSTART_MODEL} python -m uvicorn inference_engine.server.app_v2:app --port ${ENGINE_PORT}`;

const ENDPOINTS = [
  { method: 'POST', path: '/generate',       color: '#60A5FA', note: 'Native · SSE' },
  { method: 'POST', path: '/v1/completions', color: '#22D3EE', note: 'OpenAI' },
  { method: 'GET',  path: '/health',         color: '#4ADE80', note: 'Liveness' },
  { method: 'GET',  path: '/metrics',        color: '#A78BFA', note: 'p50–p99' },
];

// ── Component ──────────────────────────────────────────────────────────────────

export function APICodeSection() {
  const [activeTab, setActiveTab] = useState<string>('stream');
  const tabRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const [underline, setUnderline] = useState({ left: 0, width: 0 });

  const tab = TABS.find((t) => t.id === activeTab) ?? TABS[0];

  useLayoutEffect(() => {
    const el = tabRefs.current[activeTab];
    if (el) setUnderline({ left: el.offsetLeft, width: el.offsetWidth });
  }, [activeTab]);

  return (
    <section id="api" className="relative px-5 py-24 sm:px-6 sm:py-32">
      <div className="mx-auto max-w-5xl">
        <SectionHeader
          index="09"
          label="HTTP API"
          title={<>Stream it, <Accent gradient>or talk OpenAI.</Accent></>}
          subtitle={`Native /generate with SSE streaming, an OpenAI-compatible /v1/completions, /health and /metrics, on port ${ENGINE_PORT}. Client disconnects free KV memory. The Phase 1 baseline still runs on :8000.`}
        />

        <div className="grid gap-5 lg:grid-cols-[1fr_300px] [&>*]:min-w-0">
          {/* Code window */}
          <Reveal>
            <div className="overflow-hidden rounded-2xl border border-line-2 bg-ink-900 shadow-[0_30px_80px_-30px_rgba(0,0,0,0.8)]">
              <div className="flex items-center gap-3 border-b border-line bg-white/[0.015] pl-4 pr-3">
                <div className="flex gap-1.5" aria-hidden="true">
                  <span className="h-2.5 w-2.5 rounded-full bg-[#ff5f57]/80" />
                  <span className="h-2.5 w-2.5 rounded-full bg-[#febc2e]/80" />
                  <span className="h-2.5 w-2.5 rounded-full bg-[#28c840]/80" />
                </div>
                <div className="relative flex overflow-x-auto" role="tablist">
                  {TABS.map((t) => (
                    <button
                      key={t.id}
                      ref={(el) => { tabRefs.current[t.id] = el; }}
                      role="tab"
                      aria-selected={t.id === activeTab}
                      onClick={() => setActiveTab(t.id)}
                      className={`whitespace-nowrap px-3 py-3 font-mono text-[11.5px] transition-colors duration-300 ${
                        t.id === activeTab ? 'text-fg' : 'text-fg-3 hover:text-fg-2'
                      }`}
                    >
                      {t.label}
                    </button>
                  ))}
                  <span
                    className="absolute bottom-0 h-[2px] rounded-full bg-gradient-to-r from-mint to-cyan transition-all duration-500 [transition-timing-function:var(--ease-spring)]"
                    style={{ left: underline.left + 10, width: Math.max(0, underline.width - 20) }}
                  />
                </div>
                <div className="ml-auto flex items-center gap-2">
                  <Badge label="Demo data" variant="demo" className="hidden md:inline-flex" />
                  <CopyButton text={tab.code} />
                </div>
              </div>
              <div className="flex items-center justify-between border-b border-line px-5 py-2 font-mono text-[10.5px] text-fg-4">
                <span>{tab.file}</span>
                <span>{tab.lang}</span>
              </div>
              <div key={tab.id} className="[animation:log-in_0.35s_var(--ease-out)_both]">
                <CodeBlock code={tab.code} language={tab.lang} lineNumbers />
              </div>
            </div>
          </Reveal>

          {/* Sidebar */}
          <div className="flex flex-col gap-4">
            <Reveal delay={80}>
              <Card pad={false} glow={false}>
                <p className="border-b border-line px-4 py-3 font-mono text-[10.5px] uppercase tracking-[0.16em] text-fg-3">Endpoints</p>
                <ul className="p-2">
                  {ENDPOINTS.map((ep) => (
                    <li key={ep.path} className="flex items-center gap-3 rounded-xl px-2 py-2 transition-colors hover:bg-white/[0.03]">
                      <span
                        className="w-12 rounded-md py-0.5 text-center font-mono text-[9.5px] font-semibold"
                        style={{ color: ep.color, backgroundColor: `${ep.color}16` }}
                      >
                        {ep.method}
                      </span>
                      <code className="flex-1 font-mono text-[12px] text-fg">{ep.path}</code>
                      <span className="text-[11.5px] text-fg-4">{ep.note}</span>
                    </li>
                  ))}
                </ul>
              </Card>
            </Reveal>

            <Reveal delay={140}>
              <Card pad={false} glow={false}>
                <div className="flex items-baseline justify-between border-b border-line px-4 py-3">
                  <p className="font-mono text-[10.5px] uppercase tracking-[0.16em] text-fg-3">Config defaults</p>
                  <p className="font-mono text-[9.5px] text-fg-4">CUDA · MPS/CPU</p>
                </div>
                <dl className="divide-y divide-line">
                  {ENGINE_DEFAULTS.map((row) => (
                    <div key={row.key} className="group px-4 py-2" title={row.note}>
                      <div className="flex items-baseline justify-between gap-3">
                        <dt className="truncate font-mono text-[10.5px] text-fg-3 transition-colors group-hover:text-fg-2">{row.key}</dt>
                        <dd className="shrink-0 font-mono text-[11px] text-fg">
                          {row.cuda === row.other || row.other === '—' ? row.cuda : <>{row.cuda}<span className="text-fg-4"> · </span>{row.other}</>}
                        </dd>
                      </div>
                    </div>
                  ))}
                </dl>
              </Card>
            </Reveal>

            <Reveal delay={200}>
              <Card pad={false} glow={false}>
                <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
                  <p className="font-mono text-[10.5px] uppercase tracking-[0.16em] text-fg-3">Quick start</p>
                  <CopyButton text={QUICK_START} />
                </div>
                <pre className="overflow-x-auto px-4 py-3 font-mono text-[11px] leading-[1.9] text-fg-2">
                  {QUICK_START.split('\n').map((l) => (
                    <div key={l} className="whitespace-pre-wrap break-all pl-3.5 -indent-3.5"><span className="select-none text-mint">$ </span>{l.replace('https://github.com/TryingtobeingNikhil/', '…/')}</div>
                  ))}
                </pre>
                <a
                  href={COLAB_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group flex items-center justify-between border-t border-line px-4 py-2.5 text-[12.5px] text-fg-2 transition-colors hover:bg-white/[0.02] hover:text-fg"
                >
                  <span>No GPU? Run it on Colab</span>
                  <span className="transition-transform duration-300 group-hover:translate-x-0.5">↗</span>
                </a>
              </Card>
            </Reveal>
          </div>
        </div>
      </div>
    </section>
  );
}
