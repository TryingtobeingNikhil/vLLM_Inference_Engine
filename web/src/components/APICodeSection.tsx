'use client';

import { useLayoutEffect, useRef, useState } from 'react';
import { SectionHeader, Accent } from '@/components/ui/SectionHeader';
import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { Reveal } from '@/components/ui/Reveal';
import { CodeBlock, CopyButton } from '@/components/ui/CodeBlock';
import { ENGINE_CONFIG } from '@/data/benchmarks';

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
    id: 'curl',
    label: 'cURL',
    lang: 'bash',
    file: 'generate.sh',
    code: `# POST /generate — single request
curl -s -X POST http://localhost:8000/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "Explain paged attention in one sentence:",
    "max_new_tokens": 64
  }' | jq .

# Response
# {
#   "request_id": "req-0001",
#   "generated_text": "Paged attention …",
#   "tokens_generated": 64,
#   "ttft_ms": 16.5,
#   "total_latency_ms": 514.4,
#   "throughput_tps": 97.3
# }`,
  },
  {
    id: 'python',
    label: 'Python',
    lang: 'python',
    file: 'client.py',
    code: `import asyncio, aiohttp

async def generate(prompt: str, max_new_tokens: int = 64) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/generate",
            json={"prompt": prompt, "max_new_tokens": max_new_tokens},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

async def main():
    # Fire 4 requests concurrently — PageServe batches them automatically
    results = await asyncio.gather(*[
        generate("Explain KV caching:", max_new_tokens=50)
        for _ in range(${ENGINE_CONFIG.max_batch_size})
    ])
    for r in results:
        print(f"ttft={r['ttft_ms']:.1f}ms  tps={r['throughput_tps']:.1f}")

asyncio.run(main())`,
  },
  {
    id: 'typescript',
    label: 'TypeScript',
    lang: 'typescript',
    file: 'client.ts',
    code: `interface GenerateRequest {
  prompt: string;
  max_new_tokens?: number;
}

interface GenerateResponse {
  request_id: string;
  generated_text: string;
  tokens_generated: number;
  ttft_ms: number;
  total_latency_ms: number;
  throughput_tps: number;
}

async function generate(req: GenerateRequest): Promise<GenerateResponse> {
  const res = await fetch("http://localhost:8000/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(\`HTTP \${res.status}\`);
  return res.json();
}

// Concurrent batch — scheduler merges these automatically
const results = await Promise.all(
  Array.from({ length: 4 }, () =>
    generate({ prompt: "What is continuous batching?", max_new_tokens: 50 })
  )
);
console.log(results.map((r) => \`\${r.ttft_ms.toFixed(1)} ms TTFT\`));`,
  },
  {
    id: 'health',
    label: 'Health',
    lang: 'bash',
    file: 'health.sh',
    code: `# GET /health — liveness probe
curl http://localhost:8000/health

# {
#   "status": "ok",
#   "model": "${ENGINE_CONFIG.model_name}",
#   "max_batch_size": ${ENGINE_CONFIG.max_batch_size},
#   "kv_blocks_free": 198,
#   "kv_blocks_total": ${ENGINE_CONFIG.kv_num_blocks},
#   "cpu_blocks_total": ${ENGINE_CONFIG.kv_num_cpu_blocks},
#   "queue_depth": 0
# }

# GET /metrics — Prometheus-compatible counters
curl http://localhost:8000/metrics

# pageserve_requests_total 42
# pageserve_tokens_generated_total 2100
# pageserve_ttft_ms_sum 872.0
# pageserve_throughput_tps 97.25`,
  },
];

// ── Engine config rows ─────────────────────────────────────────────────────────

const CONFIG_ROWS: { key: string; value: string | number; note: string }[] = [
  { key: 'model_name',            value: ENGINE_CONFIG.model_name,            note: 'HuggingFace model id' },
  { key: 'max_batch_size',        value: ENGINE_CONFIG.max_batch_size,        note: 'Max seqs in flight' },
  { key: 'kv_block_size',         value: ENGINE_CONFIG.kv_block_size,         note: 'Tokens per KV block' },
  { key: 'kv_num_blocks',         value: ENGINE_CONFIG.kv_num_blocks,         note: 'GPU KV pool capacity' },
  { key: 'kv_num_cpu_blocks',     value: ENGINE_CONFIG.kv_num_cpu_blocks,     note: 'CPU swap pool capacity' },
  { key: 'prefill_chunk_size',    value: ENGINE_CONFIG.prefill_chunk_size,    note: 'Tokens per prefill chunk' },
  { key: 'prefill_budget_tokens', value: ENGINE_CONFIG.prefill_budget_tokens, note: 'Max prefill tok per iter' },
  { key: 'decode_batch_limit',    value: ENGINE_CONFIG.decode_batch_limit,    note: 'Max decode seqs per iter' },
];

const QUICK_START = `git clone https://github.com/TryingtobeingNikhil/vLLM_Inference_Engine
cd vLLM_Inference_Engine
pip install -r requirements.txt
python -m uvicorn inference_engine.server.app:app --port 8000`;

const ENDPOINTS = [
  { method: 'POST', path: '/generate', color: '#60A5FA', note: 'Inference' },
  { method: 'GET',  path: '/health',   color: '#4ADE80', note: 'Liveness' },
  { method: 'GET',  path: '/metrics',  color: '#A78BFA', note: 'Prometheus' },
];

// ── Component ──────────────────────────────────────────────────────────────────

export function APICodeSection() {
  const [activeTab, setActiveTab] = useState<string>('curl');
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
          index="07"
          label="HTTP API"
          title={<>Talk to it over <Accent gradient>plain HTTP.</Accent></>}
          subtitle={`POST /generate · GET /health · GET /metrics — serving ${ENGINE_CONFIG.model_name} on port 8000.`}
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
                <p className="border-b border-line px-4 py-3 font-mono text-[10.5px] uppercase tracking-[0.16em] text-fg-3">Engine config</p>
                <dl className="divide-y divide-line">
                  {CONFIG_ROWS.map((row) => (
                    <div key={row.key} className="group px-4 py-2" title={row.note}>
                      <div className="flex items-baseline justify-between gap-3">
                        <dt className="font-mono text-[11px] text-fg-3 transition-colors group-hover:text-fg-2">{row.key}</dt>
                        <dd className="truncate font-mono text-[11.5px] text-fg">{row.value}</dd>
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
              </Card>
            </Reveal>
          </div>
        </div>
      </div>
    </section>
  );
}
