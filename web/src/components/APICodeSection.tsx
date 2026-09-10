'use client';

import { useState } from 'react';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { Badge } from '@/components/ui/Badge';
import { ENGINE_CONFIG } from '@/data/benchmarks';

// ── Tab definitions ────────────────────────────────────────────────────────────

interface Tab {
  id: string;
  label: string;
  lang: string;
  code: string;
}

const TABS: Tab[] = [
  {
    id: 'curl',
    label: 'cURL',
    lang: 'bash',
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

// ── Component ──────────────────────────────────────────────────────────────────

export function APICodeSection() {
  const [activeTab, setActiveTab] = useState<string>('curl');

  const tab = TABS.find((t) => t.id === activeTab) ?? TABS[0];

  return (
    <section id="api" className="border-b border-[#1e1e1e] px-6 py-20 sm:px-10 lg:px-16">
      <div className="mx-auto max-w-5xl">
        <SectionHeader
          label="// http api"
          title="HTTP API"
          subtitle={`POST /generate · GET /health · GET /metrics — serving ${ENGINE_CONFIG.model_name} on port 8000.`}
        />

        <div className="grid gap-6 lg:grid-cols-[1fr_280px]">
          {/* ── Code panel ───────────────────────────────────────────── */}
          <div className="border border-[#2a2a2a] bg-[#0d0d0d]">
            {/* Tab bar */}
            <div className="flex items-center justify-between border-b border-[#1e1e1e]">
              <div className="flex">
                {TABS.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => setActiveTab(t.id)}
                    className={`px-4 py-2.5 font-mono text-[11px] uppercase tracking-widest transition-colors ${
                      t.id === activeTab
                        ? 'border-b border-[#e8e8e8] text-[#e8e8e8]'
                        : 'text-[#444444] hover:text-[#666666]'
                    }`}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
              <div className="px-3">
                <Badge label="Demo Data" variant="demo" />
              </div>
            </div>

            {/* Code body */}
            <pre className="overflow-x-auto p-5 font-mono text-[11px] leading-relaxed text-[#888888]">
              <code>{tab.code}</code>
            </pre>
          </div>

          {/* ── Config sidebar ────────────────────────────────────────── */}
          <div className="flex flex-col gap-4">
            {/* Server info */}
            <div className="border border-[#2a2a2a] bg-[#0d0d0d]">
              <div className="border-b border-[#1e1e1e] px-3 py-2">
                <span className="font-mono text-[10px] uppercase tracking-widest text-[#333333]">
                  Server Endpoints
                </span>
              </div>
              <div className="divide-y divide-[#1a1a1a] px-3">
                {[
                  { method: 'POST', path: '/generate', color: '#60A5FA', note: 'Inference' },
                  { method: 'GET',  path: '/health',   color: '#4ADE80', note: 'Liveness' },
                  { method: 'GET',  path: '/metrics',  color: '#a78bfa', note: 'Prometheus' },
                ].map((ep) => (
                  <div key={ep.path} className="flex items-center gap-2 py-2">
                    <span
                      className="w-10 font-mono text-[9px] uppercase"
                      style={{ color: ep.color }}
                    >
                      {ep.method}
                    </span>
                    <code className="flex-1 font-mono text-[10px] text-[#888888]">
                      {ep.path}
                    </code>
                    <span className="font-mono text-[9px] text-[#333333]">{ep.note}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Engine config */}
            <div className="border border-[#2a2a2a] bg-[#0d0d0d]">
              <div className="border-b border-[#1e1e1e] px-3 py-2">
                <span className="font-mono text-[10px] uppercase tracking-widest text-[#333333]">
                  Engine Config
                </span>
              </div>
              <div className="divide-y divide-[#1a1a1a]">
                {CONFIG_ROWS.map((row) => (
                  <div key={row.key} className="px-3 py-1.5">
                    <div className="flex items-baseline justify-between">
                      <code className="font-mono text-[9px] text-[#555555]">{row.key}</code>
                      <span className="font-mono text-[10px] text-[#888888]">
                        {row.value}
                      </span>
                    </div>
                    <p className="mt-0.5 font-mono text-[9px] text-[#333333]">{row.note}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick start */}
            <div className="border border-[#2a2a2a] bg-[#0d0d0d] px-3 py-3">
              <p className="mb-2 font-mono text-[10px] uppercase tracking-widest text-[#333333]">
                Quick Start
              </p>
              <pre className="font-mono text-[10px] leading-relaxed text-[#555555]">
                {`git clone …/vLLM_Inference_Engine
cd PageServe
pip install -r requirements.txt
python -m inference_engine.server`}
              </pre>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
