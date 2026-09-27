/**
 * engine.ts — Engine configuration shown on the site.
 *
 * ENGINE_DEFAULTS mirrors the v3 defaults in inference_engine/config.py (see the
 * README "Configuration" table). Every field is overridable by an upper-case
 * environment variable of the same name.
 *
 * DEMO_CONFIG is a deliberately small, fixed configuration used only by the
 * animated visualisations (all labelled "Demo data"). It is not a benchmark
 * setting — real pools are sized automatically from free GPU memory.
 */

export interface EngineDefault {
  key: string;
  cuda: string;
  other: string; // MPS / CPU
  note: string;
}

export const ENGINE_DEFAULTS: EngineDefault[] = [
  { key: 'MAX_BATCH_SIZE',         cuda: '64',     other: '8',       note: 'Max sequences in the running batch' },
  { key: 'PREFILL_BUDGET_TOKENS',  cuda: '2048',   other: '512',     note: 'Prompt tokens per step (all sequences)' },
  { key: 'PREFILL_CHUNK_SIZE',     cuda: '512',    other: '128',     note: 'Prompt tokens per sequence per step' },
  { key: 'KV_BLOCK_SIZE',          cuda: '16',     other: '16',      note: 'Tokens per KV block' },
  { key: 'KV_NUM_BLOCKS',          cuda: 'auto',   other: 'auto',    note: 'Profiled from GPU memory / KV_CACHE_MAX_MEMORY_MB' },
  { key: 'GPU_MEMORY_UTILIZATION', cuda: '0.85',   other: '—',       note: 'Share of GPU memory the engine may use' },
  { key: 'ENABLE_PREFIX_CACHING',  cuda: '1',      other: '1',       note: 'Share KV of common prompt prefixes' },
  { key: 'PREEMPTION_MODE',        cuda: 'swap',   other: 'swap',    note: 'swap (falls back to recompute) or recompute' },
  { key: 'SPECULATIVE_METHOD',     cuda: 'off',    other: 'off',     note: 'ngram or draft' },
  { key: 'NUM_SPECULATIVE_TOKENS', cuda: '4',      other: '4',       note: 'Drafts per step' },
];

export const ENGINE_PORT = 8001;
export const BASELINE_PORT = 8000;
export const DEFAULT_QUICKSTART_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct';

export const DEMO_CONFIG = {
  model_name: 'Qwen/Qwen2-0.5B',
  max_batch_size: 4,
  kv_block_size: 16,
  kv_num_blocks: 256,
  kv_num_cpu_blocks: 128,
  prefill_budget_tokens: 512,
  prefill_chunk_size: 128,
} as const;

export const TESTS_PASSING = 141;
export const TOTAL_PHASES = 12;

export const COLAB_URL =
  'https://colab.research.google.com/github/TryingtobeingNikhil/vLLM_Inference_Engine/blob/main/colab/PageServe_Colab.ipynb';
