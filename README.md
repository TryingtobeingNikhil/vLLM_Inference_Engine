<div align="center">

<img src="https://img.shields.io/badge/PageServe-LLM%20Inference%20Engine-2F80ED?style=for-the-badge&logoColor=white" alt="PageServe">

# 🧠 PageServe

### *An LLM inference engine built from scratch: continuous batching over a paged KV cache, prefix caching, preemption and speculative decoding — in readable PyTorch.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.51–5.x-yellow?style=flat-square)](https://huggingface.co)
[![Colab](https://img.shields.io/badge/Colab-T4%20%7C%20L4%20%7C%20A100-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](colab/PageServe_Colab.ipynb)

[Quick start](#-quick-start) · [Colab](#-run-it-on-a-colab-gpu) · [How it works](#-how-it-works) · [Benchmarks](#-benchmarking) · [API](#-api-reference) · [Configuration](#%EF%B8%8F-configuration)

</div>

---

## 🎯 The problem

Serving an LLM with a plain `model.generate()` wrapper wastes the GPU:

1. **One request at a time.** Decoding one token for one sequence is memory-bandwidth bound: the GPU streams every weight to produce a single token and sits mostly idle. Serving 32 users costs almost 32× the time of one.
2. **Static batching wastes work.** Padding requests into a batch forces everyone to wait for the longest prompt *and* the longest output.
3. **Contiguous KV caches fragment memory.** Reserving `max_length` of KV per sequence leaves most of it unused, so few sequences fit.

PageServe fixes these the way production engines (vLLM, SGLang, TGI) do — built step by step so every mechanism is visible.

## ✨ Features

- **Continuous batching** — every scheduler step packs *all* running work (prefill chunks of new prompts + one decode token per running sequence) into **one** forward pass. Requests join and leave the batch between steps.
- **Paged KV cache that attention actually uses** — K/V live in fixed-size blocks in one pre-allocated pool; a custom attention backend plugged into HuggingFace models writes and reads the pool through per-sequence block tables. No per-sequence `past_key_values`.
- **Chunked prefill** — long prompts are split across steps so they can't stall running decodes.
- **Automatic prefix caching** — full blocks are content-hashed (radix-tree semantics); requests sharing a prompt prefix reuse its KV instead of recomputing it.
- **Preemption** — when the pool is full, the newest request is **swapped** to CPU (pinned memory) or **dropped for recompute**, so the engine degrades gracefully instead of OOMing.
- **Speculative decoding** — n-gram (prompt-lookup) or draft-model proposals, verified in one batched pass; output is identical to greedy decoding.
- **Streaming API** — SSE on the native `/generate`, plus an **OpenAI-compatible** `/v1/completions`; client disconnects free KV memory.
- **Honest measurement** — client-side TTFT / TPOT / ITL / E2E p50–p99, goodput, GPU utilisation & memory; open-loop Poisson or closed-loop load; reproducible seeded workloads.
- **Runs anywhere PyTorch does** — CUDA (T4/L4/A100), Apple MPS, CPU. Tested against transformers 4.51 and 5.x.

---

## 🚀 Quick start

```bash
git clone https://github.com/TryingtobeingNikhil/vLLM_Inference_Engine.git
cd vLLM_Inference_Engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Start the engine (port 8001) and stream a completion:

```bash
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct python -m uvicorn inference_engine.server.app_v2:app --port 8001
```

```bash
curl -N localhost:8001/generate -H 'content-type: application/json' \
  -d '{"prompt": "Explain paged attention in one sentence.", "max_new_tokens": 64, "stream": true}'
```

The Phase 1 baseline (sequential HF `generate()`) runs the same way on port 8000:

```bash
python -m uvicorn inference_engine.server.app:app --port 8000
```

## ☁️ Run it on a Colab GPU

Open **[`colab/PageServe_Colab.ipynb`](colab/PageServe_Colab.ipynb)** in Colab, pick a T4 / L4 / A100 runtime and *Run all*. It will:

1. clone the repo and run [`scripts/colab_setup.sh`](scripts/colab_setup.sh) (keeps Colab's CUDA torch, installs the rest),
2. run the unit tests and the **GPU smoke test** ([`scripts/gpu_smoke_test.py`](scripts/gpu_smoke_test.py)) — exact-match checks of every engine feature *on the GPU* plus a real-model sanity check,
3. run the offline ablation and the serving benchmark with a model sized to the GPU,
4. render charts and tables, and zip the results for download.

Default models: T4 → `Qwen2.5-1.5B-Instruct` (fp16 — T4 has no bf16), L4 → `Qwen2.5-3B-Instruct`, A100 → `Qwen2.5-7B-Instruct` (bf16). The KV pool is sized automatically from free GPU memory.

From a terminal instead of the notebook:

```bash
bash scripts/colab_setup.sh
python scripts/gpu_smoke_test.py
python -m benchmarks.bench_offline
python -m benchmarks.bench_serving
python -m benchmarks.plot_results results/*.json
```

---

## 🏗️ How it works

```
 client ──► FastAPI (app_v2.py) ──► RequestQueue (FIFO, timeouts, backpressure)
                                          │
                  ┌───────────────────────▼─────────────────────────────┐
                  │ ContinuousBatchingScheduler — one step:             │
                  │  1. plan    decodes: 1 token each (+k speculative)  │
                  │             prefills: next chunk, ≤ token budget    │
                  │             admit new requests if blocks allow      │
                  │             (prefix-cache hits shared, not redone)  │
                  │             out of blocks? preempt newest: swap/    │
                  │             recompute                               │
                  │  2. execute ONE packed forward pass (worker thread) │
                  │  3. apply   append/verify tokens, stream them,      │
                  │             publish full blocks to prefix cache,    │
                  │             finish + free                           │
                  └──────┬───────────────────────────────▲──────────────┘
                         │ SequenceInputs                │ sampled tokens
                  ┌──────▼───────────────────────────────┴──────────────┐
                  │ ModelRunner: pack tokens → [1, T], positions,       │
                  │   slot mapping; HF model with the paged attention   │
                  │   backend; logits only where sampled                │
                  └──────┬──────────────────────────────────────────────┘
                         │ write new K/V / gather context via block tables
                  ┌──────▼─────────────────────┐  swap  ┌───────────────┐
                  │ PagedKVCacheManager        │ ◄────► │ CPUSwapManager│
                  │ [layers, blocks, 16, H, D] │        │ (pinned RAM)  │
                  └──────▲─────────────────────┘        └───────────────┘
                         │ block tables, ref counts, content hashes
                  ┌──────┴─────────────────────┐
                  │ BlockAllocator + prefix    │
                  │ cache (LRU of freed blocks)│
                  └────────────────────────────┘
```

### One number drives scheduling

Each sequence tracks `num_computed_tokens` — how many of its tokens already have K/V in the pool. Every step it processes some *uncomputed* tokens: many for a prefill chunk, exactly one for a decode, all of them again after a recompute preemption. Whenever a step reaches the last token, its logits predict the next one. Prefill, chunked prefill, decode and recompute are the *same operation*, which is what lets one forward pass mix them. ([`sequence.py`](inference_engine/engine/sequence.py))

### Packed batches and paged attention

The runner concatenates every scheduled token into a single `[1, T]` batch with explicit `position_ids` — no padding flows through the MLPs. HuggingFace models dispatch attention through a registry, so PageServe registers its own backend ([`attention_wrapper.py`](inference_engine/engine/attention_wrapper.py)). Each layer hands it the new keys/values; it scatters them into the pool at each token's *slot* (`block_table[pos // 16] * 16 + pos % 16`) and attends each query to its sequence's context through the block table:

* **short queries** (decode, speculative verification): all sequences batched — gather contexts, one masked SDPA call (fp32 accumulation), GQA handled by folding query heads into the query axis;
* **long queries** (prefill chunks): one fused `scaled_dot_product_attention` per sequence.

It's a readable reference in plain PyTorch; production engines swap step 2 for a fused kernel (PagedAttention / FlashInfer) that reads the pool in place.

### Prefix caching

A full block's hash is `hash(parent_hash, its 16 tokens)`, so a hash names the whole prefix up to that block — a radix tree over blocks. New requests walk their prompt block by block and share every hit (ref-counted). Only full blocks are shared and sequences only write their private last block, so no copy-on-write is needed. Freed blocks keep their contents in an LRU pool until memory is needed. ([`block_allocator.py`](inference_engine/engine/block_allocator.py))

### Preemption

FCFS admission, LIFO preemption: when a running sequence can't grow, the most recently arrived one is preempted — swapped to a pinned CPU pool if it's decoding, else its blocks are dropped and it re-prefills later (usually cheap, since its blocks stay in the prefix cache). Nothing new is admitted while swapped sequences wait.

### Speculative decoding

Decode is bandwidth-bound: verifying 5 tokens costs about as much as generating 1. A proposer guesses k tokens; the target scores `[last, d1..dk]` in one pass; drafts are accepted while they match the target's greedy choice, plus one free "bonus" token. Output is identical to greedy decoding. ([`spec_decode.py`](inference_engine/engine/spec_decode.py))

* `SPECULATIVE_METHOD=ngram` — prompt lookup: continue the most recent earlier occurrence of the last n tokens. Free; great for summarisation, RAG, code edits.
* `SPECULATIVE_METHOD=draft` — a small same-family model (e.g. Qwen2.5-0.5B for Qwen2.5-7B) with its own paged pool sharing the target's block tables.

Speculation pays off most when the GPU is latency-bound (small batches); at large batch sizes decode becomes compute-bound and verification work competes with other requests. It applies to greedy requests only.

---

## 📁 Project structure

```
inference_engine/
├── config.py                 # All tunables; every field overridable by env var
├── models/loader.py          # Direct-to-GPU loading, dtype auto (bf16/fp16/fp32), EOS ids
├── engine/
│   ├── sequential.py         # Phase 1: naive prefill → decode loop (+ streaming)
│   ├── sequence.py           # Per-request state, SamplingParams, latency metrics
│   ├── request_queue.py      # FIFO queue: timeouts, cancellation, backpressure
│   ├── scheduler.py          # Continuous batching: plan → execute → apply
│   ├── model_runner.py       # Packed batched forward pass, KV pool auto-sizing
│   ├── attention_wrapper.py  # Paged attention backend for HF models
│   ├── sampler.py            # Greedy / temperature / top-k / top-p
│   ├── spec_decode.py        # n-gram and draft-model proposers
│   ├── block_allocator.py    # Block tables, ref counts, prefix cache
│   ├── paged_kv_cache.py     # The device tensor pool
│   ├── cpu_swap_manager.py   # Pinned CPU pool for swapping
│   ├── kv_cache_config.py    # Bytes-per-token sizing from the model config
│   ├── kv_cache_tracker.py   # Logical KV accounting
│   ├── stage_tracker.py      # Per-step prefill/decode telemetry
│   └── metrics_aggregator.py # /metrics: percentiles, throughput, SLOs, step stats
├── server/
│   ├── app.py                # Phase 1 server (:8000)
│   ├── app_v2.py             # Engine server (:8001), SSE + OpenAI API
│   └── detokenizer.py        # Incremental detokenization for streaming
└── tests/                    # pytest suite (tiny offline models + real-model checks)
load_test/                    # Workloads, arrival processes, streaming client, reports, GPU monitor
benchmarks/
├── bench_offline.py          # In-process ablation: batching / prefix / speculation
├── bench_serving.py          # Launches servers, sweeps Poisson rates or concurrency
├── plot_results.py           # Charts from result JSON
└── legacy/                   # Original M2 micro-benchmarks
scripts/                      # colab_setup.sh, gpu_smoke_test.py
colab/PageServe_Colab.ipynb   # End-to-end Colab runbook
run_load_test.py              # Load-test CLI
run_validation*.py            # Original Phase 1 / Phase 2 validation scripts
```

## 🛠️ Development phases

1. **Sequential baseline** — single-request lock, synchronous token loop.
2. **Continuous batching scheduler** — background loop, iteration-level scheduling.
3. **Request queue** — FIFO with timeouts and backpressure.
4. **Prefill / decode separation** — per-step prefill budget and chunked prefill.
5. **KV memory tracking.**
6. **Block allocator** — fixed 16-token blocks, block tables.
7. **Paged KV cache** — one pre-allocated device pool.
8. **Paged attention integration** — the model reads/writes the pool directly; one packed forward pass per step.
9. **CPU swapping** — preemption without losing work.
10. **Unified metrics.**
11. **Load testing.**
12. **Prefix caching, speculative decoding, streaming/OpenAI API, GPU benchmarks.**

---

## 📊 Benchmarking

All workloads are seeded; results record GPU, versions and git commit.

### Offline ablation — what each optimisation buys

```bash
python -m benchmarks.bench_offline [--suites batching,prefix,spec] [--draft-model auto] [--quick]
```

| Suite | Workload | Systems compared |
|---|---|---|
| `batching` | random prompts (256–512 tok), outputs 128–256 tok | HF sequential · HF static batching · engine with batch size 1 · continuous batching |
| `prefix` | 1k-token shared system prompt + short questions | prefix caching off vs on |
| `spec` | copy-heavy prompts, and chat questions — outputs end at EOS | none · n-gram · draft model |

Reports output tok/s, speedup, TTFT/TPOT/ITL percentiles, peak GPU memory, mean GPU utilisation, prefix-hit and acceptance rates.

### Serving — latency under load

```bash
python -m benchmarks.bench_serving [--configs sequential,continuous,prefix] \
    [--workload random|shared_prefix|repetitive|chat] [--rates 2 4 8 16 inf | --concurrency 1 8 32 64]
# speculative decoding: compare with natural stopping
python -m benchmarks.bench_serving --configs prefix,ngram --workload chat --natural-stop
```

Launches a server per configuration, sweeps Poisson request rates (open loop) or fixed concurrencies (closed loop) with a streaming client, and reports per level: throughput, TTFT / TPOT / ITL / E2E p50–p99, goodput (share of requests meeting TTFT ≤ 2 s and TPOT ≤ 100 ms), GPU utilisation and memory, plus engine stats from `/metrics`.

### Ad-hoc load tests

```bash
python run_load_test.py --server phase2 --profile poisson --rps 8 --workload random \
    --num-requests 200 --stream --ignore-eos
python run_load_test.py --server phase2 --concurrency 32 --stream          # closed loop
python run_load_test.py --server custom --url http://localhost:8000 --api openai \
    --model Qwen/Qwen2.5-1.5B-Instruct --stream                             # e.g. vLLM
```

### Published results

Raw result files from real runs live in [`benchmarks/published/`](benchmarks/published) (JSON + Markdown + charts, each stamped with GPU, versions and git commit). The website reads a trimmed copy produced by:

```bash
python -m benchmarks.export_site_data benchmarks/published/T4_2026-09-27 web/src/data/results/t4.json
```

**NVIDIA T4 (Colab), Qwen2.5-1.5B-Instruct fp16** — offline: continuous batching **780 tok/s vs 27.5 for HF sequential (28×)** and 277 for HF static batching (2.8×); prefix caching **1.9×** on a shared 1k-token system prompt; n-gram speculation **1.5×** on copy-heavy output and no gain on free-form chat. Serving: capacity **3.7 req/s vs 0.14** for the sequential server, 100% of requests within SLO up to 4 req/s. fp16 accuracy matches HF's own (99.7% top-1 agreement with fp32).

### How the numbers are measured

- **TTFT** — scheduled send time → first streamed token (includes queueing).
- **TPOT** — (last token − first token) / (tokens − 1), per request.
- **ITL** — every gap between streamed chunks, pooled across requests.
- **E2E** — scheduled send time → last token.
- Latency is measured from the *scheduled* send time, so a server that falls behind is charged for it (no coordinated omission). Batching and prefix-caching benchmarks use `ignore_eos` so every system generates the same number of tokens. Speculative-decoding benchmarks let outputs end at EOS instead: forcing generation past EOS makes models repeat themselves, which n-gram drafts predict almost perfectly and would overstate the speedup (greedy speculation is exact, so all configs still emit identical text).
- **GPU utilisation** is NVML's "time a kernel was running" — a busy-ness signal, not FLOP efficiency.

---

## 📡 API reference

### `POST /generate` (both servers)

```json
{"prompt": "…", "max_new_tokens": 128, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
 "ignore_eos": false, "stream": false}
```

Non-streaming response: `generated_text`, `prompt_tokens`, `generated_tokens`, `ttft_ms`, `tpot_ms`, `total_latency_ms`, `per_token_latencies_ms` (ITL), `queue_wait_time_ms`, `cached_prompt_tokens`, `spec_draft_tokens`, `spec_accepted_tokens`, `finish_reason`, GPU memory.

With `"stream": true`: Server-Sent Events `data: {"token_ids": […], "text": "…"}` per step, then one `{"done": true, …}` event with the full result, then `data: [DONE]`. (Phase 1 supports `max_new_tokens`, `ignore_eos` and `stream`.)

### `POST /v1/completions` (engine)

OpenAI-style: `prompt`, `max_tokens`, `temperature`, `top_p`, `stream`, plus `ignore_eos`. Streams OpenAI chunks ending with a `usage` chunk.

### `GET /metrics` (engine)

`system` (in-flight, waiting, throughput), `e2e_latency` (TTFT/TPOT/ITL/E2E/queue p50–p99), `engine` (KV blocks, prefix hit rate, preemptions), `speculative_decoding` (acceptance, tokens per step), `engine_steps` (avg batch size, tokens/step, KV utilisation, step latency), `gpu_memory`, `paged_kv_cache`, `cpu_swap`, `queue_stats`, `slo_compliance`.

### `GET /health`

Model, device, batch occupancy, queue depth, KV block usage, enabled features.

---

## ⚙️ Configuration

Every field in [`config.py`](inference_engine/config.py) can be set via an upper-case environment variable (explicit constructor arguments win). The important ones:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen2-0.5B` | HF model id or local path (Llama/Qwen2/Qwen2.5/Qwen3/Mistral-style decoders) |
| `DEVICE` | auto | `cuda`, `mps` or `cpu` |
| `DTYPE` | `auto` | bf16 on Ampere+, fp16 on T4/MPS, fp32 on CPU |
| `MAX_BATCH_SIZE` | 64 (CUDA) / 8 | max sequences in the running batch |
| `PREFILL_BUDGET_TOKENS` | 2048 / 512 | prompt tokens processed per step (all sequences) |
| `PREFILL_CHUNK_SIZE` | 512 / 128 | prompt tokens per sequence per step |
| `MAX_MODEL_LEN` | min(model max, 4096) | longest prompt + output |
| `KV_BLOCK_SIZE` | 16 | tokens per KV block |
| `KV_NUM_BLOCKS` | auto | 0 = profile GPU memory (CUDA) / `KV_CACHE_MAX_MEMORY_MB` (else) |
| `GPU_MEMORY_UTILIZATION` | 0.85 | share of GPU memory the engine may use |
| `KV_CACHE_MAX_MEMORY_MB` | 1024 | KV pool size on CPU/MPS |
| `ENABLE_PREFIX_CACHING` | 1 | share KV of common prompt prefixes |
| `PREEMPTION_MODE` | `swap` | `swap` (falls back to recompute) or `recompute` |
| `SWAP_SPACE_GB` | 2 | CPU swap pool size |
| `SPECULATIVE_METHOD` | off | `ngram` or `draft` |
| `NUM_SPECULATIVE_TOKENS` | 4 | drafts per step |
| `DRAFT_MODEL_NAME` | – | draft model for `draft` |
| `REQUEST_TIMEOUT_MS` | 60000 | max queue wait before 504 |
| `MAX_QUEUE_SIZE` | 16 × batch | waiting requests before 503 |

---

## 🔬 Tests

```bash
pytest -m "not slow"     # ~20 s, offline: tiny random models, exact match vs HF generate()
pytest                   # + real Qwen2-0.5B checks (downloads ~1 GB once)
```

The engine is checked token-for-token against HuggingFace `generate()` (float64 tiny models) under every combination that changes *how* tokens are computed: chunked prefill, batching, prefix caching, swap and recompute preemption, n-gram and draft speculation. Other suites cover the attention kernel against a naive reference (including sliding window and GQA), allocator ref-counting and hashing, sampling, streaming, aborts, timeouts, and both HTTP servers end to end.

---

## 🧾 v3 audit — what was fixed

The v2 engine worked on small demos but did not do what its architecture claimed:

- **The paged KV cache was not used for attention.** Each sequence kept its own HF `DynamicCache`, and the pool was a shadow copy written token-by-token with Python loops (≈20k tiny kernel launches for a 400-token prompt). Memory accounting was therefore fictional, and every sequence ran its own forward pass — no GPU batching. → The model now reads and writes the pool directly and each step is one packed forward pass.
- **Swap race:** a sequence swapped out during a decode step could end up in both `running` and `swapped_out`, then be swapped back in as a duplicate.
- **Preemption policy:** new arrivals evicted the *largest* running sequence, so old requests could be starved; now FCFS admission + LIFO preemption, with recompute fallback.
- **Metrics:** TTFT excluded queueing; "per-token latency" was forward time, not inter-token latency; throughput always divided by 60 s (under-reporting after startup); the token window grew without bound unless `/metrics` was polled; Phase 1 TTFT ignored time waiting for the lock.
- **Sizing & loading:** `head_dim` was derived as `hidden/heads` (wrong for several families); CUDA used `device_map="auto"` (silent CPU offload) and always fp16; multiple EOS ids (instruct models) were ignored.
- **Load tester:** latency was measured after a client-side semaphore (coordinated omission) and TTFT was taken from the server.
- **Found while testing on MPS:** non-blocking host→device copies of temporary buffers raced (garbage outputs); `index_copy_` on MPS copies the entire destination (1000× slower than `index_put_`).

## 🗺️ Next steps

- A fused paged-attention kernel (Triton / FlashInfer) instead of gather + matmul.
- CUDA graphs for decode steps (removes Python/launch overhead at small batch sizes).
- Rejection sampling so speculation also applies to sampled requests.
- Tensor parallelism for models that don't fit one GPU.

---

<div align="center">

**Built by [Nikhil Mourya](https://github.com/TryingtobeingNikhil)**

*PageServe is what LLM serving looks like when you build the engine block-by-block.*

</div>
