<div align="center">

<img src="https://img.shields.io/badge/PageServe-LLM%20Inference%20Engine-2F80ED?style=for-the-badge&logoColor=white" alt="PageServe">

# 🧠 PageServe

### *A high-performance LLM inference engine built from scratch featuring continuous batching, paged KV-cache, and CPU swapping.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.40+-yellow?style=flat-square)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

**From a naive single-request bottleneck to a production-grade scheduling engine. PageServe is an educational and modular showcase of how modern LLM serving architectures work under the hood.**

[Quick Start](#-quick-start) · [Architecture](#-architecture) · [Development Phases](#-development-phases) · [Engineering Decisions](#-key-engineering-decisions) · [API Reference](#-api-reference) · [Validation & Benchmarks](#-validation--benchmarking)

</div>

---

## 🎯 The Problem

Naive LLM deployment uses a simple wrapper around HuggingFace's `generate()` method. This design has critical limitations:
1. **Head-of-Line Blocking**: One client requesting 500 tokens blocks another client requesting 5 tokens for the entire duration of the generation.
2. **Dynamic Memory Waste (KV-Cache Fragmentation)**: Standard Transformers allocate KV-caches as contiguous arrays in GPU memory. Because output lengths are unpredictable, this leads to significant internal fragmentation, reserving more space than needed and limiting concurrency.
3. **Out-of-Memory (OOM) Vulnerability**: Under load, concurrent request allocation spikes GPU memory, crashing the server.

**PageServe solves this.** By implementing an iteration-level scheduler with Paged KV-Cache allocation and CPU-side swapping, it maximizes GPU utilization and concurrency while gracefully handling memory pressure.

---

## ✨ Features

- **Continuous Batching**: Iteration-level scheduling allows new requests to be admitted during the decode steps of running requests.
- **Paged KV-Cache**: Eliminates memory fragmentation by breaking down the key-value cache of active sequences into fixed-sized logical blocks (size 16) mapped to physical tensors.
- **CPU-GPU Swap Pool**: Prevents OOM crashes under high load. Swaps idle or preempted sequences out to a CPU-side staging pool, freeing up device blocks for active generation.
- **Independent Prefill/Decode Budgets**: Avoids overloading GPU execution by capping the number of prompt tokens prefilled per iteration and limiting max decode batch size.
- **Unified Telemetry & Metrics**: Real-time rollups of throughput (tokens/sec), queue depth, latency percentiles ($P_{50}$ / $P_{95}$ / $P_{99}$), and SLO compliance rates.
- **Multi-Profile Load Tester**: Simulates constant, ramp, and burst load patterns to benchmark engine behavior and compare different phases.

---

## 🏗️ Architecture

```
                                 PAGESERVE SYSTEM
                                 
       Request ────► [ FastAPI Server (app_v2.py on :8001) ]
                                      │
                                      ▼
                      [ RequestQueue (FIFO with timeout) ]
                                      │
                                      ▼
                     [ ContinuousBatchingScheduler ] ◄─── (run_loop background task)
                                      │
                ┌─────────────────────┴─────────────────────┐
                │                                           │
                ▼ (Admit sequence)                          ▼ (Single-token step)
     [ Prefill Stage ]                              [ Decode Stage ]
      - Prefill Budget                               - Decode batch limit
      - Allocate blocks                              - Update attention KV cache
                │                                           │
                └─────────────────────┬─────────────────────┘
                                      │
                                      ▼
                    [ PagedKVCacheManager (Device Pool) ]
                                    ▲   │
                           swap_in  │   │  swap_out
                           (retry)  │   ▼  (memory pressure)
                     [ CPUSwapManager (CPU Staging Pool) ]
```

---

## 📁 Project Structure

```
inference_engine/
├── config.py               # Central config dataclass (all tunable parameters)
├── models/
│   └── loader.py           # Device-agnostic model and tokenizer loader (supports MPS/CUDA/CPU)
├── metrics/
│   └── collector.py        # Thread-safe raw metrics accumulator
├── server/
│   ├── app.py              # Phase 1: Sequential serving FastAPI app (:8000)
│   └── app_v2.py           # Phase 2+: Continuous batching FastAPI app (:8001)
├── engine/
│   ├── sequential.py       # Phase 1: Naive generation baseline
│   ├── sequence.py         # Sequence tracking structure (states, tokens, time, blocks)
│   ├── scheduler.py        # Continuous batching scheduling loop
│   ├── request_queue.py    # FIFO request queue with request timeout constraints
│   ├── stage_tracker.py    # Tracks whether sequences are in prefill or decode stages
│   ├── kv_cache_config.py  # Calculates default memory limits and block sizing
│   ├── kv_cache_tracker.py # Monitors current KV memory footprint
│   ├── block_allocator.py  # Thread-safe logical block allocation manager
│   ├── paged_kv_cache.py   # Physical GPU tensor block storage
│   ├── cpu_swap_manager.py # Swapping mechanism between GPU and CPU staging pool
│   ├── prefill_utils.py    # Multi-sequence prefill utilities
│   ├── attention_wrapper.py# Paged attention layer wrapper logic
│   └── metrics_aggregator.py# Phase 10: Unified telemetry & derived metrics rollup
└── tests/                  # Pytest unit testing suite
run_validation.py           # Validates sequential ordering baseline (Phase 1)
run_validation_v2.py        # Benchmarks continuous batching speedup vs Phase 1
run_load_test.py            # Generates synthetic load configurations and comparisons
requirements.txt            # System dependencies
```

---

## 🛠️ Development Phases

PageServe was constructed incrementally through 11 modular phases to map the evolution of inference engine designs:

* **Phase 1: Sequential Serving Baseline** - Single-request lock, synchronous token loop.
* **Phase 2: Continuous Batching Scheduler** - Background execution task, iteration-level scheduling.
* **Phase 3: Request Queue** - FIFO queueing to handle concurrent client requests gracefully without dropping connections.
* **Phase 4: Prefill / Decode Separation** - Independent budgets for prompt tokens (prefill) and decode batches to prevent engine stalling.
* **Phase 5: KV Cache Memory Tracking** - Monitors physical KV-cache footprint during execution.
* **Phase 6: Block Allocator** - Dynamic block allocation mapping sequence tokens to logical blocks of size 16.
* **Phase 7: Paged KV Cache** - Maps logical block indices to physical tensors stored on device.
* **Phase 8: Batch Attention Integration** - Hooks up model forward passes to read/write paged attention segments.
* **Phase 9: CPU Staging Pool & Swapping** - Swaps preempted sequence caches to host RAM when GPU memory limit is reached.
* **Phase 10: Unified Metrics Aggregation** - End-to-end telemetry surface (P50/P95/P99 latency, throughput, SLO compliance).
* **Phase 11: Load Testing Tool** - Simulates constant, ramp, and burst load patterns.

---

## 🔑 Key Engineering Decisions

### 1. Paged KV-Cache block allocation
Instead of pre-allocating a static maximum sequence length tensor for every sequence, PageServe uses a logical `BlockAllocator` coupled with `PagedKVCacheManager` that manages memory in blocks of 16 tokens.
- **Why**: Standard Transformers allocate memory for `max_sequence_length` up-front, resulting in up to 60-80% memory waste due to unused generation padding. Logical block mapping eliminates external fragmentation and permits high-density batching.

---

### 2. CPU Staging Pool over Request Dropping
When the GPU runs out of physical KV-cache blocks, PageServe does not reject incoming requests or crash. The `CPUSwapManager` copies the KV-cache of lower-priority (youngest) sequences to system RAM and clears their GPU blocks.
- **Why**: Sustained burst traffic will eventually exhaust any GPU cache. Swapping allows the system to degrade gracefully by introducing latency (preemption) instead of throwing HTTP 500 errors or failing via GPU OOM.

---

### 3. Separation of Prefill and Decode Budgets
The scheduler processes a limited number of prefill tokens (`prefill_budget_tokens=512`) and active decodes (`decode_batch_limit=8`) per step.
- **Why**: Prefill is compute-bound (takes longer to run initial prompt tokens), while decode is memory-bandwidth bound. Mixing too many large prompts in a single execution step causes spikes in iteration time, degrading the Time-to-First-Token (TTFT) for all queued requests.

---

### 4. Graceful Error Handling Design

| System Component | On Failure | Mitigation |
| --- | --- | --- |
| **Request Queue** | Queue Full | Rejects request with `429 Too Many Requests` or `QueueFullError` |
| **Block Allocator** | GPU Block Exhaustion | Triggers preemption: swaps out oldest/youngest sequence to CPU memory |
| **Swap Pool** | CPU Pool Exhaustion | Swapped sequences wait; new requests are blocked until active ones complete |
| **HTTP client / Server** | Timeout | Clears sequence state on disconnect, releasing blocks |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- PyTorch 2.1+ (with CUDA or Apple Silicon MPS support)
- HuggingFace access token (optional, for gated models)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/TryingtobeingNikhil/vLLM_Inference_Engine.git
   cd vLLM_Inference_Engine
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

---

### Running the Engines

#### Running Phase 1 (Sequential Baseline Server)
```bash
# Starts the Phase 1 server on port 8000
python -m uvicorn inference_engine.server.app:app --host 0.0.0.0 --port 8000
```

#### Running Phase 2+ (Continuous Batching Server)
```bash
# Starts the Phase 2 server on port 8001
python -m uvicorn inference_engine.server.app_v2:app --host 0.0.0.0 --port 8001
```

*Note: The model (`Qwen/Qwen2-0.5B` by default) will auto-download on the first launch. You can change this by setting the environment variable `export MODEL_NAME="your/choice"`.*

---

## 📡 API Reference

### POST `/generate`
Submits a prompt for token generation. Blocks until generation is complete.

- **URL**: `http://localhost:8001/generate` (Phase 2) or `http://localhost:8000/generate` (Phase 1)
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "prompt": "Write a short poem about latency.",
    "max_new_tokens": 50
  }
  ```
- **Response**:
  ```json
  {
    "prompt": "Write a short poem about latency.",
    "generated_text": "...",
    "prompt_tokens": 8,
    "generated_tokens": 50,
    "ttft_ms": 110.2,
    "total_latency_ms": 870.5,
    "tokens_per_second": 57.4,
    "per_token_latencies_ms": [110.2, 15.5, 15.6, ...],
    "gpu_memory_allocated_mb": 1150.4,
    "gpu_memory_reserved_mb": 2048.0,
    "timestamp": "2026-06-24T00:44:00Z"
  }
  ```

---

### GET `/metrics`
Exposes system-wide telemetry computed by the `MetricsAggregator`.

- **URL**: `http://localhost:8001/metrics`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "summary": {
      "total_requests": 25,
      "p50_latency_ms": 780.2,
      "p95_latency_ms": 1205.4,
      "p99_latency_ms": 1450.0,
      "mean_ttft_ms": 95.3,
      "p50_ttft_ms": 90.1,
      "p95_ttft_ms": 115.0
    },
    "throughput": {
      "rolling_token_throughput": 124.5,
      "rolling_request_throughput": 2.4
    },
    "slo": {
      "ttft_compliance_fraction": 0.98,
      "total_latency_compliance_fraction": 0.95
    },
    "system": {
      "oom_count": 0,
      "swap_out_count": 2,
      "active_requests": 3,
      "waiting_requests": 0
    },
    "kv_cache": {
      "gpu_blocks_used": 48,
      "gpu_blocks_free": 208,
      "cpu_blocks_used": 16,
      "cpu_blocks_free": 112
    }
  }
  ```

---

## ⚙️ Configuration

Tunable parameters located in [inference_engine/config.py](file:///Users/nikhilmourya/Desktop/PageServe/inference_engine/config.py) can be overridden using environment variables:

| Environment Variable | Default Value | Description |
| --- | --- | --- |
| `MODEL_NAME` | `"Qwen/Qwen2-0.5B"` | HuggingFace model identifier to load |
| `DEVICE` | `Auto-detected` | Hardware device to use (`cuda`, `mps`, or `cpu`) |
| `PORT` | `8000` | Port for the API server |
| `METRICS_OUTPUT_PATH`| `"baseline_metrics.json"` | Path to write historical benchmark output |
| `MAX_BATCH_SIZE` | `4` | Maximum number of concurrent sequences active in execution |
| `PREFILL_BUDGET_TOKENS` | `512` | Max prompt tokens admitted for prefill per scheduler step |
| `DECODE_BATCH_LIMIT` | `8` | Max active sequences allowed in decode batch |
| `KV_BLOCK_SIZE` | `16` | Token capacity per cache block |
| `KV_NUM_BLOCKS` | `256` | Number of physical GPU cache blocks |
| `KV_NUM_CPU_BLOCKS` | `128` | Number of physical host memory cache blocks (swapping pool) |

---

## 📊 Validation & Benchmarking

### 1. Sequential ordering validation
Validates that the sequential baseline (Phase 1) processes requests in strict serialization.
```bash
python run_validation.py
```

---

### 2. Continuous batching speedup benchmark
Sends 8 concurrent requests to the Phase 2 server, measures performance, and compares it directly with the Phase 1 sequential baseline stored in `baseline_metrics.json`.
```bash
# Make sure Phase 2 server is running in another terminal
python run_validation_v2.py
```
This generates comparison metrics and plots a `batch_size_over_time.png` graph tracking scheduler concurrency.

---

### 3. Load testing suite
Executes load generation profiles against either server.
```bash
# Target Phase 2 server with a ramp load from 1 to 10 RPS over 30s
python run_load_test.py --server phase2 --profile ramp --start-rps 1.0 --end-rps 10.0 --duration 30.0

# Compare Phase 1 and Phase 2 reports
python run_load_test.py --server phase1 --output p1.json
python run_load_test.py --server phase2 --output p2.json --compare-with p1.json
```

---

## 🔬 Run Unit Tests
To verify all scheduler, block allocation, and swap logic functions correctly:
```bash
pytest inference_engine/tests/ -v
```

---

## 🗺️ What I'd Do Differently

- **Tensor-Level Batching**: Implement actual tensor batching using padded tensors or jagged attention operators (e.g., FlashAttention-2 or vLLM custom CUDA kernels) instead of thread-based sequence serialization.
- **Asynchronous Completion Events**: Replace polling in FastAPI `/generate` endpoints with asyncio `Event` hooks triggered by the scheduler loop, eliminating busy-wait polling overhead.
- **Speculative Decoding**: Integrate a smaller draft model to verify candidate tokens in parallel, improving decode step speeds.
- **Dynamic Block Sizing**: Experiment with block sizes of 8 and 32 to study the impact of chunk overhead on compute performance and cache mapping.

---

<div align="center">

**Built by [Nikhil Mourya](https://github.com/TryingtobeingNikhil)** · June 2026

*PageServe is what LLM serving looks like when you build the engine block-by-block.*

</div>
