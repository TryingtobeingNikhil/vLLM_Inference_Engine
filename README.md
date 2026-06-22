# PageServe — Phase 1: Sequential Serving Baseline

A minimal, correctly-instrumented LLM inference server built from scratch
using raw HuggingFace `transformers`. This is the **intentionally naive
baseline** for all future optimisation phases.

---

## Project Structure

```
inference_engine/
├── config.py               # Central config dataclass
├── models/
│   └── loader.py           # Model + tokenizer loading (MPS-safe)
├── engine/
│   └── sequential.py       # prefill / decode / generate + GenerationResult
├── metrics/
│   └── collector.py        # Thread-safe metrics store + JSON persistence
├── server/
│   └── app.py              # FastAPI server
└── tests/
    └── test_sequential.py  # pytest unit tests

run_validation.py           # Sends 5 requests, verifies sequentiality
requirements.txt
```

---

## Setup

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set environment variables to override defaults
export MODEL_NAME="Qwen/Qwen2-0.5B"    # or TinyLlama/TinyLlama-1.1B-Chat-v1.0
export DEVICE="mps"                     # auto-detected if not set
```

---

## Running the Server

```bash
cd inference_engine
uvicorn server.app:app --host 0.0.0.0 --port 8000 --log-level info
```

The server will:
1. Detect device automatically (cuda → mps → cpu)
2. Download and load the model (~500 MB for Qwen2-0.5B)
3. Start serving on port 8000

---

## API

### `POST /generate`

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_new_tokens": 50}'
```

Response:
```json
{
  "prompt": "The capital of France is",
  "generated_text": "Paris",
  "prompt_tokens": 6,
  "generated_tokens": 1,
  "ttft_ms": 124.3,
  "total_latency_ms": 890.1,
  "tokens_per_second": 11.2,
  "per_token_latencies_ms": [88.4, 91.2, ...],
  "gpu_memory_allocated_mb": 1024.0,
  "gpu_memory_reserved_mb": 2048.0,
  "timestamp": "2026-06-17T20:00:00+00:00"
}
```

### `GET /metrics`

Returns all stored results plus summary statistics (p50/p95/p99).

### `GET /health`

```json
{"status": "ok", "model": "Qwen/Qwen2-0.5B", "device": "mps", "requests_served": 5}
```

---

## Running Tests

```bash
cd inference_engine
pytest tests/ -v
```

Tests load the model once (session scope) and run all 5 checks. Expect ~1–3
minutes on M2 for the first run (model download + load).

---

## Running Validation

With the server running in one terminal:

```bash
# Terminal 2
python run_validation.py
# or with custom args:
python run_validation.py --host 127.0.0.1 --port 8000 --output baseline_metrics.json
```

**Output includes:**

1. A summary table per request:
   ```
   | Req | TTFT (ms) | Total (ms) | tok/s | Prompt tok | Gen tok | Alloc MB | Resv MB |
   ```

2. A sequentiality verification section:
   ```
   Sequential ordering verification:
     Seq check 1→2: gap=+5.2 ms  ✓
     Seq check 2→3: gap=+3.8 ms  ✓
     ...
     ✓ PASS — all requests were served sequentially
   ```

3. `baseline_metrics.json` with raw results and wall-clock log.

---

## Memory Instrumentation (M2 / MPS)

On Apple Silicon, `torch.cuda.*` is not available. The engine uses:

| Metric | Source |
|---|---|
| `gpu_memory_allocated_mb` | `torch.mps.current_allocated_memory()` (PyTorch ≥ 2.1) |
| `gpu_memory_reserved_mb` | `psutil.Process().memory_info().rss` (process RSS) |

If PyTorch < 2.1, both fields fall back to `psutil`.

---

## What This Baseline Does NOT Do (Intentionally)

- ❌ No batching
- ❌ No custom KV cache management (uses HuggingFace `past_key_values` as-is)
- ❌ No Flash Attention or attention optimisation
- ❌ No speculative decoding
- ❌ No continuous batching

The goal is a correct, measurable, intentionally naive baseline so that each
subsequent phase can be compared against it.
