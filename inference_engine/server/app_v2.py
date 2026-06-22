"""
server/app_v2.py — FastAPI server for Phase 2 continuous-batching inference.

Differences from Phase 1 (app.py)
----------------------------------
* No inference_lock, no single-request serialisation.
* On startup a ContinuousBatchingScheduler is created and its run_loop() is
  started as a background asyncio Task.
* POST /generate submits a request to the scheduler and polls for completion.
* GET /metrics exposes scheduler-level telemetry (batch_size_over_time,
  scheduler_step_latency_ms, per-sequence stats) in addition to the per-
  request summary statistics from Phase 1.
* GET /health includes current_batch_size and queue_depth.
* Runs on port 8001 so Phase 1 (port 8000) and Phase 2 can run side-by-side
  for direct comparison.

Polling design (known limitation)
-----------------------------------
The /generate handler uses:

    while seq.state != "finished":
        await asyncio.sleep(POLL_INTERVAL_S)

This is a busy-wait with a configurable 5 ms sleep.  It is an intentional
Phase 2 simplification.  The correct production alternative — an asyncio.Event
per Sequence that the scheduler sets on completion — is deferred to a later
phase.  The 5 ms sleep keeps CPU utilisation minimal while not adding
noticeable latency relative to typical generation times.

Endpoints
---------
POST /generate      Submit prompt; blocks until generation finishes.
GET  /metrics       Scheduler telemetry + per-sequence summary stats.
GET  /health        Liveness check with current batch occupancy.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from inference_engine.config import Config
from inference_engine.engine.request_queue import QueueFullError
from inference_engine.engine.kv_cache_config import format_kv_cache_report
from inference_engine.engine.scheduler import ContinuousBatchingScheduler
from inference_engine.engine.sequence import Sequence
from inference_engine.models.loader import LoadedModel, load_model_and_tokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── Module-level singletons (populated during lifespan startup) ───────────────

_config: Optional[Config] = None
_loaded_model: Optional[LoadedModel] = None
_scheduler: Optional[ContinuousBatchingScheduler] = None

# Polling interval for /generate completion checks (5 ms)
_GENERATE_POLL_INTERVAL_S = 0.005


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model, create scheduler, start background loop.
    Shutdown: stop scheduler gracefully.
    """
    global _config, _loaded_model, _scheduler

    _config = Config()
    logger.info(
        "Phase 2 server starting: model=%s device=%s max_batch_size=%d",
        _config.model_name,
        _config.device,
        _config.max_batch_size,
    )

    # Model loading is blocking — run in a temporary executor so startup
    # doesn't block the event loop.  The scheduler gets its own executor later.
    loop = asyncio.get_event_loop()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as tmp_exec:
        _loaded_model = await loop.run_in_executor(
            tmp_exec, load_model_and_tokenizer, _config
        )
    logger.info("Model loaded successfully on device=%s", _loaded_model.device)

    _scheduler = ContinuousBatchingScheduler(
        model=_loaded_model.model,
        tokenizer=_loaded_model.tokenizer,
        config=_config,
    )
    print(format_kv_cache_report(_scheduler.kv_cache_config))
    _scheduler.start()   # creates asyncio.Task for run_loop()
    logger.info("Scheduler started (max_batch_size=%d)", _config.max_batch_size)

    yield  # ── server is running ────────────────────────────────────────────

    logger.info("Phase 2 server shutting down …")
    await _scheduler.stop()
    logger.info("Scheduler stopped. %d sequences finished.", len(_scheduler.finished))


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Continuous Batching LLM Inference Server",
    description=(
        "Phase 2: iteration-level continuous batching scheduler. "
        "Multiple requests are batched dynamically; no rewrite of Phase 1."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt text")
    max_new_tokens: int = Field(
        default=50, ge=1, le=512, description="Maximum tokens to generate"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sequence_to_result_dict(seq: Sequence, device: str) -> dict:
    """Convert a finished Sequence into a GenerationResult-compatible dict."""
    from inference_engine.engine.sequential import get_memory_stats

    generated_text = seq.prompt  # start with prompt — Phase 1 decode() does same
    if seq.generated_token_ids:
        generated_text = _scheduler.tokenizer.decode(  # type: ignore[union-attr]
            seq.generated_token_ids, skip_special_tokens=True
        )

    total_ms = (
        (seq.per_token_latencies_ms[-1] if seq.per_token_latencies_ms else 0.0)
        + seq.ttft_ms
        + sum(seq.per_token_latencies_ms)
    )
    # More precise: wall clock from arrival to last token
    # Use ttft + sum(per_token) as a lower bound
    total_latency_ms = seq.ttft_ms + sum(seq.per_token_latencies_ms)
    n_gen = len(seq.generated_token_ids)
    tps = (n_gen / total_latency_ms * 1000.0) if total_latency_ms > 0 else 0.0

    allocated_mb, reserved_mb = get_memory_stats(device)

    return {
        "seq_id": seq.seq_id,
        "prompt": seq.prompt,
        "generated_text": generated_text,
        "prompt_tokens": len(seq.prompt_token_ids),
        "generated_tokens": n_gen,
        "ttft_ms": seq.ttft_ms,
        "total_latency_ms": total_latency_ms,
        "tokens_per_second": tps,
        "per_token_latencies_ms": seq.per_token_latencies_ms,
        "gpu_memory_allocated_mb": allocated_mb,
        "gpu_memory_reserved_mb": reserved_mb,
        "finish_reason": seq.finish_reason,
        "queue_wait_time_ms": seq.queue_wait_time_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/generate", response_class=JSONResponse)
async def endpoint_generate(request: GenerateRequest):
    """Submit *prompt* to the continuous batching scheduler.

    The handler polls seq.state every 5 ms until the scheduler marks it
    'finished'.  This polling is a known Phase 2/3 limitation.

    Phase 3 additions:
    - QueueFullError → HTTP 503 (server at capacity)
    - The asyncio.Future returned by add_request is wired but not awaited;
      full future-based completion signalling is deferred to a later phase.
    """
    if _scheduler is None or _loaded_model is None:
        raise HTTPException(status_code=503, detail="Scheduler not ready")

    try:
        seq, _future = await _scheduler.add_request(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
        )
    except QueueFullError:
        raise HTTPException(
            status_code=503,
            detail="Server at capacity, retry later",
        )
    except Exception as exc:
        logger.exception("Failed to enqueue request: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Busy-wait poll — known Phase 3 limitation (future is wired but unused)
    while not seq.is_finished():
        await asyncio.sleep(_GENERATE_POLL_INTERVAL_S)

    return JSONResponse(
        content=_sequence_to_result_dict(seq, _loaded_model.device)
    )


@app.get("/metrics", response_class=JSONResponse)
async def endpoint_metrics():
    """Return scheduler telemetry and per-sequence statistics.

    Response structure
    ------------------
    {
        "scheduler": {
            "batch_size_over_time":     [(timestamp, batch_size), ...],
            "scheduler_step_latency_ms": [float, ...],
            "running_count":            int,
            "finished_count":           int,
            "queue_depth":              int,
        },
        "sequences": [
            {
                "seq_id":               str,
                "ttft_ms":              float,
                "total_latency_ms":     float,
                "tokens_per_second":    float,
                "generated_tokens":     int,
                "queue_wait_time_ms":   float,
                "finish_reason":        str,
            },
            ...
        ],
        "summary": {
            "count":            int,
            "ttft_ms":          {"mean": float, "p50": ..., "p95": ..., "p99": ...},
            "total_latency_ms": {...},
            "tokens_per_second": {...},
            "queue_wait_time_ms": {...},
        }
    }
    """
    if _scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not ready")

    import numpy as np

    finished = list(_scheduler.finished)

    # Per-sequence stats (lightweight — no generated text)
    seq_stats = []
    for seq in finished:
        total_latency_ms = seq.ttft_ms + sum(seq.per_token_latencies_ms)
        n_gen = len(seq.generated_token_ids)
        tps = (n_gen / total_latency_ms * 1000.0) if total_latency_ms > 0 else 0.0
        seq_stats.append({
            "seq_id": seq.seq_id,
            "ttft_ms": seq.ttft_ms,
            "total_latency_ms": total_latency_ms,
            "tokens_per_second": tps,
            "generated_tokens": n_gen,
            "queue_wait_time_ms": seq.queue_wait_time_ms,
            "finish_reason": seq.finish_reason,
        })

    # Summary statistics
    def _pcts(values: list) -> dict:
        if not values:
            return {}
        arr = np.array(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "p50":  float(np.percentile(arr, 50)),
            "p95":  float(np.percentile(arr, 95)),
            "p99":  float(np.percentile(arr, 99)),
            "min":  float(np.min(arr)),
            "max":  float(np.max(arr)),
        }

    summary: dict = {"count": len(finished)}
    if finished:
        summary["ttft_ms"] = _pcts([s["ttft_ms"] for s in seq_stats])
        summary["total_latency_ms"] = _pcts([s["total_latency_ms"] for s in seq_stats])
        summary["tokens_per_second"] = _pcts([s["tokens_per_second"] for s in seq_stats])
        summary["queue_wait_time_ms"] = _pcts([s["queue_wait_time_ms"] for s in seq_stats])

    return JSONResponse(content={
        "scheduler": {
            "batch_size_over_time": _scheduler.batch_size_over_time,
            "scheduler_step_latency_ms": _scheduler.scheduler_step_latency_ms,
            "running_count": len(_scheduler.running),
            "finished_count": len(_scheduler.finished),
            "queue_depth": _scheduler.request_queue.stats()["queue_depth"],
        },
        "queue_stats": _scheduler.request_queue.stats(),
        "stage_breakdown": _scheduler.stage_tracker.full_report(),
        "kv_cache": _scheduler.kv_tracker.stats(),
        "paged_kv_cache": _scheduler.paged_kv_cache.stats(),
        "sequences": seq_stats,
        "summary": summary,
    })


@app.get("/health", response_class=JSONResponse)
async def endpoint_health():
    """Liveness check with current scheduler occupancy."""
    if _loaded_model is None or _scheduler is None or _config is None:
        return JSONResponse(status_code=503, content={"status": "loading"})

    return JSONResponse(content={
        "status": "ok",
        "model": _config.model_name,
        "device": _loaded_model.device,
        "max_batch_size": _config.max_batch_size,
        "current_batch_size": len(_scheduler.running),
        "queue_depth": _scheduler.request_queue.stats()["queue_depth"],
        "sequences_finished": len(_scheduler.finished),
    })
