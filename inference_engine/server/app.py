"""
server/app.py — FastAPI server for sequential LLM inference.

Sequential constraint — enforced at two layers
----------------------------------------------
1. asyncio.Lock (inference_lock): makes the "one request at a time" rule
   explicit and visible at the application layer.  If a second request arrives
   while inference is running, it waits on the lock before entering the
   executor — it does NOT return a 503.

2. ThreadPoolExecutor(max_workers=1): even if the lock were somehow bypassed,
   the executor can only run one blocking call at a time.

Running inference in an executor means the event loop is never blocked, so
health-checks and metrics endpoints remain responsive during long generations.

Endpoints
---------
POST /generate      Run inference; returns GenerationResult as JSON.
GET  /metrics       Returns last N results + summary statistics.
GET  /health        Returns model name, device, and status.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from inference_engine.config import Config
from inference_engine.engine.sequential import GenerationResult, generate
from inference_engine.metrics.collector import MetricsCollector
from inference_engine.models.loader import LoadedModel, load_model_and_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Module-level singletons (populated during lifespan startup) ───────────────

_config: Optional[Config] = None
_loaded_model: Optional[LoadedModel] = None
_collector: Optional[MetricsCollector] = None

# The executor intentionally has max_workers=1 — this is the physical
# enforcement of sequential serving (independent of the asyncio lock).
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")

# Explicit sequential lock.  async with inference_lock serialises all
# /generate calls at the application layer.
inference_lock = asyncio.Lock()


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load model + tokenizer, initialise metrics collector.
    Shutdown: flush metrics to disk.
    """
    global _config, _loaded_model, _collector

    _config = Config()
    logger.info("Starting server: model=%s device=%s", _config.model_name, _config.device)

    # Model loading is blocking — run in executor so startup doesn't block
    # the event loop (though in practice uvicorn handles this fine at startup).
    loop = asyncio.get_event_loop()
    _loaded_model = await loop.run_in_executor(
        _executor, load_model_and_tokenizer, _config
    )
    logger.info("Model loaded successfully")

    _collector = MetricsCollector(history_size=_config.metrics_history_size)

    yield  # ── server is running ────────────────────────────────────────────

    # Shutdown: persist metrics
    logger.info("Shutting down — writing metrics to %s", _config.metrics_output_path)
    _collector.dump_to_json(_config.metrics_output_path)
    _executor.shutdown(wait=False)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sequential LLM Inference Server",
    description=(
        "Phase 1 baseline: intentionally naive sequential serving. "
        "One request at a time, no batching, no KV cache management."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt text")
    max_new_tokens: int = Field(
        default=50, ge=1, le=512, description="Maximum tokens to generate"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _result_to_dict(result: GenerationResult) -> dict:
    """Convert GenerationResult dataclass to a JSON-serialisable dict."""
    return dataclasses.asdict(result)


def _run_generate(prompt: str, max_new_tokens: int) -> GenerationResult:
    """
    Blocking wrapper that calls generate().

    This is the function submitted to the ThreadPoolExecutor so it runs
    on a worker thread, keeping the event loop free.
    """
    assert _loaded_model is not None, "Model not loaded"
    assert _config is not None, "Config not initialised"

    return generate(
        model=_loaded_model.model,
        tokenizer=_loaded_model.tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=_loaded_model.device,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/generate", response_class=JSONResponse)
async def endpoint_generate(request: GenerateRequest):
    """
    Run sequential inference on *prompt*.

    Requests queue behind the inference_lock — only one runs at a time.
    This is intentional: we are measuring the sequential baseline.
    """
    if _loaded_model is None or _collector is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    async with inference_lock:
        loop = asyncio.get_event_loop()
        try:
            result: GenerationResult = await loop.run_in_executor(
                _executor,
                _run_generate,
                request.prompt,
                request.max_new_tokens,
            )
        except Exception as exc:
            logger.exception("Inference error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    _collector.append(result)

    return JSONResponse(content=_result_to_dict(result))


@app.get("/metrics", response_class=JSONResponse)
async def endpoint_metrics():
    """
    Return all stored GenerationResult objects plus summary statistics.

    Summary includes p50 / p95 / p99 for TTFT and total latency.
    """
    if _collector is None:
        raise HTTPException(status_code=503, detail="Collector not ready")

    results = _collector.get_all()
    summary = _collector.compute_summary()

    return JSONResponse(
        content={
            "summary": summary,
            "results": [_result_to_dict(r) for r in results],
        }
    )


@app.get("/health", response_class=JSONResponse)
async def endpoint_health():
    """
    Lightweight health-check.  Returns 200 when the model is loaded.
    """
    if _loaded_model is None or _config is None:
        return JSONResponse(status_code=503, content={"status": "loading"})

    return JSONResponse(
        content={
            "status": "ok",
            "model": _config.model_name,
            "device": _loaded_model.device,
            "requests_served": len(_collector) if _collector else 0,
        }
    )
