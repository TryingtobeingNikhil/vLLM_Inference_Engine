"""
server/app_v2.py — FastAPI server for the continuous-batching engine.

Differences from Phase 1 (app.py)
----------------------------------
* No inference lock: every request becomes a Sequence in the
  ContinuousBatchingScheduler, whose background loop batches all of them
  into shared forward passes over a paged KV cache.
* Streaming (Server-Sent Events) so clients can measure real TTFT and
  inter-token latency.
* An OpenAI-compatible ``/v1/completions`` endpoint, so standard load
  generators — and this repo's own load tester — can benchmark PageServe and
  e.g. vLLM with identical client code.
* Client disconnects abort the request and free its KV blocks.
* Runs on port 8001 so Phase 1 (port 8000) and Phase 2 can run side-by-side.

Endpoints
---------
POST /generate        Native API (``"stream": true`` for SSE).
POST /v1/completions  OpenAI-style completions (streaming supported).
GET  /metrics         Engine telemetry (latency percentiles, KV cache,
                      prefix cache, speculative decoding, per-step stats).
GET  /health          Liveness check with current batch occupancy.

Configuration comes from environment variables (see config.py), e.g.::

    MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct MAX_BATCH_SIZE=64 \\
    SPECULATIVE_METHOD=ngram uvicorn inference_engine.server.app_v2:app --port 8001
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from inference_engine.config import Config
from inference_engine.engine.request_queue import QueueFullError
from inference_engine.engine.scheduler import ContinuousBatchingScheduler
from inference_engine.engine.sequence import SamplingParams, Sequence
from inference_engine.engine.sequential import get_memory_stats
from inference_engine.models.loader import LoadedModel, load_model, load_model_and_tokenizer
from inference_engine.server.detokenizer import IncrementalDetokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── Module-level singletons (populated during lifespan startup) ───────────────

_config: Optional[Config] = None
_loaded_model: Optional[LoadedModel] = None
_scheduler: Optional[ContinuousBatchingScheduler] = None

# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model(s), create scheduler, start background loop.
    Shutdown: stop scheduler gracefully.
    """
    global _config, _loaded_model, _scheduler

    _config = Config()
    logger.info(
        "Engine server starting: model=%s device=%s max_batch_size=%d",
        _config.model_name, _config.device, _config.max_batch_size,
    )

    loop = asyncio.get_running_loop()
    _loaded_model = await loop.run_in_executor(None, load_model_and_tokenizer, _config)
    draft_model = None
    if _config.speculative_method == "draft":
        draft_model = await loop.run_in_executor(
            None, load_model, _config.draft_model_name, _config.device, _config.dtype
        )
    logger.info("Model loaded successfully on device=%s", _loaded_model.device)

    _scheduler = ContinuousBatchingScheduler(
        model=_loaded_model.model,
        tokenizer=_loaded_model.tokenizer,
        config=_config,
        draft_model=draft_model,
    )
    _scheduler.start()
    logger.info("Scheduler started")

    yield  # ── server is running ────────────────────────────────────────────

    logger.info("Engine server shutting down …")
    await _scheduler.stop()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PageServe — Continuous Batching LLM Inference Server",
    description="Iteration-level continuous batching over a paged KV cache.",
    version="3.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt text")
    max_new_tokens: int = Field(default=50, ge=1, le=16384)
    temperature: float = Field(default=0.0, ge=0.0, description="0 = greedy")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=-1, description="-1 = disabled")
    ignore_eos: bool = Field(default=False, description="Always generate max_new_tokens")
    stream: bool = False


class CompletionRequest(BaseModel):
    """Subset of the OpenAI completions API (plus vLLM's ``ignore_eos``)."""

    prompt: str = Field(..., min_length=1)
    model: Optional[str] = None
    max_tokens: int = Field(default=16, ge=1, le=16384)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = -1
    ignore_eos: bool = False
    stream: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────


def _require_scheduler() -> ContinuousBatchingScheduler:
    if _scheduler is None or _loaded_model is None:
        raise HTTPException(status_code=503, detail="Scheduler not ready")
    return _scheduler


async def _submit(prompt: str, sampling: SamplingParams, stream: bool):
    scheduler = _require_scheduler()
    try:
        return await scheduler.add_request(prompt=prompt, sampling=sampling, stream=stream)
    except QueueFullError:
        raise HTTPException(status_code=503, detail="Server at capacity, retry later")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _abort_in_background(seq: Sequence) -> None:
    if _scheduler is not None and not seq.is_finished():
        asyncio.ensure_future(_scheduler.abort(seq.seq_id))


def _sequence_to_result_dict(seq: Sequence, device: str) -> dict:
    """Convert a finished Sequence into a GenerationResult-compatible dict."""
    generated_text = _scheduler.tokenizer.decode(  # type: ignore[union-attr]
        seq.generated_token_ids, skip_special_tokens=True
    )
    total_latency_ms = seq.e2e_latency_ms
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
        "tpot_ms": seq.tpot_ms,
        "total_latency_ms": total_latency_ms,
        "tokens_per_second": tps,
        "per_token_latencies_ms": seq.itl_ms,
        "queue_wait_time_ms": seq.queue_wait_time_ms,
        "cached_prompt_tokens": seq.num_cached_tokens,
        "num_preemptions": seq.num_preemptions,
        "spec_draft_tokens": seq.num_draft_tokens,
        "spec_accepted_tokens": seq.num_accepted_tokens,
        "gpu_memory_allocated_mb": allocated_mb,
        "gpu_memory_reserved_mb": reserved_mb,
        "finish_reason": seq.finish_reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _raise_for_failure(seq: Sequence) -> None:
    if seq.finish_reason == "oom":
        raise HTTPException(status_code=503, detail="Insufficient KV-cache capacity")
    if seq.finish_reason == "error":
        logger.error("Generation failed for seq_id=%s: %s", seq.seq_id, seq.error_message)
        raise HTTPException(status_code=500, detail="Generation failed")


def _sse(payload) -> str:
    return f"data: {payload if isinstance(payload, str) else json.dumps(payload)}\n\n"


async def _token_deltas(seq: Sequence, future: asyncio.Future) -> AsyncIterator[tuple[list, str]]:
    """Yield ``(new_token_ids, new_text)`` as the scheduler produces tokens."""
    detok = IncrementalDetokenizer(_scheduler.tokenizer)  # type: ignore[union-attr]
    assert seq.stream is not None
    while True:
        item = await seq.stream.get()
        if item is None:
            break
        yield item, detok.add(item)
    tail = detok.flush()
    if tail:
        yield [], tail


async def _await_result(seq: Sequence, future: asyncio.Future, http_request: Request) -> Sequence:
    """Wait for completion; abort the sequence if the client goes away.

    uvicorn does not cancel a non-streaming handler when its client
    disconnects, so poll for it — otherwise an abandoned request would keep
    its batch slot and KV blocks until it finished.
    """
    try:
        while True:
            done, _ = await asyncio.wait({future}, timeout=0.5)
            if done:
                return future.result()
            if await http_request.is_disconnected():
                _abort_in_background(seq)
                raise HTTPException(status_code=499, detail="Client disconnected")
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Request timed out in queue") from exc
    except asyncio.CancelledError:
        _abort_in_background(seq)
        raise


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/generate")
async def endpoint_generate(request: GenerateRequest, http_request: Request):
    """Submit *prompt* to the continuous batching scheduler."""
    sampling = SamplingParams(
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        ignore_eos=request.ignore_eos,
    )
    seq, future = await _submit(request.prompt, sampling, stream=request.stream)
    device = _loaded_model.device  # type: ignore[union-attr]

    if not request.stream:
        seq = await _await_result(seq, future, http_request)
        _raise_for_failure(seq)
        return JSONResponse(content=_sequence_to_result_dict(seq, device))

    async def event_stream():
        try:
            async for token_ids, text in _token_deltas(seq, future):
                yield _sse({"token_ids": token_ids, "text": text})
            if future.done() and not future.cancelled() and future.exception() is not None:
                yield _sse({"error": "timeout", "detail": str(future.exception())})
            elif seq.finish_reason in ("oom", "error"):
                yield _sse({"error": seq.finish_reason, "detail": seq.error_message})
            else:
                result = _sequence_to_result_dict(seq, device)
                result["done"] = True
                yield _sse(result)
            yield _sse("[DONE]")
        finally:
            _abort_in_background(seq)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/completions")
async def endpoint_completions(request: CompletionRequest, http_request: Request):
    """OpenAI-compatible text completion."""
    sampling = SamplingParams(
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        ignore_eos=request.ignore_eos,
    )
    seq, future = await _submit(request.prompt, sampling, stream=request.stream)
    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_name = request.model or _config.model_name  # type: ignore[union-attr]

    def chunk(text: str, finish_reason=None, usage=None) -> dict:
        body = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "text": text, "logprobs": None,
                         "finish_reason": finish_reason}],
        }
        if usage is not None:
            body["usage"] = usage
        return body

    def usage_of(s: Sequence) -> dict:
        prompt_tokens = len(s.prompt_token_ids)
        completion_tokens = len(s.generated_token_ids)
        return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens}

    def openai_finish_reason(s: Sequence) -> str:
        return "length" if s.finish_reason == "length" else "stop"

    if not request.stream:
        seq = await _await_result(seq, future, http_request)
        _raise_for_failure(seq)
        text = _scheduler.tokenizer.decode(seq.generated_token_ids, skip_special_tokens=True)  # type: ignore[union-attr]
        return JSONResponse(content=chunk(text, openai_finish_reason(seq), usage_of(seq)))

    async def event_stream():
        try:
            async for _, text in _token_deltas(seq, future):
                if text:
                    yield _sse(chunk(text))
            if seq.finish_reason in ("", "oom", "error"):
                yield _sse({"error": {"message": seq.error_message or "request failed"}})
            else:
                yield _sse(chunk("", openai_finish_reason(seq), usage_of(seq)))
            yield _sse("[DONE]")
        finally:
            _abort_in_background(seq)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/metrics", response_class=JSONResponse)
async def endpoint_metrics():
    """Unified engine metrics (see MetricsAggregator.full_report)."""
    return JSONResponse(content=_require_scheduler().get_metrics())


@app.get("/health", response_class=JSONResponse)
async def endpoint_health():
    """Liveness check with current scheduler occupancy."""
    if _loaded_model is None or _scheduler is None or _config is None:
        return JSONResponse(status_code=503, content={"status": "loading"})

    allocator = _scheduler.block_allocator
    return JSONResponse(content={
        "status": "ok",
        "model": _config.model_name,
        "device": _loaded_model.device,
        "max_batch_size": _config.max_batch_size,
        "max_model_len": _scheduler.max_model_len,
        "current_batch_size": len(_scheduler.running),
        "queue_depth": len(_scheduler.request_queue),
        "sequences_finished": _scheduler.total_finished,
        "kv_blocks_used": allocator.num_used_blocks(),
        "kv_blocks_total": allocator.num_blocks,
        "prefix_caching": _config.enable_prefix_caching,
        "speculative_method": _config.speculative_method or "off",
    })
