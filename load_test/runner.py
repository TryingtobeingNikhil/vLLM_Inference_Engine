"""
load_test/runner.py — Async load test executor.

Fires LoadRequests at a server and records what a *client* experiences.

Measurement rules (these matter for honest numbers)
---------------------------------------------------
* Latency is measured on the client, from the request's **scheduled** send
  time.  If the client or server falls behind, that delay counts against
  the server — otherwise a slow server hides its own queueing
  ("coordinated omission").
* With ``stream=True`` the server streams tokens (SSE) and we timestamp every
  chunk: TTFT = first chunk, ITL = gaps between chunks, E2E = last chunk.
  Without streaming, TTFT falls back to the server-reported value.
* Two traffic models:
    - open loop  (``run``)            — requests arrive on a schedule
                                        (e.g. Poisson) regardless of how the
                                        server is doing: realistic traffic.
    - closed loop (``run_closed_loop``) — N virtual users, each sends its next
                                        request when the previous finishes:
                                        measures throughput at a concurrency.

Works against the native ``/generate`` API (both PageServe servers) or any
OpenAI-compatible ``/v1/completions`` server (PageServe, vLLM, …).
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import httpx

if TYPE_CHECKING:
    from load_test.profiles import LoadRequest


# ── RequestResult ─────────────────────────────────────────────────────────────


@dataclass
class RequestResult:
    """Outcome of a single load-test request (all times in ms).

    ``ttft_ms`` / ``total_latency_ms`` are client-measured when streaming.
    ``sent_at`` / ``completed_at`` are ``time.time()`` wall-clock stamps.
    """

    request_id: str
    success: bool
    status_code: int | None
    ttft_ms: float | None
    total_latency_ms: float | None
    tokens_generated: int | None
    error: str | None
    sent_at: float
    completed_at: float | None
    # Added with streaming support (defaults keep old constructors working)
    itl_ms: List[float] = field(default_factory=list)
    prompt_tokens: int | None = None
    server_ttft_ms: float | None = None
    queue_wait_ms: float | None = None
    cached_prompt_tokens: int | None = None

    @property
    def tpot_ms(self) -> float | None:
        """Mean time per output token after the first."""
        if (self.ttft_ms is None or self.total_latency_ms is None
                or not self.tokens_generated or self.tokens_generated < 2):
            return None
        return (self.total_latency_ms - self.ttft_ms) / (self.tokens_generated - 1)


def _parse_sse_line(line: str):
    if not line.startswith("data: "):
        return None
    payload = line[len("data: "):].strip()
    if payload == "[DONE]":
        return "[DONE]"
    return json.loads(payload)


# ── LoadTestRunner ────────────────────────────────────────────────────────────


class LoadTestRunner:
    """Async load test executor.

    Parameters
    ----------
    base_url:
        Root URL of the target server, e.g. ``"http://localhost:8001"``.
    max_concurrent:
        Cap on in-flight requests (open loop).  None = unbounded.
    request_timeout_s:
        Per-request HTTP timeout.
    api:
        ``"native"`` (``/generate``) or ``"openai"`` (``/v1/completions``).
    stream:
        Use streaming responses and measure TTFT / ITL on the client.
    ignore_eos:
        Ask the server to always generate ``max_new_tokens`` (fixed-length
        outputs make runs comparable).
    """

    def __init__(
        self,
        base_url: str,
        max_concurrent: Optional[int] = 50,
        request_timeout_s: float = 120.0,
        api: str = "native",
        stream: bool = False,
        ignore_eos: bool = False,
        model: Optional[str] = None,
    ) -> None:
        if max_concurrent is not None and max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than zero")
        if request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be greater than zero")
        if api not in ("native", "openai"):
            raise ValueError("api must be 'native' or 'openai'")

        self.base_url = base_url.rstrip("/")
        self.max_concurrent = max_concurrent
        self.request_timeout_s = request_timeout_s
        self.api = api
        self.stream = stream
        self.ignore_eos = ignore_eos
        self.model = model

        self._semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None
        self._client: httpx.AsyncClient | None = None

    # ── Request encoding ──────────────────────────────────────────────────────

    def _endpoint_and_body(self, load_req: "LoadRequest") -> tuple[str, dict]:
        if self.api == "openai":
            body = {
                "prompt": load_req.prompt,
                "max_tokens": load_req.max_new_tokens,
                "temperature": 0.0,
                "stream": self.stream,
                "ignore_eos": self.ignore_eos,
            }
            if self.model:
                body["model"] = self.model
            return f"{self.base_url}/v1/completions", body
        return f"{self.base_url}/generate", {
            "prompt": load_req.prompt,
            "max_new_tokens": load_req.max_new_tokens,
            "ignore_eos": self.ignore_eos,
            "stream": self.stream,
        }

    # ── Core request dispatch ─────────────────────────────────────────────────

    async def _send_one(self, load_req: "LoadRequest", t_start: Optional[float] = None) -> RequestResult:
        """Fire a single request.  Latency is measured from *t_start*
        (perf_counter; defaults to now).  Never raises."""
        t0 = time.perf_counter() if t_start is None else t_start
        sent_at = time.time() - (time.perf_counter() - t0)
        url, body = self._endpoint_and_body(load_req)
        assert self._client is not None, "Client not initialised"

        def failure(status, error) -> RequestResult:
            return RequestResult(
                request_id=load_req.request_id, success=False, status_code=status,
                ttft_ms=None, total_latency_ms=(time.perf_counter() - t0) * 1000.0,
                tokens_generated=None, error=error, sent_at=sent_at,
                completed_at=time.time(),
            )

        try:
            if not self.stream:
                response = await self._client.post(url, json=body, timeout=self.request_timeout_s)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if response.status_code != 200:
                    return failure(response.status_code, f"HTTP {response.status_code}")
                data = response.json()
                if self.api == "openai":
                    usage = data.get("usage", {})
                    tokens, prompt_tokens, server_ttft = (
                        usage.get("completion_tokens"), usage.get("prompt_tokens"), None)
                else:
                    tokens = data.get("generated_tokens")
                    prompt_tokens = data.get("prompt_tokens")
                    server_ttft = data.get("ttft_ms")
                return RequestResult(
                    request_id=load_req.request_id, success=True, status_code=200,
                    ttft_ms=server_ttft, total_latency_ms=elapsed_ms,
                    tokens_generated=tokens, error=None, sent_at=sent_at,
                    completed_at=time.time(), prompt_tokens=prompt_tokens,
                    server_ttft_ms=server_ttft,
                    queue_wait_ms=data.get("queue_wait_time_ms"),
                    cached_prompt_tokens=data.get("cached_prompt_tokens"),
                )

            chunk_times: List[float] = []
            final: dict = {}
            error: Optional[str] = None
            async with self._client.stream("POST", url, json=body,
                                           timeout=self.request_timeout_s) as response:
                if response.status_code != 200:
                    await response.aread()
                    return failure(response.status_code, f"HTTP {response.status_code}")
                async for line in response.aiter_lines():
                    event = _parse_sse_line(line)
                    if event is None or event == "[DONE]":
                        continue
                    now = time.perf_counter()
                    if "error" in event:
                        error = str(event["error"])
                        continue
                    if self.api == "openai":
                        choice = event.get("choices", [{}])[0]
                        if choice.get("text"):
                            chunk_times.append(now)
                        if "usage" in event:
                            final = event
                    elif event.get("done"):
                        final = event
                    elif event.get("token_ids"):
                        chunk_times.append(now)
            if error is not None or not chunk_times:
                return failure(200, error or "no tokens received")

            if self.api == "openai":
                usage = final.get("usage", {})
                tokens, prompt_tokens = usage.get("completion_tokens"), usage.get("prompt_tokens")
            else:
                tokens, prompt_tokens = final.get("generated_tokens"), final.get("prompt_tokens")
            return RequestResult(
                request_id=load_req.request_id, success=True, status_code=200,
                ttft_ms=(chunk_times[0] - t0) * 1000.0,
                total_latency_ms=(chunk_times[-1] - t0) * 1000.0,
                tokens_generated=tokens if tokens is not None else len(chunk_times),
                error=None, sent_at=sent_at, completed_at=time.time(),
                itl_ms=[(b - a) * 1000.0 for a, b in zip(chunk_times, chunk_times[1:])],
                prompt_tokens=prompt_tokens,
                server_ttft_ms=final.get("ttft_ms"),
                queue_wait_ms=final.get("queue_wait_time_ms"),
                cached_prompt_tokens=final.get("cached_prompt_tokens"),
            )
        except Exception as exc:
            return failure(None, f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__)

    async def _scheduled_send(self, load_req: "LoadRequest", test_start: float) -> RequestResult:
        """Wait until the scheduled send time, then dispatch the request."""
        target_time = test_start + load_req.scheduled_send_time
        now = time.perf_counter()
        if target_time > now:
            await asyncio.sleep(target_time - now)
        if self._semaphore is None:
            return await self._send_one(load_req, t_start=target_time)
        async with self._semaphore:
            return await self._send_one(load_req, t_start=target_time)

    def _make_client(self) -> httpx.AsyncClient:
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        return httpx.AsyncClient(limits=limits, timeout=self.request_timeout_s)

    # ── Main entry points ─────────────────────────────────────────────────────

    async def run(self, load_requests: list["LoadRequest"]) -> list[RequestResult]:
        """Open loop: fire every request at its scheduled time.

        Returns one result per load request, in input schedule order.
        """
        async with self._make_client() as client:
            self._client = client
            test_start = time.perf_counter()
            tasks = [self._scheduled_send(req, test_start) for req in load_requests]
            results: list[RequestResult] = await asyncio.gather(*tasks)
        self._client = None
        return list(results)

    async def run_closed_loop(
        self, load_requests: list["LoadRequest"], concurrency: int
    ) -> list[RequestResult]:
        """Closed loop: *concurrency* users send requests back-to-back.

        ``scheduled_send_time`` is ignored.  Results are in input order.
        """
        if concurrency <= 0:
            raise ValueError("concurrency must be greater than zero")
        results: list[Optional[RequestResult]] = [None] * len(load_requests)
        queue: asyncio.Queue = asyncio.Queue()
        for i, req in enumerate(load_requests):
            queue.put_nowait((i, req))

        async def user() -> None:
            while True:
                try:
                    i, req = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                results[i] = await self._send_one(req)

        async with self._make_client() as client:
            self._client = client
            await asyncio.gather(*[user() for _ in range(min(concurrency, len(load_requests)))])
        self._client = None
        return [r for r in results if r is not None]
