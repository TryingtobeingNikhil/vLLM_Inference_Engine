"""
load_test/runner.py — Phase 11: Async load test executor.

Fires a prepared list of LoadRequest objects against any PageServe server
according to their scheduled_send_time, collecting per-request latency and
status into RequestResult records.

Uses httpx.AsyncClient (already in requirements.txt) for non-blocking HTTP.
Concurrency is bounded by an asyncio.Semaphore to avoid overwhelming the
OS with too many open connections.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from load_test.profiles import LoadRequest


# ── RequestResult ─────────────────────────────────────────────────────────────


@dataclass
class RequestResult:
    """Outcome of a single load-test request.

    Fields
    ------
    request_id
        Matches the LoadRequest.request_id that produced this result.
    success
        True when the server returned HTTP 200.
    status_code
        HTTP status code; None if a network-level error occurred.
    ttft_ms
        Time-to-first-token in milliseconds, as reported by the server in
        the JSON body.  None on failure or if the server omits this field.
    total_latency_ms
        End-to-end latency in milliseconds.  Falls back to the client-side
        elapsed time when the server does not return it.
    tokens_generated
        Number of tokens generated, from the server response body.  None
        on failure.
    error
        Human-readable error string; None on success.
    sent_at
        ``time.time()`` wall-clock timestamp when the request was dispatched.
    completed_at
        ``time.time()`` when the response (or error) was received; None
        on network-level exception.
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


# ── LoadTestRunner ────────────────────────────────────────────────────────────


class LoadTestRunner:
    """Async load test executor.

    Parameters
    ----------
    base_url:
        Root URL of the target server, e.g. ``"http://localhost:8001"``.
    max_concurrent:
        Maximum number of in-flight requests at any given moment.
        Enforced via asyncio.Semaphore.
    request_timeout_s:
        Per-request HTTP timeout in seconds.  Requests that exceed this
        are recorded as failures with the exception message as error.
    """

    def __init__(
        self,
        base_url: str,
        max_concurrent: int = 50,
        request_timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_concurrent = max_concurrent
        self.request_timeout_s = request_timeout_s

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client: httpx.AsyncClient | None = None

    # ── Core request dispatch ─────────────────────────────────────────────────

    async def _send_one(self, load_req: "LoadRequest") -> RequestResult:
        """Fire a single request and return its result.

        The semaphore limits how many of these run concurrently.
        All exceptions are caught and converted to failure RequestResults —
        the runner never propagates network errors to the caller.
        """
        async with self._semaphore:
            sent_at = time.time()
            t0 = time.perf_counter()
            try:
                assert self._client is not None, "Client not initialised"
                response = await self._client.post(
                    f"{self.base_url}/generate",
                    json={
                        "prompt": load_req.prompt,
                        "max_new_tokens": load_req.max_new_tokens,
                    },
                    timeout=self.request_timeout_s,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                if response.status_code == 200:
                    body = response.json()
                    return RequestResult(
                        request_id=load_req.request_id,
                        success=True,
                        status_code=200,
                        ttft_ms=body.get("ttft_ms"),
                        total_latency_ms=body.get("total_latency_ms", elapsed_ms),
                        tokens_generated=body.get("generated_tokens"),
                        error=None,
                        sent_at=sent_at,
                        completed_at=time.time(),
                    )
                else:
                    return RequestResult(
                        request_id=load_req.request_id,
                        success=False,
                        status_code=response.status_code,
                        ttft_ms=None,
                        total_latency_ms=elapsed_ms,
                        tokens_generated=None,
                        error=f"HTTP {response.status_code}",
                        sent_at=sent_at,
                        completed_at=time.time(),
                    )

            except Exception as exc:
                return RequestResult(
                    request_id=load_req.request_id,
                    success=False,
                    status_code=None,
                    ttft_ms=None,
                    total_latency_ms=None,
                    tokens_generated=None,
                    error=str(exc),
                    sent_at=sent_at,
                    completed_at=time.time(),
                )

    async def _scheduled_send(
        self, load_req: "LoadRequest", test_start: float
    ) -> RequestResult:
        """Wait until the scheduled send time, then dispatch the request."""
        target_time = test_start + load_req.scheduled_send_time
        now = time.perf_counter()
        if target_time > now:
            await asyncio.sleep(target_time - now)
        return await self._send_one(load_req)

    # ── Main entry point ──────────────────────────────────────────────────────

    async def run(self, load_requests: list["LoadRequest"]) -> list[RequestResult]:
        """Execute the full list of load requests according to their schedules.

        Creates the HTTP client, fires all requests concurrently (respecting
        the semaphore and each request's scheduled_send_time), then closes
        the client before returning.

        Parameters
        ----------
        load_requests:
            Prepared list from one of the profile generators.

        Returns
        -------
        list[RequestResult]
            One result per load request, in the order they completed.
        """
        async with httpx.AsyncClient() as client:
            self._client = client
            test_start = time.perf_counter()

            tasks = [
                self._scheduled_send(req, test_start) for req in load_requests
            ]
            results: list[RequestResult] = await asyncio.gather(*tasks)

        self._client = None
        return list(results)
