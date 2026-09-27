"""
engine/metrics_aggregator.py — Phase 10: Unified metrics aggregation surface.

Pulls data from every existing tracker/manager (Phase 1–9) and computes
derived system-wide metrics that none of the individual trackers provide on
their own:

  * Latency percentiles (p50/p90/p95/p99) across finished requests: TTFT,
    end-to-end latency, time-per-output-token (TPOT), inter-token latency
    (ITL) and queue wait
  * System-wide token throughput and request throughput (rolling window)
  * SLO compliance fractions (TTFT and total latency)
  * Cumulative OOM, preemption, prefix-cache and speculative-decoding counters
  * Per-step engine telemetry (batch size, tokens per step, KV utilisation)

Design
------
* Does NOT replace any existing tracker — purely aggregates.
* No torch dependency — only numeric data from stats() / get_all() calls.
* Thread-safe via threading.Lock for all window mutations.
* full_report() is the single method the /metrics endpoint calls.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from inference_engine.engine.block_allocator import BlockAllocator
    from inference_engine.engine.cpu_swap_manager import CPUSwapManager
    from inference_engine.engine.kv_cache_tracker import KVCacheTracker
    from inference_engine.engine.paged_kv_cache import PagedKVCacheManager
    from inference_engine.engine.request_queue import RequestQueue
    from inference_engine.engine.stage_tracker import StageTracker
    from inference_engine.metrics.collector import MetricsCollector


def percentiles(values) -> dict:
    """p50/p90/p95/p99/mean of *values* (all zeros when empty)."""
    if len(values) == 0:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    arr = np.asarray(values, dtype=float)
    p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99])
    return {
        "p50": float(p50),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
        "mean": float(arr.mean()),
    }


# ── SystemSnapshot ────────────────────────────────────────────────────────────


@dataclass
class SystemSnapshot:
    """Immutable point-in-time snapshot of system-wide scheduler state.

    Fields
    ------
    timestamp
        ``time.perf_counter()`` at snapshot creation.
    wall_clock
        UTC ISO-8601 timestamp string.
    requests_in_flight
        Number of sequences currently in the running list.
    requests_waiting
        Depth of the request queue at snapshot time.
    requests_finished_total
        Cumulative count of sequences that reached state="finished".
    requests_oom_total
        Cumulative count of sequences killed with finish_reason="oom".
    requests_swapped_total
        Cumulative count of swap-out operations (from CPUSwapManager stats).
    throughput_tokens_per_sec
        Generated tokens per second over the most recent history window.
    throughput_requests_per_sec
        Completed requests per second over the most recent history window.
    """

    timestamp: float
    wall_clock: str
    requests_in_flight: int
    requests_waiting: int
    requests_finished_total: int
    requests_oom_total: int
    requests_swapped_total: int
    throughput_tokens_per_sec: float
    throughput_requests_per_sec: float


# ── MetricsAggregator ─────────────────────────────────────────────────────────


class MetricsAggregator:
    """Unified metrics surface aggregating all Phase 1–9 trackers.

    Parameters
    ----------
    metrics_collector:
        Phase 1 collector — provides per-request GenerationResult records
        via ``get_all()``.
    stage_tracker:
        Phase 4 tracker — provides prefill/decode stage breakdown.
    kv_tracker:
        Phase 5 tracker — provides KV memory accounting.
    block_allocator:
        Phase 6 allocator — provides block pool statistics.
    paged_kv_cache:
        Phase 7 manager — provides tensor pool statistics.
    cpu_swap_manager:
        Phase 9 manager — provides swap-out/in statistics.
    request_queue:
        Phase 3 queue — provides queue depth and throughput counters.
    history_window_seconds:
        Rolling window size for throughput computation (default 60 s).
    """

    def __init__(
        self,
        metrics_collector: "MetricsCollector",
        stage_tracker: "StageTracker",
        kv_tracker: "KVCacheTracker",
        block_allocator: "BlockAllocator",
        paged_kv_cache: "PagedKVCacheManager",
        cpu_swap_manager: "CPUSwapManager",
        request_queue: "RequestQueue",
        history_window_seconds: float = 60.0,
    ) -> None:
        self.metrics_collector = metrics_collector
        self.stage_tracker = stage_tracker
        self.kv_tracker = kv_tracker
        self.block_allocator = block_allocator
        self.paged_kv_cache = paged_kv_cache
        self.cpu_swap_manager = cpu_swap_manager
        self.request_queue = request_queue
        self.history_window_seconds = history_window_seconds

        # Rolling windows: deque of (timestamp, value) tuples, pruned by age on
        # every write *and* read so memory stays bounded even if nobody ever
        # polls /metrics.
        self._throughput_window: deque[tuple[float, int]] = deque()
        self._request_completion_window: deque[tuple[float]] = deque()
        # Throughput is averaged over min(window, uptime) — dividing by the
        # full window during the first minute would under-report.
        self._start_time: float = time.perf_counter()

        self._cumulative_finished: int = 0
        self._cumulative_oom: int = 0
        self._cumulative_tokens: int = 0
        self._finish_reasons: dict[str, int] = {}

        # Per-step engine telemetry (bounded)
        self._steps: deque[dict] = deque(maxlen=2000)
        self._total_steps: int = 0
        # Lifetime sums, so a client can difference two /metrics snapshots and
        # get exact averages for just the interval between them.
        self._step_sums: dict[str, float] = {}

        self._lock = threading.Lock()

    # ── Recording hooks (called by the scheduler) ─────────────────────────────

    def record_token_generated(self, count: int = 1) -> None:
        """Record that *count* tokens were generated in this decode step.

        Called by the scheduler once per decode step per active sequence.
        Thread-safe.
        """
        with self._lock:
            self._throughput_window.append((time.perf_counter(), count))
            self._cumulative_tokens += count
            self._prune_window(self._throughput_window)

    def record_step(self, **fields) -> None:
        """Record one scheduler step (batch composition, latency, KV usage)."""
        fields["timestamp"] = time.perf_counter()
        with self._lock:
            self._steps.append(fields)
            self._total_steps += 1
            for key in ("num_seqs", "num_tokens", "num_prefill_tokens", "num_decode_seqs",
                        "kv_utilization", "forward_ms"):
                self._step_sums[key] = self._step_sums.get(key, 0.0) + fields.get(key, 0)

    def record_request_finished(self, finish_reason: str) -> None:
        """Record that a sequence has just transitioned to state='finished'.

        Parameters
        ----------
        finish_reason:
            The finish_reason of the completed sequence (e.g. "eos", "length",
            "oom", "cancelled").
        """
        with self._lock:
            self._request_completion_window.append((time.perf_counter(),))
            self._prune_window(self._request_completion_window)
            self._cumulative_finished += 1
            self._finish_reasons[finish_reason] = self._finish_reasons.get(finish_reason, 0) + 1
            if finish_reason == "oom":
                self._cumulative_oom += 1

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _prune_window(self, window: deque) -> None:
        """Remove entries older than history_window_seconds from the front.

        Mutates *window* in place via popleft().  Must be called while holding
        self._lock (or on a local snapshot).
        """
        cutoff = time.perf_counter() - self.history_window_seconds
        while window and window[0][0] < cutoff:
            window.popleft()

    # ── Derived metrics ───────────────────────────────────────────────────────

    def compute_throughput(self) -> tuple[float, float]:
        """Compute rolling-window token and request throughput.

        The window length is ``min(history_window_seconds, uptime)``.

        Returns
        -------
        (tokens_per_sec, requests_per_sec)
            Both 0.0 when the windows are empty.
        """
        with self._lock:
            self._prune_window(self._throughput_window)
            self._prune_window(self._request_completion_window)

            if not self._throughput_window and not self._request_completion_window:
                return (0.0, 0.0)

            uptime = time.perf_counter() - self._start_time
            window = max(1e-3, min(self.history_window_seconds, uptime))
            total_tokens = sum(count for _, count in self._throughput_window)
            tokens_per_sec = total_tokens / window
            requests_per_sec = len(self._request_completion_window) / window

        return (tokens_per_sec, requests_per_sec)

    def compute_e2e_latency_percentiles(self) -> dict:
        """Compute latency percentiles over all retained request records.

        Returns
        -------
        dict with keys "ttft_ms", "total_latency_ms", "tpot_ms", "itl_ms",
        "queue_wait_ms", each containing p50/p90/p95/p99/mean (zeros when no
        data).  ITL pools every inter-token gap of every request.
        """
        records = self.metrics_collector.get_all()
        ttft = [r.ttft_ms for r in records]
        total = [r.total_latency_ms for r in records]
        tpot = [getattr(r, "tpot_ms", 0.0) for r in records
                if getattr(r, "generated_tokens", 0) > 1]
        queue = [getattr(r, "queue_wait_time_ms", 0.0) for r in records]
        itl = [gap for r in records for gap in getattr(r, "per_token_latencies_ms", [])]
        return {
            "ttft_ms": percentiles(ttft),
            "total_latency_ms": percentiles(total),
            "tpot_ms": percentiles(tpot),
            "itl_ms": percentiles(itl),
            "queue_wait_ms": percentiles(queue),
        }

    def step_summary(self) -> dict:
        """Aggregate the recent per-step telemetry."""
        with self._lock:
            steps = list(self._steps)
            total_steps = self._total_steps
            sums = dict(self._step_sums)
        if not steps:
            return {"total_steps": total_steps, "cumulative": sums}

        def mean(key: str) -> float:
            values = [s.get(key, 0) for s in steps]
            return float(np.mean(values)) if values else 0.0

        return {
            "total_steps": total_steps,
            "recent_steps": len(steps),
            "avg_batch_size": mean("num_seqs"),
            "avg_tokens_per_step": mean("num_tokens"),
            "avg_prefill_tokens_per_step": mean("num_prefill_tokens"),
            "avg_decode_seqs_per_step": mean("num_decode_seqs"),
            "avg_kv_utilization": mean("kv_utilization"),
            "step_latency_ms": percentiles([s.get("step_ms", 0.0) for s in steps]),
            "forward_latency_ms": percentiles([s.get("forward_ms", 0.0) for s in steps]),
            "cumulative": sums,
        }

    def compute_slo_compliance(
        self,
        ttft_slo_ms: float = 200.0,
        total_latency_slo_ms: float = 5000.0,
    ) -> dict:
        """Compute the fraction of requests meeting each SLO threshold.

        Parameters
        ----------
        ttft_slo_ms:
            Maximum allowed TTFT in milliseconds (default 200 ms).
        total_latency_slo_ms:
            Maximum allowed end-to-end latency in milliseconds (default 5 s).

        Returns
        -------
        dict with keys: ttft_slo_ms, ttft_compliance_pct,
        total_latency_slo_ms, total_latency_compliance_pct, sample_size.
        When no data is available, compliance is defined as 100.0% and
        sample_size is 0 (avoids division-by-zero errors).
        """
        records = self.metrics_collector.get_all()
        n = len(records)

        if n == 0:
            return {
                "ttft_slo_ms": ttft_slo_ms,
                "ttft_compliance_pct": 100.0,
                "total_latency_slo_ms": total_latency_slo_ms,
                "total_latency_compliance_pct": 100.0,
                "sample_size": 0,
            }

        ttft_ok = sum(1 for r in records if r.ttft_ms <= ttft_slo_ms)
        lat_ok = sum(1 for r in records if r.total_latency_ms <= total_latency_slo_ms)

        return {
            "ttft_slo_ms": ttft_slo_ms,
            "ttft_compliance_pct": 100.0 * ttft_ok / n,
            "total_latency_slo_ms": total_latency_slo_ms,
            "total_latency_compliance_pct": 100.0 * lat_ok / n,
            "sample_size": n,
        }

    # ── Snapshot & report ─────────────────────────────────────────────────────

    def snapshot(self, requests_in_flight: int, requests_waiting: int) -> SystemSnapshot:
        """Build a point-in-time SystemSnapshot.

        Parameters
        ----------
        requests_in_flight:
            ``len(scheduler.running)`` — passed in because MetricsAggregator
            does not hold a reference to the scheduler.
        requests_waiting:
            Depth of the request queue — typically ``len(scheduler.request_queue)``.
        """
        tokens_per_sec, requests_per_sec = self.compute_throughput()
        swap_stats = self.cpu_swap_manager.stats()

        with self._lock:
            finished_total = self._cumulative_finished
            oom_total = self._cumulative_oom

        return SystemSnapshot(
            timestamp=time.perf_counter(),
            wall_clock=datetime.now(timezone.utc).isoformat(),
            requests_in_flight=requests_in_flight,
            requests_waiting=requests_waiting,
            requests_finished_total=finished_total,
            requests_oom_total=oom_total,
            requests_swapped_total=swap_stats["total_swap_outs"],
            throughput_tokens_per_sec=tokens_per_sec,
            throughput_requests_per_sec=requests_per_sec,
        )

    def full_report(self, requests_in_flight: int, requests_waiting: int) -> dict:
        """Return the single unified metrics dict for the /metrics endpoint.

        This is the ONLY method the server layer needs to call.  All tracker
        stats and derived metrics are assembled here.

        Parameters
        ----------
        requests_in_flight:
            ``len(scheduler.running)``
        requests_waiting:
            ``len(scheduler.request_queue)``

        Returns
        -------
        dict with keys:
            system, e2e_latency, slo_compliance, stage_breakdown,
            kv_cache, paged_kv_cache, cpu_swap, queue_stats
        """
        with self._lock:
            finish_reasons = dict(self._finish_reasons)
            cumulative_tokens = self._cumulative_tokens
        return {
            "system": asdict(self.snapshot(requests_in_flight, requests_waiting)),
            "tokens_generated_total": cumulative_tokens,
            "finish_reasons": finish_reasons,
            "engine_steps": self.step_summary(),
            "e2e_latency": self.compute_e2e_latency_percentiles(),
            "slo_compliance": self.compute_slo_compliance(),
            "stage_breakdown": self.stage_tracker.full_report(),
            "kv_cache": self.kv_tracker.stats(),
            "paged_kv_cache": self.paged_kv_cache.stats(),
            "cpu_swap": self.cpu_swap_manager.stats(),
            "queue_stats": self.request_queue.stats(),
        }
