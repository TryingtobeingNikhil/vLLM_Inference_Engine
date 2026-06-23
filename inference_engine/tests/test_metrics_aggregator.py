"""
tests/test_metrics_aggregator.py — Unit tests for Phase 10: MetricsAggregator.

9 synchronous pytest tests.  No async, no model loading.
All injected trackers are replaced by lightweight fakes/stubs.

Run with:
    cd /Users/nikhilmourya/Desktop/PageServe
    pytest inference_engine/tests/test_metrics_aggregator.py -v
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from inference_engine.engine.metrics_aggregator import MetricsAggregator


# ── Fake tracker stubs ────────────────────────────────────────────────────────


class FakeMetricsCollector:
    """Stub for metrics.collector.MetricsCollector."""

    def __init__(self, results=None):
        self._results = results or []

    def get_all(self):
        return self._results


class FakeStageTracker:
    """Stub for engine.stage_tracker.StageTracker."""

    def full_report(self):
        return {"prefill": {}, "decode": {}}


class FakeKVTracker:
    """Stub for engine.kv_cache_tracker.KVCacheTracker."""

    def stats(self):
        return {"active_sequences": 0}


class FakeBlockAllocator:
    """Stub for engine.block_allocator.BlockAllocator."""

    def stats(self):
        return {}


class FakePagedKVCache:
    """Stub for engine.paged_kv_cache.PagedKVCacheManager."""

    def stats(self):
        return {"pool_size_mb": 10.0}


class FakeCPUSwapManager:
    """Stub for engine.cpu_swap_manager.CPUSwapManager."""

    def stats(self):
        return {"total_swap_outs": 0}


class FakeRequestQueue:
    """Stub for engine.request_queue.RequestQueue."""

    def stats(self):
        return {"queue_depth": 0}

    def __len__(self):
        return 0


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def aggregator() -> MetricsAggregator:
    """Fresh MetricsAggregator backed entirely by stubs — no real trackers."""
    return MetricsAggregator(
        metrics_collector=FakeMetricsCollector(),
        stage_tracker=FakeStageTracker(),
        kv_tracker=FakeKVTracker(),
        block_allocator=FakeBlockAllocator(),
        paged_kv_cache=FakePagedKVCache(),
        cpu_swap_manager=FakeCPUSwapManager(),
        request_queue=FakeRequestQueue(),
        history_window_seconds=60.0,
    )


# ── Test 1: record_token_generated appends to window ─────────────────────────


def test_record_token_generated_appends_window(aggregator: MetricsAggregator) -> None:
    """record_token_generated must append (timestamp, count) to _throughput_window."""
    aggregator.record_token_generated(5)

    assert len(aggregator._throughput_window) == 1, (
        "Expected exactly 1 entry in _throughput_window after one call"
    )
    ts, count = aggregator._throughput_window[0]
    assert count == 5, f"Expected count=5, got {count}"
    assert isinstance(ts, float), "Timestamp must be a float"


# ── Test 2: record_request_finished increments counters correctly ─────────────


def test_record_request_finished_increments_counters(aggregator: MetricsAggregator) -> None:
    """record_request_finished must correctly track finished/oom counters."""
    aggregator.record_request_finished("length")
    assert aggregator._cumulative_finished == 1, "Cumulative finished should be 1"
    assert aggregator._cumulative_oom == 0, "No OOM yet — oom counter should be 0"

    aggregator.record_request_finished("oom")
    assert aggregator._cumulative_finished == 2, "Cumulative finished should be 2"
    assert aggregator._cumulative_oom == 1, "One OOM recorded — oom counter should be 1"


# ── Test 3: compute_throughput returns zeros when windows are empty ────────────


def test_compute_throughput_empty(aggregator: MetricsAggregator) -> None:
    """compute_throughput must return (0.0, 0.0) with no recorded events."""
    tokens_per_sec, requests_per_sec = aggregator.compute_throughput()
    assert tokens_per_sec == 0.0, f"Expected 0.0 tokens/sec, got {tokens_per_sec}"
    assert requests_per_sec == 0.0, f"Expected 0.0 requests/sec, got {requests_per_sec}"


# ── Test 4: compute_throughput returns non-zero with recorded tokens ───────────


def test_compute_throughput_nonzero(aggregator: MetricsAggregator) -> None:
    """tokens_per_sec must equal total_tokens / history_window_seconds."""
    # Record 10 tokens three times = 30 tokens total
    aggregator.record_token_generated(10)
    aggregator.record_token_generated(10)
    aggregator.record_token_generated(10)

    tokens_per_sec, _ = aggregator.compute_throughput()

    # 30 tokens / 60 s = 0.5 tokens/sec
    assert tokens_per_sec == pytest.approx(30.0 / 60.0, rel=1e-2), (
        f"Expected ~{30.0/60.0:.4f} tokens/sec, got {tokens_per_sec}"
    )


# ── Test 5: _prune_window removes old entries ──────────────────────────────────


def test_prune_window_removes_old_entries(aggregator: MetricsAggregator) -> None:
    """Entries older than history_window_seconds must be pruned by _prune_window."""
    # Insert a very old entry (120 s ago — older than the 60 s window)
    old_ts = time.perf_counter() - 120.0
    aggregator._throughput_window.append((old_ts, 99))

    assert len(aggregator._throughput_window) == 1, "Should have 1 entry before pruning"

    aggregator._prune_window(aggregator._throughput_window)

    assert len(aggregator._throughput_window) == 0, (
        "Old entry must be removed after _prune_window — got "
        f"{len(aggregator._throughput_window)} entries remaining"
    )


# ── Test 6: compute_e2e_latency_percentiles returns zeros with no data ─────────


def test_e2e_latency_percentiles_empty(aggregator: MetricsAggregator) -> None:
    """compute_e2e_latency_percentiles must return zeroed values when no records exist."""
    result = aggregator.compute_e2e_latency_percentiles()

    assert result["ttft_ms"]["p50"] == 0.0, (
        f"Expected p50=0.0 when no data, got {result['ttft_ms']['p50']}"
    )
    assert result["total_latency_ms"]["p99"] == 0.0, (
        f"Expected p99=0.0 when no data, got {result['total_latency_ms']['p99']}"
    )


# ── Test 7: compute_e2e_latency_percentiles with real data ────────────────────


def test_e2e_latency_percentiles_with_data(aggregator: MetricsAggregator) -> None:
    """Percentiles must be computed correctly from three fake result records."""
    results = [
        SimpleNamespace(ttft_ms=10.0, total_latency_ms=100.0),
        SimpleNamespace(ttft_ms=20.0, total_latency_ms=200.0),
        SimpleNamespace(ttft_ms=30.0, total_latency_ms=300.0),
    ]
    aggregator.metrics_collector = FakeMetricsCollector(results)

    result = aggregator.compute_e2e_latency_percentiles()

    # Median of [10, 20, 30] is 20
    assert result["ttft_ms"]["p50"] == pytest.approx(20.0, rel=1e-2), (
        f"Expected p50≈20.0 for ttft_ms, got {result['ttft_ms']['p50']}"
    )
    # Median of [100, 200, 300] is 200
    assert result["total_latency_ms"]["p50"] == pytest.approx(200.0, rel=1e-2), (
        f"Expected p50≈200.0 for total_latency_ms, got {result['total_latency_ms']['p50']}"
    )


# ── Test 8: compute_slo_compliance with no data returns 100% ──────────────────


def test_slo_compliance_no_data(aggregator: MetricsAggregator) -> None:
    """compute_slo_compliance must return 100% compliance and sample_size=0 when empty."""
    result = aggregator.compute_slo_compliance()

    assert result["sample_size"] == 0, (
        f"Expected sample_size=0 with no data, got {result['sample_size']}"
    )
    assert result["ttft_compliance_pct"] == 100.0, (
        f"Expected 100.0% TTFT compliance with no data, got {result['ttft_compliance_pct']}"
    )
    assert result["total_latency_compliance_pct"] == 100.0, (
        f"Expected 100.0% total latency compliance with no data, "
        f"got {result['total_latency_compliance_pct']}"
    )


# ── Test 9: full_report returns all required top-level keys ───────────────────


def test_full_report_structure(aggregator: MetricsAggregator) -> None:
    """full_report must return a dict with all 8 required top-level keys."""
    result = aggregator.full_report(requests_in_flight=2, requests_waiting=1)

    required_keys = {
        "system", "e2e_latency", "slo_compliance", "stage_breakdown",
        "kv_cache", "paged_kv_cache", "cpu_swap", "queue_stats",
    }
    missing = required_keys - result.keys()
    assert not missing, f"full_report missing keys: {missing}"

    # Verify the system snapshot is correctly populated
    assert result["system"]["requests_in_flight"] == 2, (
        f"Expected requests_in_flight=2, got {result['system']['requests_in_flight']}"
    )
    assert result["system"]["requests_waiting"] == 1, (
        f"Expected requests_waiting=1, got {result['system']['requests_waiting']}"
    )
    assert isinstance(result["system"]["wall_clock"], str), (
        "wall_clock must be an ISO-format string"
    )
    assert isinstance(result["system"]["throughput_tokens_per_sec"], float), (
        "throughput_tokens_per_sec must be a float"
    )
