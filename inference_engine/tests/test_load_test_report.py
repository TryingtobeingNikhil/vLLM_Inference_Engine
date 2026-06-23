"""
tests/test_load_test_report.py — Unit tests for Phase 11: load_test/report.py

6 synchronous pytest tests.  No async, no network.
Constructs fake RequestResult objects directly to avoid any model or server
dependency.

Run with:
    cd /Users/nikhilmourya/Desktop/PageServe
    pytest inference_engine/tests/test_load_test_report.py -v
"""

from __future__ import annotations

import json
import os
import sys
import time

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from load_test.report import build_report, compare_reports, save_report_json
from load_test.runner import RequestResult


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_result(
    *,
    success: bool = True,
    ttft_ms: float | None = 50.0,
    total_latency_ms: float | None = 200.0,
    tokens_generated: int | None = 20,
    error: str | None = None,
    status_code: int | None = 200,
) -> RequestResult:
    """Build a minimal RequestResult for testing without a live server."""
    return RequestResult(
        request_id="test-id",
        success=success,
        status_code=status_code,
        ttft_ms=ttft_ms,
        total_latency_ms=total_latency_ms,
        tokens_generated=tokens_generated,
        error=error,
        sent_at=time.time(),
        completed_at=time.time(),
    )


# ── Test 1: build_report — all success ───────────────────────────────────────


def test_build_report_all_success() -> None:
    """All-success report must have correct counts and token throughput."""
    results = [
        _make_result(ttft_ms=30.0, total_latency_ms=100.0, tokens_generated=20),
        _make_result(ttft_ms=40.0, total_latency_ms=150.0, tokens_generated=20),
        _make_result(ttft_ms=50.0, total_latency_ms=200.0, tokens_generated=20),
    ]

    report = build_report(results, test_duration_s=10.0, server_label="test")

    assert report["successful"] == 3, f"Expected 3 successes, got {report['successful']}"
    assert report["failed"] == 0, f"Expected 0 failures, got {report['failed']}"
    assert report["success_rate_pct"] == pytest.approx(100.0, abs=1e-6)
    # 3 × 20 tokens / 10 s = 6.0 tok/s
    assert report["throughput_tokens_per_sec"] == pytest.approx(60.0 / 10.0, rel=1e-3), (
        f"Expected 6.0 tok/s, got {report['throughput_tokens_per_sec']}"
    )


# ── Test 2: build_report — with failures ─────────────────────────────────────


def test_build_report_with_failures() -> None:
    """Failed requests must be counted and their error messages tallied."""
    results = [
        _make_result(success=True,  ttft_ms=30.0, total_latency_ms=100.0, tokens_generated=10),
        _make_result(success=True,  ttft_ms=40.0, total_latency_ms=120.0, tokens_generated=10),
        _make_result(success=False, ttft_ms=None, total_latency_ms=None,
                     tokens_generated=None, error="timeout", status_code=None),
    ]

    report = build_report(results, test_duration_s=10.0, server_label="test")

    assert report["failed"] == 1, f"Expected 1 failure, got {report['failed']}"
    assert report["errors"].get("timeout") == 1, (
        f"Expected errors['timeout']=1, got {report['errors']}"
    )
    assert report["success_rate_pct"] == pytest.approx(66.67, rel=1e-2), (
        f"Expected ~66.67%, got {report['success_rate_pct']}"
    )


# ── Test 3: build_report — zero results ──────────────────────────────────────


def test_build_report_zero_results() -> None:
    """Empty result list must produce a zeroed report without raising."""
    report = build_report([], test_duration_s=10.0, server_label="test")

    assert report["total_requests"] == 0
    assert report["latency"]["ttft_ms"]["p50"] == 0.0, (
        f"Expected p50=0.0 for empty results, got {report['latency']['ttft_ms']['p50']}"
    )
    # Verify no exception was raised (implicitly — test passes if we reach here)


# ── Test 4: compare_reports — throughput delta ────────────────────────────────


def test_compare_reports_throughput_delta() -> None:
    """compare_reports must compute correct percentage deltas."""
    report_a = {
        "server_label": "a",
        "throughput_tokens_per_sec": 100.0,
        "latency": {"ttft_ms": {"p50": 50.0, "p99": 100.0}},
        "success_rate_pct": 95.0,
    }
    report_b = {
        "server_label": "b",
        "throughput_tokens_per_sec": 150.0,
        "latency": {"ttft_ms": {"p50": 40.0, "p99": 80.0}},
        "success_rate_pct": 98.0,
    }

    delta = compare_reports(report_a, report_b)

    # (150 - 100) / 100 * 100 = +50%
    assert delta["throughput_delta_pct"] == pytest.approx(50.0, rel=1e-2), (
        f"Expected throughput_delta_pct=50.0, got {delta['throughput_delta_pct']}"
    )
    # (40 - 50) / 50 * 100 = -20% (lower TTFT = improvement)
    assert delta["ttft_p50_delta_pct"] == pytest.approx(-20.0, rel=1e-2), (
        f"Expected ttft_p50_delta_pct=-20.0, got {delta['ttft_p50_delta_pct']}"
    )


# ── Test 5: compare_reports — zero-division guard ─────────────────────────────


def test_compare_reports_zero_division_guard() -> None:
    """compare_reports must return None for deltas where report_a value is 0."""
    report_a = {
        "server_label": "a",
        "throughput_tokens_per_sec": 0.0,   # zero → division by zero potential
        "latency": {"ttft_ms": {"p50": 0.0, "p99": 0.0}},
        "success_rate_pct": 0.0,
    }
    report_b = {
        "server_label": "b",
        "throughput_tokens_per_sec": 150.0,
        "latency": {"ttft_ms": {"p50": 40.0, "p99": 80.0}},
        "success_rate_pct": 98.0,
    }

    delta = compare_reports(report_a, report_b)

    assert delta["throughput_delta_pct"] is None, (
        f"Expected None for zero-denominator throughput delta, "
        f"got {delta['throughput_delta_pct']}"
    )


# ── Test 6: save_report_json round-trip ───────────────────────────────────────


def test_save_report_json(tmp_path) -> None:
    """save_report_json must write valid JSON that round-trips correctly."""
    payload = {"key": "value", "number": 42}
    out_path = str(tmp_path / "out.json")

    save_report_json(payload, out_path)

    with open(out_path, encoding="utf-8") as fh:
        loaded = json.load(fh)

    assert loaded == payload, f"Round-trip mismatch: {loaded} != {payload}"
