"""
load_test/report.py — Phase 11: Result aggregation and report generation.

Takes a list of RequestResult objects from the runner and computes:
- Success/failure counts
- Throughput (requests/sec and tokens/sec)
- Latency percentiles (p50/p95/p99) for TTFT and total latency
- Error breakdown

Also provides side-by-side comparison (compare_reports), pretty-printing
(print_report_table) and JSON serialisation (save_report_json).

Never raises on empty or all-failed result sets.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from load_test.runner import RequestResult


# ── Build report ──────────────────────────────────────────────────────────────


def build_report(
    results: list["RequestResult"],
    test_duration_s: float,
    server_label: str,
) -> dict:
    """Aggregate RequestResult list into a structured report dict.

    Parameters
    ----------
    results:
        All results returned by LoadTestRunner.run().
    test_duration_s:
        Wall-clock duration of the test run (perf_counter delta).
    server_label:
        Human-readable name for the server under test (e.g. "phase1").

    Returns
    -------
    dict with keys: server_label, total_requests, successful, failed,
    success_rate_pct, test_duration_s, throughput_requests_per_sec,
    throughput_tokens_per_sec, latency (ttft_ms + total_latency_ms p-tiles),
    errors (message → count).
    """
    _zero_pcts = {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}

    total = len(results)
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    n_ok = len(successful)
    n_fail = len(failed)

    success_rate = (100.0 * n_ok / total) if total > 0 else 0.0

    # Throughput
    tokens_total = sum(
        r.tokens_generated for r in successful if r.tokens_generated is not None
    )
    req_tps = n_ok / test_duration_s if test_duration_s > 0 else 0.0
    tok_tps = tokens_total / test_duration_s if test_duration_s > 0 else 0.0

    # Latency percentiles — only from successful requests with non-None values
    ttft_values = [r.ttft_ms for r in successful if r.ttft_ms is not None]
    lat_values = [r.total_latency_ms for r in successful if r.total_latency_ms is not None]

    def _pcts(values: list[float]) -> dict:
        if not values:
            return dict(_zero_pcts)
        arr = np.array(values, dtype=float)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(np.mean(arr)),
        }

    # Error breakdown
    errors: dict[str, int] = {}
    for r in failed:
        key = r.error or "unknown"
        errors[key] = errors.get(key, 0) + 1

    return {
        "server_label": server_label,
        "total_requests": total,
        "successful": n_ok,
        "failed": n_fail,
        "success_rate_pct": success_rate,
        "test_duration_s": test_duration_s,
        "throughput_requests_per_sec": req_tps,
        "throughput_tokens_per_sec": tok_tps,
        "latency": {
            "ttft_ms": _pcts(ttft_values),
            "total_latency_ms": _pcts(lat_values),
        },
        "errors": errors,
    }


# ── Compare two reports ───────────────────────────────────────────────────────


def compare_reports(report_a: dict, report_b: dict) -> dict:
    """Compute percentage deltas between two reports (B relative to A).

    A positive throughput_delta_pct means B was faster (better).
    A negative ttft_delta_pct means B had lower TTFT (better).

    Division-by-zero is guarded: if report_a's value is 0, the corresponding
    delta is None rather than raising ZeroDivisionError.

    Parameters
    ----------
    report_a:
        Baseline report (e.g. Phase 1 sequential server).
    report_b:
        Candidate report (e.g. Phase 2 continuous batching server).

    Returns
    -------
    dict with keys: label_a, label_b, throughput_delta_pct,
    ttft_p50_delta_pct, ttft_p99_delta_pct, success_rate_delta_pct.
    """

    def _delta(a_val, b_val) -> float | None:
        if a_val == 0 or a_val is None:
            return None
        return (b_val - a_val) / a_val * 100.0

    thr_a = report_a.get("throughput_tokens_per_sec", 0.0)
    thr_b = report_b.get("throughput_tokens_per_sec", 0.0)

    ttft_a_p50 = report_a.get("latency", {}).get("ttft_ms", {}).get("p50", 0.0)
    ttft_b_p50 = report_b.get("latency", {}).get("ttft_ms", {}).get("p50", 0.0)
    ttft_a_p99 = report_a.get("latency", {}).get("ttft_ms", {}).get("p99", 0.0)
    ttft_b_p99 = report_b.get("latency", {}).get("ttft_ms", {}).get("p99", 0.0)

    sr_a = report_a.get("success_rate_pct", 0.0)
    sr_b = report_b.get("success_rate_pct", 0.0)

    return {
        "label_a": report_a.get("server_label", "A"),
        "label_b": report_b.get("server_label", "B"),
        "throughput_delta_pct": _delta(thr_a, thr_b),
        "ttft_p50_delta_pct": _delta(ttft_a_p50, ttft_b_p50),
        "ttft_p99_delta_pct": _delta(ttft_a_p99, ttft_b_p99),
        "success_rate_delta_pct": _delta(sr_a, sr_b),
    }


# ── Pretty-print ──────────────────────────────────────────────────────────────


def print_report_table(report: dict) -> None:
    """Pretty-print a report as an aligned text table.

    Uses only f-strings — no external formatting libraries required.
    """
    w = 56  # total table width
    sep = "─" * w

    def row(label: str, value: str) -> str:
        return f"  {label:<38}{value:>14}"

    lat = report.get("latency", {})
    ttft = lat.get("ttft_ms", {})
    total_lat = lat.get("total_latency_ms", {})

    lines = [
        sep,
        f"  Load Test Report — {report.get('server_label', 'unknown')}",
        sep,
        row("Total requests:", str(report.get("total_requests", 0))),
        row("Successful:", str(report.get("successful", 0))),
        row("Failed:", str(report.get("failed", 0))),
        row("Success rate:", f"{report.get('success_rate_pct', 0.0):.1f}%"),
        row("Test duration:", f"{report.get('test_duration_s', 0.0):.2f}s"),
        sep,
        row("Throughput (req/s):", f"{report.get('throughput_requests_per_sec', 0.0):.2f}"),
        row("Throughput (tok/s):", f"{report.get('throughput_tokens_per_sec', 0.0):.2f}"),
        sep,
        "  TTFT (ms)",
        row("  p50:", f"{ttft.get('p50', 0.0):.1f}"),
        row("  p95:", f"{ttft.get('p95', 0.0):.1f}"),
        row("  p99:", f"{ttft.get('p99', 0.0):.1f}"),
        row("  mean:", f"{ttft.get('mean', 0.0):.1f}"),
        sep,
        "  Total Latency (ms)",
        row("  p50:", f"{total_lat.get('p50', 0.0):.1f}"),
        row("  p95:", f"{total_lat.get('p95', 0.0):.1f}"),
        row("  p99:", f"{total_lat.get('p99', 0.0):.1f}"),
        row("  mean:", f"{total_lat.get('mean', 0.0):.1f}"),
    ]

    errors = report.get("errors", {})
    if errors:
        lines.append(sep)
        lines.append("  Errors")
        for msg, count in errors.items():
            lines.append(row(f"  {msg[:36]}:", str(count)))

    lines.append(sep)
    print("\n".join(lines))


def print_comparison_table(delta: dict) -> None:
    """Pretty-print a comparison delta as an aligned text table."""
    w = 56
    sep = "─" * w

    def row(label: str, value: str) -> str:
        return f"  {label:<38}{value:>14}"

    def fmt_delta(val) -> str:
        if val is None:
            return "N/A"
        sign = "+" if val >= 0 else ""
        return f"{sign}{val:.1f}%"

    lines = [
        sep,
        f"  Comparison: {delta.get('label_a', 'A')} → {delta.get('label_b', 'B')}",
        sep,
        row("Throughput delta (tok/s):", fmt_delta(delta.get("throughput_delta_pct"))),
        row("TTFT p50 delta:", fmt_delta(delta.get("ttft_p50_delta_pct"))),
        row("TTFT p99 delta:", fmt_delta(delta.get("ttft_p99_delta_pct"))),
        row("Success rate delta:", fmt_delta(delta.get("success_rate_delta_pct"))),
        sep,
        "  (negative latency delta = improvement; positive throughput = improvement)",
        sep,
    ]
    print("\n".join(lines))


# ── JSON persistence ──────────────────────────────────────────────────────────


def save_report_json(report: dict, filepath: str) -> None:
    """Write *report* to *filepath* as pretty-printed JSON."""
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Report saved to {filepath}")
