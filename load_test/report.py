"""
load_test/report.py — Phase 11: Result aggregation and report generation.

Takes a list of RequestResult objects from the runner and computes:
- Success/failure counts
- Throughput (requests/sec, output tokens/sec, input tokens/sec)
- Latency percentiles (p50/p90/p95/p99) for TTFT, total latency (E2E),
  time-per-output-token (TPOT) and inter-token latency (ITL)
- Goodput: requests/sec that met the TTFT and TPOT SLOs
- Error breakdown

Also provides side-by-side comparison (compare_reports), pretty-printing
(print_report_table) and JSON serialisation (save_report_json).

Never raises on empty or all-failed result sets.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx

import numpy as np

if TYPE_CHECKING:
    from load_test.runner import RequestResult


# ── Build report ──────────────────────────────────────────────────────────────


def _pcts(values: list[float]) -> dict:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    arr = np.array(values, dtype=float)
    p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99])
    return {"p50": float(p50), "p90": float(p90), "p95": float(p95),
            "p99": float(p99), "mean": float(np.mean(arr))}


def build_report(
    results: list["RequestResult"],
    test_duration_s: float,
    server_label: str,
    slo_ttft_ms: float = 2000.0,
    slo_tpot_ms: float = 100.0,
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
    errors (message → count), plus itl/tpot percentiles and goodput.
    """

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

    prompt_tokens_total = sum(
        getattr(r, "prompt_tokens", None) or 0 for r in successful
    )
    cached_tokens_total = sum(
        getattr(r, "cached_prompt_tokens", None) or 0 for r in successful
    )

    # Latency percentiles — only from successful requests with non-None values
    ttft_values = [r.ttft_ms for r in successful if r.ttft_ms is not None]
    lat_values = [r.total_latency_ms for r in successful if r.total_latency_ms is not None]
    itl_values = [gap for r in successful for gap in (getattr(r, "itl_ms", None) or [])]
    tpot_values = [t for r in successful if (t := getattr(r, "tpot_ms", None)) is not None]

    # Goodput: requests that met both SLOs (TPOT only checked when defined).
    def _meets_slo(r) -> bool:
        tpot = getattr(r, "tpot_ms", None)
        return (r.ttft_ms is not None and r.ttft_ms <= slo_ttft_ms
                and (tpot is None or tpot <= slo_tpot_ms))

    n_good = sum(1 for r in successful if _meets_slo(r))

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
        "throughput_input_tokens_per_sec": (
            prompt_tokens_total / test_duration_s if test_duration_s > 0 else 0.0
        ),
        "total_output_tokens": tokens_total,
        "total_input_tokens": prompt_tokens_total,
        "cached_prompt_tokens": cached_tokens_total,
        "goodput": {
            "slo_ttft_ms": slo_ttft_ms,
            "slo_tpot_ms": slo_tpot_ms,
            "requests_meeting_slo": n_good,
            "pct": (100.0 * n_good / total) if total else 0.0,
            "requests_per_sec": n_good / test_duration_s if test_duration_s > 0 else 0.0,
        },
        "latency": {
            "ttft_ms": _pcts(ttft_values),
            "total_latency_ms": _pcts(lat_values),
            "tpot_ms": _pcts(tpot_values),
            "itl_ms": _pcts(itl_values),
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
    itl = lat.get("itl_ms", {})
    tpot = lat.get("tpot_ms", {})

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
        row("Throughput (output tok/s):", f"{report.get('throughput_tokens_per_sec', 0.0):.2f}"),
        row("Throughput (input tok/s):", f"{report.get('throughput_input_tokens_per_sec', 0.0):.2f}"),
        row("Goodput (req/s in SLO):",
            f"{report.get('goodput', {}).get('requests_per_sec', 0.0):.2f}"),
        sep,
        f"  {'Latency (ms)':<14}{'p50':>9}{'p90':>9}{'p95':>9}{'p99':>9}",
    ]
    for name, stats in (("TTFT", ttft), ("TPOT", tpot), ("ITL", itl), ("E2E", total_lat)):
        lines.append(
            f"  {name:<14}{stats.get('p50', 0.0):>9.1f}{stats.get('p90', 0.0):>9.1f}"
            f"{stats.get('p95', 0.0):>9.1f}{stats.get('p99', 0.0):>9.1f}"
        )

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


# ── Server-side metrics ──────────────────────────────────────────────────────


async def scrape_metrics(base_url: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            return (await client.get(f"{base_url}/metrics")).json()
    except Exception:
        return {}


def _counters(metrics: dict) -> dict:
    engine = metrics.get("engine", {})
    spec = metrics.get("speculative_decoding", {})
    steps = metrics.get("engine_steps", {})
    alloc = metrics.get("paged_kv_cache", {}).get("block_allocator", {})
    return {
        "prefix_query_tokens": alloc.get("prefix_cache_query_tokens", 0),
        "prefix_hit_tokens": alloc.get("prefix_cache_hit_tokens", 0),
        "preemptions_swap": engine.get("preemptions_swap", 0),
        "preemptions_recompute": engine.get("preemptions_recompute", 0),
        "spec_draft_tokens": spec.get("draft_tokens", 0),
        "spec_accepted_tokens": spec.get("accepted_tokens", 0),
        "steps": steps.get("total_steps", 0),
        **{f"sum_{k}": v for k, v in steps.get("cumulative", {}).items()},
    }


def engine_summary(metrics: dict, before: dict | None = None) -> dict:
    """Headline engine-side numbers from /metrics.

    With *before* (an earlier /metrics snapshot of the same server), every
    number covers only the interval between the two snapshots — e.g. one load
    level of a sweep — instead of the server's lifetime.
    """
    if "engine" not in metrics:
        return {}
    now = _counters(metrics)
    if before and "engine" in before:
        prev = _counters(before)
        now = {k: v - prev.get(k, 0) for k, v in now.items()}
    steps = now["steps"] or 0
    drafted = now["spec_draft_tokens"]
    queried = now["prefix_query_tokens"]
    return {
        "interval": "since previous snapshot" if before else "server lifetime",
        "prefix_cache_hit_rate": now["prefix_hit_tokens"] / queried if queried else None,
        "preemptions_swap": now["preemptions_swap"],
        "preemptions_recompute": now["preemptions_recompute"],
        "spec_acceptance_rate": now["spec_accepted_tokens"] / drafted if drafted else None,
        "engine_steps": steps,
        "avg_batch_size": now.get("sum_num_seqs", 0) / steps if steps else None,
        "avg_tokens_per_step": now.get("sum_num_tokens", 0) / steps if steps else None,
        "avg_kv_utilization": now.get("sum_kv_utilization", 0) / steps if steps else None,
        "gpu_memory": metrics.get("gpu_memory"),
    }
