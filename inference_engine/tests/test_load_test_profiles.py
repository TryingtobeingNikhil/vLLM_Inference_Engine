"""
tests/test_load_test_profiles.py — Unit tests for Phase 11: load_test/profiles.py

7 synchronous pytest tests.  No async, no network, no model.
load_test/ lives at the project root; pytest is run from there so imports
resolve naturally.

Run with:
    cd /Users/nikhilmourya/Desktop/PageServe
    pytest inference_engine/tests/test_load_test_profiles.py -v
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on sys.path so load_test is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from load_test.profiles import (
    burst_load,
    constant_load,
    default_prompt_pool,
    ramp_load,
)


# ── Test 1: constant_load count and spacing ───────────────────────────────────


def test_constant_load_count_and_spacing() -> None:
    """constant_load must produce exactly num_requests items with correct gaps."""
    reqs = constant_load(num_requests=10, requests_per_second=2.0, prompts=["a"])

    assert len(reqs) == 10, f"Expected 10 requests, got {len(reqs)}"
    assert reqs[0].scheduled_send_time == pytest.approx(0.0, abs=1e-9), (
        f"First request must fire at t=0, got {reqs[0].scheduled_send_time}"
    )
    # At 2 rps, gap = 0.5 s → second request fires at t=0.5
    assert reqs[1].scheduled_send_time == pytest.approx(0.5, rel=1e-3), (
        f"Second request must fire at t=0.5, got {reqs[1].scheduled_send_time}"
    )


# ── Test 2: constant_load cycles through prompts ─────────────────────────────


def test_constant_load_cycles_prompts() -> None:
    """Prompts must be cycled with modulo when num_requests > len(prompts)."""
    reqs = constant_load(num_requests=5, requests_per_second=1.0, prompts=["a", "b"])

    assert reqs[0].prompt == "a"
    assert reqs[1].prompt == "b"
    assert reqs[2].prompt == "a", "Prompts must wrap around (index 2 → 'a')"
    assert reqs[3].prompt == "b"
    assert reqs[4].prompt == "a"


# ── Test 3: ramp_load density and ordering ───────────────────────────────────


def test_ramp_load_increasing_density() -> None:
    """ramp_load must produce non-decreasing timestamps within a sensible range."""
    reqs = ramp_load(
        num_requests=20,
        start_rps=1.0,
        end_rps=10.0,
        duration_seconds=20.0,
        prompts=["a"],
    )

    assert len(reqs) == 20, f"Expected 20 requests, got {len(reqs)}"

    # Timestamps must be non-decreasing (monotone schedule)
    times = [r.scheduled_send_time for r in reqs]
    for i in range(1, len(times)):
        assert times[i] >= times[i - 1], (
            f"Timestamps must be non-decreasing: "
            f"t[{i}]={times[i]:.4f} < t[{i-1}]={times[i-1]:.4f}"
        )

    # The last send time must be within a reasonable bound
    # (sanity check: should not exceed first + 25s for 20 requests ramping 1→10 rps)
    assert times[-1] < times[0] + 25.0, (
        f"Last timestamp {times[-1]:.2f} s too far from first {times[0]:.2f} s"
    )


# ── Test 4: burst_load groups share timestamp ─────────────────────────────────


def test_burst_load_groups_share_timestamp() -> None:
    """All requests within a burst must share the same scheduled_send_time."""
    reqs = burst_load(
        num_bursts=2,
        requests_per_burst=3,
        burst_interval_seconds=10.0,
        prompts=["a"],
    )

    assert len(reqs) == 6, f"Expected 6 requests (2 bursts × 3), got {len(reqs)}"

    # First burst — all at t=0.0
    assert reqs[0].scheduled_send_time == pytest.approx(0.0, abs=1e-9)
    assert reqs[1].scheduled_send_time == pytest.approx(0.0, abs=1e-9)
    assert reqs[2].scheduled_send_time == pytest.approx(0.0, abs=1e-9)

    # Second burst — all at t=10.0
    assert reqs[3].scheduled_send_time == pytest.approx(10.0, rel=1e-3)
    assert reqs[4].scheduled_send_time == pytest.approx(10.0, rel=1e-3)


# ── Test 5: default_prompt_pool returns 10 non-empty strings ─────────────────


def test_default_prompt_pool_returns_list() -> None:
    """default_prompt_pool must return exactly 10 non-empty strings."""
    prompts = default_prompt_pool()

    assert len(prompts) == 10, f"Expected 10 prompts, got {len(prompts)}"
    assert all(isinstance(p, str) and len(p) > 0 for p in prompts), (
        "All prompts must be non-empty strings"
    )


# ── Test 6: all request IDs are unique ───────────────────────────────────────


def test_load_request_has_unique_ids() -> None:
    """Every LoadRequest must have a globally unique request_id (uuid4 hex)."""
    reqs = constant_load(num_requests=20, requests_per_second=5.0, prompts=["a"])

    ids = [r.request_id for r in reqs]
    assert len(set(ids)) == 20, (
        f"Expected 20 unique IDs, got {len(set(ids))} unique out of {len(ids)}"
    )


# ── Test 7: burst_load total count ───────────────────────────────────────────


def test_burst_load_total_count() -> None:
    """burst_load must produce num_bursts × requests_per_burst requests."""
    reqs = burst_load(
        num_bursts=4,
        requests_per_burst=5,
        burst_interval_seconds=1.0,
        prompts=["a", "b"],
    )

    assert len(reqs) == 20, f"Expected 20 requests (4 × 5), got {len(reqs)}"
