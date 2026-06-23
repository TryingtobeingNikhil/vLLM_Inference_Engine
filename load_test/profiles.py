"""
load_test/profiles.py — Phase 11: Load pattern generators.

Pure functions — no I/O, no network, deterministic for a given seed
(except request_id which uses uuid4, acceptable randomness).

Three profiles:
  constant_load  — even spacing at a fixed requests-per-second rate
  ramp_load      — linearly increasing rate from start_rps to end_rps
  burst_load     — groups of simultaneous requests at fixed intervals
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass


# ── LoadRequest ───────────────────────────────────────────────────────────────


@dataclass
class LoadRequest:
    """A single request to be fired at a specific time during the load test.

    Fields
    ------
    request_id
        UUID4 hex string — globally unique across the test run.
    prompt
        Text prompt to send to the /generate endpoint.
    max_new_tokens
        Upper bound on tokens to generate.
    scheduled_send_time
        Seconds from test start when this request should be dispatched.
    """

    request_id: str
    prompt: str
    max_new_tokens: int
    scheduled_send_time: float


# ── Profile generators ────────────────────────────────────────────────────────


def constant_load(
    num_requests: int,
    requests_per_second: float,
    prompts: list[str],
    max_new_tokens: int = 50,
) -> list[LoadRequest]:
    """Generate evenly-spaced requests at a fixed rate.

    Parameters
    ----------
    num_requests:
        Total number of requests to create.
    requests_per_second:
        Dispatch rate.  Inter-request gap = 1 / requests_per_second.
    prompts:
        Pool of prompts to cycle through.  If num_requests > len(prompts),
        prompts are repeated with modulo indexing.
    max_new_tokens:
        Max tokens to generate per request.

    Returns
    -------
    list[LoadRequest] sorted by scheduled_send_time.
    """
    requests: list[LoadRequest] = []
    n_prompts = len(prompts)
    for i in range(num_requests):
        requests.append(
            LoadRequest(
                request_id=uuid.uuid4().hex,
                prompt=prompts[i % n_prompts],
                max_new_tokens=max_new_tokens,
                scheduled_send_time=i / requests_per_second,
            )
        )
    return requests


def ramp_load(
    num_requests: int,
    start_rps: float,
    end_rps: float,
    duration_seconds: float,
    prompts: list[str],
    max_new_tokens: int = 50,
) -> list[LoadRequest]:
    """Generate requests with a linearly-ramping dispatch rate.

    The request rate interpolates from start_rps to end_rps over
    duration_seconds.  Higher rate → denser request spacing toward the end.

    Algorithm
    ---------
    For each request i (0-indexed):
      fraction = i / num_requests
      avg_rps_so_far = start_rps + (end_rps - start_rps) * fraction / 2
      scheduled_send_time = i / max(avg_rps_so_far, 0.01)

    This piecewise approximation is accurate enough for benchmarking purposes.

    Parameters
    ----------
    num_requests:
        Total number of requests.
    start_rps:
        Initial request rate (requests per second).
    end_rps:
        Final request rate (requests per second).
    duration_seconds:
        Target total test duration (used for documentation; the schedule
        emerges from the ramp formula, not clamped to this value).
    prompts:
        Pool of prompts; cycled with modulo.
    max_new_tokens:
        Max tokens per request.

    Returns
    -------
    list[LoadRequest] in non-decreasing scheduled_send_time order.
    """
    requests: list[LoadRequest] = []
    n_prompts = len(prompts)
    for i in range(num_requests):
        fraction = i / max(num_requests, 1)
        avg_rps_so_far = start_rps + (end_rps - start_rps) * fraction / 2
        scheduled_send_time = i / max(avg_rps_so_far, 0.01)
        requests.append(
            LoadRequest(
                request_id=uuid.uuid4().hex,
                prompt=prompts[i % n_prompts],
                max_new_tokens=max_new_tokens,
                scheduled_send_time=scheduled_send_time,
            )
        )
    return requests


def burst_load(
    num_bursts: int,
    requests_per_burst: int,
    burst_interval_seconds: float,
    prompts: list[str],
    max_new_tokens: int = 50,
) -> list[LoadRequest]:
    """Generate bursts of simultaneous requests at fixed intervals.

    All requests within a burst share the same scheduled_send_time so they
    are fired concurrently.  Bursts are separated by burst_interval_seconds.

    Total requests = num_bursts × requests_per_burst.

    Parameters
    ----------
    num_bursts:
        Number of burst events.
    requests_per_burst:
        Number of simultaneous requests per burst.
    burst_interval_seconds:
        Gap between consecutive bursts (first burst fires at t=0).
    prompts:
        Pool of prompts; cycled with modulo across all requests.
    max_new_tokens:
        Max tokens per request.

    Returns
    -------
    list[LoadRequest] ordered by burst index.
    """
    requests: list[LoadRequest] = []
    n_prompts = len(prompts)
    flat_index = 0
    for burst_idx in range(num_bursts):
        send_time = burst_idx * burst_interval_seconds
        for _ in range(requests_per_burst):
            requests.append(
                LoadRequest(
                    request_id=uuid.uuid4().hex,
                    prompt=prompts[flat_index % n_prompts],
                    max_new_tokens=max_new_tokens,
                    scheduled_send_time=send_time,
                )
            )
            flat_index += 1
    return requests


def default_prompt_pool() -> list[str]:
    """Return a hardcoded pool of 10 varied prompts of different lengths.

    Used as the default prompt set when the user does not supply their own.
    Covers short, medium, and long prompts to exercise variable-length
    generation within a single load test run.
    """
    return [
        # Short prompts
        "The capital of France is",
        "The largest planet in our solar system is",
        "Water boils at",
        # Medium prompts
        "Explain the difference between TCP and UDP in networking.",
        "What are the main principles of object-oriented programming?",
        "Describe the process of photosynthesis in simple terms.",
        "What is the significance of the Turing test in artificial intelligence?",
        # Longer prompts
        "Write a short story about a robot learning to paint for the first time.",
        "Explain how a transformer neural network architecture works, "
        "focusing on the self-attention mechanism.",
        "Compare and contrast the advantages and disadvantages of "
        "relational databases versus NoSQL document stores, "
        "giving concrete examples of use cases for each.",
    ]
