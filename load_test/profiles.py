"""
load_test/profiles.py — Phase 11: Load pattern generators.

Pure functions — no I/O, no network, deterministic for a given seed
(except request_id which uses uuid4, acceptable randomness).

Profiles:
  constant_load  — even spacing at a fixed requests-per-second rate
  ramp_load      — linearly increasing rate from start_rps to end_rps
  burst_load     — groups of simultaneous requests at fixed intervals
  poisson_load   — exponential inter-arrival gaps (memoryless traffic, the
                   standard model for independent users; seeded)

``schedule_workload`` combines any arrival process with a workload from
``load_test.workloads`` (per-request prompts and output lengths).
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass
from typing import List, Sequence


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
    if num_requests < 0:
        raise ValueError("num_requests must be non-negative")
    if requests_per_second <= 0:
        raise ValueError("requests_per_second must be greater than zero")
    if num_requests and not prompts:
        raise ValueError("prompts must not be empty")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")

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

    The schedule inverts the cumulative integral of the linear rate curve,
    then scales it to ``duration_seconds``.

    Parameters
    ----------
    num_requests:
        Total number of requests.
    start_rps:
        Initial request rate (requests per second).
    end_rps:
        Final request rate (requests per second).
    duration_seconds:
        Target total test duration.
    prompts:
        Pool of prompts; cycled with modulo.
    max_new_tokens:
        Max tokens per request.

    Returns
    -------
    list[LoadRequest] in non-decreasing scheduled_send_time order.
    """
    if num_requests < 0:
        raise ValueError("num_requests must be non-negative")
    if start_rps <= 0 or end_rps <= 0:
        raise ValueError("start_rps and end_rps must be greater than zero")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero")
    if num_requests and not prompts:
        raise ValueError("prompts must not be empty")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")

    requests: list[LoadRequest] = []
    n_prompts = len(prompts)
    rate_delta = end_rps - start_rps
    average_rate = (start_rps + end_rps) / 2.0
    for i in range(num_requests):
        fraction = i / max(num_requests, 1)
        if abs(rate_delta) < 1e-12:
            normalized_time = fraction
        else:
            discriminant = (
                start_rps ** 2 + 2.0 * rate_delta * fraction * average_rate
            )
            normalized_time = (
                -start_rps + max(discriminant, 0.0) ** 0.5
            ) / rate_delta
        scheduled_send_time = normalized_time * duration_seconds
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
    if num_bursts < 0 or requests_per_burst < 0:
        raise ValueError("burst counts must be non-negative")
    if burst_interval_seconds < 0:
        raise ValueError("burst_interval_seconds must be non-negative")
    if num_bursts and requests_per_burst and not prompts:
        raise ValueError("prompts must not be empty")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")

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


def poisson_arrivals(num_requests: int, requests_per_second: float, seed: int = 0) -> List[float]:
    """Send times (s) with exponentially distributed gaps.  ``inf`` rate → all at t=0."""
    if num_requests < 0:
        raise ValueError("num_requests must be non-negative")
    if requests_per_second <= 0:
        raise ValueError("requests_per_second must be greater than zero")
    if math.isinf(requests_per_second):
        return [0.0] * num_requests
    rng = random.Random(seed)
    t, times = 0.0, []
    for _ in range(num_requests):
        times.append(t)
        t += rng.expovariate(requests_per_second)
    return times


def poisson_load(
    num_requests: int,
    requests_per_second: float,
    prompts: list[str],
    max_new_tokens: int = 50,
    seed: int = 0,
) -> list[LoadRequest]:
    """Poisson arrivals cycling through *prompts*."""
    if num_requests and not prompts:
        raise ValueError("prompts must not be empty")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")
    return [
        LoadRequest(uuid.uuid4().hex, prompts[i % len(prompts)], max_new_tokens, t)
        for i, t in enumerate(poisson_arrivals(num_requests, requests_per_second, seed))
    ]


def schedule_workload(items: Sequence, send_times: Sequence[float]) -> list[LoadRequest]:
    """Pair workload items (``.prompt``, ``.max_new_tokens``) with send times."""
    if len(items) != len(send_times):
        raise ValueError("items and send_times must have the same length")
    return [
        LoadRequest(uuid.uuid4().hex, item.prompt, item.max_new_tokens, t)
        for item, t in zip(items, send_times)
    ]
