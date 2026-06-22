"""Per-stage scheduler timing and throughput statistics."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque

import numpy as np


@dataclass
class PrefillRecord:
    timestamp: float
    sequences_prefilled: int
    tokens_prefilled: int
    latency_ms: float
    budget_tokens: int
    budget_utilization: float


@dataclass
class DecodeRecord:
    timestamp: float
    sequences_decoded: int
    latency_ms: float
    batch_limit: int


class StageTracker:
    """Keep a bounded, thread-safe history of prefill and decode stages."""

    def __init__(self, history_size: int = 500):
        self._prefill_records: Deque[PrefillRecord] = deque(maxlen=history_size)
        self._decode_records: Deque[DecodeRecord] = deque(maxlen=history_size)
        self._lock = Lock()

    def record_prefill(
        self,
        sequences_prefilled: int,
        tokens_prefilled: int,
        latency_ms: float,
        budget_tokens: int,
    ) -> None:
        utilization = tokens_prefilled / budget_tokens if budget_tokens > 0 else 0.0
        record = PrefillRecord(
            timestamp=time.perf_counter(),
            sequences_prefilled=sequences_prefilled,
            tokens_prefilled=tokens_prefilled,
            latency_ms=latency_ms,
            budget_tokens=budget_tokens,
            budget_utilization=utilization,
        )
        with self._lock:
            self._prefill_records.append(record)

    def record_decode(
        self, sequences_decoded: int, latency_ms: float, batch_limit: int
    ) -> None:
        record = DecodeRecord(
            timestamp=time.perf_counter(),
            sequences_decoded=sequences_decoded,
            latency_ms=latency_ms,
            batch_limit=batch_limit,
        )
        with self._lock:
            self._decode_records.append(record)

    def prefill_summary(self) -> dict:
        with self._lock:
            records = list(self._prefill_records)
        if not records:
            return {
                "total_iterations": 0,
                "total_tokens_prefilled": 0,
                "total_sequences_prefilled": 0,
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "avg_budget_utilization": 0.0,
            }

        latencies = [record.latency_ms for record in records]
        return {
            "total_iterations": len(records),
            "total_tokens_prefilled": sum(r.tokens_prefilled for r in records),
            "total_sequences_prefilled": sum(r.sequences_prefilled for r in records),
            "avg_latency_ms": float(np.mean(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "avg_budget_utilization": float(
                np.mean([record.budget_utilization for record in records])
            ),
        }

    def decode_summary(self) -> dict:
        with self._lock:
            records = list(self._decode_records)
        if not records:
            return {
                "total_iterations": 0,
                "total_sequences_decoded": 0,
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "avg_batch_size": 0.0,
            }

        latencies = [record.latency_ms for record in records]
        return {
            "total_iterations": len(records),
            "total_sequences_decoded": sum(r.sequences_decoded for r in records),
            "avg_latency_ms": float(np.mean(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "avg_batch_size": float(
                np.mean([record.sequences_decoded for record in records])
            ),
        }

    def full_report(self) -> dict:
        return {
            "prefill": self.prefill_summary(),
            "decode": self.decode_summary(),
        }
