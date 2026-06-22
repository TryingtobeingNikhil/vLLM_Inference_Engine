import math

import pytest
import torch

from inference_engine.engine.kv_cache_config import (
    KVCacheConfig,
    estimate_max_sequences,
)
from inference_engine.engine.kv_cache_tracker import KVCacheTracker


@pytest.fixture
def sample_kv_config():
    return KVCacheConfig(
        num_layers=24,
        num_kv_heads=8,
        head_dim=64,
        dtype=torch.float16,
        device="cpu",
    )


def test_bytes_per_token_calculation(sample_kv_config):
    assert sample_kv_config.bytes_per_token == 49152


def test_bytes_per_token_mb(sample_kv_config):
    assert sample_kv_config.bytes_per_token_mb == pytest.approx(
        49152 / (1024 * 1024), rel=1e-5
    )


def test_estimate_max_sequences(sample_kv_config):
    expected = math.floor(
        1024 / (sample_kv_config.bytes_per_token_mb * 512)
    )
    assert estimate_max_sequences(sample_kv_config, 1024, 512) == expected


def test_tracker_register_and_memory(sample_kv_config):
    tracker = KVCacheTracker(sample_kv_config, max_memory_mb=1024)
    tracker.register_sequence("seq1", 100)
    assert tracker.sequence_memory_mb("seq1") == pytest.approx(
        100 * sample_kv_config.bytes_per_token_mb, rel=1e-5
    )


def test_tracker_update_sequence(sample_kv_config):
    tracker = KVCacheTracker(sample_kv_config, max_memory_mb=1024)
    tracker.register_sequence("seq1", 100)
    tracker.update_sequence("seq1", 150)
    assert tracker.sequence_memory_mb("seq1") == pytest.approx(
        150 * sample_kv_config.bytes_per_token_mb, rel=1e-5
    )


def test_tracker_unregister_idempotent(sample_kv_config):
    tracker = KVCacheTracker(sample_kv_config, max_memory_mb=1024)
    tracker.register_sequence("seq1", 100)
    tracker.unregister_sequence("seq1")
    tracker.unregister_sequence("seq1")
    assert tracker.sequence_memory_mb("seq1") == 0.0


def test_eviction_candidates_ordering(sample_kv_config):
    tracker = KVCacheTracker(sample_kv_config, max_memory_mb=1024)
    tracker.register_sequence("small", 50)
    tracker.register_sequence("large", 200)
    tracker.register_sequence("medium", 100)
    assert tracker.eviction_candidates(3)[0] == "large"


def test_memory_pressure_clamped(sample_kv_config):
    tracker = KVCacheTracker(sample_kv_config, max_memory_mb=1.0)
    tracker.register_sequence("seq1", 100000)
    assert tracker.memory_pressure() == 1.0
