"""
tests/test_sequential.py — Unit tests for the sequential inference engine.

Tests are designed to run without a GPU (they will use CPU / MPS depending
on the machine), and they load the model once per session via a pytest fixture
to keep runtime manageable.

Run with:
    cd inference_engine
    pytest tests/ -v

Tests
-----
1. test_prefill_returns_valid_kv_and_token
2. test_decode_token_count
3. test_ttft_less_than_total_latency
4. test_generation_result_fully_populated
5. test_metrics_file_written_on_dump
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from inference_engine.config import Config
from inference_engine.engine.sequential import GenerationResult, decode, generate, prefill
from inference_engine.metrics.collector import MetricsCollector
from inference_engine.models.loader import LoadedModel, load_model_and_tokenizer

try:
    # transformers >= 4.38 returns DynamicCache instead of a plain tuple
    from transformers.cache_utils import Cache as HFCache
except ImportError:
    HFCache = None  # older transformers — tuple is the only type


# ── Session-scoped model fixture (loaded once for the entire test run) ────────


@pytest.fixture(scope="session")
def cfg() -> Config:
    return Config()


@pytest.fixture(scope="session")
def loaded(cfg: Config) -> LoadedModel:
    """Load model once; reused by all tests in the session."""
    return load_model_and_tokenizer(cfg)


# ── Helpers ───────────────────────────────────────────────────────────────────

SHORT_PROMPT = "The capital of France is"
MEDIUM_PROMPT = (
    "Explain the significance of the Turing test in the history of artificial intelligence. "
    "Alan Turing proposed this test as a criterion for machine intelligence."
)


# ── Test 1: prefill returns a non-empty KV cache and a valid token id ─────────


def test_prefill_returns_valid_kv_and_token(loaded: LoadedModel):
    """
    prefill() must return:
    - past_key_values that is a non-empty tuple (one entry per layer)
    - first_token_id that is a non-negative integer within vocab range
    - ttft_ms that is strictly positive
    """
    model, tokenizer, device = loaded

    past_kv, first_token_id, ttft_ms = prefill(model, tokenizer, SHORT_PROMPT)

    # KV cache must be a non-None object with at least one layer.
    # transformers < 4.38  → plain tuple of per-layer tuples
    # transformers >= 4.38 → DynamicCache object (subclass of Cache)
    # Both support len() and are truthy when non-empty.
    assert past_kv is not None, "past_key_values must not be None"

    valid_type = isinstance(past_kv, (tuple, list))
    if HFCache is not None:
        valid_type = valid_type or isinstance(past_kv, HFCache)
    assert valid_type, (
        f"past_key_values must be tuple/list/DynamicCache, got {type(past_kv).__name__}"
    )
    assert len(past_kv) > 0, "past_key_values must have at least one layer"

    # first_token_id must be a valid vocabulary index
    vocab_size = model.config.vocab_size
    assert isinstance(first_token_id, int), "first_token_id must be int"
    assert 0 <= first_token_id < vocab_size, (
        f"first_token_id {first_token_id} out of vocab range [0, {vocab_size})"
    )

    # TTFT must be positive
    assert ttft_ms > 0.0, f"ttft_ms must be positive, got {ttft_ms}"


# ── Test 2: decode produces the correct number of tokens ──────────────────────


@pytest.mark.parametrize("max_new_tokens", [1, 5, 20])
def test_decode_token_count(loaded: LoadedModel, max_new_tokens: int):
    """
    decode() must return at most max_new_tokens token ids.
    The generated_ids list always includes the first_token_id, so its length
    is between 1 and max_new_tokens (EOS may truncate early).
    """
    model, tokenizer, device = loaded

    past_kv, first_token_id, _ = prefill(model, tokenizer, SHORT_PROMPT)
    generated_ids, per_token_latencies = decode(
        model=model,
        past_key_values=past_kv,
        first_token_id=first_token_id,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )

    assert 1 <= len(generated_ids) <= max_new_tokens, (
        f"Expected 1–{max_new_tokens} tokens, got {len(generated_ids)}"
    )

    # per_token_latencies has one entry per decode step (excludes first token)
    assert len(per_token_latencies) == len(generated_ids) - 1, (
        "per_token_latencies length mismatch"
    )

    # All latencies must be positive
    for i, lat in enumerate(per_token_latencies):
        assert lat > 0.0, f"Step {i} latency must be positive, got {lat}"


# ── Test 3: TTFT < total latency ──────────────────────────────────────────────


def test_ttft_less_than_total_latency(loaded: LoadedModel, cfg: Config):
    """
    TTFT (time to first token) must always be strictly less than total_latency_ms
    because total includes TTFT plus at least one decode step.
    """
    model, tokenizer, device = loaded

    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=MEDIUM_PROMPT,
        max_new_tokens=10,
        device=device,
    )

    assert result.ttft_ms > 0.0, "TTFT must be positive"
    assert result.total_latency_ms > 0.0, "Total latency must be positive"
    assert result.ttft_ms < result.total_latency_ms, (
        f"TTFT ({result.ttft_ms:.2f} ms) must be < total latency "
        f"({result.total_latency_ms:.2f} ms)"
    )


# ── Test 4: GenerationResult fields are all populated and non-null ────────────


def test_generation_result_fully_populated(loaded: LoadedModel):
    """
    Every field in GenerationResult must be:
    - Not None
    - Numerically non-negative (for numeric fields)
    - Non-empty (for string / list fields)
    """
    model, tokenizer, device = loaded

    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=MEDIUM_PROMPT,
        max_new_tokens=15,
        device=device,
    )

    assert isinstance(result, GenerationResult)

    # String fields
    assert result.prompt, "prompt must be non-empty"
    assert result.generated_text is not None, "generated_text must not be None"
    assert result.timestamp, "timestamp must be non-empty"

    # Integer counts
    assert result.prompt_tokens > 0, "prompt_tokens must be > 0"
    assert result.generated_tokens > 0, "generated_tokens must be > 0"

    # Latency metrics
    assert result.ttft_ms > 0.0
    assert result.total_latency_ms > 0.0
    assert result.tokens_per_second > 0.0

    # Memory fields — non-negative (may be 0 on some CPU-only systems)
    assert result.gpu_memory_allocated_mb >= 0.0
    assert result.gpu_memory_reserved_mb >= 0.0

    # per_token_latencies_ms must be a list (possibly empty if only 1 token)
    assert isinstance(result.per_token_latencies_ms, list)


# ── Test 5: Metrics file is written correctly ─────────────────────────────────


def test_metrics_file_written_on_dump(loaded: LoadedModel):
    """
    MetricsCollector.dump_to_json() must:
    - Create the output file
    - Write valid JSON
    - Include both 'results' and 'summary' keys
    - Have the correct number of result entries
    """
    model, tokenizer, device = loaded

    collector = MetricsCollector(history_size=10)

    # Generate two results and collect them
    for prompt in [SHORT_PROMPT, MEDIUM_PROMPT]:
        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=5,
            device=device,
        )
        collector.append(result)

    assert len(collector) == 2

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        tmp_path = tmp.name

    try:
        collector.dump_to_json(tmp_path)

        assert os.path.exists(tmp_path), "Metrics file was not created"
        assert os.path.getsize(tmp_path) > 0, "Metrics file is empty"

        with open(tmp_path, encoding="utf-8") as fh:
            data = json.load(fh)

        assert "results" in data, "Missing 'results' key"
        assert "summary" in data, "Missing 'summary' key"
        assert len(data["results"]) == 2, (
            f"Expected 2 results, got {len(data['results'])}"
        )

        summary = data["summary"]
        assert summary["count"] == 2
        assert "ttft_ms" in summary
        assert "total_latency_ms" in summary
        for metric_key in ("ttft_ms", "total_latency_ms"):
            for pct_key in ("p50", "p95", "p99", "mean", "min", "max"):
                assert pct_key in summary[metric_key], (
                    f"Missing '{pct_key}' in summary['{metric_key}']"
                )

    finally:
        os.unlink(tmp_path)
