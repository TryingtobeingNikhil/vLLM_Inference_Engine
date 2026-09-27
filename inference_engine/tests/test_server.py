"""
tests/test_server.py — HTTP API of both servers, fully offline.

A tiny random checkpoint (with a byte-level tokenizer) is written to a temp
dir and loaded through the normal ``MODEL_NAME`` path, so the real lifespan,
scheduler, streaming and detokenization code all run.
"""

from __future__ import annotations

import importlib
import json

import pytest
from fastapi.testclient import TestClient
from transformers import AutoTokenizer

from inference_engine.server.detokenizer import IncrementalDetokenizer
from inference_engine.tests.tiny_models import save_tiny_checkpoint


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return save_tiny_checkpoint(str(tmp_path_factory.mktemp("tiny")))


@pytest.fixture
def engine_env(monkeypatch, checkpoint):
    for key, value in {
        "MODEL_NAME": checkpoint, "DEVICE": "cpu", "DTYPE": "float32",
        "KV_NUM_BLOCKS": "128", "MAX_BATCH_SIZE": "4", "MAX_MODEL_LEN": "256",
        "SPECULATIVE_METHOD": "ngram",
    }.items():
        monkeypatch.setenv(key, value)


def _sse_events(response):
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            payload = line[len("data: "):]
            events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


# ── Engine server (app_v2) ────────────────────────────────────────────────────


@pytest.fixture
def engine_client(engine_env):
    from inference_engine.server import app_v2

    importlib.reload(app_v2)
    with TestClient(app_v2.app) as client:
        yield client


def test_health_and_metrics(engine_client):
    health = engine_client.get("/health").json()
    assert health["status"] == "ok"
    assert health["kv_blocks_total"] == 128
    assert health["speculative_method"] == "ngram"
    engine_client.post("/generate", json={"prompt": "hi", "max_new_tokens": 3})
    metrics = engine_client.get("/metrics").json()
    assert metrics["system"]["requests_finished_total"] == 1
    assert "itl_ms" in metrics["e2e_latency"]


def test_generate_non_streaming(engine_client):
    body = engine_client.post(
        "/generate",
        json={"prompt": "hello world", "max_new_tokens": 12, "ignore_eos": True},
    ).json()
    assert body["generated_tokens"] == 12
    assert body["finish_reason"] == "length"
    assert body["prompt_tokens"] == len("hello world")
    assert body["ttft_ms"] > 0 and body["total_latency_ms"] >= body["ttft_ms"]
    # one ITL gap per emission step (speculative steps emit several tokens)
    assert 1 <= len(body["per_token_latencies_ms"]) <= 11


def test_generate_streaming(engine_client):
    with engine_client.stream(
        "POST", "/generate",
        json={"prompt": "stream me", "max_new_tokens": 10, "ignore_eos": True, "stream": True},
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _sse_events(response)
    assert events[-1] == "[DONE]"
    final = events[-2]
    assert final["done"] is True
    tokens = [t for e in events[:-2] for t in e["token_ids"]]
    text = "".join(e["text"] for e in events[:-2])
    assert len(tokens) == final["generated_tokens"] == 10
    assert text == final["generated_text"]


def test_openai_completions(engine_client):
    body = engine_client.post(
        "/v1/completions",
        json={"prompt": "abc", "max_tokens": 5, "temperature": 0, "ignore_eos": True},
    ).json()
    assert body["object"] == "text_completion"
    assert body["usage"]["completion_tokens"] == 5
    assert body["choices"][0]["finish_reason"] == "length"

    with engine_client.stream(
        "POST", "/v1/completions",
        json={"prompt": "abc", "max_tokens": 5, "temperature": 0,
              "ignore_eos": True, "stream": True},
    ) as response:
        events = _sse_events(response)
    assert events[-1] == "[DONE]"
    assert events[-2]["usage"]["completion_tokens"] == 5
    streamed = "".join(e["choices"][0]["text"] for e in events[:-1])
    assert streamed == body["choices"][0]["text"]


def test_rejects_prompt_longer_than_max_model_len(engine_client):
    response = engine_client.post("/generate", json={"prompt": "x" * 300})
    assert response.status_code == 400


def test_validation_errors(engine_client):
    assert engine_client.post("/generate", json={"prompt": ""}).status_code == 422
    assert engine_client.post(
        "/generate", json={"prompt": "a", "max_new_tokens": 0}
    ).status_code == 422


# ── Phase 1 sequential server (app) ───────────────────────────────────────────


@pytest.fixture
def sequential_client(engine_env, tmp_path, monkeypatch):
    monkeypatch.setenv("METRICS_OUTPUT_PATH", str(tmp_path / "metrics.json"))
    from inference_engine.server import app

    importlib.reload(app)
    with TestClient(app.app) as client:
        yield client


def test_sequential_server_stream_and_plain(sequential_client):
    plain = sequential_client.post(
        "/generate", json={"prompt": "hi", "max_new_tokens": 6, "ignore_eos": True}
    ).json()
    assert plain["generated_tokens"] == 6
    assert plain["queue_wait_time_ms"] >= 0

    with sequential_client.stream(
        "POST", "/generate",
        json={"prompt": "hi", "max_new_tokens": 6, "ignore_eos": True, "stream": True},
    ) as response:
        events = _sse_events(response)
    assert events[-1] == "[DONE]"
    tokens = [t for e in events[:-2] for t in e["token_ids"]]
    assert len(tokens) == 6 == events[-2]["generated_tokens"]
    assert sequential_client.get("/metrics").json()["summary"]["count"] == 2


# ── Incremental detokenizer ───────────────────────────────────────────────────


def test_incremental_detokenizer_handles_multibyte(checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    text = "héllo wörld — 你好 🙂!"
    ids = tokenizer(text)["input_ids"]
    detok = IncrementalDetokenizer(tokenizer)
    pieces = [detok.add([i]) for i in ids]
    pieces.append(detok.flush())
    assert "".join(pieces) == text
    assert all("�" not in p for p in pieces)
