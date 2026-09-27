#!/usr/bin/env python
"""
benchmarks/bench_serving.py — Online serving benchmark (latency under load).

For each server configuration this script:
  1. launches the server as a subprocess (uvicorn) with that configuration,
  2. waits for /health, sends warm-up requests,
  3. sweeps Poisson request rates (open loop) — or fixed concurrencies
     (closed loop) — with the streaming load tester, measuring TTFT, TPOT,
     ITL, E2E percentiles, throughput and goodput on the client, sampling GPU
     utilisation/memory, and scraping engine stats from /metrics,
  4. shuts the server down and moves on.

Configurations
--------------
  sequential   Phase 1 server: HF generate(), one request at a time
  no_batching  engine with MAX_BATCH_SIZE=1
  continuous   continuous batching + paged KV cache (prefix caching off)
  prefix       + automatic prefix caching
  ngram        + prefix caching + n-gram speculative decoding
  draft        + prefix caching + draft-model speculation (--draft-model)

Examples
--------
  python -m benchmarks.bench_serving                          # defaults sized to the GPU
  python -m benchmarks.bench_serving --configs continuous,prefix \\
         --workload shared_prefix --rates 2 4 8 16
  python -m benchmarks.bench_serving --concurrency 1 8 32 64 --configs continuous
  python -m benchmarks.bench_serving --server-env PREFILL_CHUNK_SIZE=256

Results: results/serving_<gpu>_<time>.json and .md (+ server logs)
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from benchmarks.common import (
    DRAFT_MODEL,
    RESULTS_DIR,
    gpu_tag,
    markdown_table,
    pick_model,
    save_results,
    system_info,
)
from load_test.gpu_monitor import GPUMonitor
from load_test.profiles import poisson_arrivals, schedule_workload
from load_test.report import build_report, engine_summary, scrape_metrics
from load_test.runner import LoadTestRunner
from load_test.workloads import load_tokenizer, make_workload

ENGINE_APP = "inference_engine.server.app_v2:app"
SEQUENTIAL_APP = "inference_engine.server.app:app"

CONFIGS: Dict[str, dict] = {
    "sequential": {"app": SEQUENTIAL_APP, "env": {}},
    "no_batching": {"app": ENGINE_APP, "env": {"MAX_BATCH_SIZE": "1", "ENABLE_PREFIX_CACHING": "0"}},
    "continuous": {"app": ENGINE_APP, "env": {"ENABLE_PREFIX_CACHING": "0"}},
    "prefix": {"app": ENGINE_APP, "env": {"ENABLE_PREFIX_CACHING": "1"}},
    "ngram": {"app": ENGINE_APP, "env": {"ENABLE_PREFIX_CACHING": "1", "SPECULATIVE_METHOD": "ngram"}},
    "draft": {"app": ENGINE_APP, "env": {"ENABLE_PREFIX_CACHING": "1", "SPECULATIVE_METHOD": "draft"}},
}
SLOW_CONFIGS = {"sequential", "no_batching"}
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Server process management ─────────────────────────────────────────────────


class ServerProcess:
    def __init__(self, app: str, env: dict, port: int, log_path: str) -> None:
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.log_path = log_path
        self._log = open(log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", app, "--host", "127.0.0.1",
             "--port", str(port), "--log-level", "warning"],
            cwd=REPO_ROOT, env={**os.environ, **env},
            stdout=self._log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def log_tail(self, n: int = 30) -> str:
        with open(self.log_path, encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-n:])

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited during startup:\n{self.log_tail()}")
            try:
                if httpx.get(f"{self.url}/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2)
        raise TimeoutError(f"server not ready after {timeout_s:.0f}s:\n{self.log_tail()}")

    def stop(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self._log.close()


# ── One load level ────────────────────────────────────────────────────────────


async def run_level(url: str, items, rate: Optional[float], concurrency: Optional[int],
                    seed: int, slo_ttft_ms: float, slo_tpot_ms: float) -> dict:
    runner = LoadTestRunner(base_url=url, max_concurrent=None, request_timeout_s=3600,
                            stream=True, ignore_eos=True)
    before = await scrape_metrics(url)
    with GPUMonitor() as gpu:
        t0 = time.perf_counter()
        if concurrency:
            requests = schedule_workload(items, [0.0] * len(items))
            results = await runner.run_closed_loop(requests, concurrency)
        else:
            requests = schedule_workload(items, poisson_arrivals(len(items), rate, seed))
            results = await runner.run(requests)
        duration = time.perf_counter() - t0
    report = build_report(results, duration, server_label="",
                          slo_ttft_ms=slo_ttft_ms, slo_tpot_ms=slo_tpot_ms)
    report.pop("server_label")
    report["gpu"] = gpu.summary()
    report["engine"] = engine_summary(await scrape_metrics(url), before=before)
    return report


def level_label(rate: Optional[float], concurrency: Optional[int]) -> str:
    if concurrency:
        return f"c={concurrency}"
    return "inf" if math.isinf(rate) else f"{rate:g}/s"


# ── Reporting ─────────────────────────────────────────────────────────────────


def config_markdown(name: str, levels: List[dict]) -> str:
    rows = []
    for level in levels:
        r = level["report"]
        lat, e, g = r["latency"], r.get("engine") or {}, r.get("gpu") or {}
        notes = []
        if e.get("prefix_cache_hit_rate"):
            notes.append(f"prefix hit {e['prefix_cache_hit_rate']:.0%}")
        if e.get("spec_acceptance_rate"):
            notes.append(f"accept {e['spec_acceptance_rate']:.0%}")
        preempt = (e.get("preemptions_swap") or 0) + (e.get("preemptions_recompute") or 0)
        if preempt:
            notes.append(f"{preempt} preemptions")
        if r["failed"]:
            notes.append(f"{r['failed']} failed")
        rows.append([
            level["label"], r["total_requests"], r["throughput_requests_per_sec"],
            r["throughput_tokens_per_sec"],
            lat["ttft_ms"]["p50"], lat["ttft_ms"]["p99"],
            lat["tpot_ms"]["p50"], lat["tpot_ms"]["p99"],
            lat["itl_ms"]["p99"], lat["total_latency_ms"]["p50"],
            r["goodput"]["pct"],
            g.get("util_mean_pct") if g.get("available") else None,
            g.get("mem_used_max_mb") if g.get("available") else None,
            ", ".join(notes),
        ])
    headers = ["load", "reqs", "req/s", "out tok/s", "TTFT p50", "TTFT p99", "TPOT p50",
               "TPOT p99", "ITL p99", "E2E p50", "goodput %", "GPU util %", "GPU mem MB", "notes"]
    return f"### {name}\n\n" + markdown_table(headers, rows) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="auto")
    p.add_argument("--draft-model", default="auto", help=f"for the 'draft' config (auto = {DRAFT_MODEL})")
    p.add_argument("--configs", default="sequential,continuous,prefix,ngram")
    p.add_argument("--workload", default="random",
                   choices=["random", "shared_prefix", "repetitive", "chat"])
    p.add_argument("--num-requests", type=int, default=200)
    p.add_argument("--sequential-requests", type=int, default=24,
                   help="requests per level for the slow configs")
    p.add_argument("--input-len", type=int, nargs=2, default=(256, 512))
    p.add_argument("--output-len", type=int, nargs=2, default=(128, 256))
    p.add_argument("--rates", type=float, nargs="+", default=[2, 4, 8, 16, float("inf")],
                   help="Poisson request rates (req/s); inf = all at once")
    p.add_argument("--concurrency", type=int, nargs="+", default=None,
                   help="closed-loop concurrency levels (instead of --rates)")
    p.add_argument("--slo-ttft-ms", type=float, default=2000.0)
    p.add_argument("--slo-tpot-ms", type=float, default=100.0)
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--server-env", nargs="*", default=[], metavar="KEY=VALUE",
                   help="extra environment for every server (e.g. MAX_BATCH_SIZE=128)")
    p.add_argument("--startup-timeout", type=float, default=1800.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=RESULTS_DIR)
    args = p.parse_args()

    model_name = pick_model(args.model)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = set(configs) - set(CONFIGS)
    if unknown:
        raise SystemExit(f"unknown configs: {sorted(unknown)}; choose from {sorted(CONFIGS)}")
    extra_env = dict(kv.split("=", 1) for kv in args.server_env)
    draft_name = DRAFT_MODEL if args.draft_model == "auto" else args.draft_model

    tokenizer = load_tokenizer(model_name)
    kwargs = {"output_len": tuple(args.output_len)}
    if args.workload == "random":
        kwargs["input_len"] = tuple(args.input_len)
    if args.workload == "repetitive":
        kwargs = {"passage_len": tuple(args.output_len)}
    levels = [(None, c) for c in args.concurrency] if args.concurrency \
        else [(r, None) for r in args.rates]
    # A fresh (seeded) prompt set per load level: identical across configs,
    # but no level can hit prefix-cache entries left behind by the previous one.
    level_items = [
        make_workload(args.workload, args.num_requests, seed=args.seed + i,
                      tokenizer=tokenizer, **kwargs)
        for i in range(len(levels))
    ]

    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = f"serving_{gpu_tag()}_{stamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    info = system_info()
    info["model"] = model_name
    payload = {"system": info, "args": vars(args), "configs": {}}
    markdown = [f"## Serving benchmark — {info.get('gpu', info['device'].upper())} — {model_name}\n",
                f"workload `{args.workload}`, streaming, fixed output lengths (ignore_eos); "
                f"SLO: TTFT ≤ {args.slo_ttft_ms:.0f} ms and TPOT ≤ {args.slo_tpot_ms:.0f} ms. "
                f"Latencies in ms, measured on the client from the scheduled send time.\n"]

    for name in configs:
        spec = CONFIGS[name]
        env = {"MODEL_NAME": model_name, **spec["env"], **extra_env,
               "REQUEST_TIMEOUT_MS": "3600000", "MAX_QUEUE_SIZE": str(args.num_requests + 64),
               # Phase 1 dumps its metrics on shutdown; keep them with the results.
               "METRICS_OUTPUT_PATH": os.path.join(args.output_dir, f"{stem}_{name}_metrics.json")}
        if name == "draft":
            if draft_name == model_name:
                print("draft model equals target; skipping 'draft'")
                continue
            env["DRAFT_MODEL_NAME"] = draft_name
        log_path = os.path.join(args.output_dir, f"{stem}_{name}.log")
        print(f"\n=== {name}: starting server ({spec['app']}) — log: {log_path}", flush=True)
        server = ServerProcess(spec["app"], env, args.port, log_path)
        results = []
        try:
            server.wait_ready(args.startup_timeout)
            for _ in range(2):   # warm-up
                httpx.post(f"{server.url}/generate", json={"prompt": "Warm up.", "max_new_tokens": 8},
                           timeout=600)
            for (rate, conc), items in zip(levels, level_items):
                subset = items[:args.sequential_requests] if name in SLOW_CONFIGS else items
                label = level_label(rate, conc)
                print(f"  load {label}: {len(subset)} requests …", flush=True)
                report = asyncio.run(run_level(server.url, subset, rate, conc, args.seed,
                                               args.slo_ttft_ms, args.slo_tpot_ms))
                lat = report["latency"]
                print(f"    {report['throughput_tokens_per_sec']:.1f} out tok/s | "
                      f"TTFT p50/p99 {lat['ttft_ms']['p50']:.0f}/{lat['ttft_ms']['p99']:.0f} ms | "
                      f"TPOT p50 {lat['tpot_ms']['p50']:.1f} ms | failed {report['failed']}",
                      flush=True)
                results.append({"label": label, "rate": rate, "concurrency": conc, "report": report})
        except Exception as exc:
            print(f"  {name} failed: {exc}", flush=True)
            payload["configs"][name] = {"error": str(exc), "levels": results}
        else:
            payload["configs"][name] = {"env": env, "levels": results}
        finally:
            server.stop()
        if results:
            markdown.append(config_markdown(name, results))

    json_path = save_results(payload, args.output_dir, stem)
    md_path = os.path.join(args.output_dir, stem + ".md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(markdown))
    print("\n" + "\n".join(markdown))
    print(f"Saved {json_path} and {md_path}")


if __name__ == "__main__":
    main()
