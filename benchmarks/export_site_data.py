#!/usr/bin/env python
"""
benchmarks/export_site_data.py — Turn benchmark result files into website data.

    python -m benchmarks.export_site_data benchmarks/published/T4_2026-09-27 \\
        web/src/data/results/t4.json

Reads every offline_*.json and serving_*.json in the directory and writes one
trimmed JSON file (only the fields the site renders), so adding a new GPU to
the site is: run the Colab notebook, copy its results into
benchmarks/published/<GPU>_<date>/, run this script, import the file in
web/src/data/gpuBenchmarks.ts.  Numbers are copied, never recomputed.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

PCT_KEYS = ("p50", "p90", "p95", "p99", "mean")


def _pcts(d):
    return {k: d[k] for k in PCT_KEYS if isinstance(d, dict) and k in d} if d else None


def _gpu(d):
    if not d or not d.get("available"):
        return {"available": False}
    return {k: d.get(k) for k in ("available", "gpu_name", "util_mean_pct", "util_max_pct",
                                  "mem_used_max_mb", "mem_total_mb")}


def _offline_metrics(m):
    out = {k: m[k] for k in ("num_requests", "wall_s", "output_tok_s", "output_tokens",
                             "input_tokens", "peak_gpu_mem_mb", "prefix_cache_hit_rate",
                             "spec_acceptance_rate", "preemptions", "num_failed", "error",
                             "batch_size", "note") if k in m}
    for k in ("ttft_ms", "tpot_ms", "itl_ms", "e2e_ms"):
        if m.get(k):
            out[k] = _pcts(m[k])
    if "gpu" in m:
        out["gpu"] = _gpu(m["gpu"])
    return out


def _serving_report(r):
    lat = r.get("latency", {})
    engine = r.get("engine") or {}
    return {
        "total_requests": r.get("total_requests"),
        "failed": r.get("failed"),
        "throughput_requests_per_sec": r.get("throughput_requests_per_sec"),
        "throughput_tokens_per_sec": r.get("throughput_tokens_per_sec"),
        "total_output_tokens": r.get("total_output_tokens"),
        "goodput": r.get("goodput"),
        "latency": {k: _pcts(lat.get(k)) for k in ("ttft_ms", "tpot_ms", "itl_ms", "total_latency_ms")},
        "gpu": _gpu(r.get("gpu")),
        "engine": {k: engine.get(k) for k in ("prefix_cache_hit_rate", "spec_acceptance_rate",
                                                "preemptions_swap", "preemptions_recompute",
                                                "avg_batch_size") if k in engine},
    }


def _finite(x):
    return None if isinstance(x, float) and math.isinf(x) else x


def export(src_dir: str) -> dict:
    data = {"system": None, "offline": None, "serving": []}
    for path in sorted(glob.glob(os.path.join(src_dir, "offline_*.json"))):
        d = json.load(open(path))
        data["system"] = d["system"]
        data["offline"] = {
            "source": os.path.basename(path),
            "args": {k: d["args"].get(k) for k in ("num_requests", "sequential_requests",
                                                   "input_len", "output_len", "seed")},
            "runs": [{"suite": r["suite"], "workload": r["workload"], "system": r["system"],
                      "metrics": _offline_metrics(r["metrics"])} for r in d["runs"]],
        }
    for path in sorted(glob.glob(os.path.join(src_dir, "serving_*.json"))):
        d = json.load(open(path))
        data["system"] = data["system"] or d["system"]
        args = d["args"]
        data["serving"].append({
            "source": os.path.basename(path),
            "workload": args["workload"],
            "natural_stop": bool(args.get("natural_stop")),
            "num_requests": args["num_requests"],
            "slo_ttft_ms": args.get("slo_ttft_ms"),
            "slo_tpot_ms": args.get("slo_tpot_ms"),
            "configs": {
                name: {"levels": [{"label": lv["label"], "rate": _finite(lv.get("rate")),
                                   "concurrency": lv.get("concurrency"),
                                   "report": _serving_report(lv["report"])}
                                  for lv in cfg.get("levels", [])],
                       **({"error": cfg["error"]} if "error" in cfg else {})}
                for name, cfg in d["configs"].items()
            },
        })
    return data


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1:]
    data = export(src)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, allow_nan=False)
    n_off = len(data["offline"]["runs"]) if data["offline"] else 0
    print(f"wrote {dst}: {n_off} offline runs, {len(data['serving'])} serving sweeps")


if __name__ == "__main__":
    main()
