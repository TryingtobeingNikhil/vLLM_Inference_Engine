"""
benchmarks/common.py — Shared helpers for the benchmark scripts.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from typing import Iterable, List, Optional, Sequence

import numpy as np

RESULTS_DIR = "results"

# Default models per GPU class: big enough to be interesting, small enough to
# leave room for a large KV cache.  All are ungated on the HF Hub.
MODEL_BY_MEMORY_GB = [
    (20, "Qwen/Qwen2.5-1.5B-Instruct"),   # T4 (16 GB)
    (32, "Qwen/Qwen2.5-3B-Instruct"),     # L4 (24 GB)
    (1e9, "Qwen/Qwen2.5-7B-Instruct"),    # A100 (40/80 GB)
]
CPU_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def pick_model(model: str) -> str:
    """Resolve ``"auto"`` to a model sized for the local accelerator."""
    if model != "auto":
        return model
    import torch

    if not torch.cuda.is_available():
        return CPU_MODEL
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    for limit, name in MODEL_BY_MEMORY_GB:
        if total_gb < limit:
            return name
    return MODEL_BY_MEMORY_GB[-1][1]


def system_info() -> dict:
    import torch
    import transformers

    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
    }
    mps = getattr(torch.backends, "mps", None)
    info["device"] = "cuda" if torch.cuda.is_available() else (
        "mps" if mps is not None and mps.is_available() else "cpu")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu": props.name,
            "gpu_memory_gb": round(props.total_memory / 2**30, 1),
            "cuda": torch.version.cuda,
            "compute_capability": f"{props.major}.{props.minor}",
        })
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).stdout.strip()
    except Exception:
        pass
    return info


def gpu_tag() -> str:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0).replace("NVIDIA ", "").replace(" ", "-")
    return "cpu"


def pcts(values: Sequence[float]) -> dict:
    if len(values) == 0:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    arr = np.asarray(values, dtype=float)
    p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99])
    return {"p50": float(p50), "p90": float(p90), "p95": float(p95),
            "p99": float(p99), "mean": float(arr.mean())}


def save_results(payload: dict, output_dir: str, stem: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{stem}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def markdown_table(headers: List[str], rows: Iterable[Sequence]) -> str:
    def fmt(value) -> str:
        if value is None:
            return "–"
        if isinstance(value, float):
            return f"{value:,.1f}" if abs(value) >= 10 else f"{value:.2f}"
        return str(value)

    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def fmt_optional(value: Optional[float], spec: str = ".2f") -> str:
    return "–" if value is None else format(value, spec)
