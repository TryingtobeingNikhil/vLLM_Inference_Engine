"""
load_test/gpu_monitor.py — Sample GPU utilisation and memory in the background.

Uses NVML (``pip install nvidia-ml-py``) when available, otherwise polls
``nvidia-smi``.  On machines without an NVIDIA GPU it records nothing and
``summary()`` returns ``{"available": False}``.

``utilization.gpu`` is NVIDIA's "fraction of time at least one kernel was
running" — a coarse busy-ness signal, not FLOP efficiency — but it is the
standard number and enough to see whether batching keeps the GPU fed.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from typing import List, Optional, Tuple

import numpy as np


class GPUMonitor:
    def __init__(self, interval_s: float = 0.2, device_index: int = 0) -> None:
        self.interval_s = interval_s
        self.device_index = device_index
        self.samples: List[Tuple[float, float, float]] = []   # (t, util %, mem MB)
        self.total_mem_mb: Optional[float] = None
        self.name: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._reader = self._make_reader()

    # ── Backends ──────────────────────────────────────────────────────────────

    def _make_reader(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            name = pynvml.nvmlDeviceGetName(handle)
            self.name = name.decode() if isinstance(name, bytes) else name
            self.total_mem_mb = pynvml.nvmlDeviceGetMemoryInfo(handle).total / 2**20

            def read():
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20
                return float(util), float(mem)

            return read
        except Exception:
            pass

        if shutil.which("nvidia-smi"):
            def read():
                out = subprocess.run(
                    ["nvidia-smi", f"--id={self.device_index}",
                     "--query-gpu=utilization.gpu,memory.used,memory.total,name",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip().split(", ")
                self.total_mem_mb = float(out[2])
                self.name = out[3]
                return float(out[0]), float(out[1])

            try:
                read()
                return read
            except Exception:
                pass
        return None

    @property
    def available(self) -> bool:
        return self._reader is not None

    # ── Sampling ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                util, mem = self._reader()
                self.samples.append((time.perf_counter(), util, mem))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> "GPUMonitor":
        if self.available and self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self.summary()

    def __enter__(self) -> "GPUMonitor":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def summary(self) -> dict:
        if not self.available or not self.samples:
            return {"available": False}
        util = np.array([s[1] for s in self.samples])
        mem = np.array([s[2] for s in self.samples])
        return {
            "available": True,
            "gpu_name": self.name,
            "num_samples": len(self.samples),
            "util_mean_pct": float(util.mean()),
            "util_p50_pct": float(np.percentile(util, 50)),
            "util_max_pct": float(util.max()),
            "mem_used_max_mb": float(mem.max()),
            "mem_total_mb": self.total_mem_mb,
        }
