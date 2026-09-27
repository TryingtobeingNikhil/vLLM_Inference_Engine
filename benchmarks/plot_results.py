#!/usr/bin/env python
"""
benchmarks/plot_results.py — Charts for benchmark result JSON files.

    python -m benchmarks.plot_results results/serving_*.json results/offline_*.json

Serving results → one PNG with three panels sharing the load axis:
output throughput, TTFT p99 and TPOT p50 (each panel has its own single
y-axis).  Offline results → one horizontal bar chart of output tokens/s per
system, one panel per suite.  PNGs are written next to the JSON files.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed categorical order (validated for colour-vision deficiency); a config
# keeps its colour in every chart regardless of which configs were run.
SERIES_COLORS = {
    "sequential": "#2a78d6", "no_batching": "#eb6834", "continuous": "#1baf7a",
    "prefix": "#eda100", "ngram": "#e87ba4", "draft": "#008300",
}
MARKERS = {"sequential": "o", "no_batching": "s", "continuous": "D",
           "prefix": "^", "ngram": "v", "draft": "P"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def _title(prefix: str, info: dict) -> str:
    parts = [prefix, info.get("gpu") or info.get("device", "").upper(), info.get("model", "")]
    return " — ".join(p for p in parts if p)


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def plot_serving(path: str, data: dict) -> str:
    panels = [
        ("Output throughput", "tokens / s", lambda r: r["throughput_tokens_per_sec"]),
        ("Time to first token (p99)", "ms", lambda r: r["latency"]["ttft_ms"]["p99"]),
        ("Time per output token (p50)", "ms", lambda r: r["latency"]["tpot_ms"]["p50"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    labels = None
    for ax, (title, unit, metric) in zip(axes, panels):
        for name, cfg in data["configs"].items():
            levels = cfg.get("levels") or []
            if not levels:
                continue
            labels = [lv["label"] for lv in levels]
            xs = list(range(len(levels)))
            ys = [metric(lv["report"]) for lv in levels]
            color = SERIES_COLORS.get(name, INK)
            ax.plot(xs, ys, color=color, linewidth=2, marker=MARKERS.get(name, "o"),
                    markersize=8, markeredgecolor="white", markeredgewidth=1.5, label=name)
            ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
        if labels:
            ax.set_xticks(range(len(labels)), labels)
        ax.set_xlabel("offered load (Poisson rate or concurrency)", color=MUTED)
        _style(ax, title, unit)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    info = data.get("system", {})
    fig.suptitle(_title("Serving", info), x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    out = os.path.splitext(path)[0] + ".png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_offline(path: str, data: dict) -> str:
    groups: dict = {}
    for run in data["runs"]:
        if "error" in run["metrics"]:
            continue
        groups.setdefault(f"{run['suite']}: {run['workload']}", []).append(run)
    fig, axes = plt.subplots(len(groups), 1, figsize=(9, 1.2 + 0.55 * sum(len(g) for g in groups.values())
                                                   + 0.6 * len(groups)), squeeze=False)
    for ax, (title, runs) in zip(axes[:, 0], groups.items()):
        names = [r["system"] for r in runs]
        values = [r["metrics"]["output_tok_s"] for r in runs]
        bars = ax.barh(range(len(runs)), values, color="#2a78d6", height=0.6)
        ax.set_yticks(range(len(runs)), names)
        ax.invert_yaxis()
        base = values[0] or 1.0
        for bar, value in zip(bars, values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f"  {value:,.0f} tok/s ({value / base:.1f}x)", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, max(values) * 1.3 if values and max(values) > 0 else 1)
        _style(ax, title, "")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
    info = data.get("system", {})
    fig.suptitle(_title("Offline output throughput", info), x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    out = os.path.splitext(path)[0] + ".png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main(paths) -> None:
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if "configs" in data:
            print("wrote", plot_serving(path, data))
        elif "runs" in data:
            print("wrote", plot_offline(path, data))


if __name__ == "__main__":
    main(sys.argv[1:])
