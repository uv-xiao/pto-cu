"""Latency statistics for ThunderKittens captures."""

from __future__ import annotations

import statistics


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def summarize_ns(samples_ns: list[float]) -> dict[str, float | int]:
    return {
        "sample_count": len(samples_ns),
        "mean_ns": statistics.fmean(samples_ns),
        "stdev_ns": statistics.stdev(samples_ns) if len(samples_ns) > 1 else 0.0,
        "min_ns": min(samples_ns),
        "max_ns": max(samples_ns),
        "p50_ns": percentile(samples_ns, 0.50),
        "p90_ns": percentile(samples_ns, 0.90),
        "p99_ns": percentile(samples_ns, 0.99),
    }
