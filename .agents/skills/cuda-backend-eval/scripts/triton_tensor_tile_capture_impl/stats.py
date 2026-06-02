"""Latency statistics for Triton tensor-tile captures."""

from __future__ import annotations

import statistics
from typing import Any

from triton_tensor_tile_capture_impl.errors import fail


def percentile_int(values: list[int], quantile: float) -> int:
    if not values:
        fail("cannot summarize an empty sample list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return int(ordered[lower] + (ordered[upper] - ordered[lower]) * weight)


def latency_summary(samples: list[dict[str, Any]], field: str, prefix: str) -> dict[str, int]:
    values = [int(sample[field]) for sample in samples]
    return {
        f"{prefix}_p50_ns": percentile_int(values, 0.50),
        f"{prefix}_p90_ns": percentile_int(values, 0.90),
        f"{prefix}_p99_ns": percentile_int(values, 0.99),
        f"{prefix}_mean_ns": int(statistics.fmean(values)),
        f"{prefix}_stdev_ns": int(statistics.stdev(values)) if len(values) > 1 else 0,
        f"{prefix}_min_ns": min(values),
        f"{prefix}_max_ns": max(values),
    }
