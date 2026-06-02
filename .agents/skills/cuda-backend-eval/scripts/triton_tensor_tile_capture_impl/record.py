"""Viewer-record conversion for Triton tensor-tile captures."""

from __future__ import annotations

from typing import Any

from triton_tensor_tile_capture_impl.constants import DEFAULT_TOLERANCE
from triton_tensor_tile_capture_impl.errors import fail
from triton_tensor_tile_capture_impl.stats import latency_summary


def require_dict(record: dict[str, Any], key: str, owner: str) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict):
        fail(f"{owner} missing {key}")
    return value


def require_string(record: dict[str, Any], key: str, owner: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        fail(f"{owner} missing {key}")
    return value


def require_samples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        fail("capture has no samples")
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            fail(f"sample {index} is not an object")
        for key in ("host_wall_ns", "device_wall_ns", "max_abs_error"):
            value = sample.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                fail(f"sample {index} has invalid {key}")
    return samples


def viewer_record(payload: dict[str, Any], raw_artifact: str) -> dict[str, Any]:
    owner = "triton tensor tile capture"
    metadata = require_dict(payload, "metadata", owner)
    hardware = require_dict(payload, "hardware", owner)
    inputs = require_dict(payload, "inputs", owner)
    samples = require_samples(payload)
    host_summary = latency_summary(samples, "host_wall_ns", "host_wall")
    device_summary = latency_summary(samples, "device_wall_ns", "device_wall")
    max_abs_error = max(float(sample["max_abs_error"]) for sample in samples)
    tolerance = float(metadata.get("tolerance", DEFAULT_TOLERANCE))

    return {
        "benchmark_id": "tensor_core_tile",
        "method_id": "triton",
        "hardware": {
            "gpu": require_string(hardware, "gpu", owner),
            "machine": require_string(hardware, "machine", owner),
            "compute_target": require_string(hardware, "compute_target", owner),
            "driver": str(hardware.get("driver", "see raw artifact")),
            "cuda_toolkit": str(hardware.get("cuda_toolkit", "see raw artifact")),
            "clock_policy": str(hardware.get("clock_policy", "not recorded")),
        },
        "commit": str(metadata.get("pto_commit", metadata.get("git_commit", "unknown"))),
        "inputs": {
            "shape": require_string(inputs, "shape", owner),
            "dtype": require_string(inputs, "dtype", owner),
            "repeat_policy": require_string(inputs, "repeat_policy", owner),
        },
        "statistic": {
            "kind": "triton_cuda_event_distribution",
            "sample_count": len(samples),
            "host_wall_ns": host_summary["host_wall_p50_ns"],
            "device_wall_ns": device_summary["device_wall_p50_ns"],
            **host_summary,
            **device_summary,
            "max_abs_error": max_abs_error,
            "tolerance": tolerance,
        },
        "raw_artifact": raw_artifact,
        "correctness": "pass" if max_abs_error <= tolerance else "fail",
    }
