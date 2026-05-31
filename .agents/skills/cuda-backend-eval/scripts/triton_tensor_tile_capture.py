#!/usr/bin/env python3
"""Capture a Triton 16x16x16 tensor-tile baseline for the viewer."""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SHAPE = "n=1024, tensor tile 16x16x16"
DEFAULT_DTYPE = "tf32 Triton tl.dot, f32 accumulator"
DEFAULT_TOLERANCE = 1.0e-3


def fail(message: str) -> None:
    raise SystemExit(f"triton tensor tile capture failed: {message}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing JSON file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"JSON root is not an object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


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


def driver_version() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "-i",
                "0",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return "not recorded"
    if result.returncode != 0:
        return "not recorded"
    return result.stdout.splitlines()[0].strip() if result.stdout.strip() else "not recorded"


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def run_capture(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        import triton
        import triton.language as tl
    except ImportError as exc:
        fail(f"missing Triton capture dependency: {exc}")

    @triton.jit
    def tensor_tile_kernel(a, b, c, block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):
        tile = tl.program_id(0)
        offs_m = tl.arange(0, block_m)
        offs_n = tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)
        a_ptrs = a + tile * block_m * block_k + offs_m[:, None] * block_k + offs_k[None, :]
        b_ptrs = b + tile * block_k * block_n + offs_k[:, None] * block_n + offs_n[None, :]
        c_ptrs = c + tile * block_m * block_n + offs_m[:, None] * block_n + offs_n[None, :]
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), input_precision="tf32")
        tl.store(c_ptrs, acc)

    if not torch.cuda.is_available():
        fail("CUDA is not available to torch")
    torch.cuda.set_device(args.device)
    rows = args.rows
    cols = args.cols
    inner = args.inner
    tile_count = args.tile_count
    total_a = tile_count * rows * inner
    total_b = tile_count * inner * cols
    a = torch.linspace(-0.01, 0.01, total_a, device="cuda", dtype=torch.float32)
    b = torch.linspace(0.01, -0.01, total_b, device="cuda", dtype=torch.float32)
    a = a.reshape(tile_count, rows, inner).contiguous()
    b = b.reshape(tile_count, inner, cols).contiguous()
    c = torch.empty((tile_count, rows, cols), device="cuda", dtype=torch.float32)
    expected = torch.matmul(a, b)
    grid = (tile_count,)

    for _ in range(args.warmup):
        tensor_tile_kernel[grid](a, b, c, rows, cols, inner)
    torch.cuda.synchronize()

    samples: list[dict[str, Any]] = []
    for _ in range(args.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        host_start = time.perf_counter_ns()
        start.record()
        tensor_tile_kernel[grid](a, b, c, rows, cols, inner)
        end.record()
        end.synchronize()
        host_end = time.perf_counter_ns()
        max_abs_error = float((c - expected).abs().max().item())
        samples.append(
            {
                "host_wall_ns": int(host_end - host_start),
                "device_wall_ns": int(start.elapsed_time(end) * 1_000_000),
                "max_abs_error": max_abs_error,
            }
        )

    props = torch.cuda.get_device_properties(args.device)
    return {
        "metadata": {
            "pto_commit": args.pto_commit or git_commit(),
            "source": "triton_tensor_tile_capture.py",
            "rows": rows,
            "cols": cols,
            "inner": inner,
            "tile_count": tile_count,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "tolerance": args.tolerance,
            "triton_version": getattr(triton, "__version__", "unknown"),
            "torch_version": getattr(torch, "__version__", "unknown"),
        },
        "hardware": {
            "gpu": props.name.split("NVIDIA ", 1)[-1].split()[0],
            "machine": socket.gethostname(),
            "compute_target": f"compute_{props.major}{props.minor}",
            "driver": driver_version(),
            "cuda_toolkit": str(torch.version.cuda or "unknown"),
            "clock_policy": "not recorded",
        },
        "inputs": {
            "shape": DEFAULT_SHAPE,
            "dtype": DEFAULT_DTYPE,
            "repeat_policy": f"{args.repeats}-repeat Triton tensor tile capture",
        },
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, help="Convert an existing raw capture.")
    parser.add_argument("--output", type=Path, help="Write raw capture JSON here.")
    parser.add_argument("--viewer-output", type=Path, help="Write viewer result records here.")
    parser.add_argument("--artifact-root", help="Repo-relative raw artifact path.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--inner", type=int, default=16)
    parser.add_argument("--tile-count", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--pto-commit", help="Commit label to store in the raw artifact.")
    args = parser.parse_args()
    for key in ("rows", "cols", "inner", "tile_count", "warmup", "repeats"):
        value = getattr(args, key)
        if value <= 0:
            fail(f"--{key.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    payload = load_json(args.input_json) if args.input_json else run_capture(args)
    if args.output:
        write_json(args.output, payload)
    raw_artifact = args.artifact_root
    if raw_artifact is None:
        source = args.output or args.input_json
        raw_artifact = repo_relative(source.parent) + "/" if source else "tmp/"
    records = [viewer_record(payload, raw_artifact)]
    if args.viewer_output:
        write_json(args.viewer_output, records)
    elif not args.output:
        print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
