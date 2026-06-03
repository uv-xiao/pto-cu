"""Triton tensor-tile runtime capture."""

from __future__ import annotations

import argparse
import socket
import subprocess
import time
from typing import Any

from triton_tensor_tile_capture_impl.constants import DEFAULT_DTYPE
from triton_tensor_tile_capture_impl.constants import DEFAULT_SHAPE
from triton_tensor_tile_capture_impl.errors import fail
from triton_tensor_tile_capture_impl.io import ROOT


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
            "shape": f"n=1024, tensor tile {rows}x{cols}x{inner}",
            "dtype": DEFAULT_DTYPE,
            "repeat_policy": f"{args.repeats}-repeat Triton tensor tile capture",
        },
        "samples": samples,
    }
