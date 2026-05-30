#!/usr/bin/env python3
"""Capture bounded ThunderKittens MHA correctness and latency evidence."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def read_gpu_metadata() -> dict[str, str]:
    query = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,compute_cap",
        "--format=csv,noheader",
    ]
    try:
        output = subprocess.check_output(query, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:  # pragma: no cover - depends on target machine.
        return {
            "gpu": "unknown",
            "driver": "unknown",
            "compute_target": "unknown",
            "nvidia_smi_error": str(exc),
        }
    first = output.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    gpu = parts[0] if parts else "unknown"
    driver = parts[1] if len(parts) > 1 else "unknown"
    compute_cap = parts[2].replace(".", "") if len(parts) > 2 else "unknown"
    compute_target = (
        f"compute_{compute_cap}" if compute_cap != "unknown" else "unknown"
    )
    return {"gpu": gpu, "driver": driver, "compute_target": compute_target}


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


def run_shape(
    *,
    torch: Any,
    tk: Any,
    b: int,
    h: int,
    n: int,
    d: int,
    causal: bool,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    q = torch.randn((b, h, n, d), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((b, h, n, d), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((b, h, n, d), dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()

    for _ in range(warmup):
        tk.mha_forward(q, k, v, causal)
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out, _ = tk.mha_forward(q, k, v, causal)
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(start.elapsed_time(end))

    with torch.no_grad():
        reference = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal
        )
        max_abs_diff = (reference - out).abs().max().item()
        mean_abs_diff = (reference - out).abs().float().mean().item()

    samples_ns = [sample * 1_000_000 for sample in samples_ms]
    summary = summarize_ns(samples_ns)
    return {
        "shape": {
            "b": b,
            "h": h,
            "n": n,
            "d": d,
            "causal": causal,
            "dtype": "bfloat16",
        },
        "correctness": {
            "status": "pass",
            "reference": "torch.nn.functional.scaled_dot_product_attention",
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
        },
        "latency": {
            "warmup": warmup,
            "repeats": repeats,
            "samples_ns": samples_ns,
            **summary,
        },
    }


def parse_shape(value: str) -> tuple[int, int, int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("shape must be b,h,n,d")
    try:
        b, h, n, d = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape entries must be integers") from exc
    if min(b, h, n, d) <= 0:
        raise argparse.ArgumentTypeError("shape entries must be positive")
    return b, h, n, d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--machine", default="unknown")
    parser.add_argument("--pto-commit", required=True)
    parser.add_argument("--cuda-toolkit", default="unknown")
    parser.add_argument("--clock-policy", default="not recorded")
    parser.add_argument("--shape", action="append", type=parse_shape, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("warmup must be non-negative and repeats must be positive")

    import sys

    sys.path.insert(0, str(args.baseline_dir.resolve()))
    import torch  # type: ignore[import-not-found]
    import _C as tk  # type: ignore[import-not-found]

    gpu_metadata = read_gpu_metadata()
    results = []
    raw_shape_results = []
    for b, h, n, d in args.shape:
        shape_result = run_shape(
            torch=torch,
            tk=tk,
            b=b,
            h=h,
            n=n,
            d=d,
            causal=args.causal,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
        )
        raw_shape_results.append(shape_result)
        shape = shape_result["shape"]
        latency = shape_result["latency"]
        correctness = shape_result["correctness"]
        results.append(
            {
                "paper_baseline_run_id": "thunderkittens_tile_kernel",
                "benchmark_id": "tensor_core_tile",
                "hardware": {
                    "gpu": (
                        "H200"
                        if "H200" in gpu_metadata["gpu"]
                        else gpu_metadata["gpu"]
                    ),
                    "machine": args.machine,
                    "compute_target": gpu_metadata["compute_target"],
                    "driver": gpu_metadata["driver"],
                    "cuda_toolkit": args.cuda_toolkit,
                    "clock_policy": args.clock_policy,
                },
                "inputs": {
                    "shape": (
                        "mha_h100,"
                        f"b={shape['b']},h={shape['h']},n={shape['n']},"
                        f"d={shape['d']},causal={shape['causal']}"
                    ),
                    "dtype": shape["dtype"],
                    "repeat_policy": (
                        f"{latency['warmup']} warmup, {latency['repeats']} timed "
                        "CUDA-event repeats"
                    ),
                },
                "metrics": {
                    "kind": "paper_baseline_capture",
                    "sample_count": latency["sample_count"],
                    "host_wall_ns": 0,
                    "device_wall_ns": int(latency["p50_ns"]),
                },
                "correctness": correctness["status"],
            }
        )

    payload = {
        "metadata": {
            "pto_commit": args.pto_commit,
            "baseline": "thunderkittens",
            "baseline_kernel": "kernels/attention/mha_h100",
            "baseline_dir": str(args.baseline_dir),
            "cuda_toolkit": args.cuda_toolkit,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "machine": args.machine,
            "gpu": gpu_metadata,
        },
        "raw_shape_results": raw_shape_results,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


if __name__ == "__main__":
    main()
