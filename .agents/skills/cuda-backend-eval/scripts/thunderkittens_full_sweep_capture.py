#!/usr/bin/env python3
"""Capture ThunderKittens selected full-sweep correctness and benchmark rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from thunderkittens_mha_capture import (  # noqa: E402
    parse_shape,
    read_gpu_metadata,
    run_shape,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def normalized_gpu_name(gpu_metadata: dict[str, str]) -> str:
    gpu = gpu_metadata["gpu"]
    return "H200" if "H200" in gpu else gpu


def attention_flops(shape: dict[str, Any]) -> int:
    batch = int(shape["b"])
    heads = int(shape["h"])
    seqlen = int(shape["n"])
    headdim = int(shape["d"])
    causal_divisor = 2 if shape["causal"] else 1
    return 4 * batch * seqlen * seqlen * heads * headdim // causal_divisor


def build_raw_result_record(
    *,
    machine: str,
    cuda_toolkit: str,
    clock_policy: str,
    gpu_metadata: dict[str, str],
    shape: dict[str, Any],
    latency: dict[str, Any],
    correctness: dict[str, Any],
) -> dict[str, Any]:
    elapsed_ns = int(latency["p50_ns"])
    flops = attention_flops(shape)
    throughput = int(flops * 1_000_000_000 / elapsed_ns)
    return {
        "paper_baseline_run_id": "thunderkittens_full_sweep",
        "benchmark_id": "tensor_core_tile",
        "hardware": {
            "gpu": normalized_gpu_name(gpu_metadata),
            "machine": machine,
            "compute_target": gpu_metadata["compute_target"],
            "driver": gpu_metadata["driver"],
            "cuda_toolkit": cuda_toolkit,
            "clock_policy": clock_policy,
        },
        "inputs": {
            "shape": (
                "mha_h100_full_sweep,"
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
            "kind": "paper_baseline_full_sweep_capture",
            "sample_count": latency["sample_count"],
            "host_wall_ns": 0,
            "device_wall_ns": elapsed_ns,
            "throughput": throughput,
            "attention_flops": flops,
            "max_abs_error": correctness["max_abs_diff"],
        },
        "correctness": correctness["status"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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

    sys.path.insert(0, str(args.baseline_dir.resolve()))
    import torch  # type: ignore[import-not-found]
    import _C as tk  # type: ignore[import-not-found]

    gpu_metadata = read_gpu_metadata()
    raw_results = []
    correctness_results = []
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
        raw_results.append(
            build_raw_result_record(
                machine=args.machine,
                cuda_toolkit=args.cuda_toolkit,
                clock_policy=args.clock_policy,
                gpu_metadata=gpu_metadata,
                shape=shape_result["shape"],
                latency=shape_result["latency"],
                correctness=shape_result["correctness"],
            )
        )
        correctness_results.append(
            {
                "shape": shape_result["shape"],
                "correctness": shape_result["correctness"],
            }
        )

    metadata = {
        "pto_commit": args.pto_commit,
        "baseline": "thunderkittens",
        "kernel": "kernels/attention/mha_h100",
        "machine": args.machine,
        "cuda_toolkit": args.cuda_toolkit,
        "clock_policy": args.clock_policy,
        "gpu_metadata": gpu_metadata,
    }
    write_json(
        args.output_dir / "correctness.json",
        {
            "metadata": metadata,
            "correctness_results": correctness_results,
        },
    )
    write_json(
        args.output_dir / "benchmark.json",
        {
            "metadata": metadata,
            "results": raw_results,
        },
    )
    print(args.output_dir / "correctness.json")
    print(args.output_dir / "benchmark.json")


if __name__ == "__main__":
    main()
