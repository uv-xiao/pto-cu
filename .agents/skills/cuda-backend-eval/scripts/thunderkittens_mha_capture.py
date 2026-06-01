#!/usr/bin/env python3
"""Capture bounded ThunderKittens MHA correctness and latency evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from thunderkittens_mha_capture_impl.gpu import read_gpu_metadata  # noqa: E402
from thunderkittens_mha_capture_impl.records import (  # noqa: E402
    build_raw_result_record,
)
from thunderkittens_mha_capture_impl.records import normalized_gpu_name  # noqa: E402
from thunderkittens_mha_capture_impl.run import run_shape  # noqa: E402
from thunderkittens_mha_capture_impl.shapes import parse_shape  # noqa: E402
from thunderkittens_mha_capture_impl.stats import percentile  # noqa: E402
from thunderkittens_mha_capture_impl.stats import summarize_ns  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--machine", default="unknown")
    parser.add_argument("--pto-commit", required=True)
    parser.add_argument("--cuda-toolkit", default="unknown")
    parser.add_argument("--clock-policy", default="not recorded")
    parser.add_argument(
        "--paper-baseline-run-id",
        default="thunderkittens_tile_kernel",
    )
    parser.add_argument("--benchmark-id", default="tensor_core_tile")
    parser.add_argument("--serving-workload-id", default="")
    parser.add_argument("--prompt-tokens", type=int, default=0)
    parser.add_argument("--decode-tokens", type=int, default=0)
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
        results.append(
            build_raw_result_record(
                paper_baseline_run_id=args.paper_baseline_run_id,
                benchmark_id=args.benchmark_id,
                machine=args.machine,
                cuda_toolkit=args.cuda_toolkit,
                clock_policy=args.clock_policy,
                gpu_metadata=gpu_metadata,
                shape=shape_result["shape"],
                latency=shape_result["latency"],
                correctness=shape_result["correctness"],
                serving_workload_id=args.serving_workload_id,
                prompt_tokens=args.prompt_tokens,
                decode_tokens=args.decode_tokens,
            )
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
