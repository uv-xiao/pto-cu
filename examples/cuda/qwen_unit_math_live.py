#!/usr/bin/env python3
"""Run Qwen unit-math task bodies on cuda/persistent_device."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from qwen_unit_math_live_impl.plan import (  # noqa: E402
    build_unit_math_live_plan,
    repo_relative,
    write_json,
)
from qwen_unit_math_live_impl.runner import run_unit_math_live  # noqa: E402


EVIDENCE_SYMBOLS = [
    "pto_qwen_unit_math_live_execution",
    "qwen_unit_math_cuda_live_execution_plan",
    "qwen_unit_math_cuda_live_execution",
    "qwen_unit_math_decode_loop_reuse_execution",
    "persistent_device_unit_math_dag_launch_plan",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arch", default="compute_80")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--build-runtime", action="store_true")
    parser.add_argument("--repeat-runs", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plan_only:
        payload = build_unit_math_live_plan(repeat_runs=args.repeat_runs)
    else:
        payload = run_unit_math_live(
            device=args.device,
            arch=args.arch,
            cache_root=args.cache_root,
            build_runtime=args.build_runtime,
            repeat_runs=args.repeat_runs,
        )
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
