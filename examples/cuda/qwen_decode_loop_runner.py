#!/usr/bin/env python3
"""Emit Qwen persistent decode-loop runner integration evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from qwen_decode_loop_runner_impl.lifecycle import (  # noqa: E402
    build_decode_loop_runner,
    repo_relative,
    write_json,
)


EVIDENCE_SYMBOLS = [
    "pto_qwen_decode_loop_runner",
    "decode_loop_owner_lifetime_order",
    "persistent_dag_submission_plan",
    "output_token_accounting_plan",
    "cuda_live_resource_bridge_contract",
    "qwen_unit_math_live_bridge_contract",
    "cuda_live_token_pointer_table_in_runner",
    "cuda_live_kv_cache_owner_in_runner",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["offline", "download", "mock"],
        default="offline",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--unit-math-live-json", type=Path)
    parser.add_argument("--run-unit-math-live", action="store_true")
    parser.add_argument("--token-cuda-live", action="store_true")
    parser.add_argument("--kv-cuda-live", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arch", default="compute_80")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--host-runtime", type=Path)
    parser.add_argument("--build-runtime", action="store_true")
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def load_unit_math_payload(args: argparse.Namespace) -> dict | None:
    if args.unit_math_live_json:
        return json.loads(args.unit_math_live_json.read_text(encoding="utf-8"))
    if not args.run_unit_math_live:
        return None
    from qwen_unit_math_live_impl.runner import run_unit_math_live

    return run_unit_math_live(
        device=args.device,
        arch=args.arch,
        cache_root=args.cache_root,
        build_runtime=args.build_runtime,
        repeat_runs=args.repeat_runs,
    )


def main() -> None:
    args = parse_args()
    payload = build_decode_loop_runner(
        mode=args.mode,
        cache_dir=args.cache_dir,
        unit_math_live_payload=load_unit_math_payload(args),
        token_cuda_live=args.token_cuda_live,
        kv_cuda_live=args.kv_cuda_live,
        device=args.device,
        host_runtime=args.host_runtime,
    )
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
