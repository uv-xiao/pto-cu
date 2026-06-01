#!/usr/bin/env python3
"""Run controlled Qwen QKV proxy evidence on cuda/persistent_device."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from qwen_persistent_proxy_live_impl.live import (  # noqa: E402
    build_live_proxy_plan,
    repo_relative,
    run_live_proxy,
    write_json,
)


EVIDENCE_SYMBOLS = [
    "pto_qwen_proxy_live_execution",
    "controlled_proxy_live_cuda_execution",
    "persistent_device_single_task_dag_launch_plan",
    "qwen_attention_qkv_mutable_kv_live_contract",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arch", default="compute_80")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--build-runtime", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plan_only:
        payload = build_live_proxy_plan()
    else:
        payload = run_live_proxy(
            device=args.device,
            arch=args.arch,
            cache_root=args.cache_root,
            build_runtime=args.build_runtime,
        )
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
