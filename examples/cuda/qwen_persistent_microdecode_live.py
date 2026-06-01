#!/usr/bin/env python3
"""Run controlled Qwen microdecode proxy evidence on cuda/persistent_device."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from qwen_persistent_microdecode_live_impl.live import (  # noqa: E402
    build_live_microdecode_plan,
    repo_relative,
    run_live_microdecode,
    write_json,
)


EVIDENCE_SYMBOLS = [
    "pto_qwen_microdecode_live_execution",
    "controlled_proxy_live_microdecode_execution",
    "controlled_proxy_live_decode_loop_execution",
    "persistent_device_proxy_decode_chain_plan",
    "qwen_attention_to_logits_dag_contract",
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
        payload = build_live_microdecode_plan(repeat_runs=args.repeat_runs)
    else:
        payload = run_live_microdecode(
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
