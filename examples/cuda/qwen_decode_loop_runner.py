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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["offline", "download", "mock"],
        default="offline",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_decode_loop_runner(
        mode=args.mode,
        cache_dir=args.cache_dir,
    )
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
