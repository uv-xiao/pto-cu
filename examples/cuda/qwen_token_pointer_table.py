#!/usr/bin/env python3
"""Emit Qwen token pointer-table lifecycle evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from qwen_token_pointer_table_impl.common import (  # noqa: E402
    DEFAULT_HOST_RUNTIME,
    repo_relative,
    write_json,
)
from qwen_token_pointer_table_impl.lifecycle import (  # noqa: E402
    build_token_pointer_table_lifecycle,
)


EVIDENCE_SYMBOLS = [
    "pto_qwen_cuda_token_pointer_table_lifecycle",
    "live_token_pointer_table_owner",
    "persistent_decode_arg_materialization_during_lifetime",
    "dry_run_pointer_lifecycle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["offline", "download", "mock"],
        default="offline",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cuda-live", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--host-runtime", type=Path, default=DEFAULT_HOST_RUNTIME)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_token_pointer_table_lifecycle(
        mode=args.mode,
        cache_dir=args.cache_dir,
        cuda_live=args.cuda_live,
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
