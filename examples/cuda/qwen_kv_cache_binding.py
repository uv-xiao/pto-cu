#!/usr/bin/env python3
"""Emit Qwen KV-cache pointer binding evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from qwen_kv_cache_binding_impl.lifecycle import (  # noqa: E402
    build_kv_cache_lifecycle,
    repo_relative,
    write_json,
)


EVIDENCE_SYMBOLS = [
    "pto_qwen_cuda_kv_cache_lifecycle",
    "kv_cache_key_value_field_binding",
    "PtoCudaPersistentDagTask::c",
    "PtoCudaPersistentDagTask::d",
    "cuda_live_kv_cache_owner",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pointer-base",
        type=lambda text: int(text, 0),
        default=0x70000000,
    )
    parser.add_argument(
        "--pointer-stride",
        type=lambda text: int(text, 0),
        default=0x10000000,
    )
    parser.add_argument("--cuda-live", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--host-runtime", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_kv_cache_lifecycle(
        pointer_base=args.pointer_base,
        pointer_stride=args.pointer_stride,
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
