#!/usr/bin/env python3
"""Emit Qwen resident weight pointer table lifecycle evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from qwen_resident_weight_table_impl.lifecycle import (  # noqa: E402
    DEFAULT_COPY_CHUNK_BYTES,
    DEFAULT_HOST_RUNTIME,
    ResidentWeightTableOwner,
    build_resident_table_lifecycle,
    write_json,
)


EVIDENCE_SYMBOLS = [
    "pto_qwen_resident_weight_table_lifecycle",
    "live_resident_weight_table_owner",
    "resident_pointer_table_materialization_bridge",
    "dry_run_pointer_lifecycle",
]


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-binding-json", type=Path)
    parser.add_argument("--weight-args-json", type=Path)
    parser.add_argument("--cuda-live", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--host-runtime", type=Path, default=DEFAULT_HOST_RUNTIME)
    parser.add_argument(
        "--copy-chunk-bytes",
        type=int,
        default=DEFAULT_COPY_CHUNK_BYTES,
    )
    parser.add_argument(
        "--pointer-base",
        type=lambda text: int(text, 0),
        default=0x50000000,
    )
    parser.add_argument(
        "--pointer-stride",
        type=lambda text: int(text, 0),
        default=0x100000,
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_resident_table_lifecycle(
        weight_binding_json=args.weight_binding_json,
        weight_args_json=args.weight_args_json,
        dry_run=not args.cuda_live,
        device=args.device,
        host_runtime=args.host_runtime,
        pointer_base=args.pointer_base,
        pointer_stride=args.pointer_stride,
        copy_chunk_bytes=args.copy_chunk_bytes,
    )
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
