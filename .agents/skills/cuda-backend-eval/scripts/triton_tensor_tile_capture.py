#!/usr/bin/env python3
"""Capture a Triton 16x16x16 tensor-tile baseline for the viewer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from triton_tensor_tile_capture_impl.constants import DEFAULT_DTYPE  # noqa: E402
from triton_tensor_tile_capture_impl.constants import DEFAULT_SHAPE  # noqa: E402
from triton_tensor_tile_capture_impl.constants import DEFAULT_TOLERANCE  # noqa: E402
from triton_tensor_tile_capture_impl.errors import fail  # noqa: E402
from triton_tensor_tile_capture_impl.io import load_json  # noqa: E402
from triton_tensor_tile_capture_impl.io import repo_relative  # noqa: E402
from triton_tensor_tile_capture_impl.io import write_json  # noqa: E402
from triton_tensor_tile_capture_impl.record import require_dict  # noqa: E402
from triton_tensor_tile_capture_impl.record import require_samples  # noqa: E402
from triton_tensor_tile_capture_impl.record import require_string  # noqa: E402
from triton_tensor_tile_capture_impl.record import viewer_record  # noqa: E402
from triton_tensor_tile_capture_impl.run import driver_version  # noqa: E402
from triton_tensor_tile_capture_impl.run import git_commit  # noqa: E402
from triton_tensor_tile_capture_impl.run import run_capture  # noqa: E402
from triton_tensor_tile_capture_impl.stats import latency_summary  # noqa: E402
from triton_tensor_tile_capture_impl.stats import percentile_int  # noqa: E402


def tensor_tile_shape(rows: int, cols: int, inner: int) -> str:
    return f"n=1024, tensor tile {rows}x{cols}x{inner}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, help="Convert an existing raw capture.")
    parser.add_argument("--output", type=Path, help="Write raw capture JSON here.")
    parser.add_argument("--viewer-output", type=Path, help="Write viewer result records here.")
    parser.add_argument("--artifact-root", help="Repo-relative raw artifact path.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--inner", type=int, default=16)
    parser.add_argument("--tile-count", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--pto-commit", help="Commit label to store in the raw artifact.")
    args = parser.parse_args()
    for key in ("rows", "cols", "inner", "tile_count", "warmup", "repeats"):
        value = getattr(args, key)
        if value <= 0:
            fail(f"--{key.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    payload = load_json(args.input_json) if args.input_json else run_capture(args)
    if args.output:
        write_json(args.output, payload)
    raw_artifact = args.artifact_root
    if raw_artifact is None:
        source = args.output or args.input_json
        raw_artifact = repo_relative(source.parent) + "/" if source else "tmp/"
    records = [viewer_record(payload, raw_artifact)]
    if args.viewer_output:
        write_json(args.viewer_output, records)
    elif not args.output:
        print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
