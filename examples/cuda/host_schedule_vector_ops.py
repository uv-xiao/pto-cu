#!/usr/bin/env python3
"""Describe the historical CUDA host_schedule vector benchmark row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_CAPTURE_COMMIT = "743709f3"
HOST_SCHEDULE_PROVENANCE = {
    "runtime": "host_schedule",
    "benchmark_id": "host_schedule_vector_ops",
    "historical_capture_commit": HISTORICAL_CAPTURE_COMMIT,
    "evidence_refs": [
        "docs/nvidia-backend/history/captures/current-head-layered-cross-743709f3.md",
        "src/cuda/platform/onboard/host/pto_runtime_c_api.cpp",
        "src/cuda/platform/include/host/pto_cuda_host_schedule_abi.h",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--block-dim", type=int, default=256)
    parser.add_argument("--arch", default="compute_80")
    parser.add_argument(
        "--op",
        choices=(
            "add",
            "mul",
            "scale",
            "square",
            "axpy",
            "affine",
            "triad",
            "quad",
            "generic_args",
            "generic_args4",
        ),
        default="add",
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--describe",
        action="store_true",
        help="emit review provenance for the historical benchmark row",
    )
    args = parser.parse_args()

    payload = {
        **HOST_SCHEDULE_PROVENANCE,
        "status": "historical_provenance_only",
        "device": args.device,
        "n": args.n,
        "block_dim": args.block_dim,
        "arch": args.arch,
        "op": args.op,
        "build_requested": args.build,
        "description": (
            "The active smoke runner used for the 743709f3 capture was removed "
            "when the CUDA eval skill was slimmed. This example now preserves "
            "the review-facing row metadata without claiming a fresh CUDA run."
        ),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
