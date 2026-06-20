#!/usr/bin/env python3
"""Describe the historical CUDA persistent layered-cross benchmark row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_CAPTURE_COMMIT = "743709f3"
GRAPH_LAYERED_CROSS_DESCRIPTOR = {
    "dispatch_func_ids": [1, 2, 11, 1, 2, 1, 6, 1, 1],
    "fanin": [0, 0, 0, 2, 3, 1, 2, 3, 2],
    "dependents": [3, 3, 4, 4, 5, 4, 6, 7, 6, 7, 7, 8, 8],
    "scalar0": 2.0,
    "tensor_alias": "c=a",
}
PERSISTENT_LAYERED_CROSS_PROVENANCE = {
    "runtime": "persistent_device",
    "benchmark_id": "graph_layered_cross",
    "historical_capture_commit": HISTORICAL_CAPTURE_COMMIT,
    "dag_shape": "graph_descriptor_layered_cross",
    "descriptor": GRAPH_LAYERED_CROSS_DESCRIPTOR,
    "evidence_refs": [
        "docs/nvidia-backend/history/captures/current-head-layered-cross-743709f3.md",
        "tests/ut/py/test_cuda_backend.py",
        "src/cuda/platform/include/host/pto_cuda_persistent_device_abi.h",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--arch", default="compute_80")
    parser.add_argument("--block-dim", type=int, default=256)
    parser.add_argument("--scheduler-blocks", type=int, default=3)
    parser.add_argument("--worker-blocks", type=int, default=4)
    parser.add_argument("--repeat-runs", type=int, default=2)
    parser.add_argument("--stream-id", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--describe",
        action="store_true",
        help="emit review provenance for the historical benchmark row",
    )
    args = parser.parse_args()

    payload = {
        **PERSISTENT_LAYERED_CROSS_PROVENANCE,
        "status": "historical_provenance_only",
        "device": args.device,
        "n": args.n,
        "arch": args.arch,
        "block_dim": args.block_dim,
        "scheduler_blocks": args.scheduler_blocks,
        "worker_blocks": args.worker_blocks,
        "repeat_runs": args.repeat_runs,
        "stream_id": args.stream_id,
        "description": (
            "The active persistent smoke runner used for the 743709f3 capture "
            "was removed when the CUDA eval skill was slimmed. This example "
            "now preserves the review-facing graph metadata without claiming "
            "a fresh CUDA run."
        ),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
