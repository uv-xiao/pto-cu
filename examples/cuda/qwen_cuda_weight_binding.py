#!/usr/bin/env python3
"""Build Qwen safetensors-to-CUDA weight binding evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen_cuda_weight_binding_impl.builder import build_weight_binding  # noqa: E402
from qwen_cuda_weight_binding_impl.common import (  # noqa: E402
    DEFAULT_COPY_CHUNK_BYTES,
    DEFAULT_HOST_RUNTIME,
    DEFAULT_INDEX,
    DEFAULT_SHARD_DIR,
    MODEL_ID,
    MODEL_REVISION,
    load_json,
    load_python_payload,
    repo_relative,
    write_json,
)
from qwen_cuda_weight_binding_impl.cuda_runtime import (  # noqa: E402
    copy_file_range_to_device,
    load_cuda_runtime,
    read_tensor_bytes,
    run_cuda_copy_probe,
    verify_device_prefix,
)
from qwen_cuda_weight_binding_impl.full_residency import (  # noqa: E402
    run_cuda_full_residency_probe,
)
from qwen_cuda_weight_binding_impl.safetensors import (  # noqa: E402
    binding_group_for_tensor,
    build_bindings,
    inventory_group_map,
    inventory_tensor_contracts,
    load_or_build_inventory,
    load_or_build_metadata,
    normalize_dtype,
    read_safetensors_header,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-json", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--weight-inventory-json", type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--host-runtime", type=Path, default=DEFAULT_HOST_RUNTIME)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--cuda-probe-mode",
        choices=("bounded", "full"),
        default="bounded",
    )
    parser.add_argument("--max-probe-tensor-bytes", type=int, default=16 * 1024)
    parser.add_argument("--max-probe-total-bytes", type=int, default=256 * 1024)
    parser.add_argument("--max-probe-tensors", type=int, default=16)
    parser.add_argument("--copy-chunk-bytes", type=int, default=DEFAULT_COPY_CHUNK_BYTES)
    parser.add_argument("--verify-tensors", type=int, default=8)
    parser.add_argument("--verify-bytes", type=int, default=4096)
    parser.add_argument("--no-cuda-probe", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_weight_binding(
        index_json=args.index_json,
        weight_inventory_json=args.weight_inventory_json,
        metadata_json=args.metadata_json,
        shard_dir=args.shard_dir,
        no_cuda_probe=args.no_cuda_probe,
        device=args.device,
        host_runtime=args.host_runtime,
        cuda_probe_mode=args.cuda_probe_mode,
        max_probe_tensor_bytes=args.max_probe_tensor_bytes,
        max_probe_total_bytes=args.max_probe_total_bytes,
        max_probe_tensors=args.max_probe_tensors,
        copy_chunk_bytes=args.copy_chunk_bytes,
        verify_tensors=args.verify_tensors,
        verify_bytes=args.verify_bytes,
    )
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
