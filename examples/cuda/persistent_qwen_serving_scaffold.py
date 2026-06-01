#!/usr/bin/env python3
"""Emit the PTO CUDA persistent-device Qwen serving lifecycle scaffold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from persistent_qwen_serving_scaffold_impl.builder import build_scaffold  # noqa: E402
from persistent_qwen_serving_scaffold_impl.common import (  # noqa: E402
    repo_relative,
    write_json,
)
from persistent_qwen_serving_scaffold_impl.loaders import (  # noqa: E402
    load_cuda_token_buffer_binding,
    load_cuda_weight_binding,
    load_decode_loop_runner,
    load_kv_cache_binding,
    load_lifecycle_plan,
    load_persistent_decode_args,
    load_persistent_weight_args,
    load_persistent_weight_materialization,
    load_prompt_accounting,
    load_resident_weight_table,
    load_runtime_input_binding,
    load_safetensors_metadata,
    load_safetensors_shards,
    load_task_bodies,
    load_token_pointer_table,
    load_weight_inventory,
)
from persistent_qwen_serving_scaffold_impl.stages import (  # noqa: E402
    serving_workload_contracts,
    stage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_scaffold()
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
