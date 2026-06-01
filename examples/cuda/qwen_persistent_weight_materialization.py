#!/usr/bin/env python3
"""Materialize Qwen persistent DAG weight descriptors with resident pointers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen_persistent_weight_materialization_impl.abi import dag_task_abi  # noqa: E402
from qwen_persistent_weight_materialization_impl.builder import (  # noqa: E402
    build_materialization_manifest,
)
from qwen_persistent_weight_materialization_impl.common import (  # noqa: E402
    DEFAULT_WEIGHT_ARGS,
    DEFAULT_WEIGHT_BINDING,
    MODEL_ID,
    MODEL_REVISION,
    load_json,
    load_python_payload,
    repo_relative,
    write_json,
)
from qwen_persistent_weight_materialization_impl.loaders import (  # noqa: E402
    binding_map,
    load_or_build_weight_args,
    load_or_build_weight_binding,
    parse_device_ptr,
    pointer_map,
)
from qwen_persistent_weight_materialization_impl.materializer import (  # noqa: E402
    materialized_descriptor,
    materialized_tensor_arg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-args-json", type=Path)
    parser.add_argument("--weight-binding-json", type=Path)
    parser.add_argument("--pointer-table-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_materialization_manifest(
        weight_args_json=args.weight_args_json,
        weight_binding_json=args.weight_binding_json,
        pointer_table_json=args.pointer_table_json,
    )
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
