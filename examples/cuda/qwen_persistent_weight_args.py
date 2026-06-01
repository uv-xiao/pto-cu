#!/usr/bin/env python3
"""Emit Qwen persistent-device weight argument binding evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen_persistent_weight_args_impl.builder import (  # noqa: E402
    build_weight_arg_manifest,
)
from qwen_persistent_weight_args_impl.common import (  # noqa: E402
    ABI_PATH,
    DEFAULT_WEIGHT_BINDING,
    MODEL_ID,
    MODEL_REVISION,
    ROOT,
    TENSOR_ARG_CAPACITY,
    load_json,
    load_python_payload,
    repo_relative,
    write_json,
)
from qwen_persistent_weight_args_impl.descriptors import (  # noqa: E402
    build_task_descriptors,
    descriptor,
    layer_tensor,
    tensor_arg_records,
)
from qwen_persistent_weight_args_impl.loaders import (  # noqa: E402
    binding_map,
    load_or_build_weight_binding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-binding-json", type=Path)
    parser.add_argument("--num-hidden-layers", type=int, default=36)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_weight_arg_manifest(
        weight_binding_json=args.weight_binding_json,
        num_hidden_layers=args.num_hidden_layers,
    )
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
