#!/usr/bin/env python3
"""Emit Qwen persistent-device weight argument binding evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHT_BINDING = (
    ROOT
    / "tmp"
    / "cuda-backend"
    / "pto-serving-weight-residency-1ae913c9"
    / "qwen-cuda-weight-residency.json"
)
ABI_PATH = "src/cuda/platform/include/host/pto_cuda_persistent_device_abi.h"
TENSOR_ARG_CAPACITY = 4
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "d117af2f304f02a8647f88fe05b61cfb405a1d9e"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_python_payload(
    path: Path,
    module_name: str,
    build_name: str,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, build_name)(*args, **kwargs)


def load_or_build_weight_binding(
    weight_binding_json: Path | None,
) -> tuple[dict[str, Any], str]:
    if weight_binding_json is not None:
        return load_json(weight_binding_json), repo_relative(weight_binding_json)
    if DEFAULT_WEIGHT_BINDING.is_file():
        return load_json(DEFAULT_WEIGHT_BINDING), repo_relative(DEFAULT_WEIGHT_BINDING)
    payload = load_python_payload(
        ROOT / "examples" / "cuda" / "qwen_cuda_weight_binding.py",
        "qwen_cuda_weight_binding",
        "build_weight_binding",
        no_cuda_probe=True,
    )
    return payload, "generated_from_examples/cuda/qwen_cuda_weight_binding.py"


def binding_map(weight_binding: dict[str, Any]) -> dict[str, dict[str, Any]]:
    bindings = weight_binding.get("bindings", [])
    if not isinstance(bindings, list):
        raise ValueError("weight binding artifact has no bindings list")
    result = {}
    for item in bindings:
        if not isinstance(item, dict) or not isinstance(item.get("tensor"), str):
            raise ValueError("weight binding contains a malformed binding record")
        result[item["tensor"]] = item
    return result


def layer_tensor(layer: int, suffix: str) -> str:
    return f"model.layers.{layer}.{suffix}.weight"


def tensor_arg_records(
    tensors: list[str],
    bindings: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for index, tensor in enumerate(tensors):
        item = bindings.get(tensor)
        if item is None:
            records.append(
                {
                    "arg": f"tensor_args[{index}]",
                    "tensor": tensor,
                    "status": "missing_weight_binding",
                }
            )
            continue
        records.append(
            {
                "arg": f"tensor_args[{index}]",
                "slot_id": item["slot_id"],
                "tensor": tensor,
            }
        )
    return records


def descriptor(
    *,
    descriptor_id: str,
    callable_name: str,
    phase: str,
    tensors: list[str],
    bindings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    tensor_args = tensor_arg_records(tensors, bindings)
    missing = [
        item["tensor"]
        for item in tensor_args
        if item.get("status") == "missing_weight_binding"
    ]
    return {
        "id": descriptor_id,
        "callable": callable_name,
        "phase": phase,
        "tensor_arg_count": len(tensor_args),
        "tensor_args": tensor_args,
        "status": "ready" if not missing else "missing_weight_binding",
        "missing_tensors": missing,
    }


def build_task_descriptors(
    *,
    bindings: dict[str, dict[str, Any]],
    num_hidden_layers: int,
) -> list[dict[str, Any]]:
    descriptors = [
        descriptor(
            descriptor_id="embedding_lookup",
            callable_name="qwen_embedding_lookup",
            phase="prefill_or_decode_input",
            tensors=["model.embed_tokens.weight"],
            bindings=bindings,
        )
    ]
    for layer in range(num_hidden_layers):
        prefix = f"layer_{layer}"
        descriptors.extend(
            [
                descriptor(
                    descriptor_id=f"{prefix}_input_norm",
                    callable_name="qwen_rmsnorm_input",
                    phase="per_layer_decode",
                    tensors=[layer_tensor(layer, "input_layernorm")],
                    bindings=bindings,
                ),
                descriptor(
                    descriptor_id=f"{prefix}_attention_qkv",
                    callable_name="qwen_attention_qkv",
                    phase="per_layer_decode",
                    tensors=[
                        layer_tensor(layer, "self_attn.q_proj"),
                        layer_tensor(layer, "self_attn.k_proj"),
                        layer_tensor(layer, "self_attn.v_proj"),
                    ],
                    bindings=bindings,
                ),
                descriptor(
                    descriptor_id=f"{prefix}_attention_qk_norm",
                    callable_name="qwen_attention_qk_norm",
                    phase="per_layer_decode",
                    tensors=[
                        layer_tensor(layer, "self_attn.q_norm"),
                        layer_tensor(layer, "self_attn.k_norm"),
                    ],
                    bindings=bindings,
                ),
                descriptor(
                    descriptor_id=f"{prefix}_attention_o",
                    callable_name="qwen_attention_o",
                    phase="per_layer_decode",
                    tensors=[layer_tensor(layer, "self_attn.o_proj")],
                    bindings=bindings,
                ),
                descriptor(
                    descriptor_id=f"{prefix}_post_attention_norm",
                    callable_name="qwen_rmsnorm_post_attention",
                    phase="per_layer_decode",
                    tensors=[layer_tensor(layer, "post_attention_layernorm")],
                    bindings=bindings,
                ),
                descriptor(
                    descriptor_id=f"{prefix}_mlp_gate_up",
                    callable_name="qwen_mlp_gate_up",
                    phase="per_layer_decode",
                    tensors=[
                        layer_tensor(layer, "mlp.gate_proj"),
                        layer_tensor(layer, "mlp.up_proj"),
                    ],
                    bindings=bindings,
                ),
                descriptor(
                    descriptor_id=f"{prefix}_mlp_down",
                    callable_name="qwen_mlp_down",
                    phase="per_layer_decode",
                    tensors=[layer_tensor(layer, "mlp.down_proj")],
                    bindings=bindings,
                ),
            ]
        )
    descriptors.extend(
        [
            descriptor(
                descriptor_id="final_norm",
                callable_name="qwen_final_norm",
                phase="per_token_decode",
                tensors=["model.norm.weight"],
                bindings=bindings,
            ),
            descriptor(
                descriptor_id="logits",
                callable_name="qwen_logits",
                phase="per_token_decode",
                tensors=["lm_head.weight"],
                bindings=bindings,
            ),
        ]
    )
    return descriptors


def build_weight_arg_manifest(
    *,
    weight_binding_json: Path | None = None,
    num_hidden_layers: int = 36,
) -> dict[str, Any]:
    weight_binding, source = load_or_build_weight_binding(weight_binding_json)
    bindings = binding_map(weight_binding)
    descriptors = build_task_descriptors(
        bindings=bindings,
        num_hidden_layers=num_hidden_layers,
    )
    missing = sorted(
        {
            tensor
            for item in descriptors
            for tensor in item.get("missing_tensors", [])
        }
    )
    covered = sorted(
        {
            arg["tensor"]
            for item in descriptors
            for arg in item["tensor_args"]
            if "slot_id" in arg
        }
    )
    uncovered_bindings = sorted(set(bindings) - set(covered))
    max_tensor_args = max(
        (item["tensor_arg_count"] for item in descriptors),
        default=0,
    )
    capacity_ok = max_tensor_args <= TENSOR_ARG_CAPACITY
    complete = not missing and not uncovered_bindings and capacity_ok
    implemented_contracts = [
        "qwen_weight_task_decomposition",
        "persistent_dag_tensor_arg_capacity_check",
    ]
    if complete:
        implemented_contracts.append("persistent_task_weight_arg_binding_manifest")
    return {
        "schema_version": 1,
        "kind": "pto_qwen_persistent_weight_args",
        "status": (
            "persistent_weight_args_ready"
            if complete
            else "persistent_weight_args_incomplete"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "weight_binding_json": source,
        "weight_binding_status": weight_binding.get("status"),
        "weight_cuda_probe": weight_binding.get("cuda_probe", {}),
        "abi": {
            "task_struct": "PtoCudaPersistentDagTask",
            "source": ABI_PATH,
            "tensor_arg_field": "tensor_args",
            "tensor_arg_capacity": TENSOR_ARG_CAPACITY,
        },
        "num_hidden_layers": num_hidden_layers,
        "task_arg_descriptor_count": len(descriptors),
        "task_arg_descriptors": descriptors,
        "covered_tensor_count": len(covered),
        "missing_tensor_count": len(missing),
        "missing_tensors": missing,
        "uncovered_binding_count": len(uncovered_bindings),
        "uncovered_bindings": uncovered_bindings[:32],
        "max_tensor_args_per_task": max_tensor_args,
        "implemented_contracts": implemented_contracts,
        "remaining_runtime_gaps": [
            "persistent_task_weight_arg_runtime_binding",
            "qwen_kernel_weight_consumption",
        ],
    }


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
