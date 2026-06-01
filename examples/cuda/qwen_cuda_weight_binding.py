#!/usr/bin/env python3
"""Build Qwen safetensors-to-CUDA weight binding evidence."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = (
    ROOT / "tmp" / "sources" / "qwen3-8b-model-safetensors-index-d117af2f.json"
)
DEFAULT_SHARD_DIR = ROOT / "tmp" / "sources" / "qwen3-8b-safetensors"
DEFAULT_HOST_RUNTIME = (
    ROOT / "build" / "lib" / "cuda" / "onboard" / "host_schedule" / "libhost_runtime.so"
)
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "d117af2f304f02a8647f88fe05b61cfb405a1d9e"
DTYPE_ALIASES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
}


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


def read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"{path} is too short for a safetensors header")
        header_size = struct.unpack("<Q", size_bytes)[0]
        header_bytes = handle.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"{path} ended before safetensors header finished")
    header = json.loads(header_bytes.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"{path} safetensors header is not a JSON object")
    return header_size, header


def normalize_dtype(dtype: Any) -> str:
    return DTYPE_ALIASES.get(str(dtype), str(dtype))


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


def load_or_build_inventory(
    *,
    index_json: Path,
    weight_inventory_json: Path | None,
) -> tuple[dict[str, Any], str]:
    if weight_inventory_json is not None:
        return load_json(weight_inventory_json), repo_relative(weight_inventory_json)
    payload = load_python_payload(
        ROOT / "examples" / "cuda" / "qwen_weight_inventory.py",
        "qwen_weight_inventory",
        "build_weight_inventory",
        index_json,
    )
    return payload, "generated_from_examples/cuda/qwen_weight_inventory.py"


def load_or_build_metadata(
    *,
    index_json: Path,
    weight_inventory_json: Path | None,
    metadata_json: Path | None,
    shard_dir: Path,
) -> tuple[dict[str, Any], str]:
    if metadata_json is not None:
        return load_json(metadata_json), repo_relative(metadata_json)
    payload = load_python_payload(
        ROOT / "examples" / "cuda" / "qwen_safetensors_metadata.py",
        "qwen_safetensors_metadata",
        "build_metadata_probe",
        index_json=index_json,
        weight_inventory_json=weight_inventory_json,
        shard_dir=shard_dir,
    )
    return payload, "generated_from_examples/cuda/qwen_safetensors_metadata.py"


def inventory_tensor_contracts(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contract = inventory.get("weight_shape_contract", {})
    tensors = contract.get("tensor_shapes", [])
    if not isinstance(tensors, list):
        raise ValueError("weight inventory has no tensor_shapes list")
    return {
        item["name"]: item
        for item in tensors
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def inventory_group_map(inventory: dict[str, Any]) -> dict[str, str]:
    groups = {}
    for group in inventory.get("binding_groups", []):
        if not isinstance(group, dict):
            continue
        group_id = group.get("id")
        if not isinstance(group_id, str):
            continue
        for tensor in group.get("sample_tensors", []):
            if isinstance(tensor, str):
                groups[tensor] = group_id
    return groups


def binding_group_for_tensor(
    tensor_name: str,
    known_groups: dict[str, str],
) -> str:
    if tensor_name in known_groups:
        return known_groups[tensor_name]
    if tensor_name.startswith("model.embed_tokens."):
        return "embedding"
    if any(
        marker in tensor_name
        for marker in (
            ".self_attn.q_proj.",
            ".self_attn.k_proj.",
            ".self_attn.v_proj.",
            ".self_attn.o_proj.",
        )
    ):
        return "attention_qkv_o"
    if any(
        marker in tensor_name
        for marker in (
            ".input_layernorm.",
            ".post_attention_layernorm.",
            ".self_attn.q_norm.",
            ".self_attn.k_norm.",
        )
    ):
        return "attention_norms"
    if any(
        marker in tensor_name
        for marker in (
            ".mlp.gate_proj.",
            ".mlp.up_proj.",
            ".mlp.down_proj.",
        )
    ):
        return "mlp_gate_up_down"
    if tensor_name.startswith("model.norm.") or tensor_name.startswith("lm_head."):
        return "norm_and_logits"
    return "unclassified"


def build_bindings(
    *,
    index_json: Path,
    weight_inventory_json: Path | None,
    metadata_json: Path | None,
    shard_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index = load_json(index_json)
    inventory, inventory_source = load_or_build_inventory(
        index_json=index_json,
        weight_inventory_json=weight_inventory_json,
    )
    metadata, metadata_source = load_or_build_metadata(
        index_json=index_json,
        weight_inventory_json=weight_inventory_json,
        metadata_json=metadata_json,
        shard_dir=shard_dir,
    )
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ValueError(f"weight_map is not an object in {index_json}")

    expected = inventory_tensor_contracts(inventory)
    known_groups = inventory_group_map(inventory)
    opened_headers = {}
    for shard_name in sorted({str(shard) for shard in weight_map.values()}):
        path = shard_dir / shard_name
        if not path.is_file():
            continue
        header_size, header = read_safetensors_header(path)
        opened_headers[shard_name] = {
            "path": path,
            "header_size": header_size,
            "data_base_offset": 8 + header_size,
            "tensors": {
                key: value
                for key, value in header.items()
                if key != "__metadata__" and isinstance(value, dict)
            },
        }

    bindings = []
    mismatches = []
    for slot_id, (tensor_name, shard_name_any) in enumerate(sorted(weight_map.items())):
        shard_name = str(shard_name_any)
        shard = opened_headers.get(shard_name)
        header_entry = shard["tensors"].get(tensor_name) if shard else None
        expected_entry = expected.get(tensor_name)
        if shard is None or header_entry is None or expected_entry is None:
            mismatches.append(tensor_name)
            continue
        offsets = header_entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
            or offsets[1] < offsets[0]
        ):
            mismatches.append(tensor_name)
            continue
        dtype = normalize_dtype(header_entry.get("dtype"))
        shape = header_entry.get("shape")
        if shape != expected_entry.get("shape") or dtype != expected_entry.get("dtype"):
            mismatches.append(tensor_name)
            continue
        absolute_offsets = [
            shard["data_base_offset"] + offsets[0],
            shard["data_base_offset"] + offsets[1],
        ]
        bindings.append(
            {
                "slot_id": slot_id,
                "tensor": tensor_name,
                "shard": shard_name,
                "shard_path": repo_relative(shard["path"]),
                "dtype": dtype,
                "shape": shape,
                "size_bytes": offsets[1] - offsets[0],
                "file_data_offsets": offsets,
                "file_absolute_offsets": absolute_offsets,
                "binding_group": binding_group_for_tensor(tensor_name, known_groups),
                "persistent_arg_role": "readonly_weight_tensor",
                "cuda_binding_state": "planned_not_resident",
            }
        )

    summary = {
        "metadata_status": metadata.get("status"),
        "index_tensor_count": len(weight_map),
        "planned_binding_count": len(bindings),
        "binding_mismatch_count": len(mismatches),
        "binding_mismatches": mismatches[:32],
        "total_weight_bytes": sum(item["size_bytes"] for item in bindings),
        "inventory_source": inventory_source,
        "metadata_source": metadata_source,
    }
    return summary, bindings


def read_tensor_bytes(binding: dict[str, Any]) -> bytes:
    path = ROOT / binding["shard_path"]
    if not path.is_file():
        path = Path(binding["shard_path"])
    start, end = binding["file_absolute_offsets"]
    with path.open("rb") as handle:
        handle.seek(start)
        data = handle.read(end - start)
    if len(data) != end - start:
        raise ValueError(f"{path} ended before {binding['tensor']} tensor bytes")
    return data


def load_cuda_runtime(host_runtime: Path) -> Any:
    runtime = ctypes.CDLL(str(host_runtime))
    runtime.create_device_context.restype = ctypes.c_void_p
    runtime.destroy_device_context.argtypes = [ctypes.c_void_p]
    runtime.simpler_init.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    runtime.simpler_init.restype = ctypes.c_int
    runtime.finalize_device.argtypes = [ctypes.c_void_p]
    runtime.finalize_device.restype = ctypes.c_int
    runtime.device_malloc_ctx.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    runtime.device_malloc_ctx.restype = ctypes.c_void_p
    runtime.device_free_ctx.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    runtime.copy_to_device_ctx.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    runtime.copy_to_device_ctx.restype = ctypes.c_int
    return runtime


def run_cuda_copy_probe(
    *,
    bindings: list[dict[str, Any]],
    device: int,
    host_runtime: Path,
    max_tensor_bytes: int,
    max_total_bytes: int,
    max_tensors: int,
) -> dict[str, Any]:
    candidates = [
        item
        for item in bindings
        if item["size_bytes"] <= max_tensor_bytes
    ]
    candidates.sort(key=lambda item: (item["size_bytes"], item["tensor"]))
    selected = []
    total = 0
    for item in candidates:
        if len(selected) >= max_tensors:
            break
        if total + item["size_bytes"] > max_total_bytes:
            continue
        selected.append(item)
        total += item["size_bytes"]

    if not selected:
        return {
            "status": "skipped",
            "reason": "no_tensor_fits_probe_limits",
            "max_tensor_bytes": max_tensor_bytes,
            "max_total_bytes": max_total_bytes,
            "max_tensors": max_tensors,
        }
    if not host_runtime.is_file():
        return {
            "status": "skipped",
            "reason": "host_runtime_missing",
            "host_runtime": repo_relative(host_runtime),
        }

    runtime = load_cuda_runtime(host_runtime)
    ctx = runtime.create_device_context()
    if not ctx:
        return {"status": "fail", "reason": "create_device_context_failed"}

    copied = []
    device_ptrs = []
    try:
        init_status = runtime.simpler_init(ctx, device, None, 0, None, 0)
        if init_status != 0:
            return {
                "status": "fail",
                "reason": "simpler_init_failed",
                "return_code": init_status,
            }
        for item in selected:
            data = read_tensor_bytes(item)
            buffer = ctypes.create_string_buffer(data, len(data))
            dev_ptr = runtime.device_malloc_ctx(ctx, len(data))
            if not dev_ptr:
                return {
                    "status": "fail",
                    "reason": "device_malloc_failed",
                    "tensor": item["tensor"],
                    "size_bytes": len(data),
                }
            device_ptrs.append(dev_ptr)
            copy_status = runtime.copy_to_device_ctx(
                ctx,
                dev_ptr,
                ctypes.cast(buffer, ctypes.c_void_p),
                len(data),
            )
            if copy_status != 0:
                return {
                    "status": "fail",
                    "reason": "copy_to_device_failed",
                    "tensor": item["tensor"],
                    "return_code": copy_status,
                }
            copied.append(
                {
                    "slot_id": item["slot_id"],
                    "tensor": item["tensor"],
                    "size_bytes": len(data),
                    "binding_group": item["binding_group"],
                }
            )
    except Exception as exc:  # noqa: BLE001 - artifact records the runtime error.
        return {
            "status": "fail",
            "reason": "exception",
            "message": str(exc),
        }
    finally:
        for dev_ptr in device_ptrs:
            runtime.device_free_ctx(ctx, dev_ptr)
        runtime.finalize_device(ctx)
        runtime.destroy_device_context(ctx)

    return {
        "status": "pass",
        "device": device,
        "host_runtime": repo_relative(host_runtime),
        "copied_tensor_count": len(copied),
        "copied_bytes": sum(item["size_bytes"] for item in copied),
        "copied_tensors": copied,
    }


def build_weight_binding(
    *,
    index_json: Path = DEFAULT_INDEX,
    weight_inventory_json: Path | None = None,
    metadata_json: Path | None = None,
    shard_dir: Path = DEFAULT_SHARD_DIR,
    no_cuda_probe: bool = False,
    device: int = 0,
    host_runtime: Path = DEFAULT_HOST_RUNTIME,
    max_probe_tensor_bytes: int = 16 * 1024,
    max_probe_total_bytes: int = 256 * 1024,
    max_probe_tensors: int = 16,
) -> dict[str, Any]:
    summary, bindings = build_bindings(
        index_json=index_json,
        weight_inventory_json=weight_inventory_json,
        metadata_json=metadata_json,
        shard_dir=shard_dir,
    )
    cuda_probe = (
        {
            "status": "skipped",
            "reason": "disabled_by_no_cuda_probe",
        }
        if no_cuda_probe
        else run_cuda_copy_probe(
            bindings=bindings,
            device=device,
            host_runtime=host_runtime,
            max_tensor_bytes=max_probe_tensor_bytes,
            max_total_bytes=max_probe_total_bytes,
            max_tensors=max_probe_tensors,
        )
    )
    implemented_contracts = [
        "safetensors_tensor_data_offsets",
        "persistent_task_weight_arg_binding_plan",
    ]
    if cuda_probe.get("status") == "pass":
        implemented_contracts.append("cuda_device_weight_copy_probe")
    status = (
        "binding_plan_ready"
        if summary["binding_mismatch_count"] == 0
        and summary["metadata_status"] == "metadata_validated"
        else "binding_plan_incomplete"
    )
    return {
        "schema_version": 1,
        "kind": "pto_qwen_cuda_weight_binding",
        "status": status,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "index_json": repo_relative(index_json),
        "weight_inventory_json": summary["inventory_source"],
        "metadata_json": summary["metadata_source"],
        "shard_dir": repo_relative(shard_dir),
        "metadata_status": summary["metadata_status"],
        "tensor_count": summary["index_tensor_count"],
        "planned_binding_count": summary["planned_binding_count"],
        "binding_mismatch_count": summary["binding_mismatch_count"],
        "binding_mismatches": summary["binding_mismatches"],
        "total_weight_bytes": summary["total_weight_bytes"],
        "cuda_probe": cuda_probe,
        "bindings": bindings,
        "implemented_contracts": implemented_contracts,
        "remaining_runtime_gaps": [
            "full_cuda_weight_residency",
            "persistent_task_weight_arg_runtime_binding",
            "qwen_kernel_weight_consumption",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-json", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--weight-inventory-json", type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--host-runtime", type=Path, default=DEFAULT_HOST_RUNTIME)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-probe-tensor-bytes", type=int, default=16 * 1024)
    parser.add_argument("--max-probe-total-bytes", type=int, default=256 * 1024)
    parser.add_argument("--max-probe-tensors", type=int, default=16)
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
        max_probe_tensor_bytes=args.max_probe_tensor_bytes,
        max_probe_total_bytes=args.max_probe_total_bytes,
        max_probe_tensors=args.max_probe_tensors,
    )
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
