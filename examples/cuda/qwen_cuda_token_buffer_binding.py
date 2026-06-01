#!/usr/bin/env python3
"""Emit Qwen host-token to CUDA-buffer binding evidence."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_INPUT_BINDING = ROOT / "examples" / "cuda" / "qwen_runtime_input_binding.py"
CUDA_WEIGHT_BINDING = ROOT / "examples" / "cuda" / "qwen_cuda_weight_binding.py"
DEFAULT_HOST_RUNTIME = (
    ROOT / "build" / "lib" / "cuda" / "onboard" / "host_schedule" / "libhost_runtime.so"
)

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def load_runtime_input_binding(*, mode: str, cache_dir: Path | None) -> dict[str, Any]:
    module = load_module(RUNTIME_INPUT_BINDING, "qwen_runtime_input_binding_for_cuda")
    kwargs: dict[str, Any] = {"mode": mode}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    return module.build_runtime_input_binding(**kwargs)

def device_buffer(record: dict[str, Any], name: str) -> dict[str, Any]:
    source = record[f"{name}_buffer"]
    return {
        "name": name,
        "dtype": source["dtype"],
        "shape": source["shape"],
        "element_count": source["element_count"],
        "byte_count": source["byte_count"],
        "source_checksum": source["checksum"],
        "binding_state": "planned_cuda_allocation",
    }


def repeated_i32(values: list[int], repeat: int) -> bytes:
    payload = values * repeat
    array_type = ctypes.c_int32 * len(payload)
    return bytes(array_type(*payload))


def record_host_buffers(record: dict[str, Any]) -> dict[str, bytes]:
    max_batch = max(int(item) for item in record["batch_sizes"])
    runtime_ids = [int(item) for item in record["runtime_prompt_token_ids"]]
    observed = int(record["target_prompt_alignment"]["observed_prompt_tokens"])
    mask = [1] * observed + [0] * (len(runtime_ids) - observed)
    output_count = max_batch * int(record["decode_tokens"])
    output = [-1] * output_count
    return {
        "input_ids": repeated_i32(runtime_ids, max_batch),
        "attention_mask": repeated_i32(mask, max_batch),
        "output_ids": repeated_i32(output, 1),
    }


def workload_records(runtime_binding: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for record in runtime_binding.get("workload_records", []):
        records.append(
            {
                "workload_id": record["workload_id"],
                "device_binding_state": "planned_not_allocated",
                "input_ids_device_buffer": device_buffer(record, "input_ids"),
                "attention_mask_device_buffer": device_buffer(
                    record,
                    "attention_mask",
                ),
                "output_ids_device_buffer": device_buffer(record, "output_ids"),
                "copy_plan": {
                    "host_to_device_buffers": ["input_ids", "attention_mask"],
                    "device_to_host_buffers": ["output_ids"],
                    "predecode_initialized_buffers": ["output_ids"],
                },
                "scalar_bindings": record["scalar_bindings"],
            }
        )
    return records

def copy_and_verify_buffer(
    *,
    runtime: Any,
    ctx: Any,
    host_bytes: bytes,
) -> int:
    host = ctypes.create_string_buffer(host_bytes, len(host_bytes))
    dev_ptr = runtime.device_malloc_ctx(ctx, len(host_bytes))
    if not dev_ptr:
        raise RuntimeError(f"device_malloc failed for {len(host_bytes)} bytes")
    status = runtime.copy_to_device_ctx(
        ctx,
        dev_ptr,
        ctypes.cast(host, ctypes.c_void_p),
        len(host_bytes),
    )
    if status != 0:
        runtime.device_free_ctx(ctx, dev_ptr)
        raise RuntimeError(f"copy_to_device failed: {status}")
    out = ctypes.create_string_buffer(len(host_bytes))
    status = runtime.copy_from_device_ctx(
        ctx,
        ctypes.cast(out, ctypes.c_void_p),
        dev_ptr,
        len(host_bytes),
    )
    if status != 0:
        runtime.device_free_ctx(ctx, dev_ptr)
        raise RuntimeError(f"copy_from_device failed: {status}")
    if out.raw != host_bytes:
        runtime.device_free_ctx(ctx, dev_ptr)
        raise RuntimeError("copy verification mismatch")
    return int(dev_ptr)

def run_cuda_probe(
    *,
    runtime_binding: dict[str, Any],
    host_runtime: Path,
    device: int,
) -> dict[str, Any]:
    if runtime_binding.get("status") != "runtime_input_binding_plan_ready":
        return {
            "mode": "token_buffer_copy",
            "status": "skipped",
            "reason": "runtime_input_binding_unavailable",
            "runtime_input_binding_status": runtime_binding.get("status"),
        }
    if not host_runtime.is_file():
        return {
            "mode": "token_buffer_copy",
            "status": "skipped",
            "reason": "host_runtime_missing",
            "host_runtime": repo_relative(host_runtime),
        }
    module = load_module(CUDA_WEIGHT_BINDING, "qwen_cuda_weight_binding_runtime")
    runtime = module.load_cuda_runtime(host_runtime)
    ctx = runtime.create_device_context()
    if not ctx:
        return {
            "mode": "token_buffer_copy",
            "status": "fail",
            "reason": "create_device_context_failed",
        }

    ptrs: list[int] = []
    copied = []
    try:
        init_status = runtime.simpler_init(ctx, device, None, 0, None, 0)
        if init_status != 0:
            return {
                "mode": "token_buffer_copy",
                "status": "fail",
                "reason": "simpler_init_failed",
                "return_code": init_status,
            }
        for record in runtime_binding.get("workload_records", []):
            for name, host_bytes in record_host_buffers(record).items():
                ptr = copy_and_verify_buffer(
                    runtime=runtime,
                    ctx=ctx,
                    host_bytes=host_bytes,
                )
                ptrs.append(ptr)
                copied.append(
                    {
                        "workload_id": record["workload_id"],
                        "buffer": name,
                        "byte_count": len(host_bytes),
                    }
                )
    except Exception as exc:  # noqa: BLE001 - artifact records runtime failure.
        return {
            "mode": "token_buffer_copy",
            "status": "fail",
            "reason": "exception",
            "message": str(exc),
        }
    finally:
        for ptr in ptrs:
            runtime.device_free_ctx(ctx, ptr)
        runtime.finalize_device(ctx)
        runtime.destroy_device_context(ctx)

    return {
        "mode": "token_buffer_copy",
        "status": "pass",
        "device": device,
        "host_runtime": repo_relative(host_runtime),
        "copied_buffer_count": len(copied),
        "copied_bytes": sum(item["byte_count"] for item in copied),
        "verified_buffers": copied,
    }

def build_cuda_token_buffer_binding(
    *,
    mode: str = "offline",
    cache_dir: Path | None = None,
    no_cuda_probe: bool = False,
    host_runtime: Path = DEFAULT_HOST_RUNTIME,
    device: int = 0,
) -> dict[str, Any]:
    runtime_binding = load_runtime_input_binding(mode=mode, cache_dir=cache_dir)
    cuda_probe = (
        {
            "mode": "token_buffer_copy",
            "status": "skipped",
            "reason": "disabled_by_no_cuda_probe",
            "host_runtime": repo_relative(host_runtime),
        }
        if no_cuda_probe
        else run_cuda_probe(
            runtime_binding=runtime_binding,
            host_runtime=host_runtime,
            device=device,
        )
    )
    implemented_contracts = ["cuda_token_buffer_plan"]
    if cuda_probe.get("status") == "pass":
        implemented_contracts.append("cuda_token_buffer_allocation_and_copy")
    return {
        "schema_version": 1,
        "kind": "pto_qwen_cuda_token_buffer_binding",
        "status": (
            "cuda_token_buffer_binding_ready"
            if cuda_probe.get("status") == "pass"
            else "token_buffer_binding_plan_ready"
        ),
        "runtime_input_binding_status": runtime_binding.get("status"),
        "workload_records": workload_records(runtime_binding),
        "cuda_probe": cuda_probe,
        "implemented_contracts": implemented_contracts,
        "remaining_runtime_gaps": (
            ["decode_loop_consumes_token_ids"]
            if cuda_probe.get("status") == "pass"
            else ["cuda_token_buffer_allocation", "decode_loop_consumes_token_ids"]
        ),
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["offline", "download", "mock"],
        default="offline",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--host-runtime", type=Path, default=DEFAULT_HOST_RUNTIME)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no-cuda-probe", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_cuda_token_buffer_binding(
        mode=args.mode,
        cache_dir=args.cache_dir,
        no_cuda_probe=args.no_cuda_probe,
        host_runtime=args.host_runtime,
        device=args.device,
    )
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))

if __name__ == "__main__":
    main()
