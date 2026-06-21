#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Structured preflight for Triton/Gluon Hopper WGMMA API availability."""

from __future__ import annotations

import argparse
import contextlib
import io
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
REQUIRED_HOPPER_IMPORTS = {
    "TensorDescriptor": (
        "triton.experimental.gluon.nvidia.hopper",
        "TensorDescriptor",
    ),
    "warpgroup_mma": (
        "triton.experimental.gluon.language.nvidia.hopper",
        "warpgroup_mma",
    ),
    "warpgroup_mma_wait": (
        "triton.experimental.gluon.language.nvidia.hopper",
        "warpgroup_mma_wait",
    ),
    "mbarrier": ("triton.experimental.gluon.language.nvidia.hopper", "mbarrier"),
    "tma": ("triton.experimental.gluon.language.nvidia.hopper", "tma"),
    "fence_async_shared": (
        "triton.experimental.gluon.language.nvidia.hopper",
        "fence_async_shared",
    ),
}
REQUIRED_GL_ATTRS = ("NVMMASharedLayout", "NVMMADistributedLayout", "bfloat16")
HOPPER_COMPUTE_MAJOR = 9
PREFERRED_FP8_GL_ATTRS = (
    "float8e4nv",
    "float8e4b15",
    "float8e4b8",
    "float8e5",
    "float8e5b16",
)
PREFERRED_FP4_GL_ATTRS = (
    "float4_e2m1fn_x2",
    "float4_e2m1fn",
    "float4e2m1",
    "fp4_to_fp",
)


def _sanitize_path_text(text: str) -> str:
    sanitized = str(text)
    try:
        cwd = str(Path.cwd())
        if cwd:
            sanitized = sanitized.replace(cwd, ".")
    except OSError:
        pass
    sanitized = re.sub(
        r"(?:/[^\s:'\"]+)+/examples/cuda/",
        "<path>/examples/cuda/",
        sanitized,
    )
    sanitized = re.sub(
        r"(?:/[^\s:'\"]+)+/site-packages/",
        "<path>/site-packages/",
        sanitized,
    )
    sanitized = re.sub(r"/private/home/[^\s:'\"]+", "<path>", sanitized)
    sanitized = re.sub(r"/home/[^\s:'\"]+", "<path>", sanitized)
    sanitized = re.sub(r"/data/[^\s:'\"]+", "<path>", sanitized)
    return sanitized


def _sanitize_command_arg(arg: Any) -> str:
    raw = str(arg)
    text = _sanitize_path_text(raw)
    marker = "examples/cuda/"
    if marker in text:
        return text[text.index(marker) :]
    if raw.startswith("/"):
        return f"<path>/{Path(raw).name}"
    return text


def _sanitize_command(command: list[str] | None) -> list[str]:
    if command is None:
        command = sys.argv
    return [_sanitize_command_arg(arg) for arg in command]


def _error_payload(exc: BaseException) -> dict[str, str]:
    return {
        "error_type": type(exc).__name__,
        "error": _sanitize_path_text(str(exc)),
    }


def _captured_stderr_payload(stderr_text: str) -> dict[str, str]:
    sanitized_stderr = _sanitize_path_text(stderr_text).strip()
    if not sanitized_stderr:
        return {}
    return {"stderr": sanitized_stderr}


def _import_module_status(module_name: str) -> tuple[dict[str, Any], Any | None]:
    stderr_capture = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr_capture):
            module = importlib.import_module(module_name)
    except BaseException as exc:  # noqa: BLE001 - import probes must not crash.
        status = {
            "imported": False,
            **_error_payload(exc),
            **_captured_stderr_payload(stderr_capture.getvalue()),
        }
        return status, None
    return {
        "imported": True,
        **_captured_stderr_payload(stderr_capture.getvalue()),
    }, module


def _import_attr_status(
    module_name: str,
    attr_name: str,
) -> tuple[dict[str, Any], Any | None]:
    module_status, module = _import_module_status(module_name)
    if not module_status["imported"]:
        return module_status, None
    try:
        value = getattr(module, attr_name)
    except BaseException as exc:  # noqa: BLE001 - import probes must not crash.
        status = {"imported": False, **_error_payload(exc)}
        return status, None
    return {"imported": True}, value


def _attr_status(module: Any | None, attr_name: str) -> dict[str, Any]:
    if module is None:
        return {"present": False, "error": "Gluon language module unavailable"}
    try:
        getattr(module, attr_name)
    except BaseException as exc:  # noqa: BLE001 - attribute probes must not crash.
        return {"present": False, **_error_payload(exc)}
    return {"present": True}


def _torch_status() -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    torch_status, torch_module = _import_module_status("torch")
    cuda_status: dict[str, Any] = {"device_count": 0, "selected_device": None}
    missing: list[str] = []

    def mark_missing(*markers: str) -> None:
        for marker in markers:
            if marker not in missing:
                missing.append(marker)

    if not torch_status["imported"]:
        mark_missing("torch", "cuda_available", "hopper_device")
        torch_status["cuda_available"] = False
        return torch_status, cuda_status, missing

    try:
        cuda_available = bool(torch_module.cuda.is_available())
    except BaseException as exc:  # noqa: BLE001 - CUDA probes must emit JSON.
        torch_status["cuda_available"] = False
        torch_status["cuda_error"] = _error_payload(exc)
        mark_missing("cuda_available", "hopper_device")
        return torch_status, cuda_status, missing

    torch_status["cuda_available"] = cuda_available
    if not cuda_available:
        mark_missing("cuda_available", "hopper_device")
        return torch_status, cuda_status, missing

    try:
        device_count = int(torch_module.cuda.device_count())
    except BaseException as exc:  # noqa: BLE001 - CUDA probes must emit JSON.
        cuda_status["device_count_error"] = _error_payload(exc)
        mark_missing("hopper_device")
        return torch_status, cuda_status, missing

    cuda_status["device_count"] = device_count
    if device_count <= 0:
        mark_missing("hopper_device")
        return torch_status, cuda_status, missing

    try:
        selected_index = int(torch_module.cuda.current_device())
        device_name = str(torch_module.cuda.get_device_name(selected_index))
        capability = list(torch_module.cuda.get_device_capability(selected_index))
        cuda_status["selected_device"] = {
            "index": selected_index,
            "name": device_name,
            "capability": capability,
        }
        if not capability or int(capability[0]) < HOPPER_COMPUTE_MAJOR:
            mark_missing("hopper_device")
    except BaseException as exc:  # noqa: BLE001 - CUDA probes must emit JSON.
        cuda_status["selected_device_error"] = _error_payload(exc)
        mark_missing("hopper_device")

    return torch_status, cuda_status, missing


def _torch_fp8_attrs(torch_module: Any | None) -> list[str]:
    if torch_module is None:
        return []
    return sorted(
        name for name in dir(torch_module) if "float8" in name.lower() or "fp8" in name.lower()
    )


def _torch_fp4_attrs(torch_module: Any | None) -> list[str]:
    if torch_module is None:
        return []
    return sorted(
        name
        for name in dir(torch_module)
        if "float4" in name.lower() or "fp4" in name.lower() or "e2m1" in name.lower()
    )


def _gl_fp8_attr_status(gl_module: Any | None) -> dict[str, dict[str, Any]]:
    attr_names = set(PREFERRED_FP8_GL_ATTRS)
    if gl_module is not None:
        attr_names.update(
            name for name in dir(gl_module) if "float8" in name.lower() or "fp8" in name.lower()
        )

    statuses: dict[str, dict[str, Any]] = {}
    for attr_name in sorted(attr_names):
        status = _attr_status(gl_module, attr_name)
        if status["present"] and gl_module is not None:
            dtype = getattr(gl_module, attr_name)
            primitive_bitwidth = getattr(dtype, "primitive_bitwidth", None)
            if primitive_bitwidth is not None:
                status["primitive_bitwidth"] = int(primitive_bitwidth)
            status["repr"] = _sanitize_path_text(repr(dtype))
        statuses[attr_name] = status
    return statuses


def _gl_fp4_attr_status(gl_module: Any | None) -> dict[str, dict[str, Any]]:
    attr_names = set(PREFERRED_FP4_GL_ATTRS)
    if gl_module is not None:
        attr_names.update(
            name
            for name in dir(gl_module)
            if "float4" in name.lower() or "fp4" in name.lower() or "e2m1" in name.lower()
        )

    statuses: dict[str, dict[str, Any]] = {}
    for attr_name in sorted(attr_names):
        status = _attr_status(gl_module, attr_name)
        if status["present"] and gl_module is not None:
            value = getattr(gl_module, attr_name)
            primitive_bitwidth = getattr(value, "primitive_bitwidth", None)
            if primitive_bitwidth is not None:
                status["primitive_bitwidth"] = int(primitive_bitwidth)
            status["repr"] = _sanitize_path_text(repr(value))
        statuses[attr_name] = status
    return statuses


def _aggregate_status(
    *,
    require_cuda: bool,
    cuda_required_missing: list[str],
    missing_required: list[str],
    runtime_errors: list[str],
) -> tuple[str, str]:
    if runtime_errors:
        return "failed", "runtime errors: " + ", ".join(runtime_errors)

    missing = cuda_required_missing + missing_required
    if missing:
        if require_cuda:
            return "failed", "missing required CUDA or WGMMA APIs: " + ", ".join(missing)
        return "skipped", "missing optional CUDA or WGMMA APIs: " + ", ".join(missing)

    return "passed", "all required WGMMA APIs are available"


def collect_preflight(
    *,
    command: list[str] | None = None,
    require_cuda: bool = False,
) -> dict[str, Any]:
    runtime_errors: list[str] = []
    torch_status, cuda_status, cuda_missing = _torch_status()
    torch_module = None
    if torch_status["imported"]:
        _, torch_module = _import_module_status("torch")

    triton_status, _triton = _import_module_status("triton")
    if triton_status["imported"]:
        triton_status["version"] = str(getattr(_triton, "__version__", "unknown"))

    gluon_status, _gluon = _import_module_status("triton.experimental.gluon")

    gl_status, gl_module = _import_module_status("triton.experimental.gluon.language")

    hopper_status: dict[str, dict[str, Any]] = {}
    missing_required: list[str] = []
    for api_name, (module_name, attr_name) in REQUIRED_HOPPER_IMPORTS.items():
        status, _value = _import_attr_status(module_name, attr_name)
        hopper_status[api_name] = status
        if not status["imported"]:
            missing_required.append(api_name)

    gl_attr_status: dict[str, dict[str, Any]] = {}
    for attr_name in REQUIRED_GL_ATTRS:
        status = _attr_status(gl_module, attr_name)
        gl_attr_status[attr_name] = status
        if not status["present"]:
            missing_required.append(f"gl.{attr_name}")

    gl_fp8_attrs = _gl_fp8_attr_status(gl_module)
    gl_fp4_attrs = _gl_fp4_attr_status(gl_module)

    if not triton_status["imported"]:
        missing_required.append("triton")
    if not gluon_status["imported"] or not gl_status["imported"]:
        missing_required.append("gluon")

    status, reason = _aggregate_status(
        require_cuda=require_cuda,
        cuda_required_missing=cuda_missing,
        missing_required=missing_required,
        runtime_errors=runtime_errors,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "command": _sanitize_command(command),
        "status": status,
        "reason": reason,
        "torch": torch_status,
        "cuda": cuda_status,
        "triton": triton_status,
        "gluon": gluon_status,
        "gl_language": gl_status,
        "hopper": hopper_status,
        "gl_attrs": gl_attr_status,
        "gl_fp8_attrs": gl_fp8_attrs,
        "torch_fp8_attrs": _torch_fp8_attrs(torch_module),
        "gl_fp4_attrs": gl_fp4_attrs,
        "torch_fp4_attrs": _torch_fp4_attrs(torch_module),
        "cuda_required_missing": cuda_missing,
        "missing_required": missing_required,
    }


def main(argv: list[str] | None = None, *, command: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check Triton/Gluon Hopper WGMMA API availability.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="return non-zero when CUDA/Hopper or WGMMA APIs are missing",
    )
    args = parser.parse_args(argv)

    try:
        payload = collect_preflight(
            command=command,
            require_cuda=args.require_cuda,
        )
    except BaseException as exc:  # noqa: BLE001 - CLI must return structured JSON.
        payload = {
            "schema_version": SCHEMA_VERSION,
            "command": _sanitize_command(command),
            "status": "failed",
            **_error_payload(exc),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] == "passed":
        return 0
    if args.require_cuda:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(command=sys.argv))
