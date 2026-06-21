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
    sanitized = re.sub(r"/home/[^\s:'\"]+", "<path>", sanitized)
    return sanitized


def _sanitize_command_arg(arg: Any) -> str:
    text = _sanitize_path_text(str(arg))
    marker = "examples/cuda/"
    if marker in text:
        return text[text.index(marker) :]
    if text.startswith("/"):
        return f"<path>/{Path(text).name}"
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


def _import_module_status(module_name: str) -> tuple[dict[str, Any], Any | None]:
    try:
        module = importlib.import_module(module_name)
    except BaseException as exc:  # noqa: BLE001 - import probes must not crash.
        status = {"imported": False, **_error_payload(exc)}
        return status, None
    return {"imported": True}, module


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

    if not torch_status["imported"]:
        missing.extend(["torch", "cuda_available", "hopper_device"])
        torch_status["cuda_available"] = False
        return torch_status, cuda_status, missing

    try:
        cuda_available = bool(torch_module.cuda.is_available())
    except BaseException as exc:  # noqa: BLE001 - CUDA probes must emit JSON.
        torch_status["cuda_available"] = False
        torch_status["cuda_error"] = _error_payload(exc)
        missing.extend(["cuda_available", "hopper_device"])
        return torch_status, cuda_status, missing

    torch_status["cuda_available"] = cuda_available
    if not cuda_available:
        missing.extend(["cuda_available", "hopper_device"])
        return torch_status, cuda_status, missing

    try:
        device_count = int(torch_module.cuda.device_count())
    except BaseException as exc:  # noqa: BLE001 - CUDA probes must emit JSON.
        cuda_status["device_count_error"] = _error_payload(exc)
        missing.append("hopper_device")
        return torch_status, cuda_status, missing

    cuda_status["device_count"] = device_count
    if device_count <= 0:
        missing.append("hopper_device")
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
            missing.append("hopper_device")
    except BaseException as exc:  # noqa: BLE001 - CUDA probes must emit JSON.
        cuda_status["selected_device_error"] = _error_payload(exc)
        missing.append("hopper_device")

    return torch_status, cuda_status, missing


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
