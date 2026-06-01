"""ctypes bindings for the CUDA persistent-device runtime."""

from __future__ import annotations

import ctypes
import subprocess
from typing import Any

from simpler_setup.runtime_builder import RuntimeBuilder


class PtoRunTiming(ctypes.Structure):
    _fields_ = [
        ("host_wall_ns", ctypes.c_uint64),
        ("device_wall_ns", ctypes.c_uint64),
    ]


def bind_persistent_runtime(runtime: Any) -> None:
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
    runtime.copy_from_device_ctx.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    runtime.copy_from_device_ctx.restype = ctypes.c_int
    runtime.prepare_callable.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
    runtime.prepare_callable.restype = ctypes.c_int
    runtime.run_prepared.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(PtoRunTiming),
    ]
    runtime.run_prepared.restype = ctypes.c_int
    runtime.unregister_callable.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    runtime.unregister_callable.restype = ctypes.c_int


def load_runtime(*, build_runtime: bool) -> tuple[Any, Any]:
    binaries = RuntimeBuilder(platform="cuda").get_binaries(
        "persistent_device",
        build=build_runtime,
    )
    runtime = ctypes.CDLL(str(binaries.host_path))
    bind_persistent_runtime(runtime)
    return runtime, binaries


def device_name(device: int) -> str:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
                "-i",
                str(device),
            ],
            text=True,
            timeout=5,
        )
    except Exception:  # noqa: BLE001 - nonessential artifact field.
        return "unknown"
    return output.strip().splitlines()[0] if output.strip() else "unknown"

