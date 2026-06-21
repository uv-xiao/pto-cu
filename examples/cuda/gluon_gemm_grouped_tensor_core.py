#!/usr/bin/env python3
"""Structured grouped GEMM boundary harness for Gluon WGMMA work."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Callable

from simpler_setup.gluon_gen import _SUPPORTED_KERNELS
from simpler_setup.kernel_compiler import KernelCompiler


KERNEL_NAME = "gemm_grouped_tensor_core_f16_f32"
DEFAULT_OUTPUT_DIR = Path("tmp/gluon-grouped-tensor-core-local")
GROUPED_GEMM_CASES = [
    {
        "case_name": "two_group_smoke",
        "groups": 2,
        "m": [64, 128],
        "n": [128, 64],
        "k": [32, 32],
    },
    {
        "case_name": "linear_style_grouped",
        "groups": 3,
        "m": [64, 64, 128],
        "n": [128, 64, 128],
        "k": [7168, 7168, 7168],
    },
]
F16_DTYPE_BOUNDARY = {
    "a": "float16",
    "b": "float16",
    "accumulator": "float32",
    "out": "float32",
}
UNSUPPORTED_BOUNDARY = {
    "kind": "gluon_grouped_gemm_source_path_unavailable",
    "expected_failure": (
        "No generated Gluon grouped GEMM WGMMA source path is registered"
    ),
}


def probe_grouped_gemm_api(output_dir: str | Path, arch: str) -> dict[str, Any]:
    output_dir = Path(output_dir)
    result: dict[str, Any] = {
        "status": "failed",
        "reason": "missing grouped GEMM WGMMA source path",
        "candidate_kernel_name": KERNEL_NAME,
        "available_kernel_names": sorted(_SUPPORTED_KERNELS),
        "gluon_language_grouped_attrs": [],
        "hopper_grouped_attrs": [],
    }

    for module_name, key in [
        (
            "triton.experimental.gluon.language",
            "gluon_language_grouped_attrs",
        ),
        (
            "triton.experimental.gluon.language.nvidia.hopper",
            "hopper_grouped_attrs",
        ),
    ]:
        status, module = _import_module_status(module_name)
        result[f"{key}_module"] = status
        if module is not None:
            result[key] = sorted(name for name in dir(module) if _is_grouped_name(name))

    try:
        artifact = KernelCompiler(platform="cuda").generate_gluon_kernel(
            KERNEL_NAME,
            output_dir=output_dir,
            arch=arch,
        )
    except Exception as exc:  # noqa: BLE001 - boundary probes must emit JSON.
        result["source_generation"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": _clean_text(str(exc)),
        }
        return result

    result["source_generation"] = {
        "status": "passed",
        "artifact": {
            "arch": artifact.arch,
            "compiler_role": artifact.compiler_role,
            "manifest_path": _relative_path(artifact.manifest_path),
            "source_path": _relative_path(artifact.source_path),
            "source_sha256": artifact.source_sha256,
            "tile_shape": list(artifact.tile_shape),
        },
    }
    return {
        **result,
        "status": "passed",
        "reason": "grouped GEMM source generation detected",
    }


def run_grouped_tensor_core_boundary(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
    api_probe: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved_output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    if resolved_output_dir.is_absolute():
        raise ValueError("--output-dir must be repo-relative")

    probe = (
        probe_grouped_gemm_api(resolved_output_dir, arch)
        if api_probe is None
        else api_probe()
    )
    result = {
        "schema_version": 1,
        "kernel_name": KERNEL_NAME,
        "status": "skipped",
        "artifact": None,
        "arch": arch,
        "grouped_cases": GROUPED_GEMM_CASES,
        "dtype_boundary": F16_DTYPE_BOUNDARY,
        "api_probe": probe,
        "unsupported_boundary": UNSUPPORTED_BOUNDARY,
    }
    if probe["status"] != "passed":
        return {
            **result,
            "reason": _clean_text(str(probe.get("reason", "grouped GEMM API unavailable"))),
        }

    return {
        **result,
        "status": "failed",
        "reason": (
            "grouped GEMM source path was detected, but no reviewed grouped "
            "GEMM runtime correctness path is implemented"
        ),
        "unsupported_boundary": {
            "kind": "gluon_grouped_gemm_correctness_path_unavailable",
            "expected_failure": (
                "Grouped GEMM correctness needs a confirmed source, lowering, "
                "and runtime path before it can be promoted"
            ),
        },
    }


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    require_cuda = False

    try:
        parser = JsonArgumentParser(description=__doc__)
        parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        parser.add_argument("--arch", default="compute_90")
        parser.add_argument(
            "--require-cuda",
            action="store_true",
            help="return a non-zero status when grouped GEMM WGMMA is unavailable",
        )
        args = parser.parse_args(raw_args)
        require_cuda = args.require_cuda

        result = run_grouped_tensor_core_boundary(
            output_dir=args.output_dir,
            arch=args.arch,
        )
    except Exception as exc:
        result = {
            "schema_version": 1,
            "kernel_name": KERNEL_NAME,
            "status": "failed",
            "command": _display_command(raw_args),
            "unsupported_boundary": UNSUPPORTED_BOUNDARY,
            "error_type": type(exc).__name__,
            "error": _clean_text(str(exc)),
        }

    print(json.dumps(result, indent=2, sort_keys=True))

    if result["status"] == "failed":
        return 1
    if result["status"] == "skipped" and require_cuda:
        return 2
    return 0


def _import_module_status(module_name: str) -> tuple[dict[str, Any], Any | None]:
    stderr_capture = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr_capture):
            module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - import probes must emit JSON.
        return {
            "imported": False,
            "error_type": type(exc).__name__,
            "error": _clean_text(str(exc)),
            **_captured_stderr_payload(stderr_capture.getvalue()),
        }, None
    return {
        "imported": True,
        **_captured_stderr_payload(stderr_capture.getvalue()),
    }, module


def _captured_stderr_payload(stderr_text: str) -> dict[str, str]:
    sanitized_stderr = _clean_text(stderr_text).strip()
    if not sanitized_stderr:
        return {}
    return {"stderr": sanitized_stderr}


def _is_grouped_name(name: str) -> bool:
    lowered = name.lower()
    return "group" in lowered or "grouped" in lowered


def _relative_path(path: str | Path) -> str:
    path = Path(path)
    if path.is_absolute():
        try:
            path = path.relative_to(Path.cwd())
        except ValueError:
            path = Path(path.name)
    return path.as_posix()


def _clean_text(text: str) -> str:
    cwd = Path.cwd().as_posix()
    home = Path.home().as_posix()
    return text.replace(cwd, ".").replace(home, "~")


def _display_command(raw_args: list[str]) -> str:
    return " ".join(["gluon_gemm_grouped_tensor_core.py", *raw_args])


if __name__ == "__main__":
    raise SystemExit(main())
