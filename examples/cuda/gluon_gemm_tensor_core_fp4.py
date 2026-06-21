#!/usr/bin/env python3
"""Structured FP4 API boundary harness for generated Gluon WGMMA GEMM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable


KERNEL_NAME = "gemm_tensor_core_tiled_fp4_f32"
DEFAULT_OUTPUT_DIR = Path("tmp/gluon-tensor-core-fp4-local")
TILED_TENSOR_CORE_TILE = (64, 128, 32)
FP4_DTYPE_BOUNDARY = {
    "a": "torch.float4_e2m1fn_x2 / Gluon FP4 WGMMA dtype unavailable",
    "b": "torch.float4_e2m1fn_x2 / Gluon FP4 WGMMA dtype unavailable",
    "accumulator": "float32",
    "out": "float32",
}
UNSUPPORTED_BOUNDARY = {
    "kind": "gluon_fp4_dtype_api_unavailable",
    "expected_failure": "Torch exposes packed FP4, but no Gluon FP4 WGMMA dtype is available",
}


def probe_fp4_dtypes() -> dict:
    result = {
        "status": "failed",
        "gl_dtype": None,
        "torch_dtype": "float4_e2m1fn_x2",
        "gl_fp4_attrs": [],
        "gl_fp4_attr_details": {},
        "torch_fp4_attrs": [],
    }

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local env
        return {**result, "reason": f"torch import failed: {_clean_text(str(exc))}"}

    result["torch_fp4_attrs"] = sorted(
        name
        for name in dir(torch)
        if "float4" in name.lower() or "fp4" in name.lower() or "e2m1" in name.lower()
    )
    if not hasattr(torch, "float4_e2m1fn_x2"):
        return {**result, "reason": "missing torch.float4_e2m1fn_x2"}

    try:
        from triton.experimental.gluon import language as gl
    except Exception as exc:  # pragma: no cover - depends on local env
        return {
            **result,
            "reason": f"triton Gluon language import failed: {_clean_text(str(exc))}",
        }

    gl_fp4_attrs = sorted(
        name
        for name in dir(gl)
        if "float4" in name.lower() or "fp4" in name.lower() or "e2m1" in name.lower()
    )
    result["gl_fp4_attrs"] = gl_fp4_attrs
    for attr_name in gl_fp4_attrs:
        value = getattr(gl, attr_name)
        detail = {"repr": _clean_text(repr(value))}
        primitive_bitwidth = getattr(value, "primitive_bitwidth", None)
        if primitive_bitwidth is not None:
            detail["primitive_bitwidth"] = int(primitive_bitwidth)
        result["gl_fp4_attr_details"][attr_name] = detail

    fp4_dtype_candidates = [
        name
        for name in gl_fp4_attrs
        if getattr(getattr(gl, name), "primitive_bitwidth", None) == 4
    ]
    if not fp4_dtype_candidates:
        return {**result, "reason": "missing Gluon FP4 WGMMA dtype API"}

    return {
        **result,
        "status": "passed",
        "gl_dtype": fp4_dtype_candidates[0],
    }


def run_fp4_tensor_core_boundary(
    *,
    output_dir: str | Path | None = None,
    arch: str = "compute_90",
    m: int = 64,
    n: int = 128,
    k: int = 32,
    atol: float = 1.0,
    rtol: float = 0.2,
    seed: int = 0,
    device: int = 0,
    dtype_probe: Callable[[], dict] | None = None,
) -> dict:
    resolved_output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    if resolved_output_dir.is_absolute():
        raise ValueError("--output-dir must be repo-relative")

    _validate_shape(m=m, n=n, k=k, tile_shape=TILED_TENSOR_CORE_TILE)
    probe = probe_fp4_dtypes if dtype_probe is None else dtype_probe
    dtype_status = probe()
    result = {
        "schema_version": 1,
        "kernel_name": KERNEL_NAME,
        "artifact": None,
        "arch": arch,
        "dtype_boundary": FP4_DTYPE_BOUNDARY,
        "shape": {"m": m, "n": n, "k": k},
        "tolerance": {"atol": atol, "rtol": rtol},
        "seed": seed,
        "device": device,
        "fp4_dtype_probe": dtype_status,
        "unsupported_boundary": UNSUPPORTED_BOUNDARY,
    }

    if dtype_status["status"] != "passed":
        return {
            **result,
            "status": "skipped",
            "reason": _clean_text(
                str(dtype_status.get("reason", "missing Gluon FP4 WGMMA dtype API"))
            ),
        }

    return {
        **result,
        "status": "failed",
        "reason": "Gluon FP4 dtype was detected, but no generated FP4 WGMMA source is registered",
        "unsupported_boundary": {
            "kind": "gluon_fp4_wgmma_generator_unavailable",
            "expected_failure": (
                "FP4 WGMMA source generation requires a confirmed Gluon FP4 "
                "operand dtype mapping"
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
        parser.add_argument("--m", type=int, default=64)
        parser.add_argument("--n", type=int, default=128)
        parser.add_argument("--k", type=int, default=32)
        parser.add_argument("--atol", type=float, default=1.0)
        parser.add_argument("--rtol", type=float, default=0.2)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--device", type=int, default=0)
        parser.add_argument(
            "--require-cuda",
            action="store_true",
            help="return a non-zero status when FP4 WGMMA is unavailable",
        )
        args = parser.parse_args(raw_args)
        require_cuda = args.require_cuda

        result = run_fp4_tensor_core_boundary(
            output_dir=args.output_dir,
            arch=args.arch,
            m=args.m,
            n=args.n,
            k=args.k,
            atol=args.atol,
            rtol=args.rtol,
            seed=args.seed,
            device=args.device,
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


def _validate_shape(*, m: int, n: int, k: int, tile_shape: tuple[int, int, int]) -> None:
    block_m, block_n, block_k = tile_shape
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n, and k must be positive")
    if k % block_k != 0:
        raise ValueError(f"expected k divisible by {block_k}, got {k}")
    if m % block_m != 0 or n % block_n != 0:
        raise ValueError(
            f"expected m,n divisible by tile ({block_m}, {block_n}), got ({m}, {n})"
        )


def _clean_text(text: str) -> str:
    cwd = Path.cwd().as_posix()
    home = Path.home().as_posix()
    return text.replace(cwd, ".").replace(home, "~")


def _display_command(raw_args: list[str]) -> str:
    return " ".join(["gluon_gemm_tensor_core_fp4.py", *raw_args])


if __name__ == "__main__":
    raise SystemExit(main())
